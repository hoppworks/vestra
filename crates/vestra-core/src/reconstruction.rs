//! The deterministic bridge from multi-view inference to durable measured chunks.

use std::collections::{BTreeMap, HashMap};

use vestra_engine::Engine;

use crate::cpp_pr2_f64::optimize_cpp_pr2_pose_graph_f64;
use crate::cpp_pr2_geometry_d::{backproject_frame_cpp_pr2_f32, camera_centre_direction_cpp_pr2};
use crate::{
    AlignmentReport, BackprojectionError, BackprojectionSettings, CameraCalibration, CppPr2Fixture,
    CppPr2Frame, CppPr2StreamBranches, FrameWindow, FusedPoint, FusedSceneChunk, FusedWindowPose,
    MeasuredFrameChunk, MeasuredView, OwnedFrame, PoseGraphEdge, PoseGraphReport,
    RelativePoseGraph, SceneBundle, SceneBundleError, SimilarityTransform, TsdfSettings,
    WindowMeasuredChunk, WindowSettings, align_overlapping_windows_cpp_pr2,
    backproject_measured_view, camera_centre_direction, fuse_normal_space_tsdf,
    infer_ordered_window, plan_windows, stitch_measured_windows_with_settings,
};

#[derive(Debug, Clone, Copy, Default)]
pub struct ReconstructionSettings {
    pub windows: WindowSettings,
    pub backprojection: BackprojectionSettings,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReconstructionProgress {
    pub window: FrameWindow,
    pub chunk_hash: String,
    pub measured_points: usize,
    /// True when a complete immutable checkpoint was already present and was
    /// deliberately reused without executing model inference again.
    pub reused: bool,
}

/// Published result of deriving a relative-scale world from the immutable
/// measured windows in a scene bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionProgress {
    pub chunk_hash: String,
    pub aligned_windows: usize,
    pub points: usize,
}

/// Reference-only pre-voxel emission produced from a `VPS1` fixture using the
/// pinned PR #2 schedule, confidence threshold, first-owner rule, and
/// sequential local-to-global transforms.
#[derive(Debug, Clone, PartialEq)]
pub struct CppPr2ReferenceCloud {
    pub alignments: Vec<AlignmentReport>,
    pub window_poses: Vec<FusedWindowPose>,
    pub points: Vec<FusedPoint>,
    pub frame_owned_points: Vec<i32>,
}

/// Diagnostic output for the PR #2 closed-loop tier before point fusion.
///
/// It retains sequential and pose-graph-optimized transforms separately. This
/// makes it impossible for a fused voxel cloud to hide a missed loop closure.
#[derive(Debug, Clone, PartialEq)]
pub struct CppPr2LoopOracle {
    pub sequential_window_poses: Vec<FusedWindowPose>,
    pub loop_edges: Vec<PoseGraphEdge>,
    pub optimized_window_poses: Vec<FusedWindowPose>,
    pub pose_graph: Option<PoseGraphReport>,
}

/// Camera trajectory emitted by the PR #2 deferred-emission contract.
#[derive(Debug, Clone, PartialEq)]
pub struct CppPr2Trajectory {
    pub window_mid_frames: Vec<i32>,
    pub window_positions: Vec<[f32; 3]>,
    pub frame_positions: Vec<[f32; 3]>,
    pub frame_forwards: Vec<[f32; 3]>,
}

#[derive(Debug, thiserror::Error)]
pub enum ReconstructionError {
    #[error("engine inference failed: {0}")]
    Engine(#[from] vestra_engine::EngineError),
    #[error("measured geometry construction failed: {0}")]
    Geometry(#[from] BackprojectionError),
    #[error("scene persistence failed: {0}")]
    Scene(#[from] SceneBundleError),
    #[error("engine produced {actual} views for a window of {expected} frames")]
    ViewCount { expected: usize, actual: usize },
    #[error("window planning failed: {0}")]
    Schedule(#[from] crate::ScheduleError),
    #[error("window stitching failed: {0}")]
    Stitch(#[from] crate::StitchError),
    #[error("oracle capture produced inconsistent output dimensions")]
    OracleOutputShape,
    #[error(
        "persisted checkpoint for window {window_index} is incompatible with this reconstruction schedule"
    )]
    CheckpointConflict { window_index: usize },
}

/// Rebuilds the derived world from the bundle's raw evidence in deterministic
/// schedule order. It never mutates or removes a measured chunk.
pub fn fuse_scene_bundle(bundle: &SceneBundle) -> Result<FusionProgress, ReconstructionError> {
    fuse_scene_bundle_with_settings(bundle, crate::StitchSettings::default())
}

/// Testable/settings-aware form of [`fuse_scene_bundle`]. Product callers use
/// the strict default quality gate above.
pub fn fuse_scene_bundle_with_settings(
    bundle: &SceneBundle,
    settings: crate::StitchSettings,
) -> Result<FusionProgress, ReconstructionError> {
    let manifest = bundle.manifest()?;
    let mut windows = manifest
        .measured_chunk_hashes
        .iter()
        .map(|hash| bundle.read_measured_window(hash))
        .collect::<Result<Vec<_>, _>>()?;
    windows.sort_by_key(|chunk| chunk.window.index);
    let fused = stitch_measured_windows_with_settings(&windows, settings)?;
    let chunk_hash = bundle.write_fused_scene(&fused)?;
    Ok(FusionProgress {
        chunk_hash,
        aligned_windows: windows.len(),
        points: fused.points.len(),
    })
}

/// Rebuilds a scene with the PR #2 relative geometry profile. It uses
/// first-owner key clouds, absolute revisit gates, scale-locked loop
/// measurements, and iterative ICP before emitting the final surface.
pub fn fuse_scene_bundle_cpp_pr2_relative(
    bundle: &SceneBundle,
    tsdf: Option<TsdfSettings>,
) -> Result<FusionProgress, ReconstructionError> {
    let manifest = bundle.manifest()?;
    let mut windows = manifest
        .measured_chunk_hashes
        .iter()
        .map(|hash| bundle.read_measured_window(hash))
        .collect::<Result<Vec<_>, _>>()?;
    windows.sort_by_key(|chunk| chunk.window.index);
    let overlap = windows
        .windows(2)
        .map(|pair| pair[0].window.end.saturating_sub(pair[1].window.start))
        .min()
        .unwrap_or(0);
    let solution = cpp_pr2_loop_oracle_for_windows(&windows, overlap, true)?;
    let poses = solution
        .optimized_window_poses
        .iter()
        .map(|pose| pose.local_to_world)
        .collect::<Vec<_>>();
    let mut emitted_frames = std::collections::HashSet::new();
    let mut points = Vec::new();
    let mut observations = Vec::new();
    let mut cameras = Vec::new();
    for (window, pose) in windows.iter().zip(poses) {
        for frame in &window.views {
            if !emitted_frames.insert(frame.frame_index) {
                continue;
            }
            if let Some(camera) = camera_centre_direction(frame.frame_index, frame.camera) {
                cameras.push(pose.apply(camera.centre_local));
            }
            for point in &frame.points {
                if !point.position.iter().all(|value| value.is_finite())
                    || !point.confidence.is_finite()
                    || point.confidence <= 0.0
                    || !point.radius.is_finite()
                    || point.radius <= 0.0
                {
                    continue;
                }
                let position = pose.apply(point.position);
                if tsdf.is_some() {
                    observations.push(crate::TsdfObservation {
                        position,
                        color_srgb: point.color_srgb,
                        confidence: point.confidence,
                        radius: point.radius * pose.scale,
                        frame_index: frame.frame_index as i32,
                    });
                } else {
                    points.push(FusedPoint {
                        position,
                        normal: pose.rotate(point.normal),
                        color_srgb: point.color_srgb,
                        confidence: point.confidence,
                        radius: point.radius * pose.scale,
                        first_observing_frame: frame.frame_index as i32,
                        contributors: 1,
                    });
                }
            }
        }
    }
    let (points, voxel_size) = if let Some(settings) = tsdf {
        let points = fuse_normal_space_tsdf(&observations, &cameras, settings)
            .into_iter()
            .map(|surfel| FusedPoint {
                position: surfel.position,
                normal: surfel.normal,
                color_srgb: surfel.color_srgb,
                confidence: 1.0,
                radius: surfel.radius,
                first_observing_frame: surfel.first_observing_frame,
                contributors: surfel.contributors,
            })
            .collect::<Vec<_>>();
        let voxel_size = points
            .first()
            .map(|point| point.radius / 0.6)
            .unwrap_or(0.0);
        (points, voxel_size)
    } else {
        (points, 0.0)
    };
    let alignments = windows
        .windows(2)
        .map(|pair| Ok(align_overlapping_windows_cpp_pr2(&pair[1], &pair[0])?))
        .collect::<Result<Vec<_>, ReconstructionError>>()?;
    let fused = FusedSceneChunk {
        alignments,
        pose_graph_edges: solution.loop_edges,
        pose_graph: solution.pose_graph,
        window_poses: solution.optimized_window_poses,
        voxel_size,
        points,
    };
    let points = fused.points.len();
    let chunk_hash = bundle.write_fused_scene(&fused)?;
    Ok(FusionProgress {
        chunk_hash,
        aligned_windows: windows.len(),
        points,
    })
}

/// Captures the window-scoped model outputs consumed by the pinned C++ PR #2
/// streaming oracle. It deliberately bypasses scene persistence and fusion:
/// this artifact is for an honest pre-voxel differential comparison only.
pub fn capture_cpp_pr2_fixture(
    engine: &mut Engine,
    frames: &[OwnedFrame],
    windows: WindowSettings,
    confidence_percentile: f64,
    point_size: f32,
    minimum_overlap_points: usize,
    branches: CppPr2StreamBranches,
) -> Result<CppPr2Fixture, ReconstructionError> {
    let schedule = plan_windows(frames.len(), windows)?;
    let mut window_views = Vec::with_capacity(schedule.len());
    let mut dimensions = None;
    for window in schedule {
        let frame_slice = &frames[window.start..window.end];
        let inference = infer_ordered_window(engine, frame_slice)?;
        if inference.views.len() != frame_slice.len() {
            return Err(ReconstructionError::ViewCount {
                expected: frame_slice.len(),
                actual: inference.views.len(),
            });
        }
        let expected_dimensions = dimensions.get_or_insert((inference.w, inference.h));
        if *expected_dimensions != (inference.w, inference.h) {
            return Err(ReconstructionError::OracleOutputShape);
        }
        window_views.push(
            frame_slice
                .iter()
                .zip(inference.views)
                .map(|(frame, output)| CppPr2Frame {
                    intrinsics: output.intrinsics,
                    world_to_camera: output.extrinsics,
                    depth: output.depth,
                    confidence: output.conf,
                    rgb_hwc_u8: rgb_at_inference_resolution(frame, output.w, output.h),
                })
                .collect(),
        );
    }
    let Some((width, height)) = dimensions else {
        return Err(ReconstructionError::OracleOutputShape);
    };
    Ok(CppPr2Fixture {
        frame_count: frames.len(),
        width,
        height,
        windows,
        confidence_percentile,
        point_size,
        minimum_overlap_points,
        branches,
        window_views,
    })
}

/// Runs an explicit Vestra geometry policy over exactly the window-scoped
/// evidence supplied to the C++ PR #2 stitcher. This keeps oracle experiments
/// separate from product defaults and makes the enabled loop branch visible in
/// durable fixture provenance.
pub fn stitch_cpp_pr2_fixture_with_settings(
    fixture: &CppPr2Fixture,
    settings: crate::StitchSettings,
) -> Result<FusedSceneChunk, ReconstructionError> {
    let windows = cpp_pr2_fixture_windows(fixture)?;
    Ok(stitch_measured_windows_with_settings(&windows, settings)?)
}

/// Runs Vestra's transform-tier estimator over the same evidence as the pinned
/// PR #2 stitcher. A V2 fixture remains a sequential no-op. V3 only enables
/// Vestra's automatic loop probe when the C++ branch was explicitly requested;
/// it does not claim PR #2 loop parity by merely sharing a file format.
pub fn stitch_cpp_pr2_fixture_as_vestra(
    fixture: &CppPr2Fixture,
) -> Result<FusedSceneChunk, ReconstructionError> {
    let settings = crate::StitchSettings {
        minimum_correspondences: fixture.minimum_overlap_points,
        minimum_inlier_ratio: 0.0,
        maximum_normalized_rms_residual: f32::INFINITY,
        minimum_scale: 1e-9,
        maximum_scale: 1e9,
        surface_fusion: crate::SurfaceFusion::Voxel,
        loop_closure: fixture
            .branches
            .loop_close
            .then_some(crate::LoopClosureSettings {
                minimum_window_gap: 3,
                candidate_distance_radii: 80.0,
                minimum_forward_cosine: 0.3,
                match_distance_radii: 4.0,
                minimum_correspondences: 150,
                minimum_inlier_ratio: 0.0,
                maximum_rms_radii: f32::INFINITY,
                information: 5.0,
            }),
    };
    stitch_cpp_pr2_fixture_with_settings(fixture, settings)
}

/// Computes only the sequential PR #2 seam reports. This deliberately avoids
/// Vestra's production outlier policy, global accumulation, and voxel fusion;
/// consumers use it to compare identical local-to-previous-window transforms
/// against the pinned C++ implementation.
pub fn cpp_pr2_fixture_alignment_reports(
    fixture: &CppPr2Fixture,
) -> Result<Vec<AlignmentReport>, ReconstructionError> {
    let windows = cpp_pr2_fixture_windows(fixture)?;
    windows
        .windows(2)
        .map(|pair| Ok(align_overlapping_windows_cpp_pr2(&pair[1], &pair[0])?))
        .collect()
}

/// Reproduces PR #2's base pre-voxel point emission from a fixture. It exists
/// solely for differential validation; production reconstruction retains its
/// stricter quality policy and independently defined surfel representation.
pub fn emit_cpp_pr2_reference_cloud(
    fixture: &CppPr2Fixture,
) -> Result<CppPr2ReferenceCloud, ReconstructionError> {
    let windows = cpp_pr2_fixture_windows(fixture)?;
    let mut alignments = Vec::with_capacity(windows.len().saturating_sub(1));
    let mut poses = Vec::with_capacity(windows.len());
    poses.push(SimilarityTransform::IDENTITY);
    for pair in windows.windows(2) {
        let report = align_overlapping_windows_cpp_pr2(&pair[1], &pair[0])?;
        let previous = *poses.last().expect("first reference pose exists");
        poses.push(previous.compose(report.transform));
        alignments.push(report);
    }
    emit_cpp_pr2_cloud_with_poses(fixture, &windows, alignments, poses)
}

/// Re-emits the immutable fixture evidence after the PR #2 loop oracle has
/// optimized its trajectory. This mirrors the reference's single deferred
/// emission pass: loop correction is applied before, never after, point output.
pub fn emit_cpp_pr2_loop_closed_reference_cloud(
    fixture: &CppPr2Fixture,
) -> Result<CppPr2ReferenceCloud, ReconstructionError> {
    let windows = cpp_pr2_fixture_windows(fixture)?;
    let oracle = cpp_pr2_closed_loop_oracle(fixture)?;
    let alignments = windows
        .windows(2)
        .map(|pair| Ok(align_overlapping_windows_cpp_pr2(&pair[1], &pair[0])?))
        .collect::<Result<Vec<_>, ReconstructionError>>()?;
    let poses = oracle
        .optimized_window_poses
        .into_iter()
        .map(|pose| pose.local_to_world)
        .collect();
    emit_cpp_pr2_cloud_with_poses(fixture, &windows, alignments, poses)
}

/// Applies the PR #2 normal-space TSDF defaults to the loop-optimized
/// first-owner fixture cloud. This is a distinct oracle tier: C++'s VPO1
/// reference must also have been produced with the harness `--tsdf` flag.
pub fn emit_cpp_pr2_loop_closed_tsdf_reference_cloud(
    fixture: &CppPr2Fixture,
) -> Result<CppPr2ReferenceCloud, ReconstructionError> {
    let windows = cpp_pr2_fixture_windows(fixture)?;
    let oracle = cpp_pr2_closed_loop_oracle(fixture)?;
    let alignments = windows
        .windows(2)
        .map(|pair| Ok(align_overlapping_windows_cpp_pr2(&pair[1], &pair[0])?))
        .collect::<Result<Vec<_>, ReconstructionError>>()?;
    let poses = oracle
        .optimized_window_poses
        .iter()
        .map(|pose| pose.local_to_world)
        .collect::<Vec<_>>();
    emit_cpp_pr2_tsdf_cloud_with_poses(fixture, &windows, alignments, poses)
}

/// Applies the PR #2 TSDF profile to either a sequential or loop-closed
/// fixture. This preserves the fixture's recorded geometry branch rather than
/// inventing a loop policy for a sequential control.
pub fn emit_cpp_pr2_tsdf_reference_cloud(
    fixture: &CppPr2Fixture,
) -> Result<CppPr2ReferenceCloud, ReconstructionError> {
    if fixture.branches.loop_close {
        return emit_cpp_pr2_loop_closed_tsdf_reference_cloud(fixture);
    }
    let windows = cpp_pr2_fixture_windows(fixture)?;
    let mut alignments = Vec::with_capacity(windows.len().saturating_sub(1));
    let mut poses = vec![SimilarityTransform::IDENTITY];
    for pair in windows.windows(2) {
        let report = align_overlapping_windows_cpp_pr2(&pair[1], &pair[0])?;
        let previous = *poses.last().expect("sequential origin exists");
        poses.push(previous.compose(report.transform));
        alignments.push(report);
    }
    emit_cpp_pr2_tsdf_cloud_with_poses(fixture, &windows, alignments, poses)
}

/// Reconstructs the PR #2 camera evidence emitted alongside its deferred cloud.
/// It is deliberately separate from point comparison because a trajectory drift
/// can be visually hidden before it changes first-owner point counts.
pub fn cpp_pr2_fixture_trajectory(
    fixture: &CppPr2Fixture,
) -> Result<CppPr2Trajectory, ReconstructionError> {
    let windows = cpp_pr2_fixture_windows(fixture)?;
    let poses = if fixture.branches.loop_close {
        cpp_pr2_closed_loop_oracle(fixture)?
            .optimized_window_poses
            .into_iter()
            .map(|pose| pose.local_to_world)
            .collect::<Vec<_>>()
    } else {
        let mut poses = vec![SimilarityTransform::IDENTITY];
        for pair in windows.windows(2) {
            let report = align_overlapping_windows_cpp_pr2(&pair[1], &pair[0])?;
            let previous = *poses.last().expect("sequential origin exists");
            poses.push(previous.compose(report.transform));
        }
        poses
    };
    let mut window_mid_frames = Vec::with_capacity(windows.len());
    let mut window_positions = Vec::with_capacity(windows.len());
    let mut frame_positions = vec![[0.0; 3]; fixture.frame_count];
    let mut frame_forwards = vec![[0.0; 3]; fixture.frame_count];
    for (window, pose) in windows.iter().zip(&poses) {
        let middle = window.views.len() / 2;
        window_mid_frames.push((window.window.start + middle) as i32);
        let mid = fixture.window_views[window.window.index]
            .get(middle)
            .and_then(|frame| {
                camera_centre_direction_cpp_pr2(window.window.start + middle, frame.world_to_camera)
            });
        window_positions
            .push(mid.map_or(pose.translation, |camera| pose.apply(camera.centre_local)));
        let first_owned = if window.window.index == 0 {
            0
        } else {
            fixture.windows.overlap.min(window.views.len())
        };
        for (local_index, frame) in window.views.iter().enumerate().skip(first_owned) {
            if let Some(camera) = camera_centre_direction_cpp_pr2(
                frame.frame_index,
                fixture.window_views[window.window.index][local_index].world_to_camera,
            ) {
                frame_positions[frame.frame_index] = pose.apply(camera.centre_local);
                frame_forwards[frame.frame_index] =
                    normalize_direction(pose.rotate(camera.forward_local));
            }
        }
    }
    Ok(CppPr2Trajectory {
        window_mid_frames,
        window_positions,
        frame_positions,
        frame_forwards,
    })
}

fn emit_cpp_pr2_tsdf_cloud_with_poses(
    fixture: &CppPr2Fixture,
    windows: &[WindowMeasuredChunk],
    alignments: Vec<AlignmentReport>,
    poses: Vec<SimilarityTransform>,
) -> Result<CppPr2ReferenceCloud, ReconstructionError> {
    let raw = emit_cpp_pr2_cloud_with_poses(fixture, windows, alignments, poses.clone())?;
    let cameras = first_owner_camera_centres(windows, &poses);
    let observations = raw
        .points
        .iter()
        .map(|point| crate::TsdfObservation {
            position: point.position,
            color_srgb: point.color_srgb,
            confidence: point.confidence,
            radius: point.radius,
            frame_index: point.first_observing_frame,
        })
        .collect::<Vec<_>>();
    let points = fuse_normal_space_tsdf(&observations, &cameras, TsdfSettings::default())
        .into_iter()
        .map(|surfel| FusedPoint {
            position: surfel.position,
            normal: surfel.normal,
            color_srgb: surfel.color_srgb,
            confidence: 1.0,
            radius: surfel.radius,
            first_observing_frame: surfel.first_observing_frame,
            contributors: surfel.contributors,
        })
        .collect::<Vec<_>>();
    let mut frame_owned_points = vec![0_i32; fixture.frame_count];
    for point in &points {
        if let Some(count) = frame_owned_points.get_mut(point.first_observing_frame.max(0) as usize)
        {
            *count = count.saturating_add(1);
        }
    }
    Ok(CppPr2ReferenceCloud {
        alignments: raw.alignments,
        window_poses: raw.window_poses,
        points,
        frame_owned_points,
    })
}

fn first_owner_camera_centres(
    windows: &[WindowMeasuredChunk],
    poses: &[SimilarityTransform],
) -> Vec<[f32; 3]> {
    let mut emitted_frames = std::collections::HashSet::new();
    let mut cameras = Vec::new();
    for (window, pose) in windows.iter().zip(poses) {
        for frame in &window.views {
            if emitted_frames.insert(frame.frame_index)
                && let Some(camera) = camera_centre_direction(frame.frame_index, frame.camera)
            {
                cameras.push(pose.apply(camera.centre_local));
            }
        }
    }
    cameras
}

fn normalize_direction(direction: [f32; 3]) -> [f32; 3] {
    let length = direction
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if length.is_finite() && length > 0.0 {
        direction.map(|value| value / length)
    } else {
        [0.0; 3]
    }
}

fn emit_cpp_pr2_cloud_with_poses(
    fixture: &CppPr2Fixture,
    windows: &[WindowMeasuredChunk],
    alignments: Vec<AlignmentReport>,
    poses: Vec<SimilarityTransform>,
) -> Result<CppPr2ReferenceCloud, ReconstructionError> {
    if poses.len() != windows.len() {
        return Err(ReconstructionError::OracleOutputShape);
    }

    let mut points = Vec::new();
    let mut counts = vec![0_i32; fixture.frame_count];
    let mut window_poses = Vec::with_capacity(windows.len());
    for ((window, views), pose) in windows.iter().zip(&fixture.window_views).zip(&poses) {
        window_poses.push(FusedWindowPose {
            window_index: window.window.index,
            local_to_world: *pose,
        });
        let confidences = views
            .iter()
            .flat_map(|view| view.confidence.iter().copied())
            .collect::<Vec<_>>();
        let threshold = cpp_pr2_percentile(&confidences, fixture.confidence_percentile);
        let first_local_frame = if window.window.index == 0 {
            0
        } else {
            fixture.windows.overlap.min(window.views.len())
        };
        for (local_index, frame) in window.views.iter().enumerate().skip(first_local_frame) {
            for point in frame
                .points
                .iter()
                .filter(|point| point.confidence >= threshold)
            {
                let source_pixel = point.source_pixel;
                let pixel = source_pixel[1] as usize * fixture.width + source_pixel[0] as usize;
                let depth = views[local_index].depth[pixel];
                let fx = views[local_index].intrinsics[0];
                let fy = views[local_index].intrinsics[4];
                let mut radius = 0.5 * (depth / fx + depth / fy) * fixture.point_size * pose.scale;
                if !radius.is_finite() || radius <= 0.0 {
                    radius = 1e-4;
                }
                points.push(FusedPoint {
                    position: pose.apply(point.position),
                    normal: [0.0, 0.0, 1.0],
                    color_srgb: point.color_srgb,
                    confidence: point.confidence,
                    radius,
                    first_observing_frame: frame.frame_index as i32,
                    contributors: 1,
                });
                let frame_index = frame.frame_index;
                counts[frame_index] = counts[frame_index]
                    .checked_add(1)
                    .expect("reference point count fits i32 fixture contract");
            }
        }
    }
    Ok(CppPr2ReferenceCloud {
        alignments,
        window_poses,
        points,
        frame_owned_points: counts,
    })
}

/// Runs the PR #2 loop *proposal and correspondence* policy over one recorded
/// fixture. It uses first-owner local key clouds, the reference's absolute
/// relative-scene gates, many-to-one nearest matches, and iterative
/// point-to-plane ICP before accepting a loop edge.
pub fn cpp_pr2_closed_loop_oracle(
    fixture: &CppPr2Fixture,
) -> Result<CppPr2LoopOracle, ReconstructionError> {
    let windows = cpp_pr2_fixture_windows(fixture)?;
    cpp_pr2_loop_oracle_for_windows(
        &windows,
        fixture.windows.overlap,
        fixture.branches.loop_close,
    )
}

/// Solves the PR #2 relative loop-closure trajectory from immutable measured
/// windows. This is the product-facing seam beneath the VPS fixture adapter:
/// callers supply the same window schedule used at inference time and receive
/// optimized poses before any surface emission is allowed.
pub fn cpp_pr2_loop_oracle_for_windows(
    windows: &[WindowMeasuredChunk],
    overlap: usize,
    loop_close: bool,
) -> Result<CppPr2LoopOracle, ReconstructionError> {
    let sequential = cpp_pr2_sequential_window_poses(&windows)?;
    let sequential_window_poses = windows
        .iter()
        .zip(&sequential)
        .map(|(window, &local_to_world)| FusedWindowPose {
            window_index: window.window.index,
            local_to_world,
        })
        .collect::<Vec<_>>();

    if !loop_close || windows.len() <= 3 {
        return Ok(CppPr2LoopOracle {
            sequential_window_poses: sequential_window_poses.clone(),
            loop_edges: Vec::new(),
            optimized_window_poses: sequential_window_poses,
            pose_graph: None,
        });
    }

    let keys = windows
        .iter()
        .map(|window| cpp_pr2_first_owner_key_cloud(window, overlap))
        .collect::<Vec<_>>();
    let paths = windows
        .iter()
        .map(|window| {
            window
                .views
                .iter()
                .filter_map(|view| camera_centre_direction(view.frame_index, view.camera))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut loop_edges = Vec::new();
    for earlier in 0..windows.len() {
        for later in earlier + 3..windows.len() {
            if !cpp_pr2_windows_overlap(
                &paths[earlier],
                &paths[later],
                sequential[earlier],
                sequential[later],
            ) {
                continue;
            }
            let Some(measurement) = cpp_pr2_loop_measurement(
                &keys[earlier],
                &keys[later],
                sequential[earlier],
                sequential[later],
            )?
            else {
                continue;
            };
            loop_edges.push(PoseGraphEdge {
                from: earlier,
                to: later,
                measurement,
                information: 5.0,
                loop_closure: true,
            });
        }
    }

    let mut graph = RelativePoseGraph {
        nodes: sequential.clone(),
        edges: sequential
            .windows(2)
            .enumerate()
            .map(|(from, pair)| PoseGraphEdge {
                from,
                to: from + 1,
                measurement: pair[0]
                    .inverse()
                    .expect("sequential PR #2 transform is invertible")
                    .compose(pair[1]),
                information: 1.0,
                loop_closure: false,
            })
            .chain(loop_edges.iter().cloned())
            .collect(),
        fixed: Vec::new(),
    };
    let pose_graph = if loop_edges.is_empty() {
        None
    } else {
        let (nodes, report) = optimize_cpp_pr2_pose_graph_f64(
            &graph.nodes,
            &graph.edges,
            crate::PoseGraphSettings::default(),
        )
        .map_err(crate::StitchError::from)?;
        graph.nodes = nodes;
        Some(report)
    };
    let optimized_window_poses = windows
        .iter()
        .zip(&graph.nodes)
        .map(|(window, &local_to_world)| FusedWindowPose {
            window_index: window.window.index,
            local_to_world,
        })
        .collect();
    Ok(CppPr2LoopOracle {
        sequential_window_poses,
        loop_edges,
        optimized_window_poses,
        pose_graph,
    })
}

fn cpp_pr2_sequential_window_poses(
    windows: &[WindowMeasuredChunk],
) -> Result<Vec<SimilarityTransform>, ReconstructionError> {
    let mut poses = Vec::with_capacity(windows.len());
    if windows.is_empty() {
        return Ok(poses);
    }
    poses.push(SimilarityTransform::IDENTITY);
    for pair in windows.windows(2) {
        let previous = *poses.last().expect("first reference pose exists");
        let report = align_overlapping_windows_cpp_pr2(&pair[1], &pair[0])?;
        poses.push(previous.compose(report.transform));
    }
    Ok(poses)
}

fn cpp_pr2_first_owner_key_cloud(window: &WindowMeasuredChunk, overlap: usize) -> Vec<[f32; 3]> {
    let first_owned = if window.window.index == 0 {
        0
    } else {
        overlap.min(window.views.len())
    };
    let owned = window
        .views
        .iter()
        .skip(first_owned)
        .flat_map(|view| view.points.iter())
        .filter_map(|point| {
            point
                .position
                .iter()
                .all(|value| value.is_finite())
                .then_some(point.position)
        })
        .collect::<Vec<_>>();
    let stride = (owned.len() / 3_000).max(1);
    owned.into_iter().step_by(stride).collect()
}

fn cpp_pr2_windows_overlap(
    earlier: &[crate::CameraCentreDirection],
    later: &[crate::CameraCentreDirection],
    earlier_pose: SimilarityTransform,
    later_pose: SimilarityTransform,
) -> bool {
    earlier.iter().any(|first| {
        later.iter().any(|second| {
            let first_position = earlier_pose.apply(first.centre_local);
            let second_position = later_pose.apply(second.centre_local);
            let delta = [
                first_position[0] - second_position[0],
                first_position[1] - second_position[1],
                first_position[2] - second_position[2],
            ];
            let distance_squared = delta.iter().map(|value| value * value).sum::<f32>();
            let first_direction = earlier_pose.rotate(first.forward_local);
            let second_direction = later_pose.rotate(second.forward_local);
            let cosine = first_direction
                .iter()
                .zip(second_direction)
                .map(|(left, right)| left * right)
                .sum::<f32>();
            distance_squared <= 3.0_f32.powi(2) && cosine >= 0.3
        })
    })
}

fn cpp_pr2_loop_measurement(
    earlier: &[[f32; 3]],
    later: &[[f32; 3]],
    earlier_pose: SimilarityTransform,
    later_pose: SimilarityTransform,
) -> Result<Option<SimilarityTransform>, ReconstructionError> {
    const MATCH_DISTANCE: f32 = 0.30;
    const MINIMUM_CORRESPONDENCES: usize = 150;
    if earlier.len() < MINIMUM_CORRESPONDENCES || later.len() < MINIMUM_CORRESPONDENCES {
        return Ok(None);
    }
    let mut cells: HashMap<(i32, i32, i32), Vec<[f32; 3]>> = HashMap::new();
    for &point in earlier {
        cells
            .entry(cpp_pr2_spatial_cell(point))
            .or_default()
            .push(point);
    }
    let seed = earlier_pose
        .inverse()
        .expect("sequential PR #2 transform is invertible")
        .compose(later_pose);
    let mut pairs = Vec::new();
    for &source in later {
        let seeded = seed.apply(source);
        let base = cpp_pr2_spatial_cell(seeded);
        let mut nearest: Option<([f32; 3], f32)> = None;
        for x in base.0 - 1..=base.0 + 1 {
            for y in base.1 - 1..=base.1 + 1 {
                for z in base.2 - 1..=base.2 + 1 {
                    for &target in cells.get(&(x, y, z)).into_iter().flatten() {
                        let squared = target
                            .iter()
                            .zip(seeded)
                            .map(|(left, right)| (left - right).powi(2))
                            .sum::<f32>();
                        if squared <= MATCH_DISTANCE.powi(2)
                            && nearest.is_none_or(|(_, best)| squared < best)
                        {
                            nearest = Some((target, squared));
                        }
                    }
                }
            }
        }
        if let Some((target, _)) = nearest {
            // PR #2 deliberately permits a target keypoint to serve multiple
            // source points. Enforcing uniqueness here changes loop support.
            pairs.push((source, target));
        }
    }
    if pairs.len() < MINIMUM_CORRESPONDENCES {
        return Ok(None);
    }
    let (transform, _) = crate::stitch::cpp_pr2_similarity_from_pairs(&pairs)?;
    let scale_locked = SimilarityTransform {
        scale: 1.0,
        ..transform
    };
    let refined = crate::icp::refine_point_to_plane(
        later,
        earlier,
        scale_locked,
        crate::icp::IcpSettings::default(),
    );
    Ok(refined
        .filter(|result| result.correspondences >= MINIMUM_CORRESPONDENCES)
        .map(|result| result.transform))
}

fn cpp_pr2_spatial_cell(point: [f32; 3]) -> (i32, i32, i32) {
    (
        (point[0] / 0.30).floor() as i32,
        (point[1] / 0.30).floor() as i32,
        (point[2] / 0.30).floor() as i32,
    )
}

fn cpp_pr2_percentile(values: &[f32], percentile: f64) -> f32 {
    debug_assert!(!values.is_empty());
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    if sorted.len() == 1 {
        return sorted[0];
    }
    let index = percentile / 100.0 * (sorted.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    let fraction = index - lower as f64;
    (f64::from(sorted[lower]) + fraction * f64::from(sorted[upper] - sorted[lower])) as f32
}

fn cpp_pr2_fixture_windows(
    fixture: &CppPr2Fixture,
) -> Result<Vec<WindowMeasuredChunk>, ReconstructionError> {
    let mut windows = Vec::with_capacity(fixture.window_views.len());
    for (window_index, views) in fixture.window_views.iter().enumerate() {
        let start = window_index * (fixture.windows.chunk_size - fixture.windows.overlap);
        let measured_views = views
            .iter()
            .enumerate()
            .map(|(offset, view)| {
                let points = backproject_frame_cpp_pr2_f32(
                    view,
                    fixture.width,
                    fixture.height,
                    BackprojectionSettings {
                        // PR #2 uses dense, finite positive-depth overlap
                        // observations for Sim(3), with confidence as a weight.
                        // Its percentile threshold applies only to final point
                        // emission, not to seam correspondences.
                        minimum_confidence: -f32::MAX,
                        pixel_stride: 1,
                        // Position and confidence are the only inputs to the
                        // sequential fit; C++ radius parity is a later raw tier.
                        surfel_radius_pixels: 1.0,
                    },
                )?;
                Ok(MeasuredFrameChunk {
                    frame_index: start + offset,
                    camera: CameraCalibration {
                        world_to_camera: view.world_to_camera,
                        intrinsics: view.intrinsics,
                    },
                    points,
                })
            })
            .collect::<Result<Vec<_>, ReconstructionError>>()?;
        windows.push(WindowMeasuredChunk {
            window: FrameWindow {
                index: window_index,
                start,
                end: start + measured_views.len(),
            },
            views: measured_views,
        });
    }
    Ok(windows)
}

/// Runs every deterministic window and immediately checkpoints direct evidence.
///
/// Windows intentionally retain their overlapping measured observations. The
/// future fusion phase decides which observations merge; it must never erase
/// the raw evidence by overwriting this layer.
pub fn reconstruct_frames(
    engine: &mut Engine,
    frames: &[OwnedFrame],
    bundle: &SceneBundle,
    settings: ReconstructionSettings,
) -> Result<Vec<ReconstructionProgress>, ReconstructionError> {
    let windows = plan_windows(frames.len(), settings.windows)?;
    let manifest = bundle.manifest()?;
    let mut checkpoints = BTreeMap::new();
    for hash in manifest.measured_chunk_hashes {
        let chunk = bundle.read_measured_window(&hash)?;
        let window_index = chunk.window.index;
        if checkpoints.insert(window_index, (hash, chunk)).is_some() {
            return Err(ReconstructionError::CheckpointConflict { window_index });
        }
    }
    let mut progress = Vec::with_capacity(windows.len());
    for window in windows {
        if let Some((chunk_hash, checkpoint)) = checkpoints.remove(&window.index) {
            progress.push(validated_checkpoint_progress(
                window, checkpoint, chunk_hash,
            )?);
            continue;
        }
        let frame_slice = &frames[window.start..window.end];
        let inference = infer_ordered_window(engine, frame_slice)?;
        if inference.views.len() != frame_slice.len() {
            return Err(ReconstructionError::ViewCount {
                expected: frame_slice.len(),
                actual: inference.views.len(),
            });
        }

        let mut views = Vec::with_capacity(frame_slice.len());
        let mut measured_points = 0;
        for (offset, (frame, output)) in frame_slice.iter().zip(inference.views).enumerate() {
            let rgb = rgb_at_inference_resolution(frame, output.w, output.h);
            let points = backproject_measured_view(
                MeasuredView {
                    rgb_hwc_u8: &rgb,
                    depth: &output.depth,
                    confidence: &output.conf,
                    width: output.w,
                    height: output.h,
                    camera: CameraCalibration {
                        world_to_camera: output.extrinsics,
                        intrinsics: output.intrinsics,
                    },
                },
                settings.backprojection,
            )?;
            measured_points += points.len();
            views.push(MeasuredFrameChunk {
                frame_index: window.start + offset,
                camera: CameraCalibration {
                    world_to_camera: output.extrinsics,
                    intrinsics: output.intrinsics,
                },
                points,
            });
        }

        let chunk_hash = bundle.write_measured_window(&WindowMeasuredChunk { window, views })?;
        progress.push(ReconstructionProgress {
            window,
            chunk_hash,
            measured_points,
            reused: false,
        });
    }
    Ok(progress)
}

fn validated_checkpoint_progress(
    window: FrameWindow,
    checkpoint: WindowMeasuredChunk,
    chunk_hash: String,
) -> Result<ReconstructionProgress, ReconstructionError> {
    let expected_frame_indices = window.start..window.end;
    let valid = checkpoint.window == window
        && checkpoint.views.len() == expected_frame_indices.len()
        && checkpoint
            .views
            .iter()
            .zip(expected_frame_indices)
            .all(|(view, frame_index)| view.frame_index == frame_index);
    if !valid {
        return Err(ReconstructionError::CheckpointConflict {
            window_index: window.index,
        });
    }
    Ok(ReconstructionProgress {
        window,
        chunk_hash,
        measured_points: checkpoint.views.iter().map(|view| view.points.len()).sum(),
        reused: true,
    })
}

/// Maps source RGB to the inference raster without inventing colour values.
/// A nearest source sample is used rather than interpolation so every stored
/// colour remains directly attributable to one captured pixel.
fn rgb_at_inference_resolution(frame: &OwnedFrame, width: usize, height: usize) -> Vec<u8> {
    if frame.width == width && frame.height == height {
        return frame.rgb_hwc_u8.clone();
    }
    let mut out = vec![0; width * height * 3];
    for y in 0..height {
        let source_y = ((y * frame.height) / height).min(frame.height.saturating_sub(1));
        for x in 0..width {
            let source_x = ((x * frame.width) / width).min(frame.width.saturating_sub(1));
            let source = (source_y * frame.width + source_x) * 3;
            let destination = (y * width + x) * 3;
            out[destination..destination + 3]
                .copy_from_slice(&frame.rgb_hwc_u8[source..source + 3]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MeasuredPoint, SceneProvenance};

    #[test]
    fn cpp_pr2_percentile_uses_linear_interpolation() {
        assert_eq!(cpp_pr2_percentile(&[1.0, 3.0, 5.0, 9.0], 0.0), 1.0);
        assert_eq!(cpp_pr2_percentile(&[1.0, 3.0, 5.0, 9.0], 100.0), 9.0);
        assert_eq!(cpp_pr2_percentile(&[9.0, 1.0, 5.0, 3.0], 50.0), 4.0);
    }

    #[test]
    fn cpp_pr2_reference_emission_keeps_first_owned_frames_once() {
        let view = |color: [u8; 3]| CppPr2Frame {
            intrinsics: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            world_to_camera: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            depth: vec![1.0, 2.0, 3.0, 4.0],
            confidence: vec![1.0; 4],
            rgb_hwc_u8: color.into_iter().cycle().take(12).collect(),
        };
        let fixture = CppPr2Fixture {
            frame_count: 3,
            width: 2,
            height: 2,
            windows: WindowSettings {
                chunk_size: 2,
                overlap: 1,
            },
            confidence_percentile: 0.0,
            point_size: 1.0,
            minimum_overlap_points: 3,
            branches: CppPr2StreamBranches::default(),
            window_views: vec![
                vec![view([1, 2, 3]), view([4, 5, 6])],
                vec![view([4, 5, 6]), view([7, 8, 9])],
            ],
        };
        let cloud = emit_cpp_pr2_reference_cloud(&fixture).unwrap();
        assert_eq!(cloud.points.len(), 12);
        assert_eq!(cloud.frame_owned_points, vec![4, 4, 4]);
        assert_eq!(cloud.points[0].color_srgb, [1, 2, 3]);
        assert_eq!(cloud.points[4].color_srgb, [4, 5, 6]);
        assert_eq!(cloud.points[8].color_srgb, [7, 8, 9]);
        assert!((cloud.points[3].radius - 4.0).abs() < 1e-5);
    }

    #[test]
    fn color_resampling_retains_direct_source_pixel_values() {
        let frame = OwnedFrame {
            rgb_hwc_u8: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            width: 2,
            height: 2,
        };
        assert_eq!(
            rgb_at_inference_resolution(&frame, 4, 1),
            vec![1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6]
        );
    }

    #[test]
    fn compatible_checkpoint_is_reused_without_inference() {
        let window = FrameWindow {
            index: 3,
            start: 9,
            end: 11,
        };
        let camera = CameraCalibration {
            world_to_camera: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            intrinsics: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        };
        let checkpoint = WindowMeasuredChunk {
            window,
            views: vec![
                MeasuredFrameChunk {
                    frame_index: 9,
                    camera,
                    points: vec![MeasuredPoint {
                        position: [0.0; 3],
                        normal: [0.0, 0.0, 1.0],
                        color_srgb: [0; 3],
                        confidence: 1.0,
                        radius: 1.0,
                        source_pixel: [0, 0],
                    }],
                },
                MeasuredFrameChunk {
                    frame_index: 10,
                    camera,
                    points: Vec::new(),
                },
            ],
        };
        let progress =
            validated_checkpoint_progress(window, checkpoint, "checkpoint".into()).unwrap();
        assert!(progress.reused);
        assert_eq!(progress.measured_points, 1);
    }

    #[test]
    fn incompatible_checkpoint_is_never_reused() {
        let window = FrameWindow {
            index: 3,
            start: 9,
            end: 11,
        };
        let checkpoint = WindowMeasuredChunk {
            window: FrameWindow { end: 10, ..window },
            views: Vec::new(),
        };
        assert!(matches!(
            validated_checkpoint_progress(window, checkpoint, "checkpoint".into()),
            Err(ReconstructionError::CheckpointConflict { window_index: 3 })
        ));
    }

    #[test]
    fn fusion_reloads_raw_windows_in_schedule_order_and_publishes_a_derived_world() {
        let root = std::env::temp_dir().join(format!(
            "vestra-reconstruction-fuse-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let bundle = SceneBundle::create(
            &root,
            SceneProvenance {
                engine_revision: "test".into(),
                kernel_revision: "test".into(),
                model_fingerprint: "test".into(),
                settings_fingerprint: "test".into(),
            },
        )
        .unwrap();
        let point = |pixel, position| MeasuredPoint {
            position,
            normal: [0.0, 0.0, 1.0],
            color_srgb: [30, 40, 50],
            confidence: 1.0,
            radius: 0.25,
            source_pixel: [pixel, 0],
        };
        let camera = CameraCalibration {
            world_to_camera: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            intrinsics: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        };
        let target = WindowMeasuredChunk {
            window: FrameWindow {
                index: 0,
                start: 0,
                end: 1,
            },
            views: vec![MeasuredFrameChunk {
                frame_index: 0,
                camera,
                points: vec![
                    point(0, [5.0, -3.0, 1.0]),
                    point(1, [5.0, -1.0, 1.0]),
                    point(2, [3.0, -3.0, 1.0]),
                    point(3, [5.0, -3.0, 3.0]),
                ],
            }],
        };
        let source = WindowMeasuredChunk {
            window: FrameWindow {
                index: 1,
                start: 0,
                end: 1,
            },
            views: vec![MeasuredFrameChunk {
                frame_index: 0,
                camera,
                points: vec![
                    point(0, [0.0, 0.0, 0.0]),
                    point(1, [1.0, 0.0, 0.0]),
                    point(2, [0.0, 1.0, 0.0]),
                    point(3, [0.0, 0.0, 1.0]),
                ],
            }],
        };
        bundle.write_measured_window(&source).unwrap();
        bundle.write_measured_window(&target).unwrap();

        let summary = fuse_scene_bundle_with_settings(
            &bundle,
            crate::StitchSettings {
                minimum_correspondences: 3,
                ..crate::StitchSettings::default()
            },
        )
        .unwrap();

        assert_eq!(summary.aligned_windows, 2);
        assert_eq!(summary.points, 4);
        let manifest = bundle.manifest().unwrap();
        assert_eq!(manifest.measured_chunk_hashes.len(), 2);
        assert_eq!(manifest.fused_chunk_hash, Some(summary.chunk_hash));
        std::fs::remove_dir_all(root).unwrap();
    }
}
