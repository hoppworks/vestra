//! The deterministic bridge from multi-view inference to durable measured chunks.

use std::collections::{BTreeMap, HashMap};

use rayon::prelude::*;
use vestra_engine::Engine;

use crate::cpp_pr2_f64::optimize_cpp_pr2_pose_graph_f64;
use crate::cpp_pr2_geometry_d::{backproject_frame_cpp_pr2_f32, camera_centre_direction_cpp_pr2};
use crate::{
    AlignmentReport, BackprojectionError, BackprojectionSettings, CameraCalibration, CppPr2Fixture,
    CppPr2Frame, CppPr2StreamBranches, FrameWindow, FusedPoint, FusedSceneChunk, FusedWindowPose,
    MeasuredFrameChunk, MeasuredView, OwnedFrame, PoseGraphEdge, PoseGraphReport, PoseSolution,
    RelativePoseGraph, SceneBundle, SceneBundleError, SimilarityTransform, TsdfSettings,
    WindowMeasuredChunk, WindowSettings, align_overlapping_windows_cpp_pr2,
    backproject_measured_view, camera_centre_direction, fuse_normal_space_tsdf,
    infer_ordered_window, plan_windows, stitch_measured_windows_with_settings,
};

#[derive(Debug, Clone, Copy, Default)]
pub struct ReconstructionSettings {
    pub windows: WindowSettings,
    pub backprojection: BackprojectionSettings,
    /// Capture dense PR #2-compatible raw evidence plus the per-window
    /// confidence percentile needed for deterministic first-owner emission.
    pub cpp_pr2_relative_capture: bool,
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

/// Acceptance gates for a global pose-provider derivative. These settings
/// deliberately validate camera evidence before any raw DA3 point is moved.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlobalPoseFusionSettings {
    /// A window needs several independently registered cameras; three-point
    /// Sim(3) is algebraically possible but unsafe for a product world.
    pub minimum_registered_cameras_per_window: usize,
    /// RMS camera-centre fit divided by the local camera-path RMS extent.
    pub maximum_normalized_camera_rms: f32,
    /// Surface product to derive after the global transforms are accepted.
    pub tsdf: Option<TsdfSettings>,
}

impl Default for GlobalPoseFusionSettings {
    fn default() -> Self {
        Self {
            minimum_registered_cameras_per_window: 6,
            maximum_normalized_camera_rms: 0.15,
            tsdf: Some(TsdfSettings::default()),
        }
    }
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
    #[error("PR #2-relative fusion requires a scene captured with the PR #2 evidence profile")]
    MissingCppPr2CaptureProfile,
    #[error("global pose solution has no registered cameras for window {window_index}")]
    MissingGlobalCameraEvidence { window_index: usize },
    #[error(
        "global pose solution has only {actual} registered cameras for window {window_index}; need at least {minimum}"
    )]
    InsufficientGlobalCameraEvidence {
        window_index: usize,
        actual: usize,
        minimum: usize,
    },
    #[error(
        "global pose fit for window {window_index} is too inaccurate: normalized RMS {normalized_rms:.4} exceeds {maximum:.4}"
    )]
    GlobalCameraFitQuality {
        window_index: usize,
        normalized_rms: f32,
        maximum: f32,
    },
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

/// Derives a separate world in a global pose-provider coordinate system.
///
/// The solution must be published by this exact bundle (and consequently use
/// the same decoded raster contract). Each local DA3 window is fitted to the
/// registered COLMAP camera centres, then all raw points are transformed once
/// and fused. It never changes the relative local product or raw evidence.
pub fn fuse_scene_bundle_with_pose_solution(
    bundle: &SceneBundle,
    pose_solution_hash: &str,
    settings: GlobalPoseFusionSettings,
) -> Result<FusionProgress, ReconstructionError> {
    let solution = bundle.read_pose_solution(pose_solution_hash)?;
    if solution.provider.kind != "colmap" {
        return Err(ReconstructionError::Scene(
            SceneBundleError::InvalidArtifact(
                "only the validated COLMAP global-pose provider is supported".to_owned(),
            ),
        ));
    }
    let manifest = bundle.manifest()?;
    let mut windows = manifest
        .measured_chunk_hashes
        .iter()
        .map(|hash| bundle.read_measured_window(hash))
        .collect::<Result<Vec<_>, _>>()?;
    windows.sort_by_key(|window| window.window.index);
    let poses = windows
        .iter()
        .map(|window| fit_window_to_global_pose(window, &solution, settings))
        .collect::<Result<Vec<_>, _>>()?;
    let fused = emit_windows_at_poses(&windows, &poses, settings.tsdf);
    let points = fused.points.len();
    let chunk_hash = bundle.write_fused_scene_as(
        &fused,
        "colmap-global-active",
        "colmap-global-ba",
        if settings.tsdf.is_some() {
            "tsdf"
        } else {
            "surfel"
        },
        Some(pose_solution_hash.to_owned()),
    )?;
    Ok(FusionProgress {
        chunk_hash,
        aligned_windows: windows.len(),
        points,
    })
}

fn fit_window_to_global_pose(
    window: &WindowMeasuredChunk,
    solution: &PoseSolution,
    settings: GlobalPoseFusionSettings,
) -> Result<SimilarityTransform, ReconstructionError> {
    let global = solution
        .frames
        .iter()
        .map(|frame| {
            (
                frame.frame_index,
                colmap_camera_centre(frame.world_to_camera),
            )
        })
        .collect::<HashMap<_, _>>();
    let pairs = window
        .views
        .iter()
        .filter_map(|view| {
            let local = camera_centre_direction(view.frame_index, view.camera)?;
            global
                .get(&view.frame_index)
                .copied()
                .map(|target| (local.centre_local, target))
        })
        .collect::<Vec<_>>();
    if pairs.is_empty() {
        return Err(ReconstructionError::MissingGlobalCameraEvidence {
            window_index: window.window.index,
        });
    }
    if pairs.len() < settings.minimum_registered_cameras_per_window {
        return Err(ReconstructionError::InsufficientGlobalCameraEvidence {
            window_index: window.window.index,
            actual: pairs.len(),
            minimum: settings.minimum_registered_cameras_per_window,
        });
    }
    let (transform, rms) = crate::stitch::cpp_pr2_similarity_from_pairs(&pairs)?;
    let local_mean = pairs
        .iter()
        .fold([0.0_f32; 3], |mut sum, (local, _)| {
            for axis in 0..3 {
                sum[axis] += local[axis];
            }
            sum
        })
        .map(|value| value / pairs.len() as f32);
    let local_extent = (pairs
        .iter()
        .map(|(local, _)| squared_distance(*local, local_mean))
        .sum::<f32>()
        / pairs.len() as f32)
        .sqrt();
    let normalized_rms = rms / local_extent.max(1e-6);
    if !normalized_rms.is_finite() || normalized_rms > settings.maximum_normalized_camera_rms {
        return Err(ReconstructionError::GlobalCameraFitQuality {
            window_index: window.window.index,
            normalized_rms,
            maximum: settings.maximum_normalized_camera_rms,
        });
    }
    Ok(transform)
}

fn emit_windows_at_poses(
    windows: &[WindowMeasuredChunk],
    poses: &[SimilarityTransform],
    tsdf: Option<TsdfSettings>,
) -> FusedSceneChunk {
    let mut emitted_frames = std::collections::HashSet::new();
    let mut raw_points = Vec::new();
    let mut observations = Vec::new();
    let mut cameras = Vec::new();
    for (window, pose) in windows.iter().zip(poses.iter().copied()) {
        for frame in &window.views {
            if !emitted_frames.insert(frame.frame_index) {
                continue;
            }
            if let Some(camera) = camera_centre_direction(frame.frame_index, frame.camera) {
                cameras.push(pose.apply(camera.centre_local));
            }
            for point in &frame.points {
                if !point.position.iter().all(|value| value.is_finite())
                    || !point.normal.iter().all(|value| value.is_finite())
                    || !point.confidence.is_finite()
                    || point.confidence <= 0.0
                    || !point.radius.is_finite()
                    || point.radius <= 0.0
                {
                    continue;
                }
                let position = pose.apply(point.position);
                if let Some(_) = tsdf {
                    observations.push(crate::TsdfObservation {
                        position,
                        color_srgb: point.color_srgb,
                        confidence: point.confidence,
                        radius: point.radius * pose.scale,
                        frame_index: frame.frame_index as i32,
                    });
                } else {
                    raw_points.push(FusedPoint {
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
        (raw_points, 0.0)
    };
    FusedSceneChunk {
        alignments: Vec::new(),
        pose_graph_edges: Vec::new(),
        pose_graph: None,
        window_poses: windows
            .iter()
            .zip(poses.iter().copied())
            .map(|(window, local_to_world)| FusedWindowPose {
                window_index: window.window.index,
                local_to_world,
            })
            .collect(),
        voxel_size,
        points,
    }
}

fn colmap_camera_centre(world_to_camera: [f64; 12]) -> [f32; 3] {
    let translation = [world_to_camera[3], world_to_camera[7], world_to_camera[11]];
    [
        -(world_to_camera[0] * translation[0]
            + world_to_camera[4] * translation[1]
            + world_to_camera[8] * translation[2]) as f32,
        -(world_to_camera[1] * translation[0]
            + world_to_camera[5] * translation[1]
            + world_to_camera[9] * translation[2]) as f32,
        -(world_to_camera[2] * translation[0]
            + world_to_camera[6] * translation[1]
            + world_to_camera[10] * translation[2]) as f32,
    ]
}

fn squared_distance(left: [f32; 3], right: [f32; 3]) -> f32 {
    (left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2) + (left[2] - right[2]).powi(2)
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
    if windows
        .iter()
        .any(|window| window.cpp_pr2_emission_confidence_threshold.is_none())
    {
        return Err(ReconstructionError::MissingCppPr2CaptureProfile);
    }
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
        let threshold = window
            .cpp_pr2_emission_confidence_threshold
            .expect("validated PR #2 capture profile");
        for frame in &window.views {
            if !emitted_frames.insert(frame.frame_index) {
                continue;
            }
            if let Some(camera) =
                camera_centre_direction_cpp_pr2(frame.frame_index, frame.camera.world_to_camera)
            {
                cameras.push(pose.apply(camera.centre_local));
            }
            for point in &frame.points {
                if !point.position.iter().all(|value| value.is_finite())
                    || !point.confidence.is_finite()
                    || point.confidence < threshold
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
    let profile = std::env::var_os("VESTRA_ORACLE_PROFILE").is_some();
    let started = std::time::Instant::now();
    let windows = cpp_pr2_fixture_windows(fixture)?;
    let windows_at = std::time::Instant::now();
    let alignments = windows
        .par_windows(2)
        .map(|pair| Ok(align_overlapping_windows_cpp_pr2(&pair[1], &pair[0])?))
        .collect::<Result<Vec<_>, ReconstructionError>>()?;
    let seams_at = std::time::Instant::now();
    let sequential = sequential_poses_from_alignments(&alignments);
    let keys = windows
        .iter()
        .enumerate()
        .map(|(index, window)| cpp_pr2_fixture_key_cloud(fixture, index, window))
        .collect::<Vec<_>>();
    let paths = windows
        .iter()
        .enumerate()
        .map(|(index, window)| cpp_pr2_fixture_camera_path(fixture, index, window))
        .collect::<Vec<_>>();
    let evidence_at = std::time::Instant::now();
    let oracle = cpp_pr2_loop_oracle_from_sequential(
        &windows,
        fixture.branches.loop_close,
        &keys,
        &paths,
        sequential,
    )?;
    let loop_at = std::time::Instant::now();
    let poses = oracle
        .optimized_window_poses
        .iter()
        .map(|pose| pose.local_to_world)
        .collect::<Vec<_>>();
    let cloud = emit_cpp_pr2_tsdf_cloud_with_poses(fixture, &windows, alignments, poses)?;
    if profile {
        eprintln!(
            "vestra_closed_loop_profile windows_ms={:.3} seams_ms={:.3} evidence_ms={:.3} loop_and_graph_ms={:.3} emit_and_tsdf_ms={:.3} total_ms={:.3}",
            windows_at.duration_since(started).as_secs_f64() * 1_000.0,
            seams_at.duration_since(windows_at).as_secs_f64() * 1_000.0,
            evidence_at.duration_since(seams_at).as_secs_f64() * 1_000.0,
            loop_at.duration_since(evidence_at).as_secs_f64() * 1_000.0,
            std::time::Instant::now()
                .duration_since(loop_at)
                .as_secs_f64()
                * 1_000.0,
            started.elapsed().as_secs_f64() * 1_000.0,
        );
    }
    Ok(cloud)
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
    let profile = std::env::var_os("VESTRA_ORACLE_PROFILE").is_some();
    let started = std::time::Instant::now();
    let raw = emit_cpp_pr2_cloud_with_poses(fixture, windows, alignments, poses.clone())?;
    let raw_at = std::time::Instant::now();
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
    let tsdf_input_at = std::time::Instant::now();
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
    let tsdf_at = std::time::Instant::now();
    let mut frame_owned_points = vec![0_i32; fixture.frame_count];
    for point in &points {
        if let Some(count) = frame_owned_points.get_mut(point.first_observing_frame.max(0) as usize)
        {
            *count = count.saturating_add(1);
        }
    }
    if profile {
        eprintln!(
            "vestra_oracle_profile raw_emit_ms={:.3} tsdf_input_ms={:.3} tsdf_ms={:.3} frame_counts_ms={:.3} total_ms={:.3}",
            raw_at.duration_since(started).as_secs_f64() * 1_000.0,
            tsdf_input_at.duration_since(raw_at).as_secs_f64() * 1_000.0,
            tsdf_at.duration_since(tsdf_input_at).as_secs_f64() * 1_000.0,
            std::time::Instant::now()
                .duration_since(tsdf_at)
                .as_secs_f64()
                * 1_000.0,
            started.elapsed().as_secs_f64() * 1_000.0,
        );
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
    let keys = windows
        .iter()
        .enumerate()
        .map(|(index, window)| cpp_pr2_fixture_key_cloud(fixture, index, window))
        .collect::<Vec<_>>();
    let paths = windows
        .iter()
        .enumerate()
        .map(|(index, window)| cpp_pr2_fixture_camera_path(fixture, index, window))
        .collect::<Vec<_>>();
    cpp_pr2_loop_oracle_with_evidence(&windows, fixture.branches.loop_close, &keys, &paths)
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
                .filter_map(|view| {
                    camera_centre_direction_cpp_pr2(view.frame_index, view.camera.world_to_camera)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    cpp_pr2_loop_oracle_with_evidence(windows, loop_close, &keys, &paths)
}

fn cpp_pr2_loop_oracle_with_evidence(
    windows: &[WindowMeasuredChunk],
    loop_close: bool,
    keys: &[Vec<[f32; 3]>],
    paths: &[Vec<crate::CameraCentreDirection>],
) -> Result<CppPr2LoopOracle, ReconstructionError> {
    if keys.len() != windows.len() || paths.len() != windows.len() {
        return Err(ReconstructionError::OracleOutputShape);
    }
    let sequential = cpp_pr2_sequential_window_poses(&windows)?;
    cpp_pr2_loop_oracle_from_sequential(windows, loop_close, keys, paths, sequential)
}

fn cpp_pr2_loop_oracle_from_sequential(
    windows: &[WindowMeasuredChunk],
    loop_close: bool,
    keys: &[Vec<[f32; 3]>],
    paths: &[Vec<crate::CameraCentreDirection>],
    sequential: Vec<SimilarityTransform>,
) -> Result<CppPr2LoopOracle, ReconstructionError> {
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

fn sequential_poses_from_alignments(alignments: &[AlignmentReport]) -> Vec<SimilarityTransform> {
    let mut poses = Vec::with_capacity(alignments.len() + 1);
    poses.push(SimilarityTransform::IDENTITY);
    for report in alignments {
        let previous = *poses.last().expect("first reference pose exists");
        poses.push(previous.compose(report.transform));
    }
    poses
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
        .filter(|point| {
            window
                .cpp_pr2_emission_confidence_threshold
                .is_none_or(|threshold| point.confidence >= threshold)
        })
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

/// Builds the loop key cloud exactly where PR #2 builds `WindowRec::key`: from
/// the first-owned, confidence-percentile-selected emission cloud. Generic
/// measured windows intentionally do not retain this window-local percentile,
/// but the recorded VPS oracle does.
fn cpp_pr2_fixture_key_cloud(
    fixture: &CppPr2Fixture,
    window_index: usize,
    window: &WindowMeasuredChunk,
) -> Vec<[f32; 3]> {
    let views = &fixture.window_views[window_index];
    let confidences = views
        .iter()
        .flat_map(|view| view.confidence.iter().copied())
        .collect::<Vec<_>>();
    let threshold = cpp_pr2_percentile(&confidences, fixture.confidence_percentile);
    let first_owned = if window.window.index == 0 {
        0
    } else {
        fixture.windows.overlap.min(window.views.len())
    };
    let owned = window
        .views
        .iter()
        .skip(first_owned)
        .flat_map(|frame| frame.points.iter())
        .filter(|point| point.confidence >= threshold)
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

fn cpp_pr2_fixture_camera_path(
    fixture: &CppPr2Fixture,
    window_index: usize,
    window: &WindowMeasuredChunk,
) -> Vec<crate::CameraCentreDirection> {
    fixture.window_views[window_index]
        .iter()
        .enumerate()
        .filter_map(|(offset, frame)| {
            camera_centre_direction_cpp_pr2(window.window.start + offset, frame.world_to_camera)
        })
        .collect()
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
    // Fixture windows are immutable, independent depth backprojections. Keep
    // their indexed collection order so every following PR #2 seam, loop and
    // first-owner emission operation remains exactly sequential/deterministic.
    fixture
        .window_views
        .par_iter()
        .enumerate()
        .map(|(window_index, views)| {
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
            Ok(WindowMeasuredChunk {
                window: FrameWindow {
                    index: window_index,
                    start,
                    end: start + measured_views.len(),
                },
                views: measured_views,
                cpp_pr2_emission_confidence_threshold: Some(cpp_pr2_percentile(
                    &views
                        .iter()
                        .flat_map(|view| view.confidence.iter().copied())
                        .collect::<Vec<_>>(),
                    fixture.confidence_percentile,
                )),
            })
        })
        .collect()
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

        let emission_threshold = settings.cpp_pr2_relative_capture.then(|| {
            let confidences = inference
                .views
                .iter()
                .flat_map(|output| output.conf.iter().copied())
                .collect::<Vec<_>>();
            cpp_pr2_percentile(&confidences, 55.0)
        });
        let mut views = Vec::with_capacity(frame_slice.len());
        let mut measured_points = 0;
        for (offset, (frame, output)) in frame_slice.iter().zip(inference.views).enumerate() {
            let rgb = rgb_at_inference_resolution(frame, output.w, output.h);
            let camera = CameraCalibration {
                world_to_camera: output.extrinsics,
                intrinsics: output.intrinsics,
            };
            let points = if settings.cpp_pr2_relative_capture {
                let frame = CppPr2Frame {
                    intrinsics: output.intrinsics,
                    world_to_camera: output.extrinsics,
                    depth: output.depth,
                    confidence: output.conf,
                    rgb_hwc_u8: rgb,
                };
                backproject_frame_cpp_pr2_f32(
                    &frame,
                    output.w,
                    output.h,
                    BackprojectionSettings {
                        minimum_confidence: -f32::MAX,
                        pixel_stride: 1,
                        surfel_radius_pixels: 1.0,
                    },
                )?
            } else {
                backproject_measured_view(
                    MeasuredView {
                        rgb_hwc_u8: &rgb,
                        depth: &output.depth,
                        confidence: &output.conf,
                        width: output.w,
                        height: output.h,
                        camera,
                    },
                    settings.backprojection,
                )?
            };
            measured_points += points.len();
            views.push(MeasuredFrameChunk {
                frame_index: window.start + offset,
                camera,
                points,
            });
        }

        let chunk_hash = bundle.write_measured_window(&WindowMeasuredChunk {
            window,
            views,
            cpp_pr2_emission_confidence_threshold: emission_threshold,
        })?;
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
    fn colmap_w2c_translation_recovers_the_global_camera_centre() {
        let centre =
            colmap_camera_centre([1.0, 0.0, 0.0, -4.0, 0.0, 1.0, 0.0, 2.5, 0.0, 0.0, 1.0, -7.0]);
        assert_eq!(centre, [4.0, -2.5, 7.0]);
    }

    #[test]
    fn global_pose_fit_maps_each_window_camera_path_without_mutating_points() {
        let local_centres = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ];
        let target = |point: [f32; 3]| {
            [
                point[0] * 2.0 + 10.0,
                point[1] * 2.0 - 5.0,
                point[2] * 2.0 + 3.0,
            ]
        };
        let window = WindowMeasuredChunk {
            window: FrameWindow {
                index: 4,
                start: 0,
                end: local_centres.len(),
            },
            views: local_centres
                .iter()
                .enumerate()
                .map(|(frame_index, centre)| MeasuredFrameChunk {
                    frame_index,
                    camera: CameraCalibration {
                        world_to_camera: [
                            1.0, 0.0, 0.0, -centre[0], 0.0, 1.0, 0.0, -centre[1], 0.0, 0.0, 1.0,
                            -centre[2],
                        ],
                        intrinsics: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    },
                    points: Vec::new(),
                })
                .collect(),
            cpp_pr2_emission_confidence_threshold: None,
        };
        let solution = PoseSolution {
            schema: "vestra.pose-solution/v1".to_owned(),
            provider: crate::PoseProvider {
                kind: "colmap".to_owned(),
                version: "test".to_owned(),
                settings_fingerprint: "test".to_owned(),
            },
            raster_fingerprint: "test".to_owned(),
            coordinate_convention: "test".to_owned(),
            frames: local_centres
                .iter()
                .enumerate()
                .map(|(frame_index, local)| {
                    let global = target(*local);
                    crate::PoseFrame {
                        frame_index,
                        image_name: format!("frame-{frame_index:06}.ppm"),
                        registered: true,
                        world_to_camera: [
                            1.0,
                            0.0,
                            0.0,
                            -f64::from(global[0]),
                            0.0,
                            1.0,
                            0.0,
                            -f64::from(global[1]),
                            0.0,
                            0.0,
                            1.0,
                            -f64::from(global[2]),
                        ],
                    }
                })
                .collect(),
            diagnostics: crate::PoseDiagnostics {
                input_frames: local_centres.len(),
                registered_frames: local_centres.len(),
                duplicate_images: 0,
            },
        };
        let transform =
            fit_window_to_global_pose(&window, &solution, GlobalPoseFusionSettings::default())
                .unwrap();
        assert!((transform.scale - 2.0).abs() < 1e-5);
        assert_eq!(transform.apply(local_centres[5]), target(local_centres[5]));
    }

    #[test]
    fn cpp_pr2_percentile_uses_linear_interpolation() {
        assert_eq!(cpp_pr2_percentile(&[1.0, 3.0, 5.0, 9.0], 0.0), 1.0);
        assert_eq!(cpp_pr2_percentile(&[1.0, 3.0, 5.0, 9.0], 100.0), 9.0);
        assert_eq!(cpp_pr2_percentile(&[9.0, 1.0, 5.0, 3.0], 50.0), 4.0);
    }

    #[test]
    fn fixture_loop_keys_use_the_same_confidence_selected_emit_cloud_as_pr2() {
        let frame = CppPr2Frame {
            intrinsics: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            world_to_camera: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            depth: vec![1.0, 1.0],
            confidence: vec![0.1, 0.9],
            rgb_hwc_u8: vec![0; 6],
        };
        let fixture = CppPr2Fixture {
            frame_count: 1,
            width: 2,
            height: 1,
            windows: WindowSettings {
                chunk_size: 2,
                overlap: 1,
            },
            confidence_percentile: 50.0,
            point_size: 1.0,
            minimum_overlap_points: 3,
            branches: CppPr2StreamBranches::default(),
            window_views: vec![vec![frame]],
        };
        let window = WindowMeasuredChunk {
            window: FrameWindow {
                index: 0,
                start: 0,
                end: 1,
            },
            views: vec![MeasuredFrameChunk {
                frame_index: 0,
                camera: CameraCalibration {
                    world_to_camera: fixture.window_views[0][0].world_to_camera,
                    intrinsics: fixture.window_views[0][0].intrinsics,
                },
                points: vec![
                    MeasuredPoint {
                        position: [0.0, 0.0, 1.0],
                        normal: [0.0, 0.0, 1.0],
                        color_srgb: [0; 3],
                        confidence: 0.1,
                        radius: 1.0,
                        source_pixel: [0, 0],
                    },
                    MeasuredPoint {
                        position: [1.0, 0.0, 1.0],
                        normal: [0.0, 0.0, 1.0],
                        color_srgb: [0; 3],
                        confidence: 0.9,
                        radius: 1.0,
                        source_pixel: [1, 0],
                    },
                ],
            }],
            cpp_pr2_emission_confidence_threshold: None,
        };
        assert_eq!(
            cpp_pr2_fixture_key_cloud(&fixture, 0, &window),
            vec![[1.0, 0.0, 1.0]]
        );
    }

    #[test]
    fn product_loop_keys_require_the_captured_pr2_confidence_threshold() {
        let camera = CameraCalibration {
            world_to_camera: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            intrinsics: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        };
        let point = |x, confidence| MeasuredPoint {
            position: [x, 0.0, 1.0],
            normal: [0.0, 0.0, 1.0],
            color_srgb: [0; 3],
            confidence,
            radius: 1.0,
            source_pixel: [x as u32, 0],
        };
        let window = WindowMeasuredChunk {
            window: FrameWindow {
                index: 0,
                start: 0,
                end: 1,
            },
            views: vec![MeasuredFrameChunk {
                frame_index: 0,
                camera,
                points: vec![point(0.0, 0.49), point(1.0, 0.5), point(2.0, 0.9)],
            }],
            cpp_pr2_emission_confidence_threshold: Some(0.5),
        };

        assert_eq!(
            cpp_pr2_first_owner_key_cloud(&window, 0),
            vec![[1.0, 0.0, 1.0], [2.0, 0.0, 1.0]]
        );
    }

    #[test]
    fn cpp_pr2_relative_fusion_rejects_legacy_capture_without_evidence_profile() {
        let root = std::env::temp_dir().join(format!(
            "vestra-pr2-profile-required-{}",
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
        bundle
            .write_measured_window(&WindowMeasuredChunk {
                window: FrameWindow {
                    index: 0,
                    start: 0,
                    end: 1,
                },
                views: Vec::new(),
                cpp_pr2_emission_confidence_threshold: None,
            })
            .unwrap();

        assert!(matches!(
            fuse_scene_bundle_cpp_pr2_relative(&bundle, None),
            Err(ReconstructionError::MissingCppPr2CaptureProfile)
        ));
        std::fs::remove_dir_all(root).unwrap();
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
            cpp_pr2_emission_confidence_threshold: None,
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
            cpp_pr2_emission_confidence_threshold: None,
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
            cpp_pr2_emission_confidence_threshold: None,
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
            cpp_pr2_emission_confidence_threshold: None,
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
