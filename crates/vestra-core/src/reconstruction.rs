//! The deterministic bridge from multi-view inference to durable measured chunks.

use std::collections::BTreeMap;

use vestra_engine::Engine;

use crate::{
    AlignmentReport, BackprojectionError, BackprojectionSettings, CameraCalibration, CppPr2Fixture,
    CppPr2Frame, FrameWindow, FusedPoint, FusedSceneChunk, FusedWindowPose, MeasuredFrameChunk,
    MeasuredView, OwnedFrame, SceneBundle, SceneBundleError, SimilarityTransform,
    WindowMeasuredChunk, WindowSettings, align_overlapping_windows_cpp_pr2,
    backproject_measured_view, infer_ordered_window, plan_windows,
    stitch_measured_windows_with_settings,
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
        window_views,
    })
}

/// Runs Vestra's sequential Sim(3) estimator over exactly the window-scoped
/// evidence supplied to the C++ PR #2 stitcher. This is transform-tier oracle
/// evidence; the returned chunk is not a claim of raw-cloud equivalence.
pub fn stitch_cpp_pr2_fixture_as_vestra(
    fixture: &CppPr2Fixture,
) -> Result<FusedSceneChunk, ReconstructionError> {
    let windows = cpp_pr2_fixture_windows(fixture)?;
    let settings = crate::StitchSettings {
        minimum_correspondences: fixture.minimum_overlap_points,
        minimum_inlier_ratio: 0.0,
        maximum_normalized_rms_residual: f32::INFINITY,
        minimum_scale: 1e-9,
        maximum_scale: 1e9,
        loop_closure: None,
    };
    Ok(stitch_measured_windows_with_settings(&windows, settings)?)
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
                let points = backproject_measured_view(
                    MeasuredView {
                        rgb_hwc_u8: &view.rgb_hwc_u8,
                        depth: &view.depth,
                        confidence: &view.confidence,
                        width: fixture.width,
                        height: fixture.height,
                        camera: CameraCalibration {
                            world_to_camera: view.world_to_camera,
                            intrinsics: view.intrinsics,
                        },
                    },
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
