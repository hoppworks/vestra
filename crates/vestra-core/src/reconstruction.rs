//! The deterministic bridge from multi-view inference to durable measured chunks.

use vestra_engine::Engine;

use crate::{
    BackprojectionError, BackprojectionSettings, CameraCalibration, FrameWindow,
    MeasuredFrameChunk, MeasuredView, OwnedFrame, SceneBundle, SceneBundleError,
    WindowMeasuredChunk, WindowSettings, backproject_measured_view, infer_ordered_window,
    plan_windows, stitch_measured_windows_with_settings,
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
}

/// Published result of deriving a relative-scale world from the immutable
/// measured windows in a scene bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionProgress {
    pub chunk_hash: String,
    pub aligned_windows: usize,
    pub points: usize,
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
    let mut progress = Vec::with_capacity(windows.len());
    for window in windows {
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
        });
    }
    Ok(progress)
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
