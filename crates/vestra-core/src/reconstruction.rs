//! The deterministic bridge from multi-view inference to durable measured chunks.

use vestra_engine::Engine;

use crate::{
    BackprojectionError, BackprojectionSettings, CameraCalibration, FrameWindow,
    MeasuredFrameChunk, MeasuredView, OwnedFrame, SceneBundle, SceneBundleError,
    WindowMeasuredChunk, WindowSettings, backproject_measured_view, infer_ordered_window,
    plan_windows,
};

#[derive(Debug, Clone, Copy)]
pub struct ReconstructionSettings {
    pub windows: WindowSettings,
    pub backprojection: BackprojectionSettings,
}

impl Default for ReconstructionSettings {
    fn default() -> Self {
        Self {
            windows: WindowSettings::default(),
            backprojection: BackprojectionSettings::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReconstructionProgress {
    pub window: FrameWindow,
    pub chunk_hash: String,
    pub measured_points: usize,
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
}
