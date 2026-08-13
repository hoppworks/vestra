//! Vestra's reconstruction job boundary.
//!
//! This crate owns videos, windows, scene data, and reconstruction state. Model
//! internals remain behind the `vestra-engine` API.

use serde::{Deserialize, Serialize};
use vestra_engine::{Engine, EngineError, MultiViewInferOut, ViewInput};

pub mod export;
pub mod geometry;
pub mod pose_graph;
pub mod reconstruction;
pub mod revisit;
pub mod scene;
pub mod stitch;
pub mod video;

pub use export::{ExportError, export_fused_ply};
pub use geometry::{
    BackprojectionError, BackprojectionSettings, CameraCalibration, MeasuredPoint, MeasuredView,
    backproject_measured_view,
};
pub use pose_graph::{
    PoseGraphEdge, PoseGraphError, PoseGraphReport, PoseGraphSettings, RelativePoseGraph,
    optimize_relative_pose_graph, pose_edge_residual,
};
pub use reconstruction::{
    FusionProgress, ReconstructionError, ReconstructionProgress, ReconstructionSettings,
    fuse_scene_bundle, fuse_scene_bundle_with_settings, reconstruct_frames,
};
pub use revisit::{
    CameraCentreDirection, RevisitProposal, RevisitProposalSettings, WindowCameraPath,
    camera_centre_direction, propose_revisits, window_camera_path,
};
pub use scene::{
    MeasuredFrameChunk, SceneBundle, SceneBundleError, SceneManifest, SceneProvenance,
    WindowMeasuredChunk,
};
pub use stitch::{
    AlignmentReport, FusedPoint, FusedSceneChunk, LoopClosureSettings, LoopMeasurement,
    LoopMeasurementSettings, SimilarityTransform, StitchError, StitchSettings,
    align_overlapping_windows, measure_loop_closure, stitch_measured_windows,
    stitch_measured_windows_with_loop_closures, stitch_measured_windows_with_settings,
    transform_points,
};
pub use video::{
    CaptureDisposition, CaptureQuality, VideoExtractionSettings, VideoFrames, VideoInputError,
    assess_capture_quality, extract_video_frames,
};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ScheduleError {
    #[error("chunk size must be at least 2")]
    ChunkTooSmall,
    #[error("overlap must be smaller than chunk size")]
    OverlapTooLarge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WindowSettings {
    pub chunk_size: usize,
    pub overlap: usize,
}

impl Default for WindowSettings {
    fn default() -> Self {
        Self {
            chunk_size: 12,
            overlap: 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameWindow {
    pub index: usize,
    pub start: usize,
    pub end: usize,
}

impl FrameWindow {
    #[must_use]
    pub fn len(self) -> usize {
        self.end - self.start
    }

    #[must_use]
    pub fn is_empty(self) -> bool {
        self.start == self.end
    }
}

/// Reproduces PR #2's `for (w0 = 0; w0 < F; w0 += chunk-overlap)` schedule,
/// including its final partial window.
pub fn plan_windows(
    frame_count: usize,
    settings: WindowSettings,
) -> Result<Vec<FrameWindow>, ScheduleError> {
    if settings.chunk_size < 2 {
        return Err(ScheduleError::ChunkTooSmall);
    }
    if settings.overlap >= settings.chunk_size {
        return Err(ScheduleError::OverlapTooLarge);
    }
    let step = settings.chunk_size - settings.overlap;
    Ok((0..frame_count)
        .step_by(step)
        .enumerate()
        .map(|(index, start)| FrameWindow {
            index,
            start,
            end: (start + settings.chunk_size).min(frame_count),
        })
        .collect())
}

#[derive(Debug, Clone)]
pub struct OwnedFrame {
    pub rgb_hwc_u8: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

impl OwnedFrame {
    fn as_engine_input(&self) -> ViewInput<'_> {
        ViewInput {
            rgb_hwc_u8: &self.rgb_hwc_u8,
            h: self.height,
            w: self.width,
        }
    }
}

/// Thin product-to-engine seam for one PR #2-compatible multi-view pass.
///
/// The engine selects and restores a saddle-balanced reference view when the
/// model enables alternating global attention. Frame selection, stitching,
/// and fusion remain separate Vestra phases.
pub fn infer_ordered_window(
    engine: &mut Engine,
    frames: &[OwnedFrame],
) -> Result<MultiViewInferOut, EngineError> {
    let inputs = frames
        .iter()
        .map(OwnedFrame::as_engine_input)
        .collect::<Vec<_>>();
    engine.infer_multi_view(&inputs)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScaleStatus {
    Relative,
    Metric,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryProvenance {
    Measured,
    Fused,
    Generated,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_schedule_matches_pinned_cpp_loop() {
        let windows = plan_windows(60, WindowSettings::default()).unwrap();
        assert_eq!(
            windows.first().unwrap(),
            &FrameWindow {
                index: 0,
                start: 0,
                end: 12
            }
        );
        assert_eq!(
            windows[1],
            FrameWindow {
                index: 1,
                start: 9,
                end: 21
            }
        );
        assert_eq!(
            windows.last().unwrap(),
            &FrameWindow {
                index: 6,
                start: 54,
                end: 60
            }
        );
    }

    #[test]
    fn sporting_workload_has_fourteen_windows_and_partial_tail() {
        let windows = plan_windows(120, WindowSettings::default()).unwrap();
        assert_eq!(windows.len(), 14);
        assert_eq!(windows.last().unwrap().start, 117);
        assert_eq!(windows.last().unwrap().end, 120);
    }

    #[test]
    fn invalid_overlap_is_rejected() {
        assert_eq!(
            plan_windows(
                10,
                WindowSettings {
                    chunk_size: 12,
                    overlap: 12
                }
            ),
            Err(ScheduleError::OverlapTooLarge)
        );
    }
}
