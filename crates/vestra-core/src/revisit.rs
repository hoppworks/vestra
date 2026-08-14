//! Conservative geometric revisit proposals from sequential camera poses.
//!
//! Proposals are deliberately cheap and permissive only in the sense that they
//! do not alter geometry. A proposal is never a loop closure: callers must
//! still establish tight spatial correspondences, fit a relative transform,
//! and pass its quality gates before adding a pose-graph edge.

use serde::{Deserialize, Serialize};

use crate::{CameraCalibration, SimilarityTransform, WindowMeasuredChunk};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CameraCentreDirection {
    pub frame_index: usize,
    /// Camera centre in the owning window's local coordinates.
    pub centre_local: [f32; 3],
    /// Unit positive camera-Z direction in the owning window's local coordinates.
    pub forward_local: [f32; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WindowCameraPath {
    pub window_index: usize,
    pub cameras: Vec<CameraCentreDirection>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RevisitProposalSettings {
    /// Candidates must be separated by at least this many window indices.
    pub minimum_window_gap: usize,
    /// Maximum global camera-centre distance in relative scene units. The
    /// caller derives this from scene scale; Vestra never labels it metres.
    pub maximum_centre_distance: f32,
    /// Required cosine of the two positive camera-Z directions.
    pub minimum_forward_cosine: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RevisitProposal {
    /// Earlier window node.
    pub earlier: usize,
    /// Later window node.
    pub later: usize,
    pub nearest_centre_distance: f32,
    pub best_forward_cosine: f32,
}

/// Collects valid camera trajectory evidence from one immutable measured
/// window. Invalid camera matrices are omitted rather than converted into
/// invented positions.
#[must_use]
pub fn window_camera_path(window: &WindowMeasuredChunk) -> WindowCameraPath {
    WindowCameraPath {
        window_index: window.window.index,
        cameras: window
            .views
            .iter()
            .filter_map(|view| camera_centre_direction(view.frame_index, view.camera))
            .collect(),
    }
}

/// Derives a local camera centre and forward direction from a W2C calibration.
pub fn camera_centre_direction(
    frame_index: usize,
    calibration: CameraCalibration,
) -> Option<CameraCentreDirection> {
    let matrix = calibration.world_to_camera;
    if !matrix.iter().all(|value| value.is_finite()) {
        return None;
    }
    // c2w = [Rᵀ | -Rᵀt], for W2C [R | t]. The third C2W column is the
    // positive camera-Z direction in local world coordinates.
    let centre_local = [
        -(matrix[0] * matrix[3] + matrix[4] * matrix[7] + matrix[8] * matrix[11]),
        -(matrix[1] * matrix[3] + matrix[5] * matrix[7] + matrix[9] * matrix[11]),
        -(matrix[2] * matrix[3] + matrix[6] * matrix[7] + matrix[10] * matrix[11]),
    ];
    let forward_local = normalize([matrix[8], matrix[9], matrix[10]])?;
    Some(CameraCentreDirection {
        frame_index,
        centre_local,
        forward_local,
    })
}

/// Returns at most one proposal per eligible pair of window paths.
pub fn propose_revisits(
    paths: &[WindowCameraPath],
    global_poses: &[SimilarityTransform],
    settings: RevisitProposalSettings,
) -> Vec<RevisitProposal> {
    if paths.len() != global_poses.len()
        || !settings.maximum_centre_distance.is_finite()
        || settings.maximum_centre_distance <= 0.0
        || !settings.minimum_forward_cosine.is_finite()
    {
        return Vec::new();
    }
    let mut proposals = Vec::new();
    for earlier in 0..paths.len() {
        for later in (earlier + settings.minimum_window_gap)..paths.len() {
            let mut nearest_centre_distance = f32::INFINITY;
            let mut best_forward_cosine = -1.0_f32;
            let mut accepted = false;
            for first in &paths[earlier].cameras {
                let first_centre = global_poses[earlier].apply(first.centre_local);
                let first_forward = global_poses[earlier].rotate(first.forward_local);
                for second in &paths[later].cameras {
                    let second_centre = global_poses[later].apply(second.centre_local);
                    let second_forward = global_poses[later].rotate(second.forward_local);
                    let distance = l2_distance(first_centre, second_centre);
                    let cosine = dot(first_forward, second_forward);
                    nearest_centre_distance = nearest_centre_distance.min(distance);
                    best_forward_cosine = best_forward_cosine.max(cosine);
                    if distance <= settings.maximum_centre_distance
                        && cosine >= settings.minimum_forward_cosine
                    {
                        accepted = true;
                    }
                }
            }
            if accepted {
                proposals.push(RevisitProposal {
                    earlier,
                    later,
                    nearest_centre_distance,
                    best_forward_cosine,
                });
            }
        }
    }
    proposals
}

fn normalize(vector: [f32; 3]) -> Option<[f32; 3]> {
    let length = dot(vector, vector).sqrt();
    if !length.is_finite() || length <= 1e-12 {
        None
    } else {
        Some(vector.map(|value| value / length))
    }
}

fn dot(left: [f32; 3], right: [f32; 3]) -> f32 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn l2_distance(left: [f32; 3], right: [f32; 3]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(first, second)| (first - second).powi(2))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera(centre_x: f32) -> CameraCentreDirection {
        CameraCentreDirection {
            frame_index: 0,
            centre_local: [centre_x, 0.0, 0.0],
            forward_local: [0.0, 0.0, 1.0],
        }
    }

    #[test]
    fn recovers_camera_centre_and_forward_from_w2c() {
        let calibration = CameraCalibration {
            world_to_camera: [1.0, 0.0, 0.0, -2.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 1.0, -4.0],
            intrinsics: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        };
        let pose = camera_centre_direction(9, calibration).unwrap();
        assert_eq!(pose.frame_index, 9);
        assert_eq!(pose.centre_local, [2.0, -3.0, 4.0]);
        assert_eq!(pose.forward_local, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn only_non_adjacent_proximate_aligned_views_are_proposed() {
        let paths = vec![
            WindowCameraPath {
                window_index: 0,
                cameras: vec![camera(0.0)],
            },
            WindowCameraPath {
                window_index: 1,
                cameras: vec![camera(1.0)],
            },
            WindowCameraPath {
                window_index: 2,
                cameras: vec![camera(0.1)],
            },
        ];
        let proposals = propose_revisits(
            &paths,
            &[SimilarityTransform::IDENTITY; 3],
            RevisitProposalSettings {
                minimum_window_gap: 2,
                maximum_centre_distance: 0.2,
                minimum_forward_cosine: 0.9,
            },
        );
        assert_eq!(proposals.len(), 1);
        assert_eq!((proposals[0].earlier, proposals[0].later), (0, 2));
    }

    #[test]
    fn opposite_camera_directions_are_not_proposed() {
        let paths = vec![
            WindowCameraPath {
                window_index: 0,
                cameras: vec![camera(0.0)],
            },
            WindowCameraPath {
                window_index: 2,
                cameras: vec![CameraCentreDirection {
                    forward_local: [0.0, 0.0, -1.0],
                    ..camera(0.0)
                }],
            },
        ];
        assert!(
            propose_revisits(
                &paths,
                &[SimilarityTransform::IDENTITY; 2],
                RevisitProposalSettings {
                    minimum_window_gap: 1,
                    maximum_centre_distance: 0.1,
                    minimum_forward_cosine: 0.9,
                },
            )
            .is_empty()
        );
    }

    #[test]
    fn window_path_retains_only_valid_camera_evidence() {
        let valid = crate::MeasuredFrameChunk {
            frame_index: 7,
            camera: CameraCalibration {
                world_to_camera: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                intrinsics: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            },
            points: Vec::new(),
        };
        let mut invalid = valid.clone();
        invalid.frame_index = 8;
        invalid.camera.world_to_camera[0] = f32::NAN;
        let window = WindowMeasuredChunk {
            window: crate::FrameWindow {
                index: 3,
                start: 0,
                end: 2,
            },
            views: vec![valid, invalid],
            cpp_pr2_emission_confidence_threshold: None,
        };
        let path = window_camera_path(&window);
        assert_eq!(path.window_index, 3);
        assert_eq!(path.cameras.len(), 1);
        assert_eq!(path.cameras[0].frame_index, 7);
    }
}
