//! Deterministic measured-geometry construction.
//!
//! This module deliberately creates no inferred or generated geometry. Each
//! emitted point is directly traceable to one depth/confidence pixel and its
//! calibrated camera. Coordinates are relative until a future metric-scale
//! phase supplies an independently verified scale anchor.

use serde::{Deserialize, Serialize};

/// Camera calibration for one inferred view.
///
/// `world_to_camera` is row-major `[R | t]`, so `camera = R * world + t`.
/// It is the pose convention emitted by Vestra Engine.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CameraCalibration {
    pub world_to_camera: [f32; 12],
    pub intrinsics: [f32; 9],
}

/// Raw, co-registered evidence from a single inferred view.
#[derive(Debug, Clone, Copy)]
pub struct MeasuredView<'a> {
    pub rgb_hwc_u8: &'a [u8],
    pub depth: &'a [f32],
    pub confidence: &'a [f32],
    pub width: usize,
    pub height: usize,
    pub camera: CameraCalibration,
}

/// Conservative deterministic selection policy for measured points.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BackprojectionSettings {
    /// Retain only depth pixels at or above this confidence.
    pub minimum_confidence: f32,
    /// Keep one pixel out of every `pixel_stride` in both dimensions.
    pub pixel_stride: usize,
    /// Radius in source-pixel units, projected into relative world units.
    pub surfel_radius_pixels: f32,
}

impl Default for BackprojectionSettings {
    fn default() -> Self {
        Self {
            minimum_confidence: 1.0,
            pixel_stride: 1,
            surfel_radius_pixels: 1.0,
        }
    }
}

/// One directly observed, relative-scale colored surfel centre.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MeasuredPoint {
    pub position: [f32; 3],
    /// Unit normal in relative world coordinates. It is oriented toward the
    /// observing camera when a depth stencil is unavailable at an edge.
    #[serde(default)]
    pub normal: [f32; 3],
    pub color_srgb: [u8; 3],
    pub confidence: f32,
    pub radius: f32,
    /// Pixel coordinate in the originating frame, `[x, y]`.
    pub source_pixel: [u32; 2],
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum BackprojectionError {
    #[error("view dimensions must be non-zero")]
    EmptyDimensions,
    #[error("RGB input must contain width * height * 3 bytes")]
    RgbLength,
    #[error("depth and confidence inputs must contain width * height values")]
    EvidenceLength,
    #[error("pixel stride must be at least one")]
    ZeroPixelStride,
    #[error("camera focal lengths must be finite and positive")]
    InvalidFocalLength,
    #[error("camera intrinsics and pose values must be finite")]
    NonFiniteCalibration,
    #[error("camera calibration matrix is not invertible")]
    NonInvertibleCalibration,
    #[error("backprojection settings must be finite, with a non-negative surfel radius")]
    InvalidSettings,
}

/// Back-projects valid depth pixels into the common world coordinate system.
///
/// The operation is intentionally pointwise and deterministic: it neither
/// aligns windows nor fills holes. Those are separately-labelled fused phases.
pub fn backproject_measured_view(
    view: MeasuredView<'_>,
    settings: BackprojectionSettings,
) -> Result<Vec<MeasuredPoint>, BackprojectionError> {
    if view.width == 0 || view.height == 0 {
        return Err(BackprojectionError::EmptyDimensions);
    }
    let pixels = view.width * view.height;
    if view.rgb_hwc_u8.len() != pixels * 3 {
        return Err(BackprojectionError::RgbLength);
    }
    if view.depth.len() != pixels || view.confidence.len() != pixels {
        return Err(BackprojectionError::EvidenceLength);
    }
    if settings.pixel_stride == 0 {
        return Err(BackprojectionError::ZeroPixelStride);
    }
    if !settings.minimum_confidence.is_finite()
        || !settings.surfel_radius_pixels.is_finite()
        || settings.surfel_radius_pixels < 0.0
    {
        return Err(BackprojectionError::InvalidSettings);
    }

    let k = view.camera.intrinsics;
    let fx = k[0];
    let fy = k[4];
    if !fx.is_finite() || !fy.is_finite() || fx <= 0.0 || fy <= 0.0 {
        return Err(BackprojectionError::InvalidFocalLength);
    }
    if !k.iter().all(|value| value.is_finite())
        || !view
            .camera
            .world_to_camera
            .iter()
            .all(|value| value.is_finite())
    {
        return Err(BackprojectionError::NonFiniteCalibration);
    }
    let cx = k[2];
    let cy = k[5];
    let [r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz] = view.camera.world_to_camera;

    let mut points = Vec::with_capacity(pixels / settings.pixel_stride.saturating_pow(2));
    for y in (0..view.height).step_by(settings.pixel_stride) {
        for x in (0..view.width).step_by(settings.pixel_stride) {
            let index = y * view.width + x;
            let depth = view.depth[index];
            let confidence = view.confidence[index];
            if !depth.is_finite()
                || depth <= 0.0
                || !confidence.is_finite()
                || confidence < settings.minimum_confidence
            {
                continue;
            }

            let camera_x = (x as f32 - cx) * depth / fx;
            let camera_y = (y as f32 - cy) * depth / fy;
            let camera_z = depth;
            // `world = R^T * (camera - t)` for the documented W2C pose.
            let dx = camera_x - tx;
            let dy = camera_y - ty;
            let dz = camera_z - tz;
            let position = [
                r00 * dx + r10 * dy + r20 * dz,
                r01 * dx + r11 * dy + r21 * dz,
                r02 * dx + r12 * dy + r22 * dz,
            ];
            let normal = estimate_world_normal(
                view,
                x,
                y,
                [r00, r01, r02, r10, r11, r12, r20, r21, r22],
                position,
            );
            let rgb_index = index * 3;
            points.push(MeasuredPoint {
                position,
                normal,
                color_srgb: [
                    view.rgb_hwc_u8[rgb_index],
                    view.rgb_hwc_u8[rgb_index + 1],
                    view.rgb_hwc_u8[rgb_index + 2],
                ],
                confidence,
                radius: settings.surfel_radius_pixels * depth / fx.min(fy),
                source_pixel: [x as u32, y as u32],
            });
        }
    }
    Ok(points)
}

fn estimate_world_normal(
    view: MeasuredView<'_>,
    x: usize,
    y: usize,
    rotation: [f32; 9],
    position: [f32; 3],
) -> [f32; 3] {
    let point_at = |px: usize, py: usize| -> Option<[f32; 3]> {
        if px >= view.width || py >= view.height {
            return None;
        }
        let depth = view.depth[py * view.width + px];
        if !depth.is_finite() || depth <= 0.0 {
            return None;
        }
        let fx = view.camera.intrinsics[0];
        let fy = view.camera.intrinsics[4];
        let cx = view.camera.intrinsics[2];
        let cy = view.camera.intrinsics[5];
        Some([
            (px as f32 - cx) * depth / fx,
            (py as f32 - cy) * depth / fy,
            depth,
        ])
    };
    let centre = point_at(x, y).expect("caller has already validated centre depth");
    let camera_normal = match (
        point_at(x.saturating_add(1), y),
        point_at(x, y.saturating_add(1)),
    ) {
        (Some(right), Some(down)) => normalize(cross(sub(down, centre), sub(right, centre))),
        _ => {
            let [r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz] =
                view.camera.world_to_camera;
            let origin = [
                -(r00 * tx + r10 * ty + r20 * tz),
                -(r01 * tx + r11 * ty + r21 * tz),
                -(r02 * tx + r12 * ty + r22 * tz),
            ];
            normalize(sub(origin, position))
        }
    };
    // W2C camera normals become world normals through R^T.
    normalize([
        rotation[0] * camera_normal[0]
            + rotation[3] * camera_normal[1]
            + rotation[6] * camera_normal[2],
        rotation[1] * camera_normal[0]
            + rotation[4] * camera_normal[1]
            + rotation[7] * camera_normal[2],
        rotation[2] * camera_normal[0]
            + rotation[5] * camera_normal[1]
            + rotation[8] * camera_normal[2],
    ])
}

fn sub(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn cross(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn normalize(vector: [f32; 3]) -> [f32; 3] {
    let length = (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt();
    if !length.is_finite() || length <= 1e-12 {
        [0.0, 0.0, 1.0]
    } else {
        vector.map(|value| value / length)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera(world_to_camera: [f32; 12]) -> CameraCalibration {
        CameraCalibration {
            world_to_camera,
            intrinsics: [2.0, 0.0, 0.5, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0],
        }
    }

    #[test]
    fn identity_camera_backprojects_colored_evidence() {
        let rgb = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let depth = [2.0, 4.0, 1.0, 3.0];
        let confidence = [1.0; 4];
        let points = backproject_measured_view(
            MeasuredView {
                rgb_hwc_u8: &rgb,
                depth: &depth,
                confidence: &confidence,
                width: 2,
                height: 2,
                camera: camera([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            },
            BackprojectionSettings::default(),
        )
        .unwrap();
        assert_eq!(points.len(), 4);
        assert_eq!(points[0].position, [-0.5, -0.5, 2.0]);
        assert_eq!(points[3].position, [0.75, 0.75, 3.0]);
        assert_eq!(points[1].color_srgb, [40, 50, 60]);
        assert_eq!(points[0].radius, 1.0);
    }

    #[test]
    fn flat_depth_raster_emits_a_unit_world_normal() {
        let rgb = [0; 12];
        let depth = [2.0; 4];
        let confidence = [1.0; 4];
        let points = backproject_measured_view(
            MeasuredView {
                rgb_hwc_u8: &rgb,
                depth: &depth,
                confidence: &confidence,
                width: 2,
                height: 2,
                camera: camera([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            },
            BackprojectionSettings::default(),
        )
        .unwrap();
        assert_eq!(points[0].normal, [0.0, 0.0, -1.0]);
    }

    #[test]
    fn world_to_camera_translation_is_inverted() {
        let rgb = [1, 2, 3];
        let depth = [2.0];
        let confidence = [2.0];
        let points = backproject_measured_view(
            MeasuredView {
                rgb_hwc_u8: &rgb,
                depth: &depth,
                confidence: &confidence,
                width: 1,
                height: 1,
                camera: camera([1.0, 0.0, 0.0, 3.0, 0.0, 1.0, 0.0, -2.0, 0.0, 0.0, 1.0, 1.0]),
            },
            BackprojectionSettings::default(),
        )
        .unwrap();
        assert_eq!(points[0].position, [-3.5, 1.5, 1.0]);
    }

    #[test]
    fn invalid_or_low_confidence_pixels_are_not_measured_geometry() {
        let rgb = [0; 12];
        let depth = [1.0, f32::NAN, -1.0, 2.0];
        let confidence = [0.5, 2.0, 2.0, 1.0];
        let points = backproject_measured_view(
            MeasuredView {
                rgb_hwc_u8: &rgb,
                depth: &depth,
                confidence: &confidence,
                width: 2,
                height: 2,
                camera: camera([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            },
            BackprojectionSettings {
                minimum_confidence: 1.0,
                ..BackprojectionSettings::default()
            },
        )
        .unwrap();
        assert_eq!(points.len(), 1);
        assert_eq!(points[0].source_pixel, [1, 1]);
    }

    #[test]
    fn invalid_calibration_or_settings_are_rejected_before_emitting_points() {
        let rgb = [0, 0, 0];
        let depth = [1.0];
        let confidence = [1.0];
        let view = MeasuredView {
            rgb_hwc_u8: &rgb,
            depth: &depth,
            confidence: &confidence,
            width: 1,
            height: 1,
            camera: camera([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        };
        assert_eq!(
            backproject_measured_view(
                view,
                BackprojectionSettings {
                    surfel_radius_pixels: -1.0,
                    ..BackprojectionSettings::default()
                }
            ),
            Err(BackprojectionError::InvalidSettings)
        );
    }
}
