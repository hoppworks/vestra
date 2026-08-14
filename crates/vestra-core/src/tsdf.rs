//! Deterministic normal-space TSDF surface fusion for relative-scale worlds.
//!
//! The implementation follows PR #2's CPU geometry contract: PCA normals are
//! oriented toward the nearest camera, points splat a truncated signed-distance
//! band, and extracted zero-crossing voxels are sorted frame-major for a stable
//! progressive reveal.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::icp::estimate_normals;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TsdfSettings {
    /// Relative fraction of the cloud bounding-box diagonal when `voxel_size`
    /// is absent. PR #2's default is 0.4 percent.
    pub voxel_fraction: f32,
    pub voxel_size: Option<f32>,
    pub truncation_multiple: f32,
    pub normal_radius: Option<f32>,
    pub minimum_hits: u32,
}

impl Default for TsdfSettings {
    fn default() -> Self {
        Self {
            voxel_fraction: 0.004,
            voxel_size: None,
            truncation_multiple: 4.0,
            normal_radius: None,
            minimum_hits: 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TsdfObservation {
    pub position: [f32; 3],
    pub color_srgb: [u8; 3],
    pub confidence: f32,
    pub radius: f32,
    pub frame_index: i32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TsdfSurfel {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color_srgb: [u8; 3],
    pub radius: f32,
    pub first_observing_frame: i32,
    pub contributors: u32,
}

/// Fuses one measured cloud into a single zero-crossing surfel layer.
///
/// Invalid evidence is ignored. Input order does not affect the final ordering:
/// output is frame-major, then voxel-key-major.
pub fn fuse_normal_space_tsdf(
    observations: &[TsdfObservation],
    camera_centres: &[[f32; 3]],
    settings: TsdfSettings,
) -> Vec<TsdfSurfel> {
    let observations = observations
        .iter()
        .copied()
        .filter(|point| {
            point.position.iter().all(|value| value.is_finite())
                && point.confidence.is_finite()
                && point.confidence > 0.0
                && point.radius.is_finite()
                && point.radius > 0.0
        })
        .collect::<Vec<_>>();
    if observations.len() < 8 {
        return observations
            .into_iter()
            .map(|point| TsdfSurfel {
                position: point.position,
                normal: [0.0, 0.0, 1.0],
                color_srgb: point.color_srgb,
                radius: point.radius,
                first_observing_frame: point.frame_index,
                contributors: 1,
            })
            .collect();
    }
    let positions = observations
        .iter()
        .map(|point| point.position)
        .collect::<Vec<_>>();
    let voxel = resolved_voxel_size(&positions, settings);
    let truncation_multiple =
        if settings.truncation_multiple.is_finite() && settings.truncation_multiple > 0.0 {
            settings.truncation_multiple
        } else {
            4.0
        };
    let truncation = truncation_multiple * voxel;
    let normal_radius = settings.normal_radius.unwrap_or(2.5 * voxel).max(1e-6);
    let mut normals = estimate_normals(&positions, normal_radius);
    orient_normals_toward_cameras(&positions, &mut normals, camera_centres);

    let band = (truncation / voxel).ceil() as i32;
    let mut field = HashMap::<VoxelKey, Cell>::with_capacity(observations.len());
    for (observation, normal) in observations.iter().zip(normals) {
        if normal == [0.0; 3] {
            continue;
        }
        // PR #2's C API deliberately uses the TSDF kernel's default weight:
        // inverse point radius. Radius already encodes local sample spacing from
        // the streaming backprojector, so near/high-confidence evidence receives
        // the stronger contribution without changing this branch's public input.
        let weight = 1.0 / (f64::from(observation.radius) + 1e-12);
        let linear = observation.color_srgb.map(srgb_to_linear);
        let mut previous = None;
        for step in -band..=band {
            let query = [
                observation.position[0] + step as f32 * voxel * normal[0],
                observation.position[1] + step as f32 * voxel * normal[1],
                observation.position[2] + step as f32 * voxel * normal[2],
            ];
            let key = VoxelKey::for_position(query, voxel);
            if previous == Some(key) {
                continue;
            }
            previous = Some(key);
            let centre = key.centre(voxel);
            let signed_distance = dot(subtract(centre, observation.position), normal);
            if signed_distance.abs() > truncation {
                continue;
            }
            let cell = field.entry(key).or_default();
            cell.weight += weight;
            cell.signed_distance += weight * f64::from(signed_distance);
            for axis in 0..3 {
                cell.normal[axis] += weight * f64::from(normal[axis]);
                cell.color_linear[axis] += weight * f64::from(linear[axis]);
            }
            cell.hits += 1;
            cell.first_frame = cell.first_frame.min(observation.frame_index);
        }
    }
    let half = voxel * 0.5;
    let mut output = field
        .into_iter()
        .filter_map(|(key, cell)| {
            if cell.hits < settings.minimum_hits.max(1) || cell.weight <= 0.0 {
                return None;
            }
            let signed_distance = (cell.signed_distance / cell.weight) as f32;
            if signed_distance.abs() > half {
                return None;
            }
            let normal = normalize(cell.normal.map(|value| (value / cell.weight) as f32))?;
            let centre = key.centre(voxel);
            Some((
                key,
                TsdfSurfel {
                    position: [
                        centre[0] - signed_distance * normal[0],
                        centre[1] - signed_distance * normal[1],
                        centre[2] - signed_distance * normal[2],
                    ],
                    normal,
                    color_srgb: cell
                        .color_linear
                        .map(|value| linear_to_srgb((value / cell.weight) as f32)),
                    radius: 0.6 * voxel,
                    first_observing_frame: cell.first_frame,
                    contributors: cell.hits,
                },
            ))
        })
        .collect::<Vec<_>>();
    if output.is_empty() {
        return observations
            .into_iter()
            .map(|point| TsdfSurfel {
                position: point.position,
                normal: [0.0, 0.0, 1.0],
                color_srgb: point.color_srgb,
                radius: point.radius,
                first_observing_frame: point.frame_index,
                contributors: 1,
            })
            .collect();
    }
    output.sort_by_key(|(key, point)| (point.first_observing_frame, key.x, key.y, key.z));
    output.into_iter().map(|(_, point)| point).collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct VoxelKey {
    x: i32,
    y: i32,
    z: i32,
}

impl VoxelKey {
    fn for_position(position: [f32; 3], voxel: f32) -> Self {
        Self {
            x: (position[0] / voxel).floor() as i32,
            y: (position[1] / voxel).floor() as i32,
            z: (position[2] / voxel).floor() as i32,
        }
    }
    fn centre(self, voxel: f32) -> [f32; 3] {
        [
            (self.x as f32 + 0.5) * voxel,
            (self.y as f32 + 0.5) * voxel,
            (self.z as f32 + 0.5) * voxel,
        ]
    }
}

#[derive(Debug)]
struct Cell {
    weight: f64,
    signed_distance: f64,
    normal: [f64; 3],
    color_linear: [f64; 3],
    hits: u32,
    first_frame: i32,
}
impl Default for Cell {
    fn default() -> Self {
        Self {
            weight: 0.0,
            signed_distance: 0.0,
            normal: [0.0; 3],
            color_linear: [0.0; 3],
            hits: 0,
            first_frame: i32::MAX,
        }
    }
}

fn bounding_diagonal(points: &[[f32; 3]]) -> f32 {
    let mut low = points[0];
    let mut high = points[0];
    for point in points {
        for axis in 0..3 {
            low[axis] = low[axis].min(point[axis]);
            high[axis] = high[axis].max(point[axis]);
        }
    }
    subtract(high, low)
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
}

fn resolved_voxel_size(points: &[[f32; 3]], settings: TsdfSettings) -> f32 {
    settings
        .voxel_size
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or_else(|| {
            let relative = settings.voxel_fraction * bounding_diagonal(points);
            if relative.is_finite() && relative > 0.0 {
                relative
            } else {
                // PR #2 only falls back to 0.03 for a degenerate extent; it
                // does not impose a minimum voxel edge on small valid scenes.
                0.03
            }
        })
}
fn orient_normals_toward_cameras(
    points: &[[f32; 3]],
    normals: &mut [[f32; 3]],
    cameras: &[[f32; 3]],
) {
    if cameras.is_empty() {
        return;
    }
    for (point, normal) in points.iter().zip(normals) {
        if *normal == [0.0; 3] {
            continue;
        }
        let camera = cameras
            .iter()
            .min_by(|left, right| {
                squared_distance(**left, *point).total_cmp(&squared_distance(**right, *point))
            })
            .expect("nonempty cameras");
        if dot(*normal, subtract(*camera, *point)) < 0.0 {
            *normal = normal.map(|value| -value);
        }
    }
}
fn subtract(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}
fn squared_distance(left: [f32; 3], right: [f32; 3]) -> f32 {
    subtract(left, right)
        .iter()
        .map(|value| value * value)
        .sum()
}
fn dot(left: [f32; 3], right: [f32; 3]) -> f32 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}
fn normalize(value: [f32; 3]) -> Option<[f32; 3]> {
    let length = dot(value, value).sqrt();
    (length.is_finite() && length > 0.0).then(|| value.map(|v| v / length))
}
fn srgb_to_linear(value: u8) -> f32 {
    let value = f32::from(value) / 255.0;
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}
fn linear_to_srgb(value: f32) -> u8 {
    let value = value.clamp(0.0, 1.0);
    let value = if value <= 0.0031308 {
        12.92 * value
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    (value * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    fn plane(z: f32, frame: i32, color: [u8; 3]) -> Vec<TsdfObservation> {
        (-5..=5)
            .flat_map(|y| {
                (-5..=5).map(move |x| TsdfObservation {
                    position: [x as f32 * 0.02, y as f32 * 0.02, z],
                    color_srgb: color,
                    confidence: 1.0,
                    radius: 0.01,
                    frame_index: frame,
                })
            })
            .collect()
    }
    #[test]
    fn tsdf_collapses_two_parallel_sheets_and_uses_first_frame() {
        let mut points = plane(-0.006, 7, [255, 0, 0]);
        points.extend(plane(0.006, 2, [0, 255, 0]));
        let output = fuse_normal_space_tsdf(
            &points,
            &[[0.0, 0.0, 1.0]],
            TsdfSettings {
                voxel_size: Some(0.02),
                minimum_hits: 1,
                ..TsdfSettings::default()
            },
        );
        assert!(!output.is_empty());
        assert!(output.iter().all(|point| point.first_observing_frame == 2));
        assert!(output.iter().all(|point| point.position[2].abs() < 0.011));
        // Linear-light averaging must be visibly brighter than a gamma-space 50% average (128).
        assert!(
            output
                .iter()
                .any(|point| point.color_srgb[0] > 150 && point.color_srgb[1] > 150)
        );
    }

    #[test]
    fn valid_small_scene_uses_the_pr_relative_voxel_size_without_clamping() {
        let points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        assert_eq!(resolved_voxel_size(&points, TsdfSettings::default()), 0.004);
    }
}
