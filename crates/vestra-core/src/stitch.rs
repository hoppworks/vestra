//! Relative-scale alignment of overlapping measured windows.
//!
//! Matching source pixels from the same overlapping video frame are direct
//! correspondence evidence. We use them to estimate a weighted Sim(3) from a
//! newly reconstructed window into the already accepted scene frame. Failed or
//! geometrically degenerate estimates are explicit errors: this module never
//! invents an alignment.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
    MeasuredPoint, PoseGraphEdge, PoseGraphError, PoseGraphReport, PoseGraphSettings,
    RelativePoseGraph, WindowMeasuredChunk, optimize_relative_pose_graph,
};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SimilarityTransform {
    pub scale: f32,
    /// Row-major rotation.
    pub rotation: [f32; 9],
    pub translation: [f32; 3],
}

impl SimilarityTransform {
    pub const IDENTITY: Self = Self {
        scale: 1.0,
        rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        translation: [0.0, 0.0, 0.0],
    };

    #[must_use]
    pub fn apply(self, point: [f32; 3]) -> [f32; 3] {
        let p = [
            point[0] * self.scale,
            point[1] * self.scale,
            point[2] * self.scale,
        ];
        [
            self.rotation[0] * p[0]
                + self.rotation[1] * p[1]
                + self.rotation[2] * p[2]
                + self.translation[0],
            self.rotation[3] * p[0]
                + self.rotation[4] * p[1]
                + self.rotation[5] * p[2]
                + self.translation[1],
            self.rotation[6] * p[0]
                + self.rotation[7] * p[1]
                + self.rotation[8] * p[2]
                + self.translation[2],
        ]
    }

    #[must_use]
    pub fn rotate(self, vector: [f32; 3]) -> [f32; 3] {
        normalize([
            self.rotation[0] * vector[0]
                + self.rotation[1] * vector[1]
                + self.rotation[2] * vector[2],
            self.rotation[3] * vector[0]
                + self.rotation[4] * vector[1]
                + self.rotation[5] * vector[2],
            self.rotation[6] * vector[0]
                + self.rotation[7] * vector[1]
                + self.rotation[8] * vector[2],
        ])
    }

    /// Returns `self ∘ inner`: apply `inner` first, then this transform.
    #[must_use]
    pub fn compose(self, inner: Self) -> Self {
        let rotation = matrix_mul_f32(self.rotation, inner.rotation);
        let rotated_translation = matrix_vec_f32(self.rotation, inner.translation);
        Self {
            scale: self.scale * inner.scale,
            rotation,
            translation: [
                rotated_translation[0] * self.scale + self.translation[0],
                rotated_translation[1] * self.scale + self.translation[1],
                rotated_translation[2] * self.scale + self.translation[2],
            ],
        }
    }

    /// Returns the exact inverse when the similarity scale is finite and
    /// positive. Invalid transforms are never silently inverted.
    pub fn inverse(self) -> Option<Self> {
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return None;
        }
        let rotation = [
            self.rotation[0],
            self.rotation[3],
            self.rotation[6],
            self.rotation[1],
            self.rotation[4],
            self.rotation[7],
            self.rotation[2],
            self.rotation[5],
            self.rotation[8],
        ];
        let translated = matrix_vec_f32(rotation, self.translation);
        Some(Self {
            scale: self.scale.recip(),
            rotation,
            translation: translated.map(|value| -value / self.scale),
        })
    }
}

fn matrix_mul_f32(left: [f32; 9], right: [f32; 9]) -> [f32; 9] {
    let mut out = [0.0; 9];
    for row in 0..3 {
        for column in 0..3 {
            out[row * 3 + column] = (0..3)
                .map(|inner| left[row * 3 + inner] * right[inner * 3 + column])
                .sum();
        }
    }
    out
}

fn matrix_vec_f32(matrix: [f32; 9], vector: [f32; 3]) -> [f32; 3] {
    [
        matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
        matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
        matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2],
    ]
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlignmentReport {
    pub transform: SimilarityTransform,
    pub correspondence_count: usize,
    pub inlier_count: usize,
    pub rms_residual: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusedPoint {
    pub position: [f32; 3],
    #[serde(default)]
    pub normal: [f32; 3],
    pub color_srgb: [u8; 3],
    pub confidence: f32,
    pub radius: f32,
    pub contributors: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FusedSceneChunk {
    pub alignments: Vec<AlignmentReport>,
    /// Present only when independently verified loop constraints were supplied
    /// and globally optimized before this fused derivative was published.
    #[serde(default)]
    pub pose_graph: Option<PoseGraphReport>,
    pub voxel_size: f32,
    pub points: Vec<FusedPoint>,
}

/// Product quality gates for one sequential overlap seam. The defaults are
/// intentionally stricter than the mathematical minimum of three points:
/// real videos must contribute enough direct pixel evidence before a fused
/// world is published.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StitchSettings {
    pub minimum_correspondences: usize,
    pub minimum_inlier_ratio: f32,
    pub minimum_scale: f32,
    pub maximum_scale: f32,
}

impl Default for StitchSettings {
    fn default() -> Self {
        Self {
            minimum_correspondences: 50,
            minimum_inlier_ratio: 0.6,
            minimum_scale: 0.1,
            maximum_scale: 10.0,
        }
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum StitchError {
    #[error("windows have fewer than three pixel correspondences")]
    InsufficientCorrespondences,
    #[error("overlap correspondence geometry is degenerate")]
    DegenerateGeometry,
    #[error("overlap seam does not meet the configured quality gate")]
    QualityGate,
    #[error("relative pose graph optimization failed: {0}")]
    PoseGraph(#[from] PoseGraphError),
}

#[derive(Clone, Copy)]
struct Match {
    source: [f64; 3],
    target: [f64; 3],
    target_normal: [f64; 3],
    weight: f64,
}

type VoxelCell = (f32, [f32; 3], [f32; 3], [f32; 3], f32, u32);

/// Fits the transform that maps `source` into `target` from shared
/// `(frame_index, source_pixel)` observations. The returned transform is
/// relative-scale only; callers must retain both measured inputs unchanged.
pub fn align_overlapping_windows(
    source: &WindowMeasuredChunk,
    target: &WindowMeasuredChunk,
) -> Result<AlignmentReport, StitchError> {
    let mut target_points = HashMap::new();
    for frame in &target.views {
        for point in &frame.points {
            target_points.insert((frame.frame_index, point.source_pixel), point);
        }
    }
    let mut matches = Vec::new();
    for frame in &source.views {
        for point in &frame.points {
            if let Some(other) = target_points.get(&(frame.frame_index, point.source_pixel)) {
                matches.push(Match {
                    source: point.position.map(f64::from),
                    target: other.position.map(f64::from),
                    target_normal: other.normal.map(f64::from),
                    weight: f64::from(point.confidence.min(other.confidence).max(0.0)),
                });
            }
        }
    }
    if matches.len() < 3 {
        return Err(StitchError::InsufficientCorrespondences);
    }
    let count = matches.len();
    let mut inliers: Vec<usize> = (0..count).collect();
    for _ in 0..3 {
        let transform = fit_similarity(&matches, &inliers)?;
        let mut residuals = inliers
            .iter()
            .map(|&i| {
                distance(
                    transform.apply(matches[i].source.map(|v| v as f32)),
                    matches[i].target.map(|v| v as f32),
                )
            })
            .collect::<Vec<_>>();
        residuals.sort_by(f32::total_cmp);
        let median = residuals[residuals.len() / 2];
        let limit = (median * 3.0).max(1e-5);
        let next = inliers
            .iter()
            .copied()
            .filter(|&i| {
                distance(
                    transform.apply(matches[i].source.map(|v| v as f32)),
                    matches[i].target.map(|v| v as f32),
                ) <= limit
            })
            .collect::<Vec<_>>();
        if next.len() < 3 || next.len() == inliers.len() {
            break;
        }
        inliers = next;
    }
    let transform = refine_point_to_plane(fit_similarity(&matches, &inliers)?, &matches, &inliers);
    let squared = inliers
        .iter()
        .map(|&i| {
            let d = distance(
                transform.apply(matches[i].source.map(|v| v as f32)),
                matches[i].target.map(|v| v as f32),
            );
            d * d
        })
        .sum::<f32>();
    Ok(AlignmentReport {
        transform,
        correspondence_count: count,
        inlier_count: inliers.len(),
        rms_residual: (squared / inliers.len() as f32).sqrt(),
    })
}

fn fit_similarity(
    matches: &[Match],
    selected: &[usize],
) -> Result<SimilarityTransform, StitchError> {
    let weight_sum = selected.iter().map(|&i| matches[i].weight).sum::<f64>();
    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return Err(StitchError::DegenerateGeometry);
    }
    let mut sx = [0.0; 3];
    let mut ty = [0.0; 3];
    for &i in selected {
        for d in 0..3 {
            sx[d] += matches[i].weight * matches[i].source[d];
            ty[d] += matches[i].weight * matches[i].target[d];
        }
    }
    for d in 0..3 {
        sx[d] /= weight_sum;
        ty[d] /= weight_sum;
    }
    let mut s = [[0.0; 3]; 3];
    let mut covariance = [[0.0; 3]; 3];
    let mut source_energy = 0.0;
    for &i in selected {
        let m = matches[i];
        let x = [
            m.source[0] - sx[0],
            m.source[1] - sx[1],
            m.source[2] - sx[2],
        ];
        let y = [
            m.target[0] - ty[0],
            m.target[1] - ty[1],
            m.target[2] - ty[2],
        ];
        source_energy += m.weight * dot(x, x);
        for r in 0..3 {
            for c in 0..3 {
                s[r][c] += m.weight * x[r] * y[c];
                covariance[r][c] += m.weight * x[r] * x[c];
            }
        }
    }
    if source_energy <= 1e-14 {
        return Err(StitchError::DegenerateGeometry);
    }
    // A rank-one source overlap cannot determine a 3D rotation. Planar
    // overlap is allowed; it has two non-zero covariance eigenvalues.
    let trace = covariance[0][0] + covariance[1][1] + covariance[2][2];
    let second_invariant = covariance[0][0] * covariance[1][1]
        + covariance[0][0] * covariance[2][2]
        + covariance[1][1] * covariance[2][2]
        - covariance[0][1] * covariance[0][1]
        - covariance[0][2] * covariance[0][2]
        - covariance[1][2] * covariance[1][2];
    if !second_invariant.is_finite() || second_invariant <= trace * trace * 1e-10 {
        return Err(StitchError::DegenerateGeometry);
    }
    let n = [
        [
            s[0][0] + s[1][1] + s[2][2],
            s[1][2] - s[2][1],
            s[2][0] - s[0][2],
            s[0][1] - s[1][0],
        ],
        [
            s[1][2] - s[2][1],
            s[0][0] - s[1][1] - s[2][2],
            s[0][1] + s[1][0],
            s[0][2] + s[2][0],
        ],
        [
            s[2][0] - s[0][2],
            s[0][1] + s[1][0],
            -s[0][0] + s[1][1] - s[2][2],
            s[1][2] + s[2][1],
        ],
        [
            s[0][1] - s[1][0],
            s[0][2] + s[2][0],
            s[1][2] + s[2][1],
            -s[0][0] - s[1][1] + s[2][2],
        ],
    ];
    let q = largest_symmetric_eigenvector(n)?;
    let r = rotation_from_quaternion(q);
    let mut numerator = 0.0;
    for &i in selected {
        let m = matches[i];
        let x = [
            m.source[0] - sx[0],
            m.source[1] - sx[1],
            m.source[2] - sx[2],
        ];
        let y = [
            m.target[0] - ty[0],
            m.target[1] - ty[1],
            m.target[2] - ty[2],
        ];
        numerator += m.weight * dot(y, mat_vec(r, x));
    }
    let scale = numerator / source_energy;
    if !scale.is_finite() || scale <= 0.0 {
        return Err(StitchError::DegenerateGeometry);
    }
    let rsx = mat_vec(r, sx);
    Ok(SimilarityTransform {
        scale: scale as f32,
        rotation: r.map(|v| v as f32),
        translation: [
            (ty[0] - scale * rsx[0]) as f32,
            (ty[1] - scale * rsx[1]) as f32,
            (ty[2] - scale * rsx[2]) as f32,
        ],
    })
}

/// One bounded point-to-plane ICP correction after the correspondence-derived
/// Sim(3). It is intentionally local: it does not create new matches, so a
/// poor revisit cannot be pulled into an unrelated surface. Missing legacy
/// normals or a singular normal equation leave the Sim(3) unchanged.
fn refine_point_to_plane(
    transform: SimilarityTransform,
    matches: &[Match],
    selected: &[usize],
) -> SimilarityTransform {
    let mut normal = [[0.0_f64; 6]; 6];
    let mut rhs = [0.0_f64; 6];
    let mut usable = 0;
    for &index in selected {
        let entry = matches[index];
        let n = entry.target_normal;
        let length = dot(n, n).sqrt();
        if !length.is_finite() || length < 0.5 {
            continue;
        }
        let n = n.map(|value| value / length);
        let p = transform
            .apply(entry.source.map(|value| value as f32))
            .map(f64::from);
        let residual = dot(
            n,
            [
                p[0] - entry.target[0],
                p[1] - entry.target[1],
                p[2] - entry.target[2],
            ],
        );
        let cross = cross64(p, n);
        let row = [n[0], n[1], n[2], cross[0], cross[1], cross[2]];
        for r in 0..6 {
            rhs[r] -= entry.weight * row[r] * residual;
            for c in 0..6 {
                normal[r][c] += entry.weight * row[r] * row[c];
            }
        }
        usable += 1;
    }
    if usable < 6 {
        return transform;
    }
    // The local correction is deliberately Tikhonov-regularized: planar or
    // near-symmetric overlap often leaves some rotation axes unobservable.
    // Those axes should stay near zero rather than making the whole seam fail.
    for (index, row) in normal.iter_mut().enumerate() {
        row[index] += 1e-6;
    }
    let Some(solution) = solve_6x6(normal, rhs) else {
        return transform;
    };
    let translation = [solution[0] as f32, solution[1] as f32, solution[2] as f32];
    let rotation_vector = [solution[3] as f32, solution[4] as f32, solution[5] as f32];
    let angle = rotation_vector
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if !angle.is_finite() || angle > 0.1 || translation.iter().any(|value| !value.is_finite()) {
        return transform;
    }
    let correction = axis_angle_rotation(rotation_vector);
    let rotation = matrix_mul(correction, transform.rotation);
    let rotated_translation = matrix_vec(correction, transform.translation);
    SimilarityTransform {
        scale: transform.scale,
        rotation,
        translation: [
            rotated_translation[0] + translation[0],
            rotated_translation[1] + translation[1],
            rotated_translation[2] + translation[2],
        ],
    }
}

fn cross64(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

#[allow(clippy::needless_range_loop)] // Gaussian elimination updates rows in place from the pivot.
fn solve_6x6(mut matrix: [[f64; 6]; 6], mut rhs: [f64; 6]) -> Option<[f64; 6]> {
    for pivot in 0..6 {
        let row = (pivot..6)
            .max_by(|&a, &b| matrix[a][pivot].abs().total_cmp(&matrix[b][pivot].abs()))?;
        if matrix[row][pivot].abs() < 1e-10 {
            return None;
        }
        matrix.swap(pivot, row);
        rhs.swap(pivot, row);
        let divisor = matrix[pivot][pivot];
        for column in pivot..6 {
            matrix[pivot][column] /= divisor;
        }
        rhs[pivot] /= divisor;
        for row in 0..6 {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            for column in pivot..6 {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    Some(rhs)
}

fn axis_angle_rotation(vector: [f32; 3]) -> [f32; 9] {
    let angle = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if angle < 1e-8 {
        return SimilarityTransform::IDENTITY.rotation;
    }
    let [x, y, z] = vector.map(|value| value / angle);
    let (sin, cos) = angle.sin_cos();
    let one_minus_cos = 1.0 - cos;
    [
        cos + x * x * one_minus_cos,
        x * y * one_minus_cos - z * sin,
        x * z * one_minus_cos + y * sin,
        y * x * one_minus_cos + z * sin,
        cos + y * y * one_minus_cos,
        y * z * one_minus_cos - x * sin,
        z * x * one_minus_cos - y * sin,
        z * y * one_minus_cos + x * sin,
        cos + z * z * one_minus_cos,
    ]
}

fn matrix_mul(left: [f32; 9], right: [f32; 9]) -> [f32; 9] {
    std::array::from_fn(|index| {
        let row = index / 3;
        let column = index % 3;
        (0..3)
            .map(|k| left[row * 3 + k] * right[k * 3 + column])
            .sum()
    })
}

fn matrix_vec(matrix: [f32; 9], vector: [f32; 3]) -> [f32; 3] {
    [
        matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
        matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
        matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[allow(clippy::needless_range_loop)] // Jacobi rotations update symmetric rows/columns in-place.
fn largest_symmetric_eigenvector(mut matrix: [[f64; 4]; 4]) -> Result<[f64; 4], StitchError> {
    let mut vectors = [[0.0; 4]; 4];
    for i in 0..4 {
        vectors[i][i] = 1.0;
    }
    for _ in 0..48 {
        let mut p = 0;
        let mut q = 1;
        let mut largest = 0.0_f64;
        for row in 0..4 {
            for column in row + 1..4 {
                if matrix[row][column].abs() > largest {
                    largest = matrix[row][column].abs();
                    p = row;
                    q = column;
                }
            }
        }
        if largest < 1e-14 {
            break;
        }
        let angle = 0.5 * (2.0 * matrix[p][q]).atan2(matrix[q][q] - matrix[p][p]);
        let (sin, cos) = angle.sin_cos();
        for row in 0..4 {
            if row != p && row != q {
                let rp = matrix[row][p];
                let rq = matrix[row][q];
                matrix[row][p] = cos * rp - sin * rq;
                matrix[p][row] = matrix[row][p];
                matrix[row][q] = sin * rp + cos * rq;
                matrix[q][row] = matrix[row][q];
            }
        }
        let pp = matrix[p][p];
        let qq = matrix[q][q];
        let pq = matrix[p][q];
        matrix[p][p] = cos * cos * pp - 2.0 * sin * cos * pq + sin * sin * qq;
        matrix[q][q] = sin * sin * pp + 2.0 * sin * cos * pq + cos * cos * qq;
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;
        for row in 0..4 {
            let vp = vectors[row][p];
            let vq = vectors[row][q];
            vectors[row][p] = cos * vp - sin * vq;
            vectors[row][q] = sin * vp + cos * vq;
        }
    }
    let index = (0..4)
        .max_by(|&left, &right| matrix[left][left].total_cmp(&matrix[right][right]))
        .expect("four eigenvalues exist");
    let vector = std::array::from_fn(|row| vectors[row][index]);
    let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !norm.is_finite() || norm <= 1e-20 {
        return Err(StitchError::DegenerateGeometry);
    }
    Ok(vector.map(|value| value / norm))
}
fn mat_vec(r: [f64; 9], x: [f64; 3]) -> [f64; 3] {
    [
        r[0] * x[0] + r[1] * x[1] + r[2] * x[2],
        r[3] * x[0] + r[4] * x[1] + r[5] * x[2],
        r[6] * x[0] + r[7] * x[1] + r[8] * x[2],
    ]
}
fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}
fn normalize(vector: [f32; 3]) -> [f32; 3] {
    let length = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if !length.is_finite() || length <= 1e-12 {
        [0.0, 0.0, 1.0]
    } else {
        vector.map(|value| value / length)
    }
}
fn rotation_from_quaternion(q: [f64; 4]) -> [f64; 9] {
    let [w, x, y, z] = q;
    [
        1. - 2. * (y * y + z * z),
        2. * (x * y - z * w),
        2. * (x * z + y * w),
        2. * (x * y + z * w),
        1. - 2. * (x * x + z * z),
        2. * (y * z - x * w),
        2. * (x * z - y * w),
        2. * (y * z + x * w),
        1. - 2. * (x * x + y * y),
    ]
}

pub fn transform_points(
    points: &[MeasuredPoint],
    transform: SimilarityTransform,
) -> Vec<MeasuredPoint> {
    points
        .iter()
        .map(|p| MeasuredPoint {
            position: transform.apply(p.position),
            normal: transform.rotate(p.normal),
            radius: p.radius * transform.scale,
            ..*p
        })
        .collect()
}

pub fn stitch_measured_windows(
    windows: &[WindowMeasuredChunk],
) -> Result<FusedSceneChunk, StitchError> {
    stitch_measured_windows_with_settings(windows, StitchSettings::default())
}

pub fn stitch_measured_windows_with_settings(
    windows: &[WindowMeasuredChunk],
    settings: StitchSettings,
) -> Result<FusedSceneChunk, StitchError> {
    stitch_measured_windows_with_loop_closures(windows, settings, &[])
}

/// Fuses raw measured windows after optionally applying independently verified
/// non-sequential pose constraints. Each supplied edge must use the convention
/// documented by [`PoseGraphEdge`]: `from` is the earlier node and its
/// measurement maps `to` local coordinates into `from` local coordinates.
///
/// Empty loop input is deliberately a strict sequential-path no-op.
pub fn stitch_measured_windows_with_loop_closures(
    windows: &[WindowMeasuredChunk],
    settings: StitchSettings,
    loop_closures: &[PoseGraphEdge],
) -> Result<FusedSceneChunk, StitchError> {
    let Some(first) = windows.first() else {
        return Ok(FusedSceneChunk {
            alignments: Vec::new(),
            pose_graph: None,
            voxel_size: 0.0,
            points: Vec::new(),
        });
    };
    // Keep the local source chunks immutable until every pose constraint has
    // been established. `previous_global` is temporary sequential evidence for
    // the next direct overlap only; final fusion below reapplies the collected
    // local-to-global poses in one deferred pass.
    let mut poses = vec![SimilarityTransform::IDENTITY];
    let mut previous_global = transform_window(first, SimilarityTransform::IDENTITY);
    let mut alignments = Vec::new();
    let mut sequential_edges = Vec::new();
    for window in &windows[1..] {
        let report = align_overlapping_windows(window, &previous_global)?;
        let inlier_ratio = report.inlier_count as f32 / report.correspondence_count as f32;
        if report.correspondence_count < settings.minimum_correspondences
            || inlier_ratio < settings.minimum_inlier_ratio
            || report.transform.scale < settings.minimum_scale
            || report.transform.scale > settings.maximum_scale
        {
            return Err(StitchError::QualityGate);
        }
        let previous_pose = *poses.last().expect("first pose exists");
        let relative = previous_pose
            .inverse()
            .expect("validated sequential similarity is invertible")
            .compose(report.transform);
        sequential_edges.push(PoseGraphEdge {
            from: poses.len() - 1,
            to: poses.len(),
            measurement: relative,
            information: 1.0,
            loop_closure: false,
        });
        previous_global = transform_window(window, report.transform);
        poses.push(report.transform);
        alignments.push(report);
    }
    let pose_graph = if loop_closures.is_empty() {
        None
    } else {
        let mut edges = sequential_edges;
        edges.extend_from_slice(loop_closures);
        let mut graph = RelativePoseGraph {
            nodes: poses,
            edges,
            fixed: Vec::new(),
        };
        let report = optimize_relative_pose_graph(&mut graph, PoseGraphSettings::default())?;
        poses = graph.nodes;
        Some(report)
    };
    let global = windows
        .iter()
        .zip(poses)
        .map(|(window, pose)| transform_window(window, pose))
        .collect::<Vec<_>>();
    let mut radii = global
        .iter()
        .flat_map(|w| w.views.iter())
        .flat_map(|f| f.points.iter())
        .map(|p| p.radius)
        .filter(|r| r.is_finite() && *r > 0.0)
        .collect::<Vec<_>>();
    if radii.is_empty() {
        return Ok(FusedSceneChunk {
            alignments,
            pose_graph,
            voxel_size: 0.0,
            points: Vec::new(),
        });
    }
    radii.sort_by(f32::total_cmp);
    let voxel_size = (radii[radii.len() / 2] * 2.0).max(1e-6);
    let mut cells: HashMap<(i32, i32, i32), VoxelCell> = HashMap::new();
    for window in &global {
        for frame in &window.views {
            for point in &frame.points {
                if !point.position.iter().all(|value| value.is_finite())
                    || !point.radius.is_finite()
                    || point.radius <= 0.0
                    || !point.confidence.is_finite()
                    || point.confidence <= 0.0
                {
                    continue;
                }
                let key = (
                    // Alignment residuals around an exact boundary must not
                    // turn one physical surfel into two adjacent voxels.
                    (point.position[0] / voxel_size).round() as i32,
                    (point.position[1] / voxel_size).round() as i32,
                    (point.position[2] / voxel_size).round() as i32,
                );
                let cell = cells
                    .entry(key)
                    .or_insert((0.0, [0.0; 3], [0.0; 3], [0.0; 3], 0.0, 0));
                let weight = point.confidence.max(0.0);
                cell.0 += weight;
                for d in 0..3 {
                    cell.1[d] += weight * point.position[d];
                    cell.2[d] += weight * f32::from(point.color_srgb[d]);
                    cell.3[d] += weight * point.normal[d];
                }
                cell.4 += point.radius;
                cell.5 += 1;
            }
        }
    }
    let mut points = cells
        .into_values()
        .map(|(weight, position, color, normal, radius, contributors)| {
            let d = weight.max(1e-6);
            FusedPoint {
                position: position.map(|v| v / d),
                normal: normalize(normal.map(|v| v / d)),
                color_srgb: color.map(|v| (v / d).round().clamp(0.0, 255.0) as u8),
                confidence: weight / contributors as f32,
                radius: radius / contributors as f32,
                contributors,
            }
        })
        .collect::<Vec<_>>();
    points.sort_by(|a, b| {
        a.position[0]
            .total_cmp(&b.position[0])
            .then(a.position[1].total_cmp(&b.position[1]))
            .then(a.position[2].total_cmp(&b.position[2]))
    });
    Ok(FusedSceneChunk {
        alignments,
        pose_graph,
        voxel_size,
        points,
    })
}

fn transform_window(
    window: &WindowMeasuredChunk,
    transform: SimilarityTransform,
) -> WindowMeasuredChunk {
    let mut out = window.clone();
    for frame in &mut out.views {
        frame.points = transform_points(&frame.points, transform);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MeasuredFrameChunk;
    fn chunk(points: Vec<MeasuredPoint>) -> WindowMeasuredChunk {
        WindowMeasuredChunk {
            window: crate::FrameWindow {
                index: 0,
                start: 0,
                end: 1,
            },
            views: vec![MeasuredFrameChunk {
                frame_index: 7,
                camera: crate::CameraCalibration {
                    world_to_camera: [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
                    intrinsics: [1., 0., 0., 0., 1., 0., 0., 0., 1.],
                },
                points,
            }],
        }
    }
    fn p(i: u32, pos: [f32; 3]) -> MeasuredPoint {
        MeasuredPoint {
            position: pos,
            normal: [0.0, 0.0, 1.0],
            color_srgb: [0; 3],
            confidence: 1.,
            radius: 1.,
            source_pixel: [i, 0],
        }
    }
    #[test]
    fn recovers_known_similarity_from_shared_pixels() {
        let source = chunk(vec![
            p(0, [0., 0., 0.]),
            p(1, [1., 0., 0.]),
            p(2, [0., 1., 0.]),
            p(3, [0., 0., 1.]),
        ]);
        let target = chunk(vec![
            p(0, [5., -3., 1.]),
            p(1, [5., -1., 1.]),
            p(2, [3., -3., 1.]),
            p(3, [5., -3., 3.]),
        ]);
        let report = align_overlapping_windows(&source, &target).unwrap();
        assert!((report.transform.scale - 2.).abs() < 1e-4);
        assert!(distance(report.transform.apply([1., 0., 0.]), [5., -1., 1.]) < 1e-4);
    }

    #[test]
    fn similarity_scales_surfel_radius_with_position() {
        let transformed = transform_points(
            &[MeasuredPoint {
                position: [1.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                color_srgb: [1, 2, 3],
                confidence: 1.0,
                radius: 0.25,
                source_pixel: [0, 0],
            }],
            SimilarityTransform {
                scale: 2.0,
                ..SimilarityTransform::IDENTITY
            },
        );
        assert_eq!(transformed[0].position, [2.0, 0.0, 0.0]);
        assert_eq!(transformed[0].radius, 0.5);
        assert_eq!(transformed[0].normal, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn similarity_composition_and_inverse_preserve_points() {
        let quarter_turn = SimilarityTransform {
            scale: 2.0,
            rotation: [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            translation: [3.0, -2.0, 1.0],
        };
        let inner = SimilarityTransform {
            scale: 0.5,
            translation: [-4.0, 2.0, 0.0],
            ..SimilarityTransform::IDENTITY
        };
        let point = [2.0, -1.0, 3.0];
        let composed = quarter_turn.compose(inner);
        assert!(
            distance(
                composed.apply(point),
                quarter_turn.apply(inner.apply(point))
            ) < 1e-5
        );
        let recovered = composed.inverse().unwrap().apply(composed.apply(point));
        assert!(distance(recovered, point) < 1e-5, "{recovered:?}");
    }

    #[test]
    fn rejects_collinear_overlap_that_cannot_determine_rotation() {
        let source = chunk(vec![
            p(0, [0.0, 0.0, 0.0]),
            p(1, [1.0, 0.0, 0.0]),
            p(2, [2.0, 0.0, 0.0]),
        ]);
        let target = chunk(vec![
            p(0, [3.0, 2.0, 1.0]),
            p(1, [5.0, 2.0, 1.0]),
            p(2, [7.0, 2.0, 1.0]),
        ]);
        assert_eq!(
            align_overlapping_windows(&source, &target),
            Err(StitchError::DegenerateGeometry)
        );
    }

    #[test]
    fn production_fusion_rejects_a_tiny_overlap() {
        let target = chunk(vec![
            p(0, [0.0, 0.0, 0.0]),
            p(1, [1.0, 0.0, 0.0]),
            p(2, [0.0, 1.0, 0.0]),
            p(3, [0.0, 0.0, 1.0]),
        ]);
        let mut source = target.clone();
        source.window.index = 1;
        assert_eq!(
            stitch_measured_windows(&[target, source]),
            Err(StitchError::QualityGate)
        );
    }

    #[test]
    fn point_to_plane_refinement_corrects_small_dense_translation() {
        let translation = [0.01_f64, -0.02, 0.03];
        let source = [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ];
        let matches = source
            .into_iter()
            .map(|point| Match {
                source: point,
                target: [
                    point[0] + translation[0],
                    point[1] + translation[1],
                    point[2] + translation[2],
                ],
                target_normal: point.map(|value: f64| value / 3.0_f64.sqrt()),
                weight: 1.0,
            })
            .collect::<Vec<_>>();
        let selected = (0..matches.len()).collect::<Vec<_>>();
        let refined = refine_point_to_plane(SimilarityTransform::IDENTITY, &matches, &selected);
        assert!(
            distance(refined.translation, translation.map(|value| value as f32)) < 1e-4,
            "{:?}",
            refined.translation
        );
    }

    #[test]
    fn sequential_stitching_defers_transforming_immutable_raw_windows() {
        let base = vec![
            p(0, [0.0, 0.0, 0.0]),
            p(1, [1.0, 0.0, 0.0]),
            p(2, [0.0, 1.0, 0.0]),
            p(3, [0.0, 0.0, 1.0]),
        ];
        let first = chunk(base.clone());
        let mut second = chunk(
            base.iter()
                .copied()
                .map(|mut point| {
                    point.position[0] -= 5.0;
                    point
                })
                .collect(),
        );
        second.window.index = 1;
        let mut third = chunk(
            base.iter()
                .copied()
                .map(|mut point| {
                    point.position[0] -= 10.0;
                    point
                })
                .collect(),
        );
        third.window.index = 2;
        let raw = vec![first, second, third];
        let fused = stitch_measured_windows_with_settings(
            &raw,
            StitchSettings {
                minimum_correspondences: 3,
                minimum_inlier_ratio: 0.5,
                ..StitchSettings::default()
            },
        )
        .unwrap();
        assert_eq!(raw[2].views[0].points[0].position, [-10.0, 0.0, 0.0]);
        assert_eq!(fused.alignments.len(), 2);
        assert!(fused.points.iter().all(|point| point.contributors == 3));
    }

    #[test]
    fn verified_loop_constraints_are_optimized_before_deferred_fusion() {
        let base = vec![
            p(0, [0.0, 0.0, 0.0]),
            p(1, [1.0, 0.0, 0.0]),
            p(2, [0.0, 1.0, 0.0]),
            p(3, [0.0, 0.0, 1.0]),
        ];
        let first = chunk(base.clone());
        let mut second = chunk(
            base.iter()
                .copied()
                .map(|mut point| {
                    point.position[0] -= 5.0;
                    point
                })
                .collect(),
        );
        second.window.index = 1;
        let mut third = chunk(
            base.iter()
                .copied()
                .map(|mut point| {
                    point.position[0] -= 10.0;
                    point
                })
                .collect(),
        );
        third.window.index = 2;
        let fused = stitch_measured_windows_with_loop_closures(
            &[first, second, third],
            StitchSettings {
                minimum_correspondences: 3,
                minimum_inlier_ratio: 0.5,
                ..StitchSettings::default()
            },
            &[PoseGraphEdge {
                from: 0,
                to: 2,
                measurement: SimilarityTransform {
                    translation: [10.0, 0.0, 0.0],
                    ..SimilarityTransform::IDENTITY
                },
                information: 1.0,
                loop_closure: true,
            }],
        )
        .unwrap();
        let report = fused.pose_graph.expect("loop graph should run");
        assert_eq!(report.loop_edges, 1);
        assert!(report.final_cost < 1e-9, "{report:?}");
        assert!(fused.points.iter().all(|point| point.contributors == 3));
    }
}
