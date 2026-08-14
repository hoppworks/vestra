//! Deterministic point-to-plane ICP matching the PR #2 geometry contract.
//!
//! This is intentionally independent from the older, bounded seam correction
//! in `stitch`: PR #2 re-establishes nearest neighbours on every iteration and
//! retains the scale of the preceding Sim(3) estimate.

use std::collections::HashMap;

use crate::SimilarityTransform;

#[derive(Debug, Clone, Copy)]
pub(crate) struct IcpSettings {
    pub max_iterations: usize,
    pub max_correspondence_distance: f32,
    pub huber_delta: f32,
    pub normal_radius: f32,
    pub minimum_correspondences: usize,
    pub rank_tolerance: f64,
    pub step_tolerance: f64,
    pub relative_rmse: f64,
}

impl Default for IcpSettings {
    fn default() -> Self {
        Self {
            max_iterations: 30,
            // `loop_match_dist` is 0.30, but PR #2's subsequent ICP keeps its
            // own tighter default IcpParams gate of 0.10.
            max_correspondence_distance: 0.10,
            huber_delta: 0.05,
            normal_radius: 0.15,
            minimum_correspondences: 20,
            rank_tolerance: 1e-3,
            step_tolerance: 1e-7,
            relative_rmse: 1e-6,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct IcpResult {
    pub transform: SimilarityTransform,
    pub rms_before: f32,
    pub rms_after: f32,
    pub correspondences: usize,
    pub iterations: usize,
    pub rank: usize,
}

/// Refines `initial` (source -> target) with the PR #2 point-to-plane method.
/// The returned transform always preserves the initial Sim(3) scale.
pub(crate) fn refine_point_to_plane(
    source: &[[f32; 3]],
    target: &[[f32; 3]],
    initial: SimilarityTransform,
    settings: IcpSettings,
) -> Option<IcpResult> {
    if source.is_empty() || target.is_empty() || !initial.scale.is_finite() || initial.scale <= 0.0
    {
        return None;
    }
    let target_normals = estimate_normals(target, settings.normal_radius);
    if target_normals.iter().all(|normal| *normal == [0.0; 3]) {
        return None;
    }
    let hash = SpatialHash::new(target, settings.max_correspondence_distance);
    let mut points = source
        .iter()
        .map(|&point| initial.apply(point))
        .collect::<Vec<_>>();
    let rms_before = plane_rms(
        &points,
        target,
        &target_normals,
        &hash,
        settings.max_correspondence_distance,
    )?;
    let mut previous_rms = rms_before;
    let mut incremental = SimilarityTransform::IDENTITY;
    let mut iterations = 0;
    let mut last_rank = 0;
    let mut last_correspondences = 0;

    for iteration in 0..settings.max_iterations {
        let mut h = [[0.0_f64; 6]; 6];
        let mut g = [0.0_f64; 6];
        let mut correspondences = 0;
        for point in &points {
            let Some(index) = hash.nearest(*point, settings.max_correspondence_distance) else {
                continue;
            };
            let normal = target_normals[index];
            if normal == [0.0; 3] {
                continue;
            }
            let residual = dot(subtract(*point, target[index]), normal) as f64;
            let point64 = point.map(f64::from);
            let normal64 = normal.map(f64::from);
            let cross = cross(point64, normal64);
            // Exactly the reference parameter order: omega first, translation second.
            let row = [
                cross[0],
                cross[1],
                cross[2],
                normal64[0],
                normal64[1],
                normal64[2],
            ];
            let weight =
                if settings.huber_delta > 0.0 && residual.abs() > f64::from(settings.huber_delta) {
                    f64::from(settings.huber_delta) / residual.abs()
                } else {
                    1.0
                };
            for left in 0..6 {
                g[left] += weight * row[left] * residual;
                for right in 0..6 {
                    h[left][right] += weight * row[left] * row[right];
                }
            }
            correspondences += 1;
        }
        last_correspondences = correspondences;
        if correspondences < settings.minimum_correspondences {
            break;
        }
        let (step, rank) = rank_truncated_solve(h, g, settings.rank_tolerance);
        last_rank = rank;
        let rotation = rodrigues([step[0], step[1], step[2]]);
        let translation = [step[3] as f32, step[4] as f32, step[5] as f32];
        let delta = SimilarityTransform {
            scale: 1.0,
            rotation,
            translation,
        };
        points
            .iter_mut()
            .for_each(|point| *point = delta.apply(*point));
        incremental = delta.compose(incremental);
        let rms = plane_rms(
            &points,
            target,
            &target_normals,
            &hash,
            settings.max_correspondence_distance,
        )?;
        iterations = iteration + 1;
        let step_norm = step.iter().map(|value| value * value).sum::<f64>().sqrt();
        if step_norm < settings.step_tolerance
            || (previous_rms > 0.0
                && f64::from(previous_rms - rms)
                    <= settings.relative_rmse * f64::from(previous_rms))
        {
            break;
        }
        previous_rms = rms;
    }
    let rms_after = plane_rms(
        &points,
        target,
        &target_normals,
        &hash,
        settings.max_correspondence_distance,
    )?;
    Some(IcpResult {
        transform: incremental.compose(initial),
        rms_before,
        rms_after,
        correspondences: last_correspondences,
        iterations,
        rank: last_rank,
    })
}

fn estimate_normals(points: &[[f32; 3]], radius: f32) -> Vec<[f32; 3]> {
    let hash = SpatialHash::new(points, radius);
    points
        .iter()
        .map(|&point| {
            let neighbours = hash.radius(point, radius);
            if neighbours.len() < 3 {
                return [0.0; 3];
            }
            let mut mean = [0.0_f64; 3];
            for &index in &neighbours {
                for axis in 0..3 {
                    mean[axis] += f64::from(points[index][axis]);
                }
            }
            for value in &mut mean {
                *value /= neighbours.len() as f64;
            }
            let mut covariance = [[0.0_f64; 3]; 3];
            for &index in &neighbours {
                let centered: [f64; 3] =
                    std::array::from_fn(|axis| f64::from(points[index][axis]) - mean[axis]);
                for row in 0..3 {
                    for column in 0..3 {
                        covariance[row][column] += centered[row] * centered[column];
                    }
                }
            }
            let (values, vectors) = jacobi_eigen_3(covariance);
            let minimum = (0..3)
                .min_by(|&left, &right| values[left].total_cmp(&values[right]))
                .unwrap_or(0);
            let normal = [
                vectors[0][minimum],
                vectors[1][minimum],
                vectors[2][minimum],
            ];
            normalize64(normal)
                .map(|normal| normal.map(|value| value as f32))
                .unwrap_or([0.0; 3])
        })
        .collect()
}

fn plane_rms(
    source: &[[f32; 3]],
    target: &[[f32; 3]],
    normals: &[[f32; 3]],
    hash: &SpatialHash,
    maximum_distance: f32,
) -> Option<f32> {
    let mut squared_error = 0.0_f64;
    let mut count = 0_usize;
    for &point in source {
        let Some(index) = hash.nearest(point, maximum_distance) else {
            continue;
        };
        if normals[index] == [0.0; 3] {
            continue;
        }
        let residual = f64::from(dot(subtract(point, target[index]), normals[index]));
        squared_error += residual * residual;
        count += 1;
    }
    (count > 0).then(|| (squared_error / count as f64).sqrt() as f32)
}

#[derive(Debug)]
struct SpatialHash {
    cell: f32,
    points: Vec<[f32; 3]>,
    cells: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl SpatialHash {
    fn new(points: &[[f32; 3]], cell: f32) -> Self {
        let cell = cell.max(1e-6);
        let mut cells = HashMap::new();
        for (index, &point) in points.iter().enumerate() {
            cells
                .entry(cell_for(point, cell))
                .or_insert_with(Vec::new)
                .push(index);
        }
        Self {
            cell,
            points: points.to_vec(),
            cells,
        }
    }

    fn radius(&self, point: [f32; 3], radius: f32) -> Vec<usize> {
        let base = cell_for(point, self.cell);
        let span = (radius / self.cell).ceil() as i32;
        let mut output = Vec::new();
        for x in base.0 - span..=base.0 + span {
            for y in base.1 - span..=base.1 + span {
                for z in base.2 - span..=base.2 + span {
                    output.extend(self.cells.get(&(x, y, z)).into_iter().flatten().copied());
                }
            }
        }
        let radius_squared = radius * radius;
        output.retain(|&index| squared_distance(self.points[index], point) <= radius_squared);
        output
    }

    fn nearest(&self, point: [f32; 3], maximum_distance: f32) -> Option<usize> {
        let base = cell_for(point, self.cell);
        let span = (maximum_distance / self.cell).ceil() as i32;
        let mut result = None;
        let mut best_squared = maximum_distance * maximum_distance;
        for x in base.0 - span..=base.0 + span {
            for y in base.1 - span..=base.1 + span {
                for z in base.2 - span..=base.2 + span {
                    for &index in self.cells.get(&(x, y, z)).into_iter().flatten() {
                        let squared = squared_distance(self.points[index], point);
                        if squared <= best_squared {
                            result = Some(index);
                            best_squared = squared;
                        }
                    }
                }
            }
        }
        result
    }
}

fn cell_for(point: [f32; 3], cell: f32) -> (i32, i32, i32) {
    (
        (point[0] / cell).floor() as i32,
        (point[1] / cell).floor() as i32,
        (point[2] / cell).floor() as i32,
    )
}
fn squared_distance(left: [f32; 3], right: [f32; 3]) -> f32 {
    subtract(left, right)
        .iter()
        .map(|value| value * value)
        .sum()
}

fn subtract(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}
fn dot(left: [f32; 3], right: [f32; 3]) -> f32 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}
fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}
fn normalize64(value: [f64; 3]) -> Option<[f64; 3]> {
    let length = value.iter().map(|v| v * v).sum::<f64>().sqrt();
    (length > 0.0 && length.is_finite()).then(|| value.map(|v| v / length))
}

fn rodrigues(vector: [f64; 3]) -> [f32; 9] {
    let theta = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
    if theta < 1e-12 {
        return SimilarityTransform::IDENTITY.rotation;
    }
    let [x, y, z] = vector.map(|value| value / theta);
    let (sine, cosine) = theta.sin_cos();
    let one_minus_cosine = 1.0 - cosine;
    [
        cosine + x * x * one_minus_cosine,
        x * y * one_minus_cosine - z * sine,
        x * z * one_minus_cosine + y * sine,
        y * x * one_minus_cosine + z * sine,
        cosine + y * y * one_minus_cosine,
        y * z * one_minus_cosine - x * sine,
        z * x * one_minus_cosine - y * sine,
        z * y * one_minus_cosine + x * sine,
        cosine + z * z * one_minus_cosine,
    ]
    .map(|value| value as f32)
}

fn rank_truncated_solve(
    matrix: [[f64; 6]; 6],
    gradient: [f64; 6],
    rank_tolerance: f64,
) -> ([f64; 6], usize) {
    let (values, vectors) = jacobi_eigen_6(matrix);
    let maximum = values.iter().copied().fold(0.0_f64, f64::max);
    let mut solution = [0.0; 6];
    let mut rank = 0;
    for column in 0..6 {
        if values[column] <= rank_tolerance * maximum || values[column] <= 1e-15 {
            continue;
        }
        let dot_gradient = (0..6)
            .map(|row| vectors[row][column] * -gradient[row])
            .sum::<f64>();
        let coefficient = dot_gradient / values[column];
        for row in 0..6 {
            solution[row] += coefficient * vectors[row][column];
        }
        rank += 1;
    }
    (solution, rank)
}

fn jacobi_eigen_3(matrix: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    jacobi_eigen(matrix)
}
fn jacobi_eigen_6(matrix: [[f64; 6]; 6]) -> ([f64; 6], [[f64; 6]; 6]) {
    jacobi_eigen(matrix)
}

fn jacobi_eigen<const N: usize>(mut matrix: [[f64; N]; N]) -> ([f64; N], [[f64; N]; N]) {
    let mut vectors =
        std::array::from_fn(|row| std::array::from_fn(|column| f64::from(row == column)));
    for _ in 0..(N * N * 32) {
        let (p, q, largest) = (0..N)
            .flat_map(|row| (row + 1..N).map(move |column| (row, column)))
            .map(|(row, column)| (row, column, matrix[row][column].abs()))
            .max_by(|left, right| left.2.total_cmp(&right.2))
            .unwrap_or((0, 0, 0.0));
        if largest <= 1e-14 {
            break;
        }
        let phi = 0.5 * (2.0 * matrix[p][q]).atan2(matrix[q][q] - matrix[p][p]);
        let (sine, cosine) = phi.sin_cos();
        for index in 0..N {
            if index == p || index == q {
                continue;
            }
            let mp = matrix[index][p];
            let mq = matrix[index][q];
            matrix[index][p] = cosine * mp - sine * mq;
            matrix[p][index] = matrix[index][p];
            matrix[index][q] = sine * mp + cosine * mq;
            matrix[q][index] = matrix[index][q];
        }
        let pp = matrix[p][p];
        let qq = matrix[q][q];
        let pq = matrix[p][q];
        matrix[p][p] = cosine * cosine * pp - 2.0 * sine * cosine * pq + sine * sine * qq;
        matrix[q][q] = sine * sine * pp + 2.0 * sine * cosine * pq + cosine * cosine * qq;
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;
        for row in 0..N {
            let vp = vectors[row][p];
            let vq = vectors[row][q];
            vectors[row][p] = cosine * vp - sine * vq;
            vectors[row][q] = sine * vp + cosine * vq;
        }
    }
    (std::array::from_fn(|index| matrix[index][index]), vectors)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(z: f32) -> Vec<[f32; 3]> {
        (-5..=5)
            .flat_map(|y| (-5..=5).map(move |x| [x as f32 * 0.02, y as f32 * 0.02, z]))
            .collect()
    }

    #[test]
    fn iterative_icp_recovers_normal_translation_and_keeps_scale() {
        let target = plane(0.0);
        let source = plane(0.03);
        let initial = SimilarityTransform {
            scale: 1.7,
            ..SimilarityTransform::IDENTITY
        };
        // Supply source in the scale-normalized local coordinates so the
        // pre-transformed initial cloud is exactly the offset plane.
        let source = source
            .into_iter()
            .map(|mut point| {
                point.iter_mut().for_each(|value| *value /= initial.scale);
                point
            })
            .collect::<Vec<_>>();
        let result = refine_point_to_plane(&source, &target, initial, IcpSettings::default())
            .expect("dense planar correspondence should be solvable");
        assert!(result.correspondences >= 20);
        assert!((result.transform.scale - 1.7).abs() < 1e-6);
        assert!((result.transform.translation[2] + 0.03).abs() < 2e-4);
        assert!(result.rms_after < result.rms_before * 0.01);
    }

    #[test]
    fn iterative_icp_refuses_insufficient_overlap() {
        let target = plane(0.0);
        let source = vec![[0.0, 0.0, 1.0]; 19];
        assert!(
            refine_point_to_plane(
                &source,
                &target,
                SimilarityTransform::IDENTITY,
                IcpSettings::default(),
            )
            .is_none()
        );
    }
}
