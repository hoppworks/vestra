//! Double-precision Sim(3) and pose-graph arithmetic for the pinned PR #2
//! oracle path. Persisted Vestra scene data stays F32; this module prevents
//! premature rounding while reproducing the reference's CPU geometry solver.
//!
//! Adapted from `depth-anything.cpp` PR #2 at commit
//! `f56e9be43a22c12ef575584d2fa57a6a5d5be7ae` (MIT). See the repository's
//! `THIRD_PARTY_NOTICES.md` for the full notice.

use crate::{
    PoseGraphEdge, PoseGraphError, PoseGraphReport, PoseGraphSettings, SimilarityTransform,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct F64Sim3 {
    pub scale: f64,
    pub rotation: [f64; 9],
    pub translation: [f64; 3],
}

impl F64Sim3 {
    pub(crate) const IDENTITY: Self = Self {
        scale: 1.0,
        rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        translation: [0.0; 3],
    };

    pub(crate) fn from_f32(value: SimilarityTransform) -> Self {
        Self {
            scale: f64::from(value.scale),
            rotation: value.rotation.map(f64::from),
            translation: value.translation.map(f64::from),
        }
    }

    pub(crate) fn to_f32(self) -> SimilarityTransform {
        SimilarityTransform {
            scale: self.scale as f32,
            rotation: self.rotation.map(|value| value as f32),
            translation: self.translation.map(|value| value as f32),
        }
    }

    pub(crate) fn compose(self, inner: Self) -> Self {
        let rotation = multiply(self.rotation, inner.rotation);
        let inner_translation = multiply_vector(self.rotation, inner.translation);
        Self {
            scale: self.scale * inner.scale,
            rotation,
            translation: [
                self.translation[0] + self.scale * inner_translation[0],
                self.translation[1] + self.scale * inner_translation[1],
                self.translation[2] + self.scale * inner_translation[2],
            ],
        }
    }

    pub(crate) fn inverse(self) -> Option<Self> {
        (self.scale.is_finite() && self.scale > 0.0).then(|| {
            let rotation = transpose(self.rotation);
            let translated = multiply_vector(rotation, self.translation);
            Self {
                scale: self.scale.recip(),
                rotation,
                translation: translated.map(|value| -value / self.scale),
            }
        })
    }
}

pub(crate) fn optimize_cpp_pr2_pose_graph_f64(
    initial_nodes: &[SimilarityTransform],
    edges: &[PoseGraphEdge],
    settings: PoseGraphSettings,
) -> Result<(Vec<SimilarityTransform>, PoseGraphReport), PoseGraphError> {
    let mut nodes = initial_nodes
        .iter()
        .copied()
        .map(F64Sim3::from_f32)
        .collect::<Vec<_>>();
    let edges = edges
        .iter()
        .map(|edge| F64Edge {
            from: edge.from,
            to: edge.to,
            measurement: F64Sim3::from_f32(edge.measurement),
            information: f64::from(edge.information),
            loop_closure: edge.loop_closure,
        })
        .collect::<Vec<_>>();
    if edges.iter().any(|edge| {
        edge.from >= nodes.len()
            || edge.to >= nodes.len()
            || !edge.information.is_finite()
            || edge.information <= 0.0
            || edge.measurement.inverse().is_none()
    }) {
        return Err(PoseGraphError::InvalidMeasurement);
    }
    let variables = nodes.len().saturating_sub(1) * 7;
    let initial_cost = cost(&nodes, &edges)?;
    let mut current_cost = initial_cost;
    let mut damping = settings.initial_damping.max(1e-12);
    let mut iterations = 0;
    for iteration in 0..settings.max_iterations {
        if variables == 0 {
            break;
        }
        let (normal, rhs) =
            normal_equations(&nodes, &edges, variables, settings.finite_difference_step)?;
        let mut accepted = false;
        for _ in 0..12 {
            let mut system = normal.clone();
            for index in 0..variables {
                system[index * variables + index] +=
                    damping * (normal[index * variables + index] + 1e-12);
            }
            let Some(step) = cholesky_solve(system, rhs.clone(), variables) else {
                damping *= 4.0;
                continue;
            };
            let trial = nodes
                .iter()
                .enumerate()
                .map(|(index, node)| {
                    if index == 0 {
                        Ok(*node)
                    } else {
                        retract(*node, step[(index - 1) * 7..index * 7].try_into().unwrap())
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            let trial_cost = cost(&trial, &edges)?;
            if trial_cost < current_cost {
                let improvement = current_cost - trial_cost;
                nodes = trial;
                current_cost = trial_cost;
                damping = (damping * 0.5).max(1e-12);
                iterations = iteration + 1;
                accepted = true;
                if improvement <= settings.convergence_relative_cost * (initial_cost + 1e-30) {
                    break;
                }
                break;
            }
            damping *= 4.0;
        }
        if !accepted {
            break;
        }
    }
    Ok((
        nodes.into_iter().map(F64Sim3::to_f32).collect(),
        PoseGraphReport {
            initial_cost,
            final_cost: current_cost,
            iterations,
            free_degrees_of_freedom: variables,
            loop_edges: edges.iter().filter(|edge| edge.loop_closure).count(),
        },
    ))
}

#[derive(Clone, Copy)]
struct F64Edge {
    from: usize,
    to: usize,
    measurement: F64Sim3,
    information: f64,
    loop_closure: bool,
}

fn residual(from: F64Sim3, to: F64Sim3, measurement: F64Sim3) -> Result<[f64; 7], PoseGraphError> {
    let error = measurement
        .inverse()
        .and_then(|inverse_measurement| {
            from.inverse()
                .map(|inverse_from| inverse_measurement.compose(inverse_from.compose(to)))
        })
        .ok_or(PoseGraphError::InvalidMeasurement)?;
    let rotation = so3_log(error.rotation);
    Ok([
        error.translation[0],
        error.translation[1],
        error.translation[2],
        rotation[0],
        rotation[1],
        rotation[2],
        error.scale.ln(),
    ])
}

fn cost(nodes: &[F64Sim3], edges: &[F64Edge]) -> Result<f64, PoseGraphError> {
    edges.iter().try_fold(0.0, |total, edge| {
        let residual = residual(nodes[edge.from], nodes[edge.to], edge.measurement)?;
        Ok(
            total
                + 0.5 * edge.information * residual.iter().map(|value| value * value).sum::<f64>(),
        )
    })
}

fn normal_equations(
    nodes: &[F64Sim3],
    edges: &[F64Edge],
    variables: usize,
    step: f64,
) -> Result<(Vec<f64>, Vec<f64>), PoseGraphError> {
    let mut normal = vec![0.0; variables * variables];
    let mut rhs = vec![0.0; variables];
    for edge in edges {
        let base_residual = residual(nodes[edge.from], nodes[edge.to], edge.measurement)?;
        let endpoints = [edge.from, edge.to];
        let mut jacobians = [[0.0; 49]; 2];
        for endpoint_index in 0..2 {
            let node = endpoints[endpoint_index];
            if node == 0 {
                continue;
            }
            for axis in 0..7 {
                let mut positive = [0.0; 7];
                positive[axis] = step;
                let mut negative = [0.0; 7];
                negative[axis] = -step;
                let plus = retract(nodes[node], positive)?;
                let minus = retract(nodes[node], negative)?;
                let plus_residual = if endpoint_index == 0 {
                    residual(plus, nodes[edge.to], edge.measurement)?
                } else {
                    residual(nodes[edge.from], plus, edge.measurement)?
                };
                let minus_residual = if endpoint_index == 0 {
                    residual(minus, nodes[edge.to], edge.measurement)?
                } else {
                    residual(nodes[edge.from], minus, edge.measurement)?
                };
                for row in 0..7 {
                    jacobians[endpoint_index][row * 7 + axis] =
                        (plus_residual[row] - minus_residual[row]) / (2.0 * step);
                }
            }
        }
        for left in 0..2 {
            if endpoints[left] == 0 {
                continue;
            }
            let left_base = (endpoints[left] - 1) * 7;
            for column in 0..7 {
                for row in 0..7 {
                    rhs[left_base + column] -=
                        edge.information * jacobians[left][row * 7 + column] * base_residual[row];
                }
            }
            for right in 0..2 {
                if endpoints[right] == 0 {
                    continue;
                }
                let right_base = (endpoints[right] - 1) * 7;
                for left_column in 0..7 {
                    for right_column in 0..7 {
                        let value = (0..7)
                            .map(|row| {
                                jacobians[left][row * 7 + left_column]
                                    * jacobians[right][row * 7 + right_column]
                            })
                            .sum::<f64>();
                        normal
                            [(left_base + left_column) * variables + right_base + right_column] +=
                            edge.information * value;
                    }
                }
            }
        }
    }
    Ok((normal, rhs))
}

fn retract(transform: F64Sim3, delta: [f64; 7]) -> Result<F64Sim3, PoseGraphError> {
    let scale = transform.scale * delta[6].exp();
    if !scale.is_finite() || scale <= 0.0 {
        return Err(PoseGraphError::InvalidMeasurement);
    }
    Ok(F64Sim3 {
        scale,
        rotation: multiply(transform.rotation, so3_exp([delta[3], delta[4], delta[5]])),
        translation: [
            transform.translation[0] + delta[0],
            transform.translation[1] + delta[1],
            transform.translation[2] + delta[2],
        ],
    })
}

fn cholesky_solve(mut matrix: Vec<f64>, mut rhs: Vec<f64>, size: usize) -> Option<Vec<f64>> {
    for row in 0..size {
        for column in 0..=row {
            let value = matrix[row * size + column]
                - (0..column)
                    .map(|inner| matrix[row * size + inner] * matrix[column * size + inner])
                    .sum::<f64>();
            if row == column {
                if !value.is_finite() || value <= 0.0 {
                    return None;
                }
                matrix[row * size + column] = value.sqrt();
            } else {
                matrix[row * size + column] = value / matrix[column * size + column];
            }
        }
    }
    for row in 0..size {
        rhs[row] = (rhs[row]
            - (0..row)
                .map(|column| matrix[row * size + column] * rhs[column])
                .sum::<f64>())
            / matrix[row * size + row];
    }
    for row in (0..size).rev() {
        rhs[row] = (rhs[row]
            - ((row + 1)..size)
                .map(|column| matrix[column * size + row] * rhs[column])
                .sum::<f64>())
            / matrix[row * size + row];
    }
    Some(rhs)
}

fn multiply(left: [f64; 9], right: [f64; 9]) -> [f64; 9] {
    std::array::from_fn(|index| {
        let row = index / 3;
        let column = index % 3;
        (0..3)
            .map(|inner| left[row * 3 + inner] * right[inner * 3 + column])
            .sum()
    })
}
fn multiply_vector(matrix: [f64; 9], vector: [f64; 3]) -> [f64; 3] {
    [
        matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
        matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
        matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2],
    ]
}
fn transpose(matrix: [f64; 9]) -> [f64; 9] {
    [
        matrix[0], matrix[3], matrix[6], matrix[1], matrix[4], matrix[7], matrix[2], matrix[5],
        matrix[8],
    ]
}
fn so3_exp(vector: [f64; 3]) -> [f64; 9] {
    let theta = vector.iter().map(|v| v * v).sum::<f64>().sqrt();
    if theta < 1e-12 {
        return F64Sim3::IDENTITY.rotation;
    }
    let [x, y, z] = vector.map(|v| v / theta);
    let c = theta.cos();
    let s = theta.sin();
    let q = 1.0 - c;
    [
        c + x * x * q,
        x * y * q - z * s,
        x * z * q + y * s,
        y * x * q + z * s,
        c + y * y * q,
        y * z * q - x * s,
        z * x * q - y * s,
        z * y * q + x * s,
        c + z * z * q,
    ]
}
fn so3_log(rotation: [f64; 9]) -> [f64; 3] {
    let c = ((rotation[0] + rotation[4] + rotation[8] - 1.0) * 0.5).clamp(-1.0, 1.0);
    let theta = c.acos();
    let vector = [
        rotation[7] - rotation[5],
        rotation[2] - rotation[6],
        rotation[3] - rotation[1],
    ];
    if theta < 1e-8 {
        return vector.map(|v| v * 0.5);
    }
    vector.map(|value| value * theta / (2.0 * theta.sin()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn f64_sim3_round_trip_preserves_a_small_transform() {
        let transform = F64Sim3 {
            scale: 1.000_000_1,
            rotation: so3_exp([1e-7, -2e-7, 3e-7]),
            translation: [1e-8, -2e-8, 3e-8],
        };
        let identity = transform.compose(transform.inverse().unwrap());
        assert!((identity.scale - 1.0).abs() < 1e-12);
        assert!(identity.translation.iter().all(|value| value.abs() < 1e-12));
    }
}
