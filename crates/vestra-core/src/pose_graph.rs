//! Relative Sim(3) pose-graph optimization for deferred world fusion.
//!
//! A sequential window chain accumulates drift even when every individual seam
//! is plausible. This module keeps every window in its local model frame,
//! optimizes the local-to-global transforms against sequential and verified
//! loop edges, and lets the caller perform one final transform-and-fuse pass.
//! It deliberately does not propose or measure loops: those stages must supply
//! geometrically verified `PoseGraphEdge` values.

use serde::{Deserialize, Serialize};

use crate::SimilarityTransform;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PoseGraphEdge {
    /// Source window node.
    pub from: usize,
    /// Target window node.
    pub to: usize,
    /// Measured transform from `to` local coordinates into `from` local
    /// coordinates. This matches the constraint `G_from⁻¹ ∘ G_to == measurement`.
    pub measurement: SimilarityTransform,
    /// Isotropic information weight. Must be finite and positive.
    pub information: f32,
    /// Distinguishes a non-adjacent, independently verified revisit edge from a
    /// normal sequential seam in exported provenance.
    pub loop_closure: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelativePoseGraph {
    /// Each node maps that window's local coordinates into the shared world.
    pub nodes: Vec<SimilarityTransform>,
    pub edges: Vec<PoseGraphEdge>,
    /// Gauge locks. An empty vector means node zero is fixed.
    pub fixed: Vec<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PoseGraphSettings {
    pub max_iterations: usize,
    pub initial_damping: f64,
    pub convergence_relative_cost: f64,
    pub finite_difference_step: f64,
}

impl Default for PoseGraphSettings {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            initial_damping: 1e-4,
            convergence_relative_cost: 1e-10,
            finite_difference_step: 1e-6,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PoseGraphReport {
    pub initial_cost: f64,
    pub final_cost: f64,
    pub iterations: usize,
    pub free_degrees_of_freedom: usize,
    pub loop_edges: usize,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PoseGraphError {
    #[error("pose graph edge references a missing node")]
    InvalidEdge,
    #[error("pose graph edge has invalid information or similarity scale")]
    InvalidMeasurement,
    #[error("pose graph linear system is singular")]
    SingularSystem,
}

/// Optimizes the graph in place, keeping the configured gauge lock fixed.
///
/// The residual is `log(Z⁻¹ ∘ G_from⁻¹ ∘ G_to)`, with three translation,
/// three rotation, and one log-scale component. Jacobians use central finite
/// differences so the implementation stays auditable while the graph remains
/// small (the default 120-frame capture creates thirteen window nodes).
pub fn optimize_relative_pose_graph(
    graph: &mut RelativePoseGraph,
    settings: PoseGraphSettings,
) -> Result<PoseGraphReport, PoseGraphError> {
    validate_graph(graph)?;
    let node_count = graph.nodes.len();
    if node_count == 0 {
        return Ok(PoseGraphReport {
            initial_cost: 0.0,
            final_cost: 0.0,
            iterations: 0,
            free_degrees_of_freedom: 0,
            loop_edges: 0,
        });
    }
    let fixed = fixed_nodes(graph, node_count);
    let mut offsets = vec![None; node_count];
    let mut variables = 0;
    for (node, is_fixed) in fixed.iter().copied().enumerate() {
        if !is_fixed {
            offsets[node] = Some(variables);
            variables += 7;
        }
    }
    let initial_cost = graph_cost(&graph.nodes, &graph.edges)?;
    let mut current_cost = initial_cost;
    let mut damping = settings.initial_damping.max(1e-12);
    let mut iterations = 0;

    if variables == 0 {
        return Ok(PoseGraphReport {
            initial_cost,
            final_cost: current_cost,
            iterations,
            free_degrees_of_freedom: 0,
            loop_edges: graph.edges.iter().filter(|edge| edge.loop_closure).count(),
        });
    }

    for iteration in 0..settings.max_iterations {
        let (normal, gradient) = normal_equations(
            &graph.nodes,
            &graph.edges,
            &fixed,
            &offsets,
            variables,
            settings.finite_difference_step,
        )?;
        let mut accepted = false;
        for _ in 0..12 {
            let mut system = normal.clone();
            for row in 0..variables {
                system[row * variables + row] += damping * (normal[row * variables + row] + 1e-12);
            }
            let Some(step) = cholesky_solve(system, gradient.clone(), variables) else {
                damping *= 4.0;
                continue;
            };
            let trial = retract_nodes(&graph.nodes, &offsets, &step)?;
            let trial_cost = graph_cost(&trial, &graph.edges)?;
            if trial_cost < current_cost {
                let improvement = current_cost - trial_cost;
                graph.nodes = trial;
                current_cost = trial_cost;
                damping = (damping * 0.5).max(1e-12);
                iterations = iteration + 1;
                accepted = true;
                if improvement <= settings.convergence_relative_cost * (initial_cost.abs() + 1e-30)
                {
                    return Ok(PoseGraphReport {
                        initial_cost,
                        final_cost: current_cost,
                        iterations,
                        free_degrees_of_freedom: variables,
                        loop_edges: graph.edges.iter().filter(|edge| edge.loop_closure).count(),
                    });
                }
                break;
            }
            damping *= 4.0;
        }
        if !accepted {
            break;
        }
    }

    Ok(PoseGraphReport {
        initial_cost,
        final_cost: current_cost,
        iterations,
        free_degrees_of_freedom: variables,
        loop_edges: graph.edges.iter().filter(|edge| edge.loop_closure).count(),
    })
}

/// Returns the seven-component residual for one relative edge.
pub fn pose_edge_residual(
    from_global: SimilarityTransform,
    to_global: SimilarityTransform,
    measurement: SimilarityTransform,
) -> Result<[f64; 7], PoseGraphError> {
    let inverse_measurement = measurement
        .inverse()
        .ok_or(PoseGraphError::InvalidMeasurement)?;
    let inverse_from = from_global
        .inverse()
        .ok_or(PoseGraphError::InvalidMeasurement)?;
    let error = inverse_measurement.compose(inverse_from.compose(to_global));
    if !error.scale.is_finite() || error.scale <= 0.0 {
        return Err(PoseGraphError::InvalidMeasurement);
    }
    let rotation = error.rotation.map(f64::from);
    let rotation_log = so3_log(rotation);
    Ok([
        f64::from(error.translation[0]),
        f64::from(error.translation[1]),
        f64::from(error.translation[2]),
        rotation_log[0],
        rotation_log[1],
        rotation_log[2],
        f64::from(error.scale).ln(),
    ])
}

fn validate_graph(graph: &RelativePoseGraph) -> Result<(), PoseGraphError> {
    for edge in &graph.edges {
        if edge.from >= graph.nodes.len() || edge.to >= graph.nodes.len() {
            return Err(PoseGraphError::InvalidEdge);
        }
        if !edge.information.is_finite()
            || edge.information <= 0.0
            || !edge.measurement.scale.is_finite()
            || edge.measurement.scale <= 0.0
        {
            return Err(PoseGraphError::InvalidMeasurement);
        }
    }
    Ok(())
}

fn fixed_nodes(graph: &RelativePoseGraph, node_count: usize) -> Vec<bool> {
    if graph.fixed.len() == node_count {
        graph.fixed.clone()
    } else {
        let mut fixed = vec![false; node_count];
        fixed[0] = true;
        fixed
    }
}

fn graph_cost(
    nodes: &[SimilarityTransform],
    edges: &[PoseGraphEdge],
) -> Result<f64, PoseGraphError> {
    edges.iter().try_fold(0.0, |cost, edge| {
        let residual = pose_edge_residual(nodes[edge.from], nodes[edge.to], edge.measurement)?;
        Ok(cost
            + 0.5
                * f64::from(edge.information)
                * residual.iter().map(|value| value * value).sum::<f64>())
    })
}

fn normal_equations(
    nodes: &[SimilarityTransform],
    edges: &[PoseGraphEdge],
    fixed: &[bool],
    offsets: &[Option<usize>],
    variables: usize,
    step: f64,
) -> Result<(Vec<f64>, Vec<f64>), PoseGraphError> {
    let mut normal = vec![0.0; variables * variables];
    let mut gradient = vec![0.0; variables];
    for edge in edges {
        let residual = pose_edge_residual(nodes[edge.from], nodes[edge.to], edge.measurement)?;
        let endpoint_jacobians = [
            endpoint_jacobian(nodes, edge, edge.from, fixed[edge.from], step)?,
            endpoint_jacobian(nodes, edge, edge.to, fixed[edge.to], step)?,
        ];
        for (left_endpoint, left_jacobian) in endpoint_jacobians.iter().enumerate() {
            let Some(left_base) = offsets[if left_endpoint == 0 {
                edge.from
            } else {
                edge.to
            }] else {
                continue;
            };
            for column in 0..7 {
                gradient[left_base + column] -= f64::from(edge.information)
                    * (0..7)
                        .map(|row| left_jacobian[row * 7 + column] * residual[row])
                        .sum::<f64>();
            }
            for (right_endpoint, right_jacobian) in endpoint_jacobians.iter().enumerate() {
                let Some(right_base) = offsets[if right_endpoint == 0 {
                    edge.from
                } else {
                    edge.to
                }] else {
                    continue;
                };
                for left_column in 0..7 {
                    for right_column in 0..7 {
                        let value = (0..7)
                            .map(|row| {
                                left_jacobian[row * 7 + left_column]
                                    * right_jacobian[row * 7 + right_column]
                            })
                            .sum::<f64>();
                        normal
                            [(left_base + left_column) * variables + right_base + right_column] +=
                            f64::from(edge.information) * value;
                    }
                }
            }
        }
    }
    Ok((normal, gradient))
}

fn endpoint_jacobian(
    nodes: &[SimilarityTransform],
    edge: &PoseGraphEdge,
    endpoint: usize,
    fixed: bool,
    step: f64,
) -> Result<[f64; 49], PoseGraphError> {
    let mut jacobian = [0.0; 49];
    if fixed {
        return Ok(jacobian);
    }
    for axis in 0..7 {
        let mut positive = [0.0; 7];
        positive[axis] = step;
        let mut negative = [0.0; 7];
        negative[axis] = -step;
        let plus = retract(nodes[endpoint], positive)?;
        let minus = retract(nodes[endpoint], negative)?;
        let positive_residual = if endpoint == edge.from {
            pose_edge_residual(plus, nodes[edge.to], edge.measurement)?
        } else {
            pose_edge_residual(nodes[edge.from], plus, edge.measurement)?
        };
        let negative_residual = if endpoint == edge.from {
            pose_edge_residual(minus, nodes[edge.to], edge.measurement)?
        } else {
            pose_edge_residual(nodes[edge.from], minus, edge.measurement)?
        };
        for row in 0..7 {
            jacobian[row * 7 + axis] =
                (positive_residual[row] - negative_residual[row]) / (2.0 * step);
        }
    }
    Ok(jacobian)
}

fn retract_nodes(
    nodes: &[SimilarityTransform],
    offsets: &[Option<usize>],
    step: &[f64],
) -> Result<Vec<SimilarityTransform>, PoseGraphError> {
    nodes
        .iter()
        .enumerate()
        .map(|(index, node)| match offsets[index] {
            Some(offset) => retract(*node, step[offset..offset + 7].try_into().unwrap()),
            None => Ok(*node),
        })
        .collect()
}

fn retract(
    transform: SimilarityTransform,
    delta: [f64; 7],
) -> Result<SimilarityTransform, PoseGraphError> {
    let incremental_rotation = so3_exp([delta[3], delta[4], delta[5]]);
    let rotation = matrix_mul(transform.rotation.map(f64::from), incremental_rotation);
    let scale = f64::from(transform.scale) * delta[6].exp();
    if !scale.is_finite() || scale <= 0.0 {
        return Err(PoseGraphError::InvalidMeasurement);
    }
    Ok(SimilarityTransform {
        scale: scale as f32,
        rotation: rotation.map(|value| value as f32),
        translation: [
            transform.translation[0] + delta[0] as f32,
            transform.translation[1] + delta[1] as f32,
            transform.translation[2] + delta[2] as f32,
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

fn matrix_mul(left: [f64; 9], right: [f64; 9]) -> [f64; 9] {
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

fn so3_exp(vector: [f64; 3]) -> [f64; 9] {
    let angle = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
    if angle < 1e-12 {
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    }
    let [x, y, z] = vector.map(|value| value / angle);
    let cosine = angle.cos();
    let sine = angle.sin();
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
}

fn so3_log(rotation: [f64; 9]) -> [f64; 3] {
    let cosine = ((rotation[0] + rotation[4] + rotation[8] - 1.0) * 0.5).clamp(-1.0, 1.0);
    let angle = cosine.acos();
    let vector = [
        rotation[7] - rotation[5],
        rotation[2] - rotation[6],
        rotation[3] - rotation[1],
    ];
    if angle < 1e-8 {
        return vector.map(|value| value * 0.5);
    }
    let factor = angle / (2.0 * angle.sin());
    vector.map(|value| value * factor)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn translation(x: f32) -> SimilarityTransform {
        SimilarityTransform {
            translation: [x, 0.0, 0.0],
            ..SimilarityTransform::IDENTITY
        }
    }

    #[test]
    fn residual_is_zero_for_satisfied_relative_edge() {
        let residual =
            pose_edge_residual(translation(1.0), translation(3.0), translation(2.0)).unwrap();
        assert!(
            residual.iter().all(|value| value.abs() < 1e-6),
            "{residual:?}"
        );
    }

    #[test]
    fn loop_edge_redistributes_sequential_drift() {
        let mut graph = RelativePoseGraph {
            nodes: vec![
                SimilarityTransform::IDENTITY,
                translation(1.0),
                translation(2.4),
            ],
            edges: vec![
                PoseGraphEdge {
                    from: 0,
                    to: 1,
                    measurement: translation(1.0),
                    information: 1.0,
                    loop_closure: false,
                },
                PoseGraphEdge {
                    from: 1,
                    to: 2,
                    measurement: translation(1.0),
                    information: 1.0,
                    loop_closure: false,
                },
                PoseGraphEdge {
                    from: 0,
                    to: 2,
                    measurement: translation(2.0),
                    information: 2.0,
                    loop_closure: true,
                },
            ],
            fixed: vec![true, false, false],
        };
        let report =
            optimize_relative_pose_graph(&mut graph, PoseGraphSettings::default()).unwrap();
        assert!(report.final_cost < report.initial_cost * 1e-6, "{report:?}");
        assert!(
            (graph.nodes[1].translation[0] - 1.0).abs() < 1e-4,
            "{:?}",
            graph.nodes
        );
        assert!(
            (graph.nodes[2].translation[0] - 2.0).abs() < 1e-4,
            "{:?}",
            graph.nodes
        );
    }

    #[test]
    fn invalid_edges_are_rejected_before_optimization() {
        let mut graph = RelativePoseGraph {
            nodes: vec![SimilarityTransform::IDENTITY],
            edges: vec![PoseGraphEdge {
                from: 0,
                to: 1,
                measurement: SimilarityTransform::IDENTITY,
                information: 1.0,
                loop_closure: true,
            }],
            fixed: Vec::new(),
        };
        assert_eq!(
            optimize_relative_pose_graph(&mut graph, PoseGraphSettings::default()),
            Err(PoseGraphError::InvalidEdge)
        );
    }
}
