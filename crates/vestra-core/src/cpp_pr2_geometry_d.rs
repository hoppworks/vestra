//! Exact numeric boundaries for the pinned C++ PR #2 geometry oracle.
//!
//! The reference inverts F32 calibration matrices in F64, rounds the inverse
//! back to F32, and only then promotes it to F64 for backprojection.  This is
//! deliberately *not* the product backprojection path: it exists so the
//! VPS/VPO differential suite can distinguish an upstream geometry mismatch
//! from a later Sim(3), ICP, or pose-graph mismatch.

use crate::{
    BackprojectionError, BackprojectionSettings, CameraCalibration, CameraCentreDirection,
    CppPr2Frame, MeasuredPoint, MeasuredView, backproject_measured_view,
};

/// C++ `inv3`: F64 cofactors, followed by the reference F32 rounding boundary.
pub(crate) fn inv3_cpp_pr2(matrix: [f32; 9]) -> Option<[f32; 9]> {
    let [a, b, c, d, e, f, g, h, i] = matrix.map(f64::from);
    let cofactor_a = e * i - f * h;
    let cofactor_b = -(d * i - f * g);
    let cofactor_c = d * h - e * g;
    let determinant = a * cofactor_a + b * cofactor_b + c * cofactor_c;
    if determinant == 0.0 || !determinant.is_finite() {
        return None;
    }
    let inverse = determinant.recip();
    Some([
        (cofactor_a * inverse) as f32,
        (-(b * i - c * h) * inverse) as f32,
        ((b * f - c * e) * inverse) as f32,
        (cofactor_b * inverse) as f32,
        ((a * i - c * g) * inverse) as f32,
        (-(a * f - c * d) * inverse) as f32,
        (cofactor_c * inverse) as f32,
        (-(a * h - b * g) * inverse) as f32,
        ((a * e - b * d) * inverse) as f32,
    ])
}

/// C++ `inv4`: partial-pivoting Gauss-Jordan in F64 with a final F32 round.
pub(crate) fn inv4_cpp_pr2(matrix: [f32; 16]) -> Option<[f32; 16]> {
    let mut augmented = [[0.0_f64; 8]; 4];
    for row in 0..4 {
        for column in 0..4 {
            augmented[row][column] = f64::from(matrix[row * 4 + column]);
            augmented[row][4 + column] = f64::from((row == column) as u8);
        }
    }
    for column in 0..4 {
        let mut pivot = column;
        let mut best = augmented[column][column].abs();
        for row in column + 1..4 {
            let candidate = augmented[row][column].abs();
            if candidate > best {
                best = candidate;
                pivot = row;
            }
        }
        if best == 0.0 || !best.is_finite() {
            return None;
        }
        if pivot != column {
            augmented.swap(column, pivot);
        }
        let divisor = augmented[column][column];
        for entry in &mut augmented[column] {
            *entry /= divisor;
        }
        for row in 0..4 {
            if row == column {
                continue;
            }
            let factor = augmented[row][column];
            if factor == 0.0 {
                continue;
            }
            for entry in 0..8 {
                augmented[row][entry] -= factor * augmented[column][entry];
            }
        }
    }
    Some(std::array::from_fn(|index| {
        augmented[index / 4][4 + index % 4] as f32
    }))
}

/// Builds the homogeneous W2C matrix from the fixture's row-major 3x4 pose.
pub(crate) fn homogeneous_w2c(world_to_camera: [f32; 12]) -> [f32; 16] {
    [
        world_to_camera[0],
        world_to_camera[1],
        world_to_camera[2],
        world_to_camera[3],
        world_to_camera[4],
        world_to_camera[5],
        world_to_camera[6],
        world_to_camera[7],
        world_to_camera[8],
        world_to_camera[9],
        world_to_camera[10],
        world_to_camera[11],
        0.0,
        0.0,
        0.0,
        1.0,
    ]
}

/// Reference camera centre and positive-Z direction after its exact inverse.
pub(crate) fn camera_centre_direction_cpp_pr2(
    frame_index: usize,
    world_to_camera: [f32; 12],
) -> Option<CameraCentreDirection> {
    if !world_to_camera.iter().all(|value| value.is_finite()) {
        return None;
    }
    let c2w = inv4_cpp_pr2(homogeneous_w2c(world_to_camera))?;
    let centre_local = [c2w[3], c2w[7], c2w[11]];
    let forward = [c2w[2], c2w[6], c2w[10]];
    let length = forward
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    (length.is_finite() && length > 1e-12).then(|| CameraCentreDirection {
        frame_index,
        centre_local,
        forward_local: forward.map(|value| value / length),
    })
}

/// Reconstructs PR #2's raw F64 world positions, preserving its inverse-round
/// boundary. Callers retain the result only for differential/oracle work.
pub(crate) fn backproject_positions_cpp_pr2_f64(
    frame: &CppPr2Frame,
    width: usize,
    height: usize,
    minimum_confidence: f32,
) -> Option<Vec<[f64; 3]>> {
    let pixels = width.checked_mul(height)?;
    if frame.depth.len() != pixels
        || frame.confidence.len() != pixels
        || frame.rgb_hwc_u8.len() != pixels.checked_mul(3)?
    {
        return None;
    }
    let inverse_intrinsics = inv3_cpp_pr2(frame.intrinsics)?;
    let c2w = inv4_cpp_pr2(homogeneous_w2c(frame.world_to_camera))?;
    let ki = inverse_intrinsics.map(f64::from);
    let cw = c2w.map(f64::from);
    let mut positions = Vec::with_capacity(pixels);
    for y in 0..height {
        for x in 0..width {
            let pixel = y * width + x;
            let depth = frame.depth[pixel];
            if !depth.is_finite()
                || depth <= 0.0
                || !(frame.confidence[pixel] >= minimum_confidence)
            {
                continue;
            }
            let x = x as f64;
            let y = y as f64;
            let depth = f64::from(depth);
            let ray = [
                ki[0] * x + ki[1] * y + ki[2],
                ki[3] * x + ki[4] * y + ki[5],
                ki[6] * x + ki[7] * y + ki[8],
            ];
            let camera = ray.map(|value| value * depth);
            positions.push([
                cw[0] * camera[0] + cw[1] * camera[1] + cw[2] * camera[2] + cw[3],
                cw[4] * camera[0] + cw[5] * camera[1] + cw[6] * camera[2] + cw[7],
                cw[8] * camera[0] + cw[9] * camera[1] + cw[10] * camera[2] + cw[11],
            ]);
        }
    }
    Some(positions)
}

/// Uses Vestra's evidence/normal policy while replacing its rigid-pose point
/// coordinates with the pinned C++ inverse and backprojection contract. The
/// final cast is intentional: the current public measured-window schema is
/// F32, so this slice isolates inversion/backprojection before the F64
/// measured-window migration.
pub(crate) fn backproject_frame_cpp_pr2_f32(
    frame: &CppPr2Frame,
    width: usize,
    height: usize,
    settings: BackprojectionSettings,
) -> Result<Vec<MeasuredPoint>, BackprojectionError> {
    let mut points = backproject_measured_view(
        MeasuredView {
            rgb_hwc_u8: &frame.rgb_hwc_u8,
            depth: &frame.depth,
            confidence: &frame.confidence,
            width,
            height,
            camera: CameraCalibration {
                world_to_camera: frame.world_to_camera,
                intrinsics: frame.intrinsics,
            },
        },
        settings,
    )?;
    let positions =
        backproject_positions_cpp_pr2_f64(frame, width, height, settings.minimum_confidence)
            .ok_or(BackprojectionError::NonInvertibleCalibration)?;
    if points.len() != positions.len() {
        return Err(BackprojectionError::NonInvertibleCalibration);
    }
    for (point, position) in points.iter_mut().zip(positions) {
        point.position = position.map(|value| value as f32);
    }
    Ok(points)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_inverse_recovers_camera_centre_without_rigid_pose_assumption() {
        let pose = camera_centre_direction_cpp_pr2(
            7,
            [
                2.0, 0.0, 0.0, 10.0, 0.0, 3.0, 0.0, 15.0, 0.0, 0.0, 4.0, 20.0,
            ],
        )
        .unwrap();
        assert_eq!(pose.frame_index, 7);
        assert_eq!(pose.centre_local, [-5.0, -5.0, -5.0]);
        assert_eq!(pose.forward_local, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn backprojection_observes_the_reference_f32_inverse_boundary() {
        let frame = CppPr2Frame {
            intrinsics: [2.0, 0.0, 1.0, 0.0, 4.0, 0.5, 0.0, 0.0, 1.0],
            world_to_camera: [1.0, 0.0, 0.0, -2.0, 0.0, 1.0, 0.0, 3.0, 0.0, 0.0, 1.0, -4.0],
            depth: vec![2.0],
            confidence: vec![1.0],
            rgb_hwc_u8: vec![0, 0, 0],
        };
        let points = backproject_positions_cpp_pr2_f64(&frame, 1, 1, 1.0).unwrap();
        assert_eq!(points, vec![[1.0, -3.25, 6.0]]);
    }
}
