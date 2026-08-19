//! Narrow, deterministic import boundary for COLMAP dense-MVS point clouds.
//!
//! The imported cloud is a diagnostic world product, never a replacement for
//! immutable Vestra/DA3 measurements.  Keeping this parser here means Studio
//! can inspect the global-camera control result before we decide whether to
//! introduce pose-conditioned dense depth.

use std::{fs, path::Path};

use crate::{FusedPoint, FusedSceneChunk};

#[derive(Debug, thiserror::Error)]
pub enum MvsImportError {
    #[error("failed to read MVS PLY: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid COLMAP MVS PLY: {0}")]
    Invalid(String),
}

#[derive(Debug, Clone, Copy)]
enum Scalar {
    F32,
    U8,
}

impl Scalar {
    const fn width(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::U8 => 1,
        }
    }
}

/// Imports COLMAP's binary-little-endian `stereo_fusion` PLY output.
///
/// Only scalar properties are accepted. At minimum `x`, `y`, and `z` are
/// required. Normals and RGB are retained when supplied by COLMAP; unknown
/// scalar properties are skipped rather than silently changing vertex stride.
pub fn import_colmap_fused_ply(path: impl AsRef<Path>) -> Result<FusedSceneChunk, MvsImportError> {
    let bytes = fs::read(path)?;
    let header_end = bytes
        .windows(b"end_header\n".len())
        .position(|window| window == b"end_header\n")
        .map(|offset| offset + b"end_header\n".len())
        .ok_or_else(|| MvsImportError::Invalid("missing end_header".to_owned()))?;
    let header = std::str::from_utf8(&bytes[..header_end])
        .map_err(|_| MvsImportError::Invalid("header is not UTF-8".to_owned()))?;
    let mut vertex_count = None;
    let mut in_vertex = false;
    let mut properties = Vec::new();
    for line in header.lines() {
        let fields = line.split_ascii_whitespace().collect::<Vec<_>>();
        match fields.as_slice() {
            ["format", "binary_little_endian", "1.0"] => {}
            ["format", ..] => {
                return Err(MvsImportError::Invalid(
                    "only binary_little_endian PLY is supported".to_owned(),
                ));
            }
            ["element", "vertex", count] => {
                vertex_count = Some(
                    count
                        .parse::<usize>()
                        .map_err(|_| MvsImportError::Invalid("invalid vertex count".to_owned()))?,
                );
                in_vertex = true;
            }
            ["element", ..] => in_vertex = false,
            ["property", kind, name] if in_vertex => {
                let scalar = match *kind {
                    "float" | "float32" => Scalar::F32,
                    "uchar" | "uint8" => Scalar::U8,
                    _ => {
                        return Err(MvsImportError::Invalid(format!(
                            "unsupported vertex property type {kind:?}"
                        )));
                    }
                };
                properties.push((scalar, (*name).to_owned()));
            }
            ["property", "list", ..] if in_vertex => {
                return Err(MvsImportError::Invalid(
                    "list-valued vertex properties are unsupported".to_owned(),
                ));
            }
            _ => {}
        }
    }
    if !header.starts_with("ply\n") {
        return Err(MvsImportError::Invalid("missing ply magic".to_owned()));
    }
    let vertex_count =
        vertex_count.ok_or_else(|| MvsImportError::Invalid("missing vertex element".to_owned()))?;
    let stride = properties
        .iter()
        .map(|(kind, _)| kind.width())
        .sum::<usize>();
    if stride == 0 || bytes.len().saturating_sub(header_end) < vertex_count.saturating_mul(stride) {
        return Err(MvsImportError::Invalid(
            "truncated vertex payload".to_owned(),
        ));
    }
    let position_index = |name: &str| {
        properties
            .iter()
            .position(|(_, candidate)| candidate == name)
    };
    let x = position_index("x").ok_or_else(|| MvsImportError::Invalid("missing x".to_owned()))?;
    let y = position_index("y").ok_or_else(|| MvsImportError::Invalid("missing y".to_owned()))?;
    let z = position_index("z").ok_or_else(|| MvsImportError::Invalid("missing z".to_owned()))?;
    if [x, y, z]
        .iter()
        .any(|index| properties[*index].0.width() != 4)
    {
        return Err(MvsImportError::Invalid("positions must be f32".to_owned()));
    }
    let offset_of = |index: usize| {
        properties[..index]
            .iter()
            .map(|(kind, _)| kind.width())
            .sum::<usize>()
    };
    let f32_at = |record: &[u8], index: Option<usize>| {
        index.and_then(|index| {
            (properties[index].0.width() == 4).then(|| {
                let offset = offset_of(index);
                f32::from_le_bytes(record[offset..offset + 4].try_into().expect("four bytes"))
            })
        })
    };
    let u8_at = |record: &[u8], index: Option<usize>| {
        index.and_then(|index| (properties[index].0.width() == 1).then(|| record[offset_of(index)]))
    };
    let nx = position_index("nx").or_else(|| position_index("normal_x"));
    let ny = position_index("ny").or_else(|| position_index("normal_y"));
    let nz = position_index("nz").or_else(|| position_index("normal_z"));
    let red = position_index("red").or_else(|| position_index("r"));
    let green = position_index("green").or_else(|| position_index("g"));
    let blue = position_index("blue").or_else(|| position_index("b"));
    let mut points = Vec::with_capacity(vertex_count);
    for record in bytes[header_end..].chunks_exact(stride).take(vertex_count) {
        let position = [
            f32_at(record, Some(x)).expect("validated x"),
            f32_at(record, Some(y)).expect("validated y"),
            f32_at(record, Some(z)).expect("validated z"),
        ];
        if !position.iter().all(|value| value.is_finite()) {
            continue;
        }
        let normal = [
            f32_at(record, nx).unwrap_or(0.0),
            f32_at(record, ny).unwrap_or(0.0),
            f32_at(record, nz).unwrap_or(0.0),
        ];
        let length = normal.iter().map(|value| value * value).sum::<f32>().sqrt();
        let normal = if length.is_finite() && length > 0.0 {
            normal.map(|value| value / length)
        } else {
            [0.0; 3]
        };
        points.push(FusedPoint {
            position,
            normal,
            color_srgb: [
                u8_at(record, red).unwrap_or(255),
                u8_at(record, green).unwrap_or(255),
                u8_at(record, blue).unwrap_or(255),
            ],
            confidence: 1.0,
            // MVS has no per-observation focal/depth radius. A scene-relative
            // diagnostic radius is assigned after reading the full cloud.
            radius: 0.0,
            first_observing_frame: -1,
            contributors: 1,
        });
    }
    if points.is_empty() {
        return Err(MvsImportError::Invalid("no finite vertices".to_owned()));
    }
    let (minimum, maximum) = points.iter().fold(
        ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]),
        |(mut minimum, mut maximum), point| {
            for axis in 0..3 {
                minimum[axis] = minimum[axis].min(point.position[axis]);
                maximum[axis] = maximum[axis].max(point.position[axis]);
            }
            (minimum, maximum)
        },
    );
    let diagonal = (0..3)
        .map(|axis| (maximum[axis] - minimum[axis]).powi(2))
        .sum::<f32>()
        .sqrt();
    let radius = (diagonal / 4_096.0).max(1e-5);
    for point in &mut points {
        point.radius = radius;
    }
    Ok(FusedSceneChunk {
        alignments: Vec::new(),
        pose_graph_edges: Vec::new(),
        pose_graph: None,
        window_poses: Vec::new(),
        voxel_size: 0.0,
        points,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imports_binary_colmap_vertices_and_normalizes_normals() {
        let path = std::env::temp_dir().join(format!("vestra-mvs-{}.ply", std::process::id()));
        let header = b"ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
        let mut bytes = header.to_vec();
        for (position, normal, color) in [
            ([0.0_f32, 0.0, 0.0], [0.0, 3.0, 4.0], [1, 2, 3]),
            ([2.0_f32, 0.0, 0.0], [0.0, 0.0, 0.0], [4, 5, 6]),
        ] {
            for value in position.into_iter().chain(normal) {
                bytes.extend(value.to_le_bytes());
            }
            bytes.extend(color);
        }
        fs::write(&path, bytes).unwrap();
        let cloud = import_colmap_fused_ply(&path).unwrap();
        fs::remove_file(path).unwrap();
        assert_eq!(cloud.points.len(), 2);
        assert_eq!(cloud.points[0].color_srgb, [1, 2, 3]);
        assert_eq!(cloud.points[0].normal, [0.0, 0.6, 0.8]);
        assert_eq!(cloud.points[1].normal, [0.0; 3]);
        assert!(cloud.points.iter().all(|point| point.radius > 0.0));
    }
}
