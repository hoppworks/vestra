//! Open export of a fused relative-scale Vestra world.

use std::{fs, path::Path};

use crate::{CameraCalibration, SceneBundle, SceneBundleError, SimilarityTransform};

#[derive(Debug, serde::Serialize)]
struct CameraJsonExport {
    schema: &'static str,
    scale: &'static str,
    coordinate_convention: &'static str,
    cameras: Vec<CameraJsonEntry>,
}

#[derive(Debug, serde::Serialize)]
struct CameraJsonEntry {
    window_index: usize,
    frame_index: usize,
    /// The stored model calibration remains W2C in the window-local frame.
    local_world_to_camera: CameraCalibration,
    /// This transform maps that local camera/world frame into the fused world.
    local_to_fused_world: SimilarityTransform,
}

#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    #[error("the scene has no fused world; run `vestra-lab fuse` first")]
    MissingFusedWorld,
    #[error("scene persistence failed: {0}")]
    Scene(#[from] SceneBundleError),
    #[error("failed to write export: {0}")]
    Io(#[from] std::io::Error),
}

/// Writes an ASCII PLY with relative coordinates, RGB, confidence, radius,
/// and contributor count. PLY intentionally preserves the product's
/// relative-scale truth rather than inventing metric units.
pub fn export_fused_ply(
    bundle: &SceneBundle,
    output: impl AsRef<Path>,
) -> Result<usize, ExportError> {
    let manifest = bundle.manifest()?;
    let hash = manifest
        .fused_chunk_hash
        .ok_or(ExportError::MissingFusedWorld)?;
    let fused = bundle.read_fused_scene(&hash)?;
    let mut bytes = format!(
        "ply\nformat ascii 1.0\ncomment Vestra relative-scale fused world\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty float confidence\nproperty float radius\nproperty uint contributors\nend_header\n",
        fused.points.len()
    );
    for point in &fused.points {
        bytes.push_str(&format!(
            "{} {} {} {} {} {} {} {} {} {} {} {}\n",
            point.position[0],
            point.position[1],
            point.position[2],
            point.normal[0],
            point.normal[1],
            point.normal[2],
            point.color_srgb[0],
            point.color_srgb[1],
            point.color_srgb[2],
            point.confidence,
            point.radius,
            point.contributors,
        ));
    }
    fs::write(output, bytes)?;
    Ok(fused.points.len())
}

/// Writes a standards-compliant glTF 2.0 binary (`.glb`) point cloud. The
/// primitive uses `POINTS`, so it carries observed positions, normals, and
/// sRGB vertex colors without claiming that the surfels form a watertight mesh.
pub fn export_fused_glb(
    bundle: &SceneBundle,
    output: impl AsRef<Path>,
) -> Result<usize, ExportError> {
    let manifest = bundle.manifest()?;
    let hash = manifest
        .fused_chunk_hash
        .ok_or(ExportError::MissingFusedWorld)?;
    let fused = bundle.read_fused_scene(&hash)?;

    let mut binary = Vec::with_capacity(fused.points.len() * 27);
    let positions_offset = binary.len();
    for point in &fused.points {
        for component in point.position {
            binary.extend_from_slice(&component.to_le_bytes());
        }
    }
    pad_binary(&mut binary);
    let normals_offset = binary.len();
    for point in &fused.points {
        for component in point.normal {
            binary.extend_from_slice(&component.to_le_bytes());
        }
    }
    pad_binary(&mut binary);
    let colors_offset = binary.len();
    for point in &fused.points {
        binary.extend_from_slice(&point.color_srgb);
    }
    pad_binary(&mut binary);

    let (minimum, maximum) = position_bounds(&fused.points);
    let count = fused.points.len();
    let json = serde_json::json!({
        "asset": {"version": "2.0", "generator": "Vestra relative-scale world exporter"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "Vestra relative-scale fused world", "mesh": 0}],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2},
            "mode": 0
        }]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": count, "type": "VEC3", "min": minimum, "max": maximum},
            {"bufferView": 1, "componentType": 5126, "count": count, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5121, "normalized": true, "count": count, "type": "VEC3"}
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": positions_offset, "byteLength": count * 12, "target": 34962},
            {"buffer": 0, "byteOffset": normals_offset, "byteLength": count * 12, "target": 34962},
            {"buffer": 0, "byteOffset": colors_offset, "byteLength": count * 3, "target": 34962}
        ],
        "buffers": [{"byteLength": binary.len()}],
        "extras": {"vestra_scale": "relative", "geometry_layer": "fused"}
    });
    let mut json = serde_json::to_vec(&json).expect("serializing a JSON value cannot fail");
    while !json.len().is_multiple_of(4) {
        json.push(b' ');
    }
    let total_length = 12 + 8 + json.len() + 8 + binary.len();
    let total_length = u32::try_from(total_length)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "GLB exceeds 4 GiB"))?;
    let mut glb = Vec::with_capacity(total_length as usize);
    glb.extend_from_slice(&0x4654_6C67_u32.to_le_bytes()); // glTF
    glb.extend_from_slice(&2_u32.to_le_bytes());
    glb.extend_from_slice(&total_length.to_le_bytes());
    glb.extend_from_slice(&(json.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x4E4F_534A_u32.to_le_bytes()); // JSON
    glb.extend_from_slice(&json);
    glb.extend_from_slice(&(binary.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x004E_4942_u32.to_le_bytes()); // BIN\0
    glb.extend_from_slice(&binary);
    fs::write(output, glb)?;
    Ok(count)
}

/// Writes the compact 32-byte-per-primitive `antimatter15` `.splat` layout:
/// position, anisotropic scale, RGBA, then a unit quaternion. Vestra maps each
/// measured surfel radius and normal into a thin oriented ellipsoid; this is a
/// visualization export, never a claim that the world was Gaussian-trained.
pub fn export_fused_splat(
    bundle: &SceneBundle,
    output: impl AsRef<Path>,
) -> Result<usize, ExportError> {
    let manifest = bundle.manifest()?;
    let hash = manifest
        .fused_chunk_hash
        .ok_or(ExportError::MissingFusedWorld)?;
    let fused = bundle.read_fused_scene(&hash)?;
    let mut bytes = Vec::with_capacity(fused.points.len() * 32);
    for point in &fused.points {
        for value in point.position {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let radius = point.radius.max(1e-6);
        for value in [radius, radius, radius * 0.1] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&point.color_srgb);
        bytes.push((point.confidence.clamp(0.0, 1.0) * 255.0).round() as u8);
        bytes.extend_from_slice(&quaternion_from_z_axis(point.normal).map(unit_to_byte));
    }
    fs::write(output, bytes)?;
    Ok(fused.points.len())
}

/// Exports camera evidence as explicitly composable JSON. W2C matrices remain
/// in their native window-local coordinates; `local_to_fused_world` is kept
/// alongside each one so consumers never mistake a local pose for global data.
pub fn export_camera_json(
    bundle: &SceneBundle,
    output: impl AsRef<Path>,
) -> Result<usize, ExportError> {
    let manifest = bundle.manifest()?;
    let hash = manifest
        .fused_chunk_hash
        .ok_or(ExportError::MissingFusedWorld)?;
    let fused = bundle.read_fused_scene(&hash)?;
    let mut cameras = Vec::new();
    for measured_hash in &manifest.measured_chunk_hashes {
        let window = bundle.read_measured_window(measured_hash)?;
        let Some(pose) = fused
            .window_poses
            .iter()
            .find(|pose| pose.window_index == window.window.index)
        else {
            continue;
        };
        cameras.extend(window.views.iter().map(|view| CameraJsonEntry {
            window_index: window.window.index,
            frame_index: view.frame_index,
            local_world_to_camera: view.camera,
            local_to_fused_world: pose.local_to_world,
        }));
    }
    let count = cameras.len();
    let payload = CameraJsonExport {
        schema: "vestra.camera/v1",
        scale: "relative",
        coordinate_convention: "W2C is window-local; local_to_fused_world maps into the fused relative world",
        cameras,
    };
    fs::write(
        output,
        serde_json::to_vec_pretty(&payload).expect("camera export is serializable"),
    )?;
    Ok(count)
}

fn quaternion_from_z_axis(normal: [f32; 3]) -> [f32; 4] {
    let length = normal.iter().map(|value| value * value).sum::<f32>().sqrt();
    if !length.is_finite() || length < 1e-6 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let [x, y, z] = normal.map(|value| value / length);
    if z < -0.999_999 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let quaternion = [-y, x, 0.0, 1.0 + z];
    let q_length = quaternion
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    quaternion.map(|value| value / q_length)
}

fn unit_to_byte(value: f32) -> u8 {
    ((value.clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0).round() as u8
}

fn pad_binary(bytes: &mut Vec<u8>) {
    while !bytes.len().is_multiple_of(4) {
        bytes.push(0);
    }
}

fn position_bounds(points: &[crate::FusedPoint]) -> ([f32; 3], [f32; 3]) {
    let mut minimum = [f32::INFINITY; 3];
    let mut maximum = [f32::NEG_INFINITY; 3];
    for point in points {
        for axis in 0..3 {
            minimum[axis] = minimum[axis].min(point.position[axis]);
            maximum[axis] = maximum[axis].max(point.position[axis]);
        }
    }
    if minimum.iter().any(|value| !value.is_finite()) {
        ([0.0; 3], [0.0; 3])
    } else {
        (minimum, maximum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CameraCalibration, FrameWindow, FusedPoint, FusedSceneChunk, FusedWindowPose,
        MeasuredFrameChunk, SceneProvenance, SimilarityTransform, WindowMeasuredChunk,
    };

    #[test]
    fn export_preserves_relative_world_attributes_in_ply() {
        let root = std::env::temp_dir().join(format!("vestra-export-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let output = root.with_extension("ply");
        let bundle = SceneBundle::create(
            &root,
            SceneProvenance {
                engine_revision: "test".into(),
                kernel_revision: "test".into(),
                model_fingerprint: "test".into(),
                settings_fingerprint: "test".into(),
            },
        )
        .unwrap();
        bundle
            .write_fused_scene(&FusedSceneChunk {
                alignments: Vec::new(),
                pose_graph_edges: Vec::new(),
                pose_graph: None,
                window_poses: Vec::new(),
                voxel_size: 0.1,
                points: vec![FusedPoint {
                    position: [1.0, 2.0, 3.0],
                    normal: [0.0, 0.0, 1.0],
                    color_srgb: [4, 5, 6],
                    confidence: 0.7,
                    radius: 0.2,
                    first_observing_frame: -1,
                    contributors: 3,
                }],
            })
            .unwrap();

        assert_eq!(export_fused_ply(&bundle, &output).unwrap(), 1);
        let text = fs::read_to_string(&output).unwrap();
        assert!(text.contains("comment Vestra relative-scale fused world"));
        assert!(text.ends_with("1 2 3 0 0 1 4 5 6 0.7 0.2 3\n"));
        fs::remove_file(output).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn glb_export_is_a_gltf2_point_cloud_with_relative_metadata() {
        let root = std::env::temp_dir().join(format!("vestra-glb-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let output = root.with_extension("glb");
        let bundle = SceneBundle::create(
            &root,
            SceneProvenance {
                engine_revision: "test".into(),
                kernel_revision: "test".into(),
                model_fingerprint: "test".into(),
                settings_fingerprint: "test".into(),
            },
        )
        .unwrap();
        bundle
            .write_fused_scene(&FusedSceneChunk {
                alignments: Vec::new(),
                pose_graph_edges: Vec::new(),
                pose_graph: None,
                window_poses: Vec::new(),
                voxel_size: 0.1,
                points: vec![FusedPoint {
                    position: [1.0, 2.0, 3.0],
                    normal: [0.0, 0.0, 1.0],
                    color_srgb: [4, 5, 6],
                    confidence: 0.7,
                    radius: 0.2,
                    first_observing_frame: -1,
                    contributors: 3,
                }],
            })
            .unwrap();
        assert_eq!(export_fused_glb(&bundle, &output).unwrap(), 1);
        let glb = fs::read(&output).unwrap();
        assert_eq!(&glb[..4], b"glTF");
        assert_eq!(u32::from_le_bytes(glb[4..8].try_into().unwrap()), 2);
        let json_length = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
        let json = &glb[20..20 + json_length];
        let json: serde_json::Value = serde_json::from_slice(json).unwrap();
        assert_eq!(json["meshes"][0]["primitives"][0]["mode"], 0);
        assert_eq!(json["extras"]["vestra_scale"], "relative");
        fs::remove_file(output).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn splat_export_has_one_compact_oriented_surfel_per_point() {
        let root = std::env::temp_dir().join(format!("vestra-splat-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let output = root.with_extension("splat");
        let bundle = SceneBundle::create(
            &root,
            SceneProvenance {
                engine_revision: "test".into(),
                kernel_revision: "test".into(),
                model_fingerprint: "test".into(),
                settings_fingerprint: "test".into(),
            },
        )
        .unwrap();
        bundle
            .write_fused_scene(&FusedSceneChunk {
                alignments: Vec::new(),
                pose_graph_edges: Vec::new(),
                pose_graph: None,
                window_poses: Vec::new(),
                voxel_size: 0.1,
                points: vec![FusedPoint {
                    position: [1.0, 2.0, 3.0],
                    normal: [0.0, 0.0, 1.0],
                    color_srgb: [4, 5, 6],
                    confidence: 0.7,
                    radius: 0.2,
                    first_observing_frame: -1,
                    contributors: 3,
                }],
            })
            .unwrap();
        assert_eq!(export_fused_splat(&bundle, &output).unwrap(), 1);
        let bytes = fs::read(&output).unwrap();
        assert_eq!(bytes.len(), 32);
        assert_eq!(f32::from_le_bytes(bytes[0..4].try_into().unwrap()), 1.0);
        assert_eq!(&bytes[24..27], &[4, 5, 6]);
        fs::remove_file(output).unwrap();
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn camera_export_keeps_local_w2c_and_the_final_fused_transform_together() {
        let root =
            std::env::temp_dir().join(format!("vestra-camera-export-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let output = root.with_extension("cameras.json");
        let bundle = SceneBundle::create(
            &root,
            SceneProvenance {
                engine_revision: "test".into(),
                kernel_revision: "test".into(),
                model_fingerprint: "test".into(),
                settings_fingerprint: "test".into(),
            },
        )
        .unwrap();
        bundle
            .write_measured_window(&WindowMeasuredChunk {
                window: FrameWindow {
                    index: 7,
                    start: 12,
                    end: 13,
                },
                views: vec![MeasuredFrameChunk {
                    frame_index: 12,
                    camera: CameraCalibration {
                        world_to_camera: [
                            1.0, 0.0, 0.0, -2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                        ],
                        intrinsics: [2.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 0.0, 1.0],
                    },
                    points: Vec::new(),
                }],
                cpp_pr2_emission_confidence_threshold: None,
            })
            .unwrap();
        bundle
            .write_fused_scene(&FusedSceneChunk {
                alignments: Vec::new(),
                pose_graph_edges: Vec::new(),
                pose_graph: None,
                window_poses: vec![FusedWindowPose {
                    window_index: 7,
                    local_to_world: SimilarityTransform {
                        scale: 0.5,
                        translation: [3.0, 4.0, 5.0],
                        ..SimilarityTransform::IDENTITY
                    },
                }],
                voxel_size: 0.1,
                points: Vec::new(),
            })
            .unwrap();

        assert_eq!(export_camera_json(&bundle, &output).unwrap(), 1);
        let json: serde_json::Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
        assert_eq!(json["schema"], "vestra.camera/v1");
        assert_eq!(json["scale"], "relative");
        assert_eq!(json["cameras"][0]["frame_index"], 12);
        assert_eq!(
            json["cameras"][0]["local_world_to_camera"]["world_to_camera"][3],
            -2.0
        );
        assert_eq!(
            json["cameras"][0]["local_to_fused_world"]["translation"][0],
            3.0
        );
        fs::remove_file(output).unwrap();
        fs::remove_dir_all(root).unwrap();
    }
}
