//! Open export of a fused relative-scale Vestra world.

use std::{fs, path::Path};

use crate::{SceneBundle, SceneBundleError};

#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    #[error("the scene has no fused world; run `vestra fuse` first")]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FusedPoint, FusedSceneChunk, SceneProvenance};

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
                pose_graph: None,
                voxel_size: 0.1,
                points: vec![FusedPoint {
                    position: [1.0, 2.0, 3.0],
                    normal: [0.0, 0.0, 1.0],
                    color_srgb: [4, 5, 6],
                    confidence: 0.7,
                    radius: 0.2,
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
}
