//! Crash-safe, content-addressed `.vestra` scene bundles.
//!
//! Chunks are immutable and become visible before the manifest refers to them.
//! The manifest itself is atomically replaced, so an interrupted write cannot
//! make a partial scene appear complete.

use std::{
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{CameraCalibration, FrameWindow, FusedSceneChunk, MeasuredPoint, ScaleStatus};

const MANIFEST_FILE: &str = "manifest.json";
const CHUNKS_DIRECTORY: &str = "chunks";
static TEMPORARY_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SceneProvenance {
    pub engine_revision: String,
    pub kernel_revision: String,
    pub model_fingerprint: String,
    pub settings_fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SceneManifest {
    pub schema: String,
    pub scale: ScaleStatus,
    pub coordinate_convention: String,
    pub provenance: SceneProvenance,
    pub measured_chunk_hashes: Vec<String>,
    /// The derived, relative-scale world built from immutable measured chunks.
    /// It is optional so v1 bundles created before fusion remain readable.
    #[serde(default)]
    pub fused_chunk_hash: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WindowMeasuredChunk {
    pub window: FrameWindow,
    pub views: Vec<MeasuredFrameChunk>,
}

/// Measured evidence owned by one source frame inside an overlapping window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasuredFrameChunk {
    pub frame_index: usize,
    pub camera: CameraCalibration,
    pub points: Vec<MeasuredPoint>,
}

#[derive(Debug, thiserror::Error)]
pub enum SceneBundleError {
    #[error("scene bundle already exists at {0}")]
    AlreadyExists(PathBuf),
    #[error("scene manifest is missing at {0}")]
    MissingManifest(PathBuf),
    #[error("failed to serialize scene data: {0}")]
    Serialize(#[from] serde_json::Error),
    #[error("scene I/O failed: {0}")]
    Io(#[from] std::io::Error),
}

/// A local scene bundle directory. It can be zipped for transport only after
/// its final manifest has been written.
#[derive(Debug, Clone)]
pub struct SceneBundle {
    root: PathBuf,
}

impl SceneBundle {
    /// Creates an empty relative-scale bundle. The initial manifest is already
    /// valid, which makes bundle creation resumable and inspectable.
    pub fn create(
        root: impl Into<PathBuf>,
        provenance: SceneProvenance,
    ) -> Result<Self, SceneBundleError> {
        let root = root.into();
        if root.exists() {
            return Err(SceneBundleError::AlreadyExists(root));
        }
        fs::create_dir_all(root.join(CHUNKS_DIRECTORY))?;
        let bundle = Self { root };
        bundle.write_manifest(&SceneManifest {
            schema: "vestra.scene/v1".to_owned(),
            scale: ScaleStatus::Relative,
            coordinate_convention: "right-handed; world coordinates; camera poses are W2C"
                .to_owned(),
            provenance,
            measured_chunk_hashes: Vec::new(),
            fused_chunk_hash: None,
        })?;
        Ok(bundle)
    }

    pub fn open(root: impl Into<PathBuf>) -> Result<Self, SceneBundleError> {
        let root = root.into();
        if !root.join(MANIFEST_FILE).is_file() {
            return Err(SceneBundleError::MissingManifest(root.join(MANIFEST_FILE)));
        }
        Ok(Self { root })
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn manifest(&self) -> Result<SceneManifest, SceneBundleError> {
        Ok(serde_json::from_slice(&fs::read(
            self.root.join(MANIFEST_FILE),
        )?)?)
    }

    /// Stores an immutable measured chunk and atomically publishes it in the
    /// manifest. Repeating the exact same call is idempotent.
    pub fn write_measured_window(
        &self,
        chunk: &WindowMeasuredChunk,
    ) -> Result<String, SceneBundleError> {
        let payload = serde_json::to_vec(chunk)?;
        let hash = sha256_hex(&payload);
        let chunk_path = self
            .root
            .join(CHUNKS_DIRECTORY)
            .join(format!("{hash}.json"));
        if !chunk_path.exists() {
            atomic_write(&chunk_path, &payload)?;
        }

        let mut manifest = self.manifest()?;
        if !manifest
            .measured_chunk_hashes
            .iter()
            .any(|existing| existing == &hash)
        {
            manifest.measured_chunk_hashes.push(hash.clone());
            manifest.measured_chunk_hashes.sort_unstable();
            self.write_manifest(&manifest)?;
        }
        Ok(hash)
    }

    pub fn read_measured_window(
        &self,
        hash: &str,
    ) -> Result<WindowMeasuredChunk, SceneBundleError> {
        let payload = fs::read(
            self.root
                .join(CHUNKS_DIRECTORY)
                .join(format!("{hash}.json")),
        )?;
        Ok(serde_json::from_slice(&payload)?)
    }

    /// Stores a derived fused world separately from the raw measurements and
    /// atomically points the manifest at it. Repeating the same fusion is
    /// idempotent; a later fusion replaces only this derived reference.
    pub fn write_fused_scene(&self, chunk: &FusedSceneChunk) -> Result<String, SceneBundleError> {
        let payload = serde_json::to_vec(chunk)?;
        let hash = sha256_hex(&payload);
        let chunk_path = self
            .root
            .join(CHUNKS_DIRECTORY)
            .join(format!("fused-{hash}.json"));
        if !chunk_path.exists() {
            atomic_write(&chunk_path, &payload)?;
        }
        let mut manifest = self.manifest()?;
        if manifest.fused_chunk_hash.as_deref() != Some(hash.as_str()) {
            manifest.fused_chunk_hash = Some(hash.clone());
            self.write_manifest(&manifest)?;
        }
        Ok(hash)
    }

    pub fn read_fused_scene(&self, hash: &str) -> Result<FusedSceneChunk, SceneBundleError> {
        let payload = fs::read(
            self.root
                .join(CHUNKS_DIRECTORY)
                .join(format!("fused-{hash}.json")),
        )?;
        Ok(serde_json::from_slice(&payload)?)
    }

    fn write_manifest(&self, manifest: &SceneManifest) -> Result<(), SceneBundleError> {
        Ok(atomic_write(
            &self.root.join(MANIFEST_FILE),
            &serde_json::to_vec_pretty(manifest)?,
        )?)
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn atomic_write(destination: &Path, bytes: &[u8]) -> Result<(), std::io::Error> {
    let parent = destination
        .parent()
        .expect("bundle paths always have a parent");
    fs::create_dir_all(parent)?;
    let suffix = TEMPORARY_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(
        ".{}.{}.{}.tmp",
        destination
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("vestra"),
        std::process::id(),
        suffix
    ));
    fs::write(&temporary, bytes)?;
    match fs::rename(&temporary, destination) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == ErrorKind::AlreadyExists => {
            let _ = fs::remove_file(&temporary);
            Ok(())
        }
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_root() -> PathBuf {
        std::env::temp_dir().join(format!(
            "vestra-scene-test-{}-{}",
            std::process::id(),
            TEMPORARY_FILE_COUNTER.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn provenance() -> SceneProvenance {
        SceneProvenance {
            engine_revision: "engine-test".to_owned(),
            kernel_revision: "kernel-test".to_owned(),
            model_fingerprint: "model-test".to_owned(),
            settings_fingerprint: "settings-test".to_owned(),
        }
    }

    #[test]
    fn measured_chunks_are_content_addressed_and_idempotently_published() {
        let root = test_root();
        let bundle = SceneBundle::create(&root, provenance()).unwrap();
        let chunk = WindowMeasuredChunk {
            window: FrameWindow {
                index: 0,
                start: 0,
                end: 2,
            },
            views: vec![MeasuredFrameChunk {
                frame_index: 0,
                camera: CameraCalibration {
                    world_to_camera: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    intrinsics: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                },
                points: vec![MeasuredPoint {
                    position: [1.0, 2.0, 3.0],
                    color_srgb: [4, 5, 6],
                    confidence: 2.0,
                    radius: 0.1,
                    source_pixel: [7, 8],
                }],
            }],
        };
        let first_hash = bundle.write_measured_window(&chunk).unwrap();
        let second_hash = bundle.write_measured_window(&chunk).unwrap();
        assert_eq!(first_hash, second_hash);
        assert_eq!(
            bundle.manifest().unwrap().measured_chunk_hashes,
            vec![first_hash.clone()]
        );
        assert_eq!(bundle.read_measured_window(&first_hash).unwrap(), chunk);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bundle_open_requires_an_atomic_manifest() {
        let root = test_root();
        assert!(matches!(
            SceneBundle::open(&root),
            Err(SceneBundleError::MissingManifest(_))
        ));
    }

    #[test]
    fn fused_scene_is_content_addressed_without_replacing_measured_evidence() {
        let root = test_root();
        let bundle = SceneBundle::create(&root, provenance()).unwrap();
        let measured = WindowMeasuredChunk {
            window: FrameWindow {
                index: 0,
                start: 0,
                end: 1,
            },
            views: Vec::new(),
        };
        let measured_hash = bundle.write_measured_window(&measured).unwrap();
        let fused = crate::FusedSceneChunk {
            alignments: Vec::new(),
            voxel_size: 0.25,
            points: vec![crate::FusedPoint {
                position: [1.0, 2.0, 3.0],
                color_srgb: [4, 5, 6],
                confidence: 0.7,
                radius: 0.1,
                contributors: 2,
            }],
        };

        let fused_hash = bundle.write_fused_scene(&fused).unwrap();

        let manifest = bundle.manifest().unwrap();
        assert_eq!(manifest.measured_chunk_hashes, vec![measured_hash.clone()]);
        assert_eq!(
            manifest.fused_chunk_hash.as_deref(),
            Some(fused_hash.as_str())
        );
        assert_eq!(
            bundle.read_measured_window(&measured_hash).unwrap(),
            measured
        );
        assert_eq!(bundle.read_fused_scene(&fused_hash).unwrap(), fused);
        fs::remove_dir_all(root).unwrap();
    }
}
