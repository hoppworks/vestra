//! Durable global-pose provider contracts.
//!
//! A pose provider is deliberately separate from Vestra's local DA3 evidence:
//! it may improve a derived world, but cannot rewrite a decoded raster or a
//! measured window. COLMAP is the first supported parser because its text
//! model makes the W2C convention explicit and auditable.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RasterManifest {
    pub schema: String,
    pub source_sha256: String,
    pub duration_seconds: f64,
    pub source_width: usize,
    pub source_height: usize,
    pub crop: RasterCrop,
    pub output_width: usize,
    pub output_height: usize,
    pub frames: Vec<RasterFrame>,
    pub raster_fingerprint: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RasterCrop {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RasterFrame {
    pub frame_index: usize,
    pub file_name: String,
    pub sha256: String,
    /// The exact video timestamp represented by this decoded raster.
    pub timestamp_millis: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoseSolution {
    pub schema: String,
    pub provider: PoseProvider,
    pub raster_fingerprint: String,
    pub coordinate_convention: String,
    pub frames: Vec<PoseFrame>,
    pub diagnostics: PoseDiagnostics,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoseProvider {
    pub kind: String,
    pub version: String,
    pub settings_fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoseFrame {
    pub frame_index: usize,
    pub image_name: String,
    pub registered: bool,
    /// Row-major 3×4 COLMAP world-to-camera matrix in f64.
    pub world_to_camera: [f64; 12],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct PoseDiagnostics {
    pub input_frames: usize,
    pub registered_frames: usize,
    pub duplicate_images: usize,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PoseError {
    #[error("COLMAP images.txt line {line} is malformed: {reason}")]
    ColmapLine { line: usize, reason: String },
    #[error("COLMAP image {image_name:?} is not an exact decoded raster")]
    UnknownRaster { image_name: String },
    #[error("COLMAP images.txt contains duplicate raster {image_name:?}")]
    DuplicateRaster { image_name: String },
    #[error("raster manifest is invalid: {0}")]
    Raster(String),
    #[error("pose solution is invalid: {0}")]
    Solution(String),
}

impl RasterManifest {
    #[must_use]
    pub fn fingerprint(&self) -> String {
        self.raster_fingerprint.clone()
    }

    pub fn validate(&self) -> Result<(), PoseError> {
        if self.schema != "vestra.raster/v1"
            || !self.duration_seconds.is_finite()
            || self.duration_seconds <= 0.0
            || self.output_width == 0
            || self.output_height == 0
            || self.crop.width == 0
            || self.crop.height == 0
        {
            return Err(PoseError::Raster(
                "invalid dimensions or duration".to_owned(),
            ));
        }
        let mut names = BTreeSet::new();
        for frame in &self.frames {
            if frame.file_name.is_empty() || !names.insert(&frame.file_name) {
                return Err(PoseError::Raster("frame names must be unique".to_owned()));
            }
        }
        let expected = raster_fingerprint(self);
        if self.raster_fingerprint != expected {
            return Err(PoseError::Raster("raster fingerprint mismatch".to_owned()));
        }
        Ok(())
    }
}

/// Finalizes a new raster manifest after the exact PPM cache has been created.
#[must_use]
pub fn finalized_raster_manifest(mut manifest: RasterManifest) -> RasterManifest {
    manifest.schema = "vestra.raster/v1".to_owned();
    manifest.raster_fingerprint = raster_fingerprint(&manifest);
    manifest
}

/// Validates an externally produced global trajectory against the immutable
/// decoded-raster contract.  Providers are deliberately allow-listed: a
/// syntactically valid JSON file must not become a geometry authority merely
/// because it contains twelve numbers per image.
pub fn validate_pose_solution(
    solution: &PoseSolution,
    raster: &RasterManifest,
) -> Result<(), PoseError> {
    raster.validate()?;
    if solution.schema != "vestra.pose-solution/v1" {
        return Err(PoseError::Solution("unsupported schema".to_owned()));
    }
    if solution.raster_fingerprint != raster.fingerprint() {
        return Err(PoseError::Solution(
            "raster fingerprint mismatch".to_owned(),
        ));
    }
    let expected_convention = match solution.provider.kind.as_str() {
        "colmap" => "COLMAP world; W2C row-major 3x4 f64",
        "droid-slam" | "vggt" => "OpenCV world; W2C row-major 3x4 f64",
        other => {
            return Err(PoseError::Solution(format!(
                "unsupported pose provider {other:?}"
            )));
        }
    };
    if solution.coordinate_convention != expected_convention {
        return Err(PoseError::Solution(format!(
            "expected coordinate convention {expected_convention:?}"
        )));
    }
    if solution.provider.version.trim().is_empty()
        || solution.provider.settings_fingerprint.trim().is_empty()
    {
        return Err(PoseError::Solution(
            "provider version and settings fingerprint are required".to_owned(),
        ));
    }
    if solution.diagnostics.input_frames != raster.frames.len() {
        return Err(PoseError::Solution(
            "diagnostic input-frame count does not match raster manifest".to_owned(),
        ));
    }
    let raster_by_index = raster
        .frames
        .iter()
        .map(|frame| (frame.frame_index, frame.file_name.as_str()))
        .collect::<BTreeMap<_, _>>();
    let mut seen = BTreeSet::new();
    let mut registered = 0_usize;
    for frame in &solution.frames {
        let Some(&expected_name) = raster_by_index.get(&frame.frame_index) else {
            return Err(PoseError::Solution(format!(
                "unknown raster frame index {}",
                frame.frame_index
            )));
        };
        if frame.image_name != expected_name {
            return Err(PoseError::Solution(format!(
                "frame {} does not name its exact decoded raster",
                frame.frame_index
            )));
        }
        if !seen.insert(frame.frame_index) {
            return Err(PoseError::Solution(format!(
                "duplicate frame index {}",
                frame.frame_index
            )));
        }
        if frame.registered {
            registered += 1;
            validate_rigid_w2c(frame.world_to_camera)?;
        }
    }
    if solution.diagnostics.registered_frames != registered {
        return Err(PoseError::Solution(
            "diagnostic registered-frame count does not match frames".to_owned(),
        ));
    }
    Ok(())
}

fn validate_rigid_w2c(matrix: [f64; 12]) -> Result<(), PoseError> {
    if !matrix.iter().all(|value| value.is_finite()) {
        return Err(PoseError::Solution(
            "W2C contains non-finite values".to_owned(),
        ));
    }
    let rows = [
        [matrix[0], matrix[1], matrix[2]],
        [matrix[4], matrix[5], matrix[6]],
        [matrix[8], matrix[9], matrix[10]],
    ];
    for row in rows {
        let norm = row.iter().map(|value| value * value).sum::<f64>().sqrt();
        if (norm - 1.0).abs() > 1e-3 {
            return Err(PoseError::Solution(
                "W2C rotation is not normalized".to_owned(),
            ));
        }
    }
    for left in 0..3 {
        for right in (left + 1)..3 {
            let dot = rows[left]
                .iter()
                .zip(rows[right])
                .map(|(a, b)| a * b)
                .sum::<f64>();
            if dot.abs() > 1e-3 {
                return Err(PoseError::Solution(
                    "W2C rotation is not orthogonal".to_owned(),
                ));
            }
        }
    }
    let determinant = rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0]);
    if (determinant - 1.0).abs() > 1e-3 {
        return Err(PoseError::Solution(
            "W2C rotation is not right-handed".to_owned(),
        ));
    }
    Ok(())
}

/// Parses COLMAP `images.txt` (not `images.bin`).  Each non-comment pose line
/// is followed by one feature-observation line, which is intentionally ignored.
/// COLMAP's `QW QX QY QZ TX TY TZ` is W2C; the resulting matrix stays in that
/// convention all the way to the fusion adapter.
pub fn parse_colmap_images_txt(
    images_txt: &str,
    raster: &RasterManifest,
    provider: PoseProvider,
) -> Result<PoseSolution, PoseError> {
    raster.validate()?;
    let by_name = raster
        .frames
        .iter()
        .map(|frame| (frame.file_name.as_str(), frame.frame_index))
        .collect::<BTreeMap<_, _>>();
    let mut frames = Vec::new();
    let mut seen = BTreeSet::new();
    let mut expect_observations = false;
    for (offset, raw) in images_txt.lines().enumerate() {
        let line = offset + 1;
        let trimmed = raw.trim();
        // COLMAP always writes one observations line after an image pose; the
        // line is legitimately empty when that image has no observations.
        if expect_observations {
            expect_observations = false;
            continue;
        }
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fields = trimmed.split_whitespace().collect::<Vec<_>>();
        if fields.len() < 10 {
            return Err(PoseError::ColmapLine {
                line,
                reason: "expected IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME".to_owned(),
            });
        }
        let parse = |index: usize, name: &str| {
            fields[index]
                .parse::<f64>()
                .map_err(|_| PoseError::ColmapLine {
                    line,
                    reason: format!("{name} is not finite f64"),
                })
        };
        let qw = parse(1, "QW")?;
        let qx = parse(2, "QX")?;
        let qy = parse(3, "QY")?;
        let qz = parse(4, "QZ")?;
        let tx = parse(5, "TX")?;
        let ty = parse(6, "TY")?;
        let tz = parse(7, "TZ")?;
        if ![qw, qx, qy, qz, tx, ty, tz]
            .iter()
            .all(|value| value.is_finite())
        {
            return Err(PoseError::ColmapLine {
                line,
                reason: "pose contains non-finite values".to_owned(),
            });
        }
        let image_name = fields[9].to_owned();
        let Some(&frame_index) = by_name.get(image_name.as_str()) else {
            return Err(PoseError::UnknownRaster { image_name });
        };
        if !seen.insert(frame_index) {
            return Err(PoseError::DuplicateRaster { image_name });
        }
        frames.push(PoseFrame {
            frame_index,
            image_name,
            registered: true,
            world_to_camera: quaternion_w2c_matrix(qw, qx, qy, qz, tx, ty, tz)?,
        });
        expect_observations = true;
    }
    frames.sort_by_key(|frame| frame.frame_index);
    let registered_frames = frames.len();
    let solution = PoseSolution {
        schema: "vestra.pose-solution/v1".to_owned(),
        provider,
        raster_fingerprint: raster.fingerprint(),
        coordinate_convention: "COLMAP world; W2C row-major 3x4 f64".to_owned(),
        frames,
        diagnostics: PoseDiagnostics {
            input_frames: raster.frames.len(),
            registered_frames,
            duplicate_images: 0,
        },
    };
    validate_pose_solution(&solution, raster)?;
    Ok(solution)
}

fn quaternion_w2c_matrix(
    qw: f64,
    qx: f64,
    qy: f64,
    qz: f64,
    tx: f64,
    ty: f64,
    tz: f64,
) -> Result<[f64; 12], PoseError> {
    let norm = (qw * qw + qx * qx + qy * qy + qz * qz).sqrt();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(PoseError::Raster(
            "COLMAP quaternion has zero norm".to_owned(),
        ));
    }
    let (w, x, y, z) = (qw / norm, qx / norm, qy / norm, qz / norm);
    Ok([
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - z * w),
        2.0 * (x * z + y * w),
        tx,
        2.0 * (x * y + z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - x * w),
        ty,
        2.0 * (x * z - y * w),
        2.0 * (y * z + x * w),
        1.0 - 2.0 * (x * x + y * y),
        tz,
    ])
}

fn raster_fingerprint(manifest: &RasterManifest) -> String {
    let canonical = serde_json::json!({
        "source_sha256": manifest.source_sha256,
        "duration_seconds": manifest.duration_seconds,
        "source_width": manifest.source_width,
        "source_height": manifest.source_height,
        "crop": manifest.crop,
        "output_width": manifest.output_width,
        "output_height": manifest.output_height,
        "frames": manifest.frames,
    });
    let bytes = serde_json::to_vec(&canonical).expect("fixed raster schema serializes");
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raster() -> RasterManifest {
        finalized_raster_manifest(RasterManifest {
            schema: String::new(),
            source_sha256: "source".to_owned(),
            duration_seconds: 2.0,
            source_width: 1920,
            source_height: 1080,
            crop: RasterCrop {
                x: 150,
                y: 0,
                width: 1620,
                height: 1080,
            },
            output_width: 504,
            output_height: 336,
            frames: vec![
                RasterFrame {
                    frame_index: 0,
                    file_name: "frame-000001.ppm".to_owned(),
                    sha256: "a".to_owned(),
                    timestamp_millis: 0,
                },
                RasterFrame {
                    frame_index: 1,
                    file_name: "frame-000002.ppm".to_owned(),
                    sha256: "b".to_owned(),
                    timestamp_millis: 125,
                },
            ],
            raster_fingerprint: String::new(),
        })
    }

    #[test]
    fn parses_colmap_w2c_quaternion_against_exact_raster_names() {
        let solution = parse_colmap_images_txt(
            "1 1 0 0 0 1 2 3 1 frame-000001.ppm\n\n2 0 0 0 1 4 5 6 1 frame-000002.ppm\n\n",
            &raster(),
            PoseProvider {
                kind: "colmap".to_owned(),
                version: "4.1.1".to_owned(),
                settings_fingerprint: "settings".to_owned(),
            },
        )
        .unwrap();
        assert_eq!(solution.frames.len(), 2);
        assert_eq!(
            solution.frames[0].world_to_camera,
            [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0]
        );
        assert!((solution.frames[1].world_to_camera[0] + 1.0).abs() < 1e-12);
        assert!((solution.frames[1].world_to_camera[5] + 1.0).abs() < 1e-12);
    }

    #[test]
    fn refuses_an_unrelated_or_duplicate_colmap_raster() {
        let provider = PoseProvider {
            kind: "colmap".to_owned(),
            version: "4.1.1".to_owned(),
            settings_fingerprint: "settings".to_owned(),
        };
        assert!(matches!(
            parse_colmap_images_txt(
                "1 1 0 0 0 0 0 0 1 other.ppm\n\n",
                &raster(),
                provider.clone()
            ),
            Err(PoseError::UnknownRaster { .. })
        ));
        assert!(matches!(
            parse_colmap_images_txt(
                "1 1 0 0 0 0 0 0 1 frame-000001.ppm\n\n2 1 0 0 0 0 0 0 1 frame-000001.ppm\n\n",
                &raster(),
                provider
            ),
            Err(PoseError::DuplicateRaster { .. })
        ));
    }

    #[test]
    fn accepts_only_a_raster_bound_supported_sidecar_solution() {
        let raster = raster();
        let solution = PoseSolution {
            schema: "vestra.pose-solution/v1".to_owned(),
            provider: PoseProvider {
                kind: "droid-slam".to_owned(),
                version: "official-2026-08".to_owned(),
                settings_fingerprint: "pinned-settings".to_owned(),
            },
            raster_fingerprint: raster.fingerprint(),
            coordinate_convention: "OpenCV world; W2C row-major 3x4 f64".to_owned(),
            frames: vec![PoseFrame {
                frame_index: 0,
                image_name: "frame-000001.ppm".to_owned(),
                registered: true,
                world_to_camera: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            }],
            diagnostics: PoseDiagnostics {
                input_frames: 2,
                registered_frames: 1,
                duplicate_images: 0,
            },
        };
        validate_pose_solution(&solution, &raster).unwrap();

        let mut mismatched = solution.clone();
        mismatched.frames[0].image_name = "wrong.ppm".to_owned();
        assert!(matches!(
            validate_pose_solution(&mismatched, &raster),
            Err(PoseError::Solution(_))
        ));

        let mut unsupported = solution;
        unsupported.provider.kind = "untrusted-sidecar".to_owned();
        assert!(matches!(
            validate_pose_solution(&unsupported, &raster),
            Err(PoseError::Solution(_))
        ));
    }
}
