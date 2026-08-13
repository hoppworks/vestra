use std::{
    fs::File,
    io::{BufReader, Read},
    path::PathBuf,
};

use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use vestra_core::{
    BackprojectionSettings, ReconstructionSettings, SceneBundle, SceneProvenance,
    VideoExtractionSettings, WindowSettings, export_camera_json, export_fused_glb,
    export_fused_ply, export_fused_splat, extract_video_frames, fuse_scene_bundle, fused_topology,
    load_decoded_frame_cache, plan_windows, reconstruct_frames,
};
use vestra_engine::{Engine, QuantPref};
use vestra_studio::serve;

const VESTRA_LOCK: &str = include_str!("../../../vestra.lock.toml");

#[derive(Debug, Parser)]
#[command(name = "vestra", about = "Native video-to-world reconstruction")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Print the deterministic multi-view window schedule.
    Plan {
        #[arg(long)]
        frames: usize,
        #[arg(long, default_value_t = 12)]
        chunk_size: usize,
        #[arg(long, default_value_t = 3)]
        overlap: usize,
    },
    /// Reconstruct a local relative-scale `.vestra` world from one video.
    Reconstruct {
        #[arg(long)]
        video: PathBuf,
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        output: PathBuf,
        /// Reuse compatible durable window checkpoints in an existing bundle.
        /// Provenance must match exactly; incompatible checkpoints are refused.
        #[arg(long)]
        resume: bool,
        #[arg(long, default_value_t = 120)]
        frames: usize,
        #[arg(long, default_value_t = 504)]
        width: usize,
        #[arg(long, default_value_t = 336)]
        height: usize,
        #[arg(long, default_value_t = 12)]
        chunk_size: usize,
        #[arg(long, default_value_t = 3)]
        overlap: usize,
        #[arg(long, default_value_t = 1.0)]
        minimum_confidence: f32,
        /// One retained depth pixel per stride-square source pixels. The
        /// local JSON/WebGL v1 default intentionally caps a 120-frame job to
        /// a practical surfel volume; use 1 only for small diagnostics.
        #[arg(long, default_value_t = 8)]
        pixel_stride: usize,
    },
    /// Rebuild the derived world from a bundle's immutable measured windows.
    Fuse {
        #[arg(long)]
        scene: PathBuf,
    },
    /// Export the fused relative-scale world as an open ASCII PLY file.
    Export {
        #[arg(long)]
        scene: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Export the fused relative-scale world as a glTF 2.0 point-cloud GLB.
    ExportGlb {
        #[arg(long)]
        scene: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Export the fused relative-scale world as compact oriented `.splat` surfels.
    ExportSplat {
        #[arg(long)]
        scene: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Export composable window-local camera evidence as JSON.
    ExportCameras {
        #[arg(long)]
        scene: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Print provenance and evidence-backed quality signals for a scene bundle.
    Inspect {
        #[arg(long)]
        scene: PathBuf,
    },
    /// Check a fused scene against a versioned relative-scale regression profile.
    Verify {
        #[arg(long)]
        scene: PathBuf,
        /// JSON quality profile. It records only evidence thresholds, never metres.
        #[arg(long)]
        profile: PathBuf,
    },
    /// Serve a local interactive browser studio for a `.vestra` bundle.
    Serve {
        #[arg(long)]
        scene: PathBuf,
        #[arg(long, default_value_t = 4317)]
        port: u16,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Command::Plan {
            frames,
            chunk_size,
            overlap,
        } => {
            let windows = plan_windows(
                frames,
                WindowSettings {
                    chunk_size,
                    overlap,
                },
            )?;
            println!("{}", serde_json::to_string_pretty(&windows)?);
        }
        Command::Reconstruct {
            video,
            model,
            output,
            resume,
            frames,
            width,
            height,
            chunk_size,
            overlap,
            minimum_confidence,
            pixel_stride,
        } => {
            install_reconstruction_interrupt_handler()?;
            let provenance = SceneProvenance {
                engine_revision: locked_revision("engine")?,
                kernel_revision: locked_revision("kernels")?,
                model_fingerprint: sha256_file(&model)?,
                settings_fingerprint: settings_fingerprint(
                    &video,
                    frames,
                    width,
                    height,
                    chunk_size,
                    overlap,
                    minimum_confidence,
                    pixel_stride,
                )?,
            };
            let bundle = if resume {
                let bundle = SceneBundle::open(&output)?;
                if bundle.manifest()?.provenance != provenance {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "resume bundle provenance does not match this video, model, engine, kernels, or reconstruction settings",
                    )
                    .into());
                }
                bundle
            } else {
                SceneBundle::create(&output, provenance)?
            };
            let decode_settings = VideoExtractionSettings {
                width,
                height,
                max_frames: frames,
            };
            let decoded_directory = output.join("decoded");
            let decoded = if resume && decoded_directory.is_dir() {
                load_decoded_frame_cache(&video, &decoded_directory, decode_settings)?
            } else {
                extract_video_frames(&video, &decoded_directory, decode_settings)?
            };
            bundle.write_capture_quality(decoded.capture_quality.clone())?;
            if decoded.frames.is_empty() {
                return Err("ffmpeg produced no frames".into());
            }
            eprintln!(
                "decoded {} frames from {:.3}s of video",
                decoded.frames.len(),
                decoded.duration_seconds
            );
            eprintln!(
                "capture indicator: {:?} (mean adjacent luma delta {:.5})",
                decoded.capture_quality.disposition,
                decoded.capture_quality.mean_adjacent_luma_delta,
            );
            let mut engine = Engine::load(&model, QuantPref::PreferF32)?;
            let progress = reconstruct_frames(
                &mut engine,
                &decoded.frames,
                &bundle,
                ReconstructionSettings {
                    windows: WindowSettings {
                        chunk_size,
                        overlap,
                    },
                    backprojection: BackprojectionSettings {
                        minimum_confidence,
                        pixel_stride,
                        ..BackprojectionSettings::default()
                    },
                },
            )?;
            for checkpoint in &progress {
                let state = if checkpoint.reused {
                    "reused"
                } else {
                    "checkpointed"
                };
                eprintln!(
                    "{state} window {} [{}..{}) with {} measured points",
                    checkpoint.window.index,
                    checkpoint.window.start,
                    checkpoint.window.end,
                    checkpoint.measured_points
                );
            }
            let fusion = fuse_scene_bundle(&bundle)?;
            eprintln!(
                "fused {} measured windows into {} relative-scale points",
                fusion.aligned_windows, fusion.points
            );
            println!(
                "{}",
                serde_json::json!({
                    "bundle": bundle.root(),
                    "decoded_frames": decoded.frames.len(),
                    "capture_quality": decoded.capture_quality,
                    "windows": progress.len(),
                    "reused_windows": progress.iter().filter(|item| item.reused).count(),
                    "inferred_windows": progress.iter().filter(|item| !item.reused).count(),
                    "measured_points": progress.iter().map(|item| item.measured_points).sum::<usize>(),
                    "fused_chunk": fusion.chunk_hash,
                    "fused_points": fusion.points,
                })
            );
        }
        Command::Fuse { scene } => {
            let bundle = SceneBundle::open(scene)?;
            let fusion = fuse_scene_bundle(&bundle)?;
            println!(
                "{}",
                serde_json::json!({
                    "bundle": bundle.root(),
                    "fused_chunk": fusion.chunk_hash,
                    "windows": fusion.aligned_windows,
                    "fused_points": fusion.points,
                })
            );
        }
        Command::Export { scene, output } => {
            let bundle = SceneBundle::open(scene)?;
            let points = export_fused_ply(&bundle, &output)?;
            println!(
                "{}",
                serde_json::json!({
                    "bundle": bundle.root(),
                    "format": "ply/ascii",
                    "output": output,
                    "points": points,
                    "scale": "relative",
                })
            );
        }
        Command::ExportGlb { scene, output } => {
            let bundle = SceneBundle::open(scene)?;
            let points = export_fused_glb(&bundle, &output)?;
            println!(
                "{}",
                serde_json::json!({
                    "bundle": bundle.root(),
                    "format": "glb/gltf-2.0-points",
                    "output": output,
                    "points": points,
                    "scale": "relative",
                })
            );
        }
        Command::ExportSplat { scene, output } => {
            let bundle = SceneBundle::open(scene)?;
            let points = export_fused_splat(&bundle, &output)?;
            println!(
                "{}",
                serde_json::json!({
                    "bundle": bundle.root(),
                    "format": "splat/antimatter15",
                    "output": output,
                    "points": points,
                    "scale": "relative",
                })
            );
        }
        Command::ExportCameras { scene, output } => {
            let bundle = SceneBundle::open(scene)?;
            let cameras = export_camera_json(&bundle, &output)?;
            println!(
                "{}",
                serde_json::json!({
                    "bundle": bundle.root(),
                    "format": "camera-json/v1",
                    "output": output,
                    "cameras": cameras,
                    "scale": "relative",
                })
            );
        }
        Command::Inspect { scene } => {
            let bundle = SceneBundle::open(scene)?;
            println!("{}", serde_json::to_string_pretty(&scene_report(&bundle)?)?);
        }
        Command::Verify { scene, profile } => {
            let bundle = SceneBundle::open(scene)?;
            let profile: SceneQualityProfile = serde_json::from_reader(File::open(profile)?)?;
            let result = verify_scene(&bundle, &profile)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            if !result.passed {
                return Err("scene does not satisfy the supplied regression profile".into());
            }
        }
        Command::Serve { scene, port } => {
            let _bundle = SceneBundle::open(&scene)?;
            eprintln!("Vestra Studio is listening at http://127.0.0.1:{port}");
            serve(scene, port)?;
        }
    }
    Ok(())
}

/// Makes command-line cancellation bounded independently of the current
/// inference kernel. Scene publication is already atomic: an interrupt can
/// leave an unreferenced temporary chunk, but never publishes a partial
/// manifest. A later `reconstruct --resume` validates and reuses complete
/// checkpoints before executing any missing window.
fn install_reconstruction_interrupt_handler() -> Result<(), ctrlc::Error> {
    ctrlc::set_handler(|| {
        eprintln!(
            "reconstruction canceled; durable checkpoints remain available for `vestra reconstruct --resume`"
        );
        std::process::exit(130);
    })
}

/// A versioned evidence contract for one fixture class. All values are counts,
/// ratios, or residuals in the scene's own relative units; the profile never
/// infers physical room dimensions.
#[derive(Debug, Clone, Deserialize)]
struct SceneQualityProfile {
    schema: String,
    name: String,
    minimum_measured_windows: usize,
    minimum_fused_points: usize,
    minimum_sequential_inlier_ratio: f32,
    maximum_sequential_rms_residual: f32,
    require_loop_closure: bool,
}

#[derive(Debug, Clone, Serialize)]
struct SceneVerification {
    schema: &'static str,
    profile: String,
    passed: bool,
    violations: Vec<String>,
    evidence: SceneVerificationEvidence,
}

#[derive(Debug, Clone, Serialize)]
struct SceneVerificationEvidence {
    measured_windows: usize,
    fused_points: usize,
    finite_points: bool,
    minimum_inlier_ratio: Option<f32>,
    maximum_rms_residual: Option<f32>,
    loop_closures: usize,
}

fn verify_scene(
    bundle: &SceneBundle,
    profile: &SceneQualityProfile,
) -> Result<SceneVerification, Box<dyn std::error::Error>> {
    if profile.schema != "vestra.scene-quality/v1" {
        return Err(format!(
            "unsupported scene quality profile schema `{}`",
            profile.schema
        )
        .into());
    }
    if !profile.minimum_sequential_inlier_ratio.is_finite()
        || !(0.0..=1.0).contains(&profile.minimum_sequential_inlier_ratio)
        || !profile.maximum_sequential_rms_residual.is_finite()
        || profile.maximum_sequential_rms_residual < 0.0
    {
        return Err("scene quality profile contains invalid thresholds".into());
    }
    let manifest = bundle.manifest()?;
    let Some(hash) = manifest.fused_chunk_hash.as_deref() else {
        return Err("scene has no fused relative world".into());
    };
    let fused = bundle.read_fused_scene(hash)?;
    let minimum_inlier_ratio = fused
        .alignments
        .iter()
        .filter_map(|alignment| {
            (alignment.correspondence_count > 0)
                .then_some(alignment.inlier_count as f32 / alignment.correspondence_count as f32)
        })
        .filter(|value| value.is_finite())
        .min_by(f32::total_cmp);
    let maximum_rms_residual = fused
        .alignments
        .iter()
        .map(|alignment| alignment.rms_residual)
        .filter(|value| value.is_finite())
        .max_by(f32::total_cmp);
    let finite_points = fused.points.iter().all(|point| {
        point.position.iter().all(|value| value.is_finite())
            && point.normal.iter().all(|value| value.is_finite())
            && point.radius.is_finite()
            && point.radius > 0.0
    });
    let loop_closures = fused
        .pose_graph
        .as_ref()
        .map_or(0, |graph| graph.loop_edges);
    let evidence = SceneVerificationEvidence {
        measured_windows: manifest.measured_chunk_hashes.len(),
        fused_points: fused.points.len(),
        finite_points,
        minimum_inlier_ratio,
        maximum_rms_residual,
        loop_closures,
    };
    let mut violations = Vec::new();
    if evidence.measured_windows < profile.minimum_measured_windows {
        violations.push(format!(
            "measured_windows {} < {}",
            evidence.measured_windows, profile.minimum_measured_windows
        ));
    }
    if evidence.fused_points < profile.minimum_fused_points {
        violations.push(format!(
            "fused_points {} < {}",
            evidence.fused_points, profile.minimum_fused_points
        ));
    }
    if !evidence.finite_points {
        violations.push("fused geometry contains non-finite or invalid-radius surfels".into());
    }
    if evidence.minimum_inlier_ratio < Some(profile.minimum_sequential_inlier_ratio) {
        violations.push(format!(
            "minimum_inlier_ratio {:?} < {}",
            evidence.minimum_inlier_ratio, profile.minimum_sequential_inlier_ratio
        ));
    }
    if evidence.maximum_rms_residual > Some(profile.maximum_sequential_rms_residual) {
        violations.push(format!(
            "maximum_rms_residual {:?} > {}",
            evidence.maximum_rms_residual, profile.maximum_sequential_rms_residual
        ));
    }
    if profile.require_loop_closure && evidence.loop_closures == 0 {
        violations.push("profile requires an accepted loop closure".into());
    }
    Ok(SceneVerification {
        schema: "vestra.scene-verification/v1",
        profile: profile.name.clone(),
        passed: violations.is_empty(),
        violations,
        evidence,
    })
}

/// Summarizes only persisted evidence. The report intentionally does not turn
/// a relative-scale scene or a capture indicator into an accuracy claim.
fn scene_report(bundle: &SceneBundle) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let manifest = bundle.manifest()?;
    let fused = manifest
        .fused_chunk_hash
        .as_deref()
        .map(|hash| bundle.read_fused_scene(hash))
        .transpose()?;
    let alignments = fused
        .as_ref()
        .map_or(&[][..], |chunk| chunk.alignments.as_slice());
    let correspondence_count = alignments
        .iter()
        .map(|alignment| alignment.correspondence_count)
        .sum::<usize>();
    let inlier_count = alignments
        .iter()
        .map(|alignment| alignment.inlier_count)
        .sum::<usize>();
    let scale = summarize(alignments.iter().map(|alignment| alignment.transform.scale));
    let rms_residual = summarize(alignments.iter().map(|alignment| alignment.rms_residual));
    let inlier_ratio = summarize(alignments.iter().filter_map(|alignment| {
        (alignment.correspondence_count > 0)
            .then_some(alignment.inlier_count as f32 / alignment.correspondence_count as f32)
    }));
    let finite_points = fused.as_ref().is_none_or(|chunk| {
        chunk.points.iter().all(|point| {
            point.position.iter().all(|value| value.is_finite())
                && point.normal.iter().all(|value| value.is_finite())
                && point.confidence.is_finite()
                && point.radius.is_finite()
                && point.radius > 0.0
        })
    });
    let topology = fused
        .as_ref()
        .map(|chunk| fused_topology(&chunk.points, chunk.voxel_size));
    let scene_state = match (&fused, finite_points) {
        (None, _) => "measured_only",
        (Some(_), false) => "invalid_fused_geometry",
        (Some(_), true) => "fused_relative_world",
    };

    Ok(serde_json::json!({
        "schema": manifest.schema,
        "scene": bundle.root(),
        "state": scene_state,
        "scale": manifest.scale,
        "coordinate_convention": manifest.coordinate_convention,
        "provenance": manifest.provenance,
        "capture_quality": manifest.capture_quality,
        "measured_window_count": manifest.measured_chunk_hashes.len(),
        "fused": fused.as_ref().map(|chunk| serde_json::json!({
            "points": chunk.points.len(),
            "progressive_point_chunks": manifest.fused_point_chunk_hashes.len(),
            "voxel_size": chunk.voxel_size,
            "finite_points": finite_points,
            "sequential_alignment_count": alignments.len(),
            "sequential_correspondence_count": correspondence_count,
            "sequential_inlier_count": inlier_count,
            "sequential_scale": scale,
            "sequential_rms_residual": rms_residual,
            "sequential_inlier_ratio": inlier_ratio,
            "pose_graph": chunk.pose_graph,
            "topology": topology,
        })),
        "interpretation": {
            "metric_accuracy": "not_claimed; v1 scenes use relative scale",
            "capture_indicator": "risk signal only; it is not geometric validation",
            "recommended_next_gate": "inspect the Studio result and validate a real room capture with known revisits"
        }
    }))
}

fn summarize(values: impl Iterator<Item = f32>) -> serde_json::Value {
    let mut values = values.filter(|value| value.is_finite()).collect::<Vec<_>>();
    values.sort_by(f32::total_cmp);
    if values.is_empty() {
        return serde_json::Value::Null;
    }
    serde_json::json!({
        "min": values[0],
        "median": values[values.len() / 2],
        "max": values[values.len() - 1],
    })
}

fn locked_revision(section: &str) -> Result<String, Box<dyn std::error::Error>> {
    let section_header = format!("[{section}]");
    let section_text = VESTRA_LOCK
        .split(&section_header)
        .nth(1)
        .ok_or_else(|| format!("missing {section_header} in vestra.lock.toml"))?;
    let revision_line = section_text
        .lines()
        .find(|line| line.trim_start().starts_with("revision ="))
        .ok_or_else(|| format!("missing revision for {section}"))?;
    revision_line
        .split('"')
        .nth(1)
        .map(str::to_owned)
        .ok_or_else(|| format!("invalid revision for {section}").into())
}

fn sha256_file(path: &PathBuf) -> Result<String, Box<dyn std::error::Error>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn settings_fingerprint(
    video: &PathBuf,
    frames: usize,
    width: usize,
    height: usize,
    chunk_size: usize,
    overlap: usize,
    minimum_confidence: f32,
    pixel_stride: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let video_hash = sha256_file(video)?;
    let settings = format!(
        "video={video_hash};frames={frames};width={width};height={height};chunk={chunk_size};overlap={overlap};minimum_confidence={minimum_confidence:?};pixel_stride={pixel_stride}"
    );
    Ok(Sha256::digest(settings.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vestra_core::{AlignmentReport, FusedPoint, FusedSceneChunk};

    #[test]
    fn lock_parser_extracts_pinned_component_revisions() {
        assert_eq!(locked_revision("engine").unwrap().len(), 40);
        assert_eq!(locked_revision("kernels").unwrap().len(), 40);
    }

    #[test]
    fn report_marks_unfused_bundles_as_measured_only() {
        let root = std::env::temp_dir().join(format!("vestra-cli-report-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let bundle = SceneBundle::create(
            &root,
            SceneProvenance {
                engine_revision: "engine".into(),
                kernel_revision: "kernels".into(),
                model_fingerprint: "model".into(),
                settings_fingerprint: "settings".into(),
            },
        )
        .unwrap();
        let report = scene_report(&bundle).unwrap();
        assert_eq!(report["state"], "measured_only");
        assert!(report["fused"].is_null());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn summary_is_order_independent_and_ignores_non_finite_values() {
        assert_eq!(
            summarize([f32::NAN, 3.0, 1.0, f32::INFINITY, 2.0].into_iter()),
            serde_json::json!({"min": 1.0, "median": 2.0, "max": 3.0})
        );
        assert!(summarize([f32::NAN].into_iter()).is_null());
    }

    #[test]
    fn verification_reports_evidence_and_rejects_a_required_missing_loop() {
        let root = std::env::temp_dir().join(format!("vestra-cli-verify-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let bundle = SceneBundle::create(
            &root,
            SceneProvenance {
                engine_revision: "engine".into(),
                kernel_revision: "kernels".into(),
                model_fingerprint: "model".into(),
                settings_fingerprint: "settings".into(),
            },
        )
        .unwrap();
        bundle
            .write_fused_scene(&FusedSceneChunk {
                alignments: vec![AlignmentReport {
                    transform: vestra_core::SimilarityTransform::IDENTITY,
                    correspondence_count: 100,
                    inlier_count: 95,
                    rms_residual: 0.01,
                    normalized_rms_residual: 0.01,
                }],
                pose_graph: None,
                window_poses: Vec::new(),
                voxel_size: 0.1,
                points: vec![FusedPoint {
                    position: [0.0; 3],
                    normal: [0.0, 0.0, 1.0],
                    color_srgb: [0; 3],
                    confidence: 1.0,
                    radius: 0.1,
                    contributors: 1,
                }],
            })
            .unwrap();
        let mut profile = SceneQualityProfile {
            schema: "vestra.scene-quality/v1".into(),
            name: "test".into(),
            minimum_measured_windows: 0,
            minimum_fused_points: 1,
            minimum_sequential_inlier_ratio: 0.9,
            maximum_sequential_rms_residual: 0.02,
            require_loop_closure: false,
        };
        assert!(verify_scene(&bundle, &profile).unwrap().passed);
        profile.require_loop_closure = true;
        let rejected = verify_scene(&bundle, &profile).unwrap();
        assert!(!rejected.passed);
        assert_eq!(
            rejected.violations,
            ["profile requires an accepted loop closure"]
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
