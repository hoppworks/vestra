use std::{
    fs::File,
    io::{BufReader, Read},
    path::PathBuf,
};

use clap::{Parser, Subcommand};
use sha2::{Digest, Sha256};
use vestra_core::{
    BackprojectionSettings, ReconstructionSettings, SceneBundle, SceneProvenance,
    VideoExtractionSettings, WindowSettings, export_camera_json, export_fused_glb,
    export_fused_ply, export_fused_splat, extract_video_frames, fuse_scene_bundle, plan_windows,
    reconstruct_frames,
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
            frames,
            width,
            height,
            chunk_size,
            overlap,
            minimum_confidence,
            pixel_stride,
        } => {
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
            let bundle = SceneBundle::create(&output, provenance)?;
            let decoded = extract_video_frames(
                &video,
                output.join("decoded"),
                VideoExtractionSettings {
                    width,
                    height,
                    max_frames: frames,
                },
            )?;
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
                eprintln!(
                    "checkpointed window {} [{}..{}) with {} measured points",
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
        Command::Serve { scene, port } => {
            let _bundle = SceneBundle::open(&scene)?;
            eprintln!("Vestra Studio is listening at http://127.0.0.1:{port}");
            serve(scene, port)?;
        }
    }
    Ok(())
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
}
