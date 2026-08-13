use std::{
    fs::File,
    io::{BufReader, Read},
    path::PathBuf,
};

use clap::{Parser, Subcommand};
use sha2::{Digest, Sha256};
use vestra_core::{
    BackprojectionSettings, ReconstructionSettings, SceneBundle, SceneProvenance,
    VideoExtractionSettings, WindowSettings, export_fused_ply, extract_video_frames,
    fuse_scene_bundle, plan_windows, reconstruct_frames,
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
        Command::Serve { scene, port } => {
            let _bundle = SceneBundle::open(&scene)?;
            eprintln!("Vestra Studio is listening at http://127.0.0.1:{port}");
            serve(scene, port)?;
        }
    }
    Ok(())
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
}
