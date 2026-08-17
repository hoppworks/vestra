use std::{
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
    time::Instant,
};

use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use vestra_core::{
    BackprojectionSettings, CppPr2CapiStreamOutput, CppPr2Fixture, CppPr2MultiViewOutput,
    CppPr2StreamOutput, GlobalPoseFusionSettings, RasterFrame, RasterManifest,
    ReconstructionSettings, SceneBundle, SceneProvenance, StitchSettings, SurfaceFusion,
    TsdfSettings, VideoExtractionSettings, WindowSettings, capture_cpp_pr2_fixture,
    cpp_pr2_fixture_alignment_reports, cpp_pr2_fixture_trajectory,
    emit_cpp_pr2_loop_closed_reference_cloud, emit_cpp_pr2_reference_cloud,
    emit_cpp_pr2_tsdf_reference_cloud, export_camera_json, export_fused_glb, export_fused_ply,
    export_fused_splat, extract_video_frames, finalized_raster_manifest,
    fuse_scene_bundle_cpp_pr2_relative, fuse_scene_bundle_with_pose_solution,
    fuse_scene_bundle_with_settings, fused_topology, global_pose_window_reports,
    load_decoded_frame_cache, load_decoded_rgb24_cache, plan_windows, reconstruct_frames,
    video_raster_metadata,
};
use vestra_engine::{Engine, QuantPref, ViewInput};
use vestra_studio::{IntakeConfig, serve, serve_intake};

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
        /// Candidate image rate before later motion/quality selection. This
        /// remains constant as video duration grows.
        #[arg(long, default_value_t = 8.0)]
        candidate_fps: f64,
        /// Safety ceiling, not a quality target. At 8 fps it covers 225 s.
        #[arg(long, default_value_t = 1800)]
        hard_max_frames: usize,
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
        /// Build PR #2 normal-space TSDF surfels instead of compatibility voxel fusion.
        #[arg(long)]
        tsdf: bool,
        /// Use the PR #2 relative seam and loop-closure profile. This is the
        /// default for new worlds. The flag remains accepted explicitly so
        /// intake subprocesses can record the selected profile unambiguously.
        #[arg(long, default_value_t = true, action = clap::ArgAction::SetTrue)]
        cpp_pr2_relative: bool,
    },
    /// Rebuild the derived world from a bundle's immutable measured windows.
    Fuse {
        #[arg(long)]
        scene: PathBuf,
        /// Rebuild this derivative with PR #2 normal-space TSDF fusion.
        #[arg(long)]
        tsdf: bool,
        /// Use the PR #2 relative seam and loop-closure profile. New derived
        /// worlds default to the same profile as browser intake jobs.
        #[arg(long, default_value_t = true, action = clap::ArgAction::SetTrue)]
        cpp_pr2_relative: bool,
    },
    /// Validate and publish a COLMAP text-model pose solution for the exact
    /// decoded rasters stored by a scene. This does not alter the local world.
    PoseImportColmap {
        #[arg(long)]
        scene: PathBuf,
        /// COLMAP `images.txt` emitted by `model_converter --output_type TXT`.
        #[arg(long)]
        images_txt: PathBuf,
        #[arg(long, default_value = "unknown")]
        provider_version: String,
        /// Hash of the versioned COLMAP command/settings contract.
        #[arg(long)]
        settings_fingerprint: String,
    },
    /// Validate and publish a provider-neutral W2C pose solution.  DROID-SLAM
    /// and VGGT sidecars must emit this exact schema against the immutable
    /// decoded-raster manifest; this command never fills missing poses.
    PoseImportJson {
        #[arg(long)]
        scene: PathBuf,
        #[arg(long)]
        solution: PathBuf,
    },
    /// Build a separate, globally posed world from an already-published
    /// COLMAP solution. Raw DA3 evidence and the local world remain intact.
    FuseColmapGlobal {
        #[arg(long)]
        scene: PathBuf,
        #[arg(long)]
        pose_solution: String,
        /// Emit raw surfels for diagnostics instead of the default TSDF layer.
        #[arg(long)]
        raw_surfels: bool,
    },
    /// Build a separate world from any validated global-pose provider.
    FuseGlobalPose {
        #[arg(long)]
        scene: PathBuf,
        #[arg(long)]
        pose_solution: String,
        #[arg(long)]
        raw_surfels: bool,
    },
    /// Report the per-window camera fit to a published COLMAP solution without
    /// changing the selected world product.
    InspectColmapGlobal {
        #[arg(long)]
        scene: PathBuf,
        #[arg(long)]
        pose_solution: String,
    },
    /// Report local-window camera fits to any validated global-pose provider.
    InspectGlobalPose {
        #[arg(long)]
        scene: PathBuf,
        #[arg(long)]
        pose_solution: String,
    },
    /// Attach the exact decoded-raster contract to a legacy scene before
    /// importing a global pose provider. This never re-runs DA3 inference.
    RasterRecord {
        #[arg(long)]
        scene: PathBuf,
        #[arg(long)]
        video: PathBuf,
        #[arg(long)]
        candidate_fps: f64,
        #[arg(long)]
        hard_max_frames: usize,
        #[arg(long, default_value_t = 504)]
        width: usize,
        #[arg(long, default_value_t = 336)]
        height: usize,
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
    /// Start the localhost browser intake for selecting and reconstructing a video.
    App {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        jobs: PathBuf,
        #[arg(long, default_value_t = 4317)]
        port: u16,
        #[arg(long, default_value_t = 8.0)]
        candidate_fps: f64,
        #[arg(long, default_value_t = 1800)]
        hard_max_frames: usize,
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
        #[arg(long, default_value_t = 8)]
        pixel_stride: usize,
        /// Publish a normal-space TSDF surfel derivative in addition to the
        /// immutable measured evidence. This improves surface continuity but
        /// never changes the recorded camera/depth observations.
        #[arg(long, default_value_t = true, action = clap::ArgAction::SetTrue)]
        tsdf: bool,
        /// Capture and fuse the dense PR #2-relative geometry profile. This
        /// is the default for new browser jobs; it preserves the evidence
        /// needed for closed-loop ICP and deferred pose-graph fusion.
        #[arg(long, default_value_t = true, action = clap::ArgAction::SetTrue)]
        cpp_pr2_relative: bool,
    },
    /// Capture window-scoped DA3 output for the pinned C++ PR #2 stitcher oracle.
    /// This is diagnostic evidence, not a production scene export.
    OracleFixture {
        #[arg(long)]
        model: PathBuf,
        /// Existing canonical RGB24 PPM cache, normally `<scene>/decoded`.
        #[arg(long)]
        decoded: PathBuf,
        /// New VPS1 artifact. Refuses to overwrite an existing file.
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
        #[arg(long, default_value_t = 55.0)]
        confidence_percentile: f64,
        #[arg(long, default_value_t = 1.2)]
        point_size: f32,
        #[arg(long, default_value_t = 50)]
        minimum_overlap_points: usize,
        /// Enable PR #2's per-seam point-to-plane ICP in the C++ oracle.
        #[arg(long)]
        icp_refine: bool,
        /// Enable PR #2's non-adjacent loop-closure and Sim(3) pose graph in the C++ oracle.
        #[arg(long)]
        loop_close: bool,
    },
    /// Bench only the repeated PR #2 multi-view model boundary. Model loading
    /// and canonical PPM decoding occur before the timed samples.
    OracleModelBench {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        decoded: PathBuf,
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
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        #[arg(long, default_value_t = 10)]
        repeat: usize,
    },
    /// Validate and summarize a VPO1 output from the pinned C++ streaming oracle.
    OracleInspect {
        #[arg(long)]
        input: PathBuf,
    },
    /// Produce Vestra's transform-tier seam reports from an exact VPS1 fixture.
    OracleStitch {
        #[arg(long)]
        input: PathBuf,
    },
    /// Compare Vestra's PR #2-compatible pre-voxel emission with one exact VPO1 cloud.
    OracleCompare {
        #[arg(long)]
        fixture: PathBuf,
        #[arg(long)]
        reference: PathBuf,
        /// Compare the normal-space TSDF tier. The C++ VPO1 must use `--tsdf`.
        #[arg(long)]
        tsdf: bool,
    },
    /// Compare a Rust VPS1 replay with the C++ C-API's CPS1 output from the
    /// same model, decoded frames, windows, and geometry branch settings.
    OracleCompareCapi {
        #[arg(long)]
        fixture: PathBuf,
        #[arg(long)]
        reference: PathBuf,
        /// Compare normal-space TSDF output rather than the pre-voxel cloud.
        #[arg(long)]
        tsdf: bool,
    },
    /// Compare C++ and Rust DA3 multi-view tensors before any geometry phase.
    /// Both sides must use the identical decoded frames and 12/3-style schedule.
    OracleCompareModel {
        #[arg(long)]
        fixture: PathBuf,
        #[arg(long)]
        reference: PathBuf,
    },
    /// Run Vestra's pinned PR #2 geometry oracle without comparison I/O.
    ///
    /// This exists for fair geometry-only benchmarking against the equivalent
    /// C++ oracle harness: it parses the same VPS1 evidence and performs the
    /// same selected raw-cloud or TSDF work, then emits only a tiny summary.
    OracleRun {
        #[arg(long)]
        fixture: PathBuf,
        /// Run PR #2 normal-space TSDF fusion after trajectory optimization.
        #[arg(long)]
        tsdf: bool,
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
            candidate_fps,
            hard_max_frames,
            width,
            height,
            chunk_size,
            overlap,
            minimum_confidence,
            pixel_stride,
            tsdf,
            cpp_pr2_relative,
        } => {
            install_reconstruction_interrupt_handler()?;
            let provenance = SceneProvenance {
                engine_revision: locked_revision("engine")?,
                kernel_revision: locked_revision("kernels")?,
                model_fingerprint: sha256_file(&model)?,
                settings_fingerprint: settings_fingerprint(
                    &video,
                    candidate_fps,
                    hard_max_frames,
                    width,
                    height,
                    chunk_size,
                    overlap,
                    minimum_confidence,
                    pixel_stride,
                    cpp_pr2_relative,
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
                candidate_fps,
                max_frames: hard_max_frames,
            };
            let decoded_directory = output.join("decoded");
            let decoded = if resume && decoded_directory.is_dir() {
                load_decoded_frame_cache(&video, &decoded_directory, decode_settings)?
            } else {
                extract_video_frames(&video, &decoded_directory, decode_settings)?
            };
            bundle.write_capture_quality(decoded.capture_quality.clone())?;
            let raster = raster_manifest_for_decoded(&video, &decoded, decode_settings)?;
            bundle.write_raster_manifest(&raster)?;
            if decoded.frames.is_empty() {
                return Err("ffmpeg produced no frames".into());
            }
            eprintln!(
                "selected {} geometry frames from {} candidates at {:.3} fps over {:.3}s",
                decoded.frames.len(),
                decoded
                    .candidate_indices
                    .last()
                    .map_or(0, |index| index + 1),
                decode_settings.candidate_fps,
                decoded.duration_seconds,
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
                        minimum_confidence: if cpp_pr2_relative {
                            -f32::MAX
                        } else {
                            minimum_confidence
                        },
                        pixel_stride: if cpp_pr2_relative { 1 } else { pixel_stride },
                        ..BackprojectionSettings::default()
                    },
                    cpp_pr2_relative_capture: cpp_pr2_relative,
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
            let fusion = if cpp_pr2_relative {
                fuse_scene_bundle_cpp_pr2_relative(
                    &bundle,
                    tsdf.then_some(TsdfSettings::default()),
                )?
            } else {
                fuse_scene_bundle_with_settings(&bundle, fusion_settings(tsdf))?
            };
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
        Command::App {
            model,
            jobs,
            port,
            candidate_fps,
            hard_max_frames,
            width,
            height,
            chunk_size,
            overlap,
            minimum_confidence,
            pixel_stride,
            tsdf,
            cpp_pr2_relative,
        } => {
            let executable = std::env::current_exe()?;
            println!("Vestra intake: http://127.0.0.1:{port}");
            serve_intake(IntakeConfig {
                executable,
                model,
                jobs_root: jobs,
                port,
                candidate_fps,
                hard_max_frames,
                width,
                height,
                chunk_size,
                overlap,
                minimum_confidence,
                pixel_stride,
                tsdf,
                cpp_pr2_relative,
            })?;
        }
        Command::OracleFixture {
            model,
            decoded,
            output,
            frames,
            width,
            height,
            chunk_size,
            overlap,
            confidence_percentile,
            point_size,
            minimum_overlap_points,
            icp_refine,
            loop_close,
        } => {
            if output.exists() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    format!(
                        "refusing to overwrite existing oracle fixture at {}",
                        output.display()
                    ),
                )
                .into());
            }
            let frames = load_decoded_rgb24_cache(
                &decoded,
                VideoExtractionSettings {
                    width,
                    height,
                    candidate_fps: 1.0,
                    max_frames: frames,
                },
            )?;
            let mut engine = Engine::load(&model, QuantPref::PreferF32)?;
            let fixture = capture_cpp_pr2_fixture(
                &mut engine,
                &frames,
                WindowSettings {
                    chunk_size,
                    overlap,
                },
                confidence_percentile,
                point_size,
                minimum_overlap_points,
                vestra_core::CppPr2StreamBranches {
                    icp_refine,
                    loop_close,
                },
            )?;
            let mut file = File::options().write(true).create_new(true).open(&output)?;
            fixture.write_vps1(&mut file)?;
            println!(
                "wrote VPS1: {} frames, {} windows, {}×{}",
                fixture.frame_count,
                fixture.window_views.len(),
                fixture.width,
                fixture.height,
            );
        }
        Command::OracleModelBench {
            model,
            decoded,
            frames: requested_frames,
            width,
            height,
            chunk_size,
            overlap,
            warmup,
            repeat,
        } => {
            if warmup == 0 || repeat == 0 {
                return Err("oracle-model-bench requires positive --warmup and --repeat".into());
            }
            let frames = load_decoded_rgb24_cache(
                &decoded,
                VideoExtractionSettings {
                    width,
                    height,
                    candidate_fps: 1.0,
                    max_frames: requested_frames,
                },
            )?;
            let schedule = plan_windows(
                frames.len(),
                WindowSettings {
                    chunk_size,
                    overlap,
                },
            )?;
            // Borrowed window inputs are deliberately prebuilt before timing:
            // this benchmark measures only the shared DA3 multi-view model
            // boundary, never PPM decoding, model loading or input assembly.
            let windows = schedule
                .iter()
                .map(|window| {
                    frames[window.start..window.end]
                        .iter()
                        .map(|frame| ViewInput {
                            rgb_hwc_u8: &frame.rgb_hwc_u8,
                            h: frame.height,
                            w: frame.width,
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let mut engine = Engine::load(&model, QuantPref::PreferF32)?;
            let mut samples_ms = Vec::with_capacity(repeat);
            let mut checksum = 0.0_f64;
            for iteration in 0..warmup.saturating_add(repeat) {
                let started = Instant::now();
                for inputs in &windows {
                    let output = engine.infer_multi_view(inputs)?;
                    let Some(first) = output.views.first() else {
                        return Err("multi-view model produced no views".into());
                    };
                    let Some((&depth, &confidence)) = first.depth.first().zip(first.conf.first())
                    else {
                        return Err(
                            "multi-view model produced an empty depth/confidence map".into()
                        );
                    };
                    checksum += f64::from(depth) + f64::from(confidence);
                }
                if iteration >= warmup {
                    samples_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
                }
            }
            println!(
                "{}",
                serde_json::json!({
                    "schema": "vestra.pr2-multiview-model-bench/v1",
                    "frames": frames.len(),
                    "windows": windows.len(),
                    "samples_ms": samples_ms,
                    "checksum": checksum,
                })
            );
        }
        Command::OracleInspect { input } => {
            let mut input = BufReader::new(File::open(input)?);
            let output = CppPr2StreamOutput::read_vpo1(&mut input)?;
            let point_count = output.radius.len();
            let finite_points = output
                .xyz
                .chunks_exact(3)
                .zip(&output.radius)
                .filter(|(position, radius)| {
                    position.iter().all(|value| value.is_finite())
                        && radius.is_finite()
                        && **radius > 0.0
                })
                .count();
            println!(
                "{}",
                serde_json::json!({
                    "schema": "vestra.cpp-pr2-oracle/v1",
                    "frames": output.frame_count,
                    "width": output.width,
                    "height": output.height,
                    "windows": output.window_mid_frame.len(),
                    "points": point_count,
                    "finite_positive_radius_points": finite_points,
                    "warnings": output.warnings,
                    "loops_found": output.loops_found,
                    "metric_scale": output.metric_scale,
                    "frame_owned_points": output.counts,
                })
            );
        }
        Command::OracleStitch { input } => {
            let mut input = BufReader::new(File::open(input)?);
            let fixture = CppPr2Fixture::read_vps1(&mut input)?;
            let alignments = cpp_pr2_fixture_alignment_reports(&fixture)?;
            let fused = vestra_core::stitch_cpp_pr2_fixture_as_vestra(&fixture)?;
            let cpp_pr2_loop = vestra_core::cpp_pr2_closed_loop_oracle(&fixture)?;
            println!(
                "{}",
                serde_json::json!({
                    "schema": "vestra.cpp-pr2-transform-oracle/v1",
                    "frames": fixture.frame_count,
                    "windows": fixture.window_views.len(),
                    "requested_icp_refine": fixture.branches.icp_refine,
                    "requested_loop_close": fixture.branches.loop_close,
                    "vestra_loop_edges": fused.pose_graph.as_ref().map(|report| report.loop_edges).unwrap_or(0),
                    "vestra_pose_graph": fused.pose_graph,
                    "vestra_window_poses": fused.window_poses,
                    "cpp_pr2_loop_oracle": {
                        "edges": cpp_pr2_loop.loop_edges,
                        "pose_graph": cpp_pr2_loop.pose_graph,
                        "sequential_window_poses": cpp_pr2_loop.sequential_window_poses,
                        "optimized_window_poses": cpp_pr2_loop.optimized_window_poses,
                    },
                    "alignments": alignments,
                })
            );
        }
        Command::OracleCompare {
            fixture,
            reference,
            tsdf,
        } => {
            let mut fixture_reader = BufReader::new(File::open(fixture)?);
            let fixture = CppPr2Fixture::read_vps1(&mut fixture_reader)?;
            let mut reference_reader = BufReader::new(File::open(reference)?);
            let reference = CppPr2StreamOutput::read_vpo1(&mut reference_reader)?;
            let rust = if tsdf {
                emit_cpp_pr2_tsdf_reference_cloud(&fixture)?
            } else if fixture.branches.loop_close {
                emit_cpp_pr2_loop_closed_reference_cloud(&fixture)?
            } else {
                emit_cpp_pr2_reference_cloud(&fixture)?
            };
            let trajectory = cpp_pr2_fixture_trajectory(&fixture)?;
            let shared = rust.points.len().min(reference.radius.len());
            let mut position_absolute_sum = 0.0_f64;
            let mut position_absolute_max = 0.0_f32;
            let mut radius_absolute_sum = 0.0_f64;
            let mut radius_absolute_max = 0.0_f32;
            let mut rgb_mismatches = 0_usize;
            for (index, point) in rust.points.iter().take(shared).enumerate() {
                for axis in 0..3 {
                    let delta = (point.position[axis] - reference.xyz[index * 3 + axis]).abs();
                    position_absolute_sum += f64::from(delta);
                    position_absolute_max = position_absolute_max.max(delta);
                }
                let radius_delta = (point.radius - reference.radius[index]).abs();
                radius_absolute_sum += f64::from(radius_delta);
                radius_absolute_max = radius_absolute_max.max(radius_delta);
                if point.color_srgb != reference.rgb[index * 3..index * 3 + 3] {
                    rgb_mismatches += 1;
                }
            }
            let position_values = shared.saturating_mul(3);
            let (window_position_mae, window_position_max_abs) =
                trajectory_difference(&trajectory.window_positions, &reference.window_pos);
            let (frame_position_mae, frame_position_max_abs) =
                trajectory_difference(&trajectory.frame_positions, &reference.frame_pos);
            let (frame_forward_mae, frame_forward_max_abs) =
                trajectory_difference(&trajectory.frame_forwards, &reference.frame_fwd);
            println!(
                "{}",
                serde_json::json!({
                    "schema": "vestra.cpp-pr2-raw-cloud-comparison/v1",
                    "reference_points": reference.radius.len(),
                    "rust_points": rust.points.len(),
                    "point_count_matches": rust.points.len() == reference.radius.len(),
                    "reference_frame_owned_points": reference.counts,
                    "rust_frame_owned_points": rust.frame_owned_points,
                    "frame_owned_points_match": rust.frame_owned_points == reference.counts,
                    "shared_ordered_points": shared,
                    "rgb_mismatches": rgb_mismatches,
                    "position_mae": if position_values == 0 { 0.0 } else { position_absolute_sum / position_values as f64 },
                    "position_max_abs": position_absolute_max,
                    "radius_mae": if shared == 0 { 0.0 } else { radius_absolute_sum / shared as f64 },
                    "radius_max_abs": radius_absolute_max,
                    "window_mid_frames_match": trajectory.window_mid_frames == reference.window_mid_frame,
                    "window_position_mae": window_position_mae,
                    "window_position_max_abs": window_position_max_abs,
                    "frame_position_mae": frame_position_mae,
                    "frame_position_max_abs": frame_position_max_abs,
                    "frame_forward_mae": frame_forward_mae,
                    "frame_forward_max_abs": frame_forward_max_abs,
                    "alignments": rust.alignments,
                })
            );
        }
        Command::OracleCompareCapi {
            fixture,
            reference,
            tsdf,
        } => {
            let mut fixture_reader = BufReader::new(File::open(fixture)?);
            let fixture = CppPr2Fixture::read_vps1(&mut fixture_reader)?;
            let mut reference_reader = BufReader::new(File::open(reference)?);
            let reference = CppPr2CapiStreamOutput::read_cps1(&mut reference_reader)?;
            let rust = if tsdf {
                emit_cpp_pr2_tsdf_reference_cloud(&fixture)?
            } else if fixture.branches.loop_close {
                emit_cpp_pr2_loop_closed_reference_cloud(&fixture)?
            } else {
                emit_cpp_pr2_reference_cloud(&fixture)?
            };
            let trajectory = cpp_pr2_fixture_trajectory(&fixture)?;
            let shared = rust.points.len().min(reference.radius.len());
            let mut position_absolute_sum = 0.0_f64;
            let mut position_absolute_max = 0.0_f32;
            let mut radius_absolute_sum = 0.0_f64;
            let mut radius_absolute_max = 0.0_f32;
            let mut rgb_mismatches = 0_usize;
            for (index, point) in rust.points.iter().take(shared).enumerate() {
                for axis in 0..3 {
                    let delta = (point.position[axis] - reference.xyz[index * 3 + axis]).abs();
                    position_absolute_sum += f64::from(delta);
                    position_absolute_max = position_absolute_max.max(delta);
                }
                let radius_delta = (point.radius - reference.radius[index]).abs();
                radius_absolute_sum += f64::from(radius_delta);
                radius_absolute_max = radius_absolute_max.max(radius_delta);
                if point.color_srgb != reference.rgb[index * 3..index * 3 + 3] {
                    rgb_mismatches += 1;
                }
            }
            let position_values = shared.saturating_mul(3);
            let (frame_position_mae, frame_position_max_abs) =
                trajectory_difference(&trajectory.frame_positions, &reference.frame_pos);
            let (frame_forward_mae, frame_forward_max_abs) =
                trajectory_difference(&trajectory.frame_forwards, &reference.frame_fwd);
            println!(
                "{}",
                serde_json::json!({
                    "schema": "vestra.cpp-pr2-capi-comparison/v1",
                    "reference_frames": reference.frame_count,
                    "fixture_frames": fixture.frame_count,
                    "reference_points": reference.radius.len(),
                    "rust_points": rust.points.len(),
                    "point_count_matches": rust.points.len() == reference.radius.len(),
                    "reference_frame_owned_points": reference.counts,
                    "rust_frame_owned_points": rust.frame_owned_points,
                    "frame_owned_points_match": rust.frame_owned_points == reference.counts,
                    "shared_ordered_points": shared,
                    "rgb_mismatches": rgb_mismatches,
                    "position_mae": if position_values == 0 { 0.0 } else { position_absolute_sum / position_values as f64 },
                    "position_max_abs": position_absolute_max,
                    "radius_mae": if shared == 0 { 0.0 } else { radius_absolute_sum / shared as f64 },
                    "radius_max_abs": radius_absolute_max,
                    "frame_position_mae": frame_position_mae,
                    "frame_position_max_abs": frame_position_max_abs,
                    "frame_forward_mae": frame_forward_mae,
                    "frame_forward_max_abs": frame_forward_max_abs,
                    "tsdf": tsdf,
                })
            );
        }
        Command::OracleCompareModel { fixture, reference } => {
            let mut fixture_reader = BufReader::new(File::open(fixture)?);
            let fixture = CppPr2Fixture::read_vps1(&mut fixture_reader)?;
            let mut reference_reader = BufReader::new(File::open(reference)?);
            let reference = CppPr2MultiViewOutput::read_mvo1(&mut reference_reader)?;
            let schedule_matches = reference.frame_count == fixture.frame_count
                && reference.windows == fixture.windows
                && reference.width == fixture.width
                && reference.height == fixture.height
                && reference.views.len() == fixture.window_views.len();
            let mut depth = DifferenceStats::default();
            let mut confidence = DifferenceStats::default();
            let mut extrinsics = DifferenceStats::default();
            let mut intrinsics = DifferenceStats::default();
            let mut confidence_selection = Vec::with_capacity(reference.views.len());
            let mut views_match = schedule_matches;
            for (cpp_window, rust_window) in reference.views.iter().zip(&fixture.window_views) {
                if cpp_window.len() != rust_window.len() {
                    views_match = false;
                    continue;
                }
                for (cpp, rust) in cpp_window.iter().zip(rust_window) {
                    views_match &= cpp.depth.len() == rust.depth.len()
                        && cpp.confidence.len() == rust.confidence.len();
                    depth.extend(&cpp.depth, &rust.depth);
                    confidence.extend(&cpp.confidence, &rust.confidence);
                    extrinsics.extend(&cpp.world_to_camera, &rust.world_to_camera);
                    intrinsics.extend(&cpp.intrinsics, &rust.intrinsics);
                }
                let cpp_confidences = cpp_window
                    .iter()
                    .flat_map(|view| view.confidence.iter().copied())
                    .collect::<Vec<_>>();
                let rust_confidences = rust_window
                    .iter()
                    .flat_map(|view| view.confidence.iter().copied())
                    .collect::<Vec<_>>();
                let cpp_threshold =
                    percentile_linear(&cpp_confidences, fixture.confidence_percentile);
                let rust_threshold =
                    percentile_linear(&rust_confidences, fixture.confidence_percentile);
                confidence_selection.push(serde_json::json!({
                    "cpp_threshold": cpp_threshold,
                    "rust_threshold": rust_threshold,
                    "threshold_abs_delta": (cpp_threshold - rust_threshold).abs(),
                    "cpp_selected": cpp_confidences.iter().filter(|value| **value >= cpp_threshold).count(),
                    "rust_selected": rust_confidences.iter().filter(|value| **value >= rust_threshold).count(),
                }));
            }
            println!(
                "{}",
                serde_json::json!({
                    "schema": "vestra.cpp-pr2-multiview-model-comparison/v1",
                    "schedule_matches": schedule_matches,
                    "view_tensor_shapes_match": views_match,
                    "windows": reference.views.len(),
                    "depth": depth.json(),
                    "confidence": confidence.json(),
                    "confidence_selection": confidence_selection,
                    "extrinsics": extrinsics.json(),
                    "intrinsics": intrinsics.json(),
                })
            );
        }
        Command::OracleRun { fixture, tsdf } => {
            let mut fixture_reader = BufReader::new(File::open(fixture)?);
            let fixture = CppPr2Fixture::read_vps1(&mut fixture_reader)?;
            let cloud = if tsdf {
                emit_cpp_pr2_tsdf_reference_cloud(&fixture)?
            } else if fixture.branches.loop_close {
                emit_cpp_pr2_loop_closed_reference_cloud(&fixture)?
            } else {
                emit_cpp_pr2_reference_cloud(&fixture)?
            };
            println!(
                "{}",
                serde_json::json!({
                    "schema": "vestra.cpp-pr2-oracle-run/v1",
                    "frames": fixture.frame_count,
                    "windows": fixture.window_views.len(),
                    "tsdf": tsdf,
                    "points": cloud.points.len(),
                    "frame_owned_points": cloud.frame_owned_points,
                    "alignment_count": cloud.alignments.len(),
                })
            );
        }
        Command::Fuse {
            scene,
            tsdf,
            cpp_pr2_relative,
        } => {
            let bundle = SceneBundle::open(scene)?;
            let fusion = if cpp_pr2_relative {
                fuse_scene_bundle_cpp_pr2_relative(
                    &bundle,
                    tsdf.then_some(TsdfSettings::default()),
                )?
            } else {
                fuse_scene_bundle_with_settings(&bundle, fusion_settings(tsdf))?
            };
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
        Command::PoseImportColmap {
            scene,
            images_txt,
            provider_version,
            settings_fingerprint,
        } => {
            let bundle = SceneBundle::open(scene)?;
            let raster = bundle.read_raster_manifest()?;
            let text = std::fs::read_to_string(images_txt)?;
            let solution = vestra_core::parse_colmap_images_txt(
                &text,
                &raster,
                vestra_core::PoseProvider {
                    kind: "colmap".to_owned(),
                    version: provider_version,
                    settings_fingerprint,
                },
            )?;
            let hash = bundle.write_pose_solution(&solution)?;
            println!(
                "{}",
                serde_json::json!({
                    "schema": "vestra.pose-import/v1",
                    "pose_solution": hash,
                    "registered_frames": solution.diagnostics.registered_frames,
                    "input_frames": solution.diagnostics.input_frames,
                })
            );
        }
        Command::PoseImportJson { scene, solution } => {
            let bundle = SceneBundle::open(scene)?;
            let raster = bundle.read_raster_manifest()?;
            let file = File::open(solution)?;
            let solution: vestra_core::PoseSolution =
                serde_json::from_reader(BufReader::new(file))?;
            vestra_core::validate_pose_solution(&solution, &raster)?;
            let hash = bundle.write_pose_solution(&solution)?;
            println!(
                "{}",
                serde_json::json!({
                    "schema": "vestra.pose-import/v1",
                    "provider": solution.provider.kind,
                    "pose_solution": hash,
                    "registered_frames": solution.diagnostics.registered_frames,
                    "input_frames": solution.diagnostics.input_frames,
                })
            );
        }
        Command::FuseColmapGlobal {
            scene,
            pose_solution,
            raw_surfels,
        }
        | Command::FuseGlobalPose {
            scene,
            pose_solution,
            raw_surfels,
        } => {
            let bundle = SceneBundle::open(scene)?;
            let fusion = fuse_scene_bundle_with_pose_solution(
                &bundle,
                &pose_solution,
                GlobalPoseFusionSettings {
                    tsdf: (!raw_surfels).then(TsdfSettings::default),
                    ..GlobalPoseFusionSettings::default()
                },
            )?;
            println!(
                "{}",
                serde_json::json!({
                    "schema": "vestra.global-pose-fuse/v1",
                    "bundle": bundle.root(),
                    "pose_solution": pose_solution,
                    "fused_chunk": fusion.chunk_hash,
                    "windows": fusion.aligned_windows,
                    "fused_points": fusion.points,
                    "surface": if raw_surfels { "surfel" } else { "tsdf" },
                })
            );
        }
        Command::InspectColmapGlobal {
            scene,
            pose_solution,
        }
        | Command::InspectGlobalPose {
            scene,
            pose_solution,
        } => {
            let bundle = SceneBundle::open(scene)?;
            let reports = global_pose_window_reports(&bundle, &pose_solution)?;
            println!(
                "{}",
                serde_json::json!({
                    "schema": "vestra.colmap-global-inspect/v1",
                    "pose_solution": pose_solution,
                    "windows": reports.iter().map(|report| serde_json::json!({
                        "window_index": report.window_index,
                        "registered_cameras": report.registered_cameras,
                        "scale": report.local_to_global.map(|pose| pose.scale),
                        "rms_camera_residual": report.rms_camera_residual,
                        "normalized_camera_rms": report.normalized_camera_rms,
                    })).collect::<Vec<_>>(),
                })
            );
        }
        Command::RasterRecord {
            scene,
            video,
            candidate_fps,
            hard_max_frames,
            width,
            height,
        } => {
            let bundle = SceneBundle::open(&scene)?;
            let settings = VideoExtractionSettings {
                width,
                height,
                candidate_fps,
                max_frames: hard_max_frames,
            };
            let decoded = load_decoded_frame_cache(&video, scene.join("decoded"), settings)?;
            let raster = raster_manifest_for_decoded(&video, &decoded, settings)?;
            let hash = bundle.write_raster_manifest(&raster)?;
            println!(
                "{}",
                serde_json::json!({
                    "schema": "vestra.raster-record/v1",
                    "raster_manifest": hash,
                    "frames": raster.frames.len(),
                    "raster_fingerprint": raster.raster_fingerprint,
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

fn trajectory_difference(left: &[[f32; 3]], right_flat: &[f32]) -> (f64, f32) {
    let shared = left.len().min(right_flat.len() / 3);
    if shared == 0 {
        return (0.0, 0.0);
    }
    let mut sum = 0.0_f64;
    let mut maximum = 0.0_f32;
    for (index, point) in left.iter().take(shared).enumerate() {
        for axis in 0..3 {
            let delta = (point[axis] - right_flat[index * 3 + axis]).abs();
            sum += f64::from(delta);
            maximum = maximum.max(delta);
        }
    }
    (sum / (shared * 3) as f64, maximum)
}

fn percentile_linear(values: &[f32], percentile: f64) -> f32 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f32::total_cmp);
    if sorted.len() <= 1 {
        return sorted.first().copied().unwrap_or(0.0);
    }
    let index = percentile.clamp(0.0, 100.0) / 100.0 * (sorted.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    let fraction = index - lower as f64;
    (f64::from(sorted[lower]) + fraction * f64::from(sorted[upper] - sorted[lower])) as f32
}

#[derive(Default)]
struct DifferenceStats {
    count: usize,
    absolute_sum: f64,
    maximum: f32,
    bitwise_mismatches: usize,
}

impl DifferenceStats {
    fn extend(&mut self, left: &[f32], right: &[f32]) {
        for (&left, &right) in left.iter().zip(right) {
            let delta = (left - right).abs();
            self.count += 1;
            self.absolute_sum += f64::from(delta);
            self.maximum = self.maximum.max(delta);
            self.bitwise_mismatches += usize::from(left.to_bits() != right.to_bits());
        }
    }

    fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "values_compared": self.count,
            "mae": if self.count == 0 { 0.0 } else { self.absolute_sum / self.count as f64 },
            "max_abs": self.maximum,
            "bitwise_mismatches": self.bitwise_mismatches,
        })
    }
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

fn sha256_file(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
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

fn raster_manifest_for_decoded(
    video: &Path,
    decoded: &vestra_core::VideoFrames,
    settings: VideoExtractionSettings,
) -> Result<RasterManifest, Box<dyn std::error::Error>> {
    let metadata = video_raster_metadata(video, settings)?;
    let frames = decoded
        .frames
        .iter()
        .enumerate()
        .map(|(frame_index, _)| {
            let file_name = format!("frame-{:06}.ppm", frame_index + 1);
            let candidate_index = decoded.candidate_indices.get(frame_index).ok_or_else(|| {
                format!("decoded frame {frame_index} is missing candidate time identity")
            })?;
            Ok(RasterFrame {
                frame_index,
                sha256: sha256_file(&decoded.decoded_directory.join(&file_name))?,
                file_name,
                timestamp_millis: ((*candidate_index as f64 / settings.candidate_fps) * 1000.0)
                    .round() as u64,
            })
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    Ok(finalized_raster_manifest(RasterManifest {
        schema: String::new(),
        source_sha256: sha256_file(video)?,
        duration_seconds: metadata.duration_seconds,
        source_width: metadata.source_width,
        source_height: metadata.source_height,
        crop: metadata.crop,
        output_width: settings.width,
        output_height: settings.height,
        frames,
        raster_fingerprint: String::new(),
    }))
}

#[allow(clippy::too_many_arguments)]
fn settings_fingerprint(
    video: &PathBuf,
    candidate_fps: f64,
    hard_max_frames: usize,
    width: usize,
    height: usize,
    chunk_size: usize,
    overlap: usize,
    minimum_confidence: f32,
    pixel_stride: usize,
    cpp_pr2_relative: bool,
) -> Result<String, Box<dyn std::error::Error>> {
    let video_hash = sha256_file(video)?;
    let settings = format!(
        "video={video_hash};candidate_fps={candidate_fps:?};hard_max_frames={hard_max_frames};geometry_keyframe_selection=v2;geometry_minimum_gap_seconds=0.4;geometry_minimum_novelty=0.015;geometry_maximum_gap_seconds=0.6;geometry_minimum_sharpness=0.012;width={width};height={height};chunk={chunk_size};overlap={overlap};minimum_confidence={minimum_confidence:?};pixel_stride={pixel_stride};cpp_pr2_relative={cpp_pr2_relative}"
    );
    Ok(Sha256::digest(settings.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn fusion_settings(tsdf: bool) -> StitchSettings {
    StitchSettings {
        surface_fusion: tsdf
            .then_some(SurfaceFusion::NormalSpaceTsdf(TsdfSettings::default()))
            .unwrap_or(SurfaceFusion::Voxel),
        ..StitchSettings::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vestra_core::{AlignmentReport, FusedPoint, FusedSceneChunk};

    #[test]
    fn new_browser_jobs_default_to_the_pr2_closed_loop_profile() {
        let cli = Cli::try_parse_from([
            "vestra",
            "app",
            "--model",
            "depth-anything-base-f32.gguf",
            "--jobs",
            "vestra-jobs",
        ])
        .unwrap();
        let Command::App {
            cpp_pr2_relative, ..
        } = cli.command
        else {
            panic!("expected app command");
        };
        assert!(cpp_pr2_relative);
    }

    #[test]
    fn explicit_closed_loop_flag_remains_compatible_with_intake_subprocesses() {
        let cli = Cli::try_parse_from([
            "vestra",
            "reconstruct",
            "--video",
            "capture.mov",
            "--model",
            "depth-anything-base-f32.gguf",
            "--output",
            "world.vestra",
            "--cpp-pr2-relative",
        ])
        .unwrap();
        let Command::Reconstruct {
            cpp_pr2_relative, ..
        } = cli.command
        else {
            panic!("expected reconstruct command");
        };
        assert!(cpp_pr2_relative);
    }

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
                pose_graph_edges: Vec::new(),
                pose_graph: None,
                window_poses: Vec::new(),
                voxel_size: 0.1,
                points: vec![FusedPoint {
                    position: [0.0; 3],
                    normal: [0.0, 0.0, 1.0],
                    color_srgb: [0; 3],
                    confidence: 1.0,
                    radius: 0.1,
                    first_observing_frame: -1,
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
