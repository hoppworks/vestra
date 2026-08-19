use std::{
    fs::{self, File},
    io::{BufReader, Read},
    path::{Path, PathBuf},
    time::Instant,
};

use clap::{Command as ClapCommand, CommandFactory, FromArgMatches, Parser, Subcommand};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use vestra_core::{
    ArchitectureSettings, BackprojectionSettings, CppPr2CapiStreamOutput, CppPr2Fixture,
    CppPr2MultiViewOutput, CppPr2StreamOutput, FusedPoint, FusedSceneChunk,
    GlobalPoseFusionSettings, RasterFrame, RasterManifest, ReconstructionSettings, SceneBundle,
    SceneProvenance, StitchSettings, SurfaceFusion, TsdfObservation, TsdfSettings,
    VideoExtractionSettings, WindowSettings, capture_cpp_pr2_fixture,
    cpp_pr2_fixture_alignment_reports, cpp_pr2_fixture_trajectory,
    emit_cpp_pr2_loop_closed_reference_cloud, emit_cpp_pr2_reference_cloud,
    emit_cpp_pr2_tsdf_reference_cloud, export_camera_json, export_fused_glb, export_fused_ply,
    export_fused_splat, extract_video_frames, finalized_raster_manifest, fuse_normal_space_tsdf,
    fuse_scene_bundle_cpp_pr2_relative, fuse_scene_bundle_with_pose_solution,
    fuse_scene_bundle_with_settings, fused_topology, global_pose_window_reports,
    import_colmap_fused_ply, load_decoded_frame_cache, load_decoded_rgb24_cache, plan_windows,
    reconstruct_frames, video_raster_metadata,
};
use vestra_engine::{Engine, QuantPref, ViewInput};
use vestra_studio::{IntakeConfig, serve, serve_intake};

const VESTRA_LOCK: &str = include_str!("../../../vestra.lock.toml");
const PRODUCT_COMMANDS: [&str; 6] = ["app", "reconstruct", "demo", "serve", "inspect", "export"];
const LAB_COMMANDS: [&str; 28] = [
    "plan",
    "fuse",
    "pose-import-colmap",
    "pose-import-colmap-model",
    "pose-import-json",
    "fuse-colmap-global",
    "fuse-global-pose",
    "inspect-colmap-global",
    "inspect-global-pose",
    "inspect-colmap-frame-global",
    "fuse-colmap-frame-global",
    "import-colmap-mvs",
    "import-da3-pose-conditioned",
    "fuse-da3-pose-conditioned-tsdf",
    "extract-architecture",
    "raster-record",
    "export-glb",
    "export-splat",
    "export-cameras",
    "verify",
    "oracle-fixture",
    "oracle-model-bench",
    "oracle-inspect",
    "oracle-stitch",
    "oracle-compare",
    "oracle-compare-capi",
    "oracle-compare-model",
    "oracle-run",
];

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
        /// One retained depth pixel per stride-square source pixels. Product
        /// jobs use 2 for a dense but bounded source cloud; 1 is reserved for
        /// small PR #2 oracle diagnostics.
        #[arg(long, default_value_t = 8)]
        pixel_stride: usize,
        /// Build PR #2 normal-space TSDF surfels instead of compatibility voxel fusion.
        #[arg(long)]
        tsdf: bool,
        /// Capture the dense PR #2 oracle evidence profile. Product captures
        /// keep this opt-in so their configured confidence/point stride are
        /// respected; parity fixtures select it explicitly.
        #[arg(long)]
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
    /// Publish the complete globally bundle-adjusted COLMAP model, including
    /// calibrated rays and sparse tracks for frame-global depth rebasing.
    PoseImportColmapModel {
        #[arg(long)]
        scene: PathBuf,
        #[arg(long)]
        cameras_txt: PathBuf,
        #[arg(long)]
        images_txt: PathBuf,
        #[arg(long)]
        points3d_txt: PathBuf,
        #[arg(long, default_value = "unknown")]
        provider_version: String,
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
    /// Report per-frame sparse-track depth calibration for frame-global
    /// COLMAP rebasing without publishing a world product.
    InspectColmapFrameGlobal {
        #[arg(long)]
        scene: PathBuf,
        #[arg(long)]
        pose_solution: String,
    },
    /// Publish a separate world that uses globally bundle-adjusted COLMAP
    /// cameras per source frame, never a window-level Sim(3) transform.
    FuseColmapFrameGlobal {
        #[arg(long)]
        scene: PathBuf,
        #[arg(long)]
        pose_solution: String,
        /// Emit raw surfels instead of the default frame-global TSDF product.
        #[arg(long)]
        raw_surfels: bool,
    },
    /// Publish a dense COLMAP-MVS control cloud as a separate browser product.
    /// It never changes DA3 measurements or any Vestra-derived world.
    ImportColmapMvs {
        #[arg(long)]
        scene: PathBuf,
        /// Binary-little-endian PLY produced by `colmap stereo_fusion`.
        #[arg(long)]
        ply: PathBuf,
        /// Explicitly label this as a photometric-only MVS control. This is
        /// required when the provider did not produce geometric consistency
        /// maps; it must never be presented as a verified geometric world.
        #[arg(long)]
        photometric: bool,
        /// Hash of the calibrated COLMAP pose solution that produced this
        /// dense reconstruction.  It supplies source-camera evidence in
        /// Studio; it does not alter any MVS vertices.
        #[arg(long)]
        pose_solution: String,
    },
    /// Publish an official DA3 pose-conditioned sidecar result as a separate
    /// COLMAP-camera world. The sidecar must bind the immutable raster and
    /// exact pose solution before its PLY can become visible in Studio.
    ImportDa3PoseConditioned {
        #[arg(long)]
        scene: PathBuf,
        /// Directory created by `tools/run_da3_pose_conditioned.py`.
        #[arg(long)]
        artifact: PathBuf,
        /// Hash of the globally bundle-adjusted COLMAP pose solution supplied
        /// to official DA3 as extrinsics and intrinsics.
        #[arg(long)]
        pose_solution: String,
    },
    /// Create a TSDF surfel derivative from the immutable DA3
    /// pose-conditioned COLMAP surfel world. It retains the raw product and
    /// selects the derivative as a separate Studio product.
    FuseDa3PoseConditionedTsdf {
        #[arg(long)]
        scene: PathBuf,
        #[arg(long)]
        pose_solution: String,
        /// Bound memory without preferentially retaining an early video
        /// prefix. The input order is already first-owner frame-major, so a
        /// regular stride samples every admitted frame and raster region.
        #[arg(long, default_value_t = 1_000_000)]
        maximum_observations: usize,
        /// Derive from the held-out verified calibrated DA3 surfel product
        /// instead of the immutable raw diagnostic product.
        #[arg(long)]
        calibrated: bool,
        /// Derive from the separately verified MVS-guided DA3 surfel product.
        /// This remains an explicitly labelled experimental surface; it never
        /// replaces the calibrated DA3 or MVS-only products.
        #[arg(long)]
        mvs_guided: bool,
    },
    /// Publish a separate, conservative architecture layer from one existing
    /// global surfel/TSDF product. Only directly supported plane cells are
    /// emitted; unsupported wall regions remain visible holes/openings.
    ExtractArchitecture {
        #[arg(long)]
        scene: PathBuf,
        /// Existing world product that supplies global geometry. It is never
        /// overwritten by this command.
        #[arg(long)]
        source_product: String,
        /// Bound the evidence sampled during plane extraction.
        #[arg(long, default_value_t = 250_000)]
        maximum_source_points: usize,
        /// Limit the number of distinct architectural planes displayed.
        #[arg(long, default_value_t = 12)]
        maximum_planes: usize,
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
    /// Open a precomputed `.vestra` scene in the local browser studio.
    /// This never downloads a model or runs inference.
    Demo {
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
        #[arg(long, default_value_t = 2)]
        pixel_stride: usize,
        /// Publish a normal-space TSDF surfel derivative in addition to the
        /// immutable measured evidence. This improves surface continuity but
        /// never changes the recorded camera/depth observations.
        #[arg(long, default_value_t = true, action = clap::ArgAction::SetTrue)]
        tsdf: bool,
        /// Capture the dense PR #2 oracle evidence profile. The local product
        /// uses its configured stride and quality gates by default.
        #[arg(long)]
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

fn command_surface(
    binary_name: &'static str,
    about: &'static str,
    command_names: &[&str],
) -> ClapCommand {
    let complete = Cli::command();
    let selected = command_names
        .iter()
        .enumerate()
        .map(|(display_order, name)| {
            complete
                .find_subcommand(name)
                .unwrap_or_else(|| panic!("missing command definition for {name}"))
                .clone()
                .display_order(display_order)
        });
    ClapCommand::new(binary_name)
        .about(about)
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommands(selected)
}

fn product_command() -> ClapCommand {
    command_surface(
        "vestra",
        "Local video-to-world reconstruction",
        &PRODUCT_COMMANDS,
    )
}

fn lab_command() -> ClapCommand {
    command_surface(
        "vestra-lab",
        "Vestra engineering, provider, and validation tools",
        &LAB_COMMANDS,
    )
}

fn parse_command<I, T>(command: ClapCommand, arguments: I) -> Result<Command, clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let mut matches = command.try_get_matches_from(arguments)?;
    Ok(Cli::from_arg_matches_mut(&mut matches)?.command)
}

fn try_parse_product_from<I, T>(arguments: I) -> Result<Command, clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    parse_command(product_command(), arguments)
}

fn try_parse_lab_from<I, T>(arguments: I) -> Result<Command, clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    parse_command(lab_command(), arguments)
}

/// Run the curated end-user command surface.
pub fn run_product() -> Result<(), Box<dyn std::error::Error>> {
    let command = try_parse_product_from(std::env::args_os()).unwrap_or_else(|error| error.exit());
    execute(command)
}

/// Run the engineering and validation command surface.
pub fn run_lab() -> Result<(), Box<dyn std::error::Error>> {
    let command = try_parse_lab_from(std::env::args_os()).unwrap_or_else(|error| error.exit());
    execute(command)
}

/// Deliberately small import contract for the official Python DA3 sidecar.
/// The Rust process verifies all identities again; it never treats a PLY as a
/// trustworthy geometry product merely because it happens to be nearby.
#[derive(Debug, Deserialize)]
struct Da3PoseConditionedArtifact {
    schema: String,
    raster_fingerprint: String,
    pose_solution_hash: String,
    align_to_input_ext_scale: bool,
    frames: Vec<usize>,
    #[serde(default)]
    published_frames: Option<Vec<usize>>,
    #[serde(default)]
    decision: Option<String>,
    #[serde(default)]
    source: Option<Da3CalibrationSource>,
    #[serde(default)]
    contract: Option<Da3CalibrationContract>,
    #[serde(default)]
    hybrid: Option<Da3MvsHybridEvidence>,
    ply: Da3PoseConditionedPly,
    depth_frames: Da3PoseConditionedDepthFrames,
}

#[derive(Debug, Deserialize)]
struct Da3CalibrationSource {
    raw_manifest_sha256: String,
    raster_fingerprint: String,
    pose_solution_hash: String,
    batch_files: Vec<Da3CalibrationBatch>,
}

#[derive(Debug, Deserialize)]
struct Da3CalibrationBatch {
    file: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct Da3CalibrationContract {
    minimum_accepted_frame_fraction: f64,
    pixel_mapping: String,
    track_split: String,
}

#[derive(Debug, Deserialize)]
struct Da3MvsHybridEvidence {
    #[serde(default)]
    mvs_ply_sha256: Option<String>,
    #[serde(default)]
    mvs_vertices: Option<usize>,
    #[serde(default)]
    mvs_depth_map_index_sha256: Option<String>,
    #[serde(default)]
    mvs_depth_map_count: Option<usize>,
    #[serde(default)]
    per_frame: Vec<Da3MvsPatchMatchFrame>,
    pixel_policy: String,
    median_mvs_coverage: f64,
}

#[derive(Debug, Deserialize)]
struct Da3MvsPatchMatchFrame {
    frame_index: usize,
    file: String,
    sha256: String,
    mvs_coverage: f64,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_calibrated_da3_contract(
    sidecar: &Da3PoseConditionedArtifact,
) -> Result<(), &'static str> {
    let source = sidecar
        .source
        .as_ref()
        .ok_or("calibrated DA3 artifact has no immutable raw-source binding")?;
    let contract = sidecar
        .contract
        .as_ref()
        .ok_or("calibrated DA3 artifact has no calibration contract")?;
    if !is_sha256(&source.raw_manifest_sha256)
        || source.raster_fingerprint != sidecar.raster_fingerprint
        || source.pose_solution_hash != sidecar.pose_solution_hash
        || source.batch_files.is_empty()
        || source.batch_files.iter().any(|batch| {
            Path::new(&batch.file)
                .file_name()
                .and_then(|name| name.to_str())
                != Some(batch.file.as_str())
                || !is_sha256(&batch.sha256)
        })
        || contract.pixel_mapping != "pixel-center-resize/v1"
        || contract.track_split != "sha256-track-id-fold/v1"
        || !contract.minimum_accepted_frame_fraction.is_finite()
        || contract.minimum_accepted_frame_fraction < 0.85
        || contract.minimum_accepted_frame_fraction > 1.0
    {
        return Err("calibrated DA3 artifact violates the V2 provenance contract");
    }
    Ok(())
}

fn validate_mvs_hybrid_evidence(sidecar: &Da3PoseConditionedArtifact) -> Result<(), &'static str> {
    let hybrid = sidecar
        .hybrid
        .as_ref()
        .ok_or("MVS-DA3 hybrid artifact has no MVS provenance")?;
    let fused_ply = hybrid.mvs_ply_sha256.as_deref().is_some_and(is_sha256)
        && hybrid.mvs_vertices.is_some_and(|count| count > 0);
    let patchmatch_maps = hybrid
        .mvs_depth_map_index_sha256
        .as_deref()
        .is_some_and(is_sha256)
        && hybrid.mvs_depth_map_count.is_some_and(|count| count > 0);
    if !(fused_ply || patchmatch_maps)
        || !matches!(
            hybrid.pixel_policy.as_str(),
            "mvs-zbuffer-where-observed-else-da3/v1"
                | "mvs-zbuffer-plus-coarse-local-ratio/v1"
                | "colmap-patchmatch-geometric-resample-else-da3/v1"
        )
        || !hybrid.median_mvs_coverage.is_finite()
        || hybrid.median_mvs_coverage <= 0.0
        || hybrid.median_mvs_coverage > 1.0
    {
        return Err("MVS-DA3 hybrid artifact violates its dense-depth provenance contract");
    }
    if hybrid.pixel_policy == "colmap-patchmatch-geometric-resample-else-da3/v1" {
        let expected_count = hybrid
            .mvs_depth_map_count
            .expect("PatchMatch evidence was checked above");
        if expected_count != sidecar.frames.len()
            || hybrid.per_frame.len() != expected_count
            || hybrid
                .per_frame
                .iter()
                .zip(&sidecar.frames)
                .any(|(map, frame)| {
                    map.frame_index != *frame
                        || map.file != format!("frame-{:06}.ppm.geometric.bin", frame + 1)
                        || !is_sha256(&map.sha256)
                        || !map.mvs_coverage.is_finite()
                        || !(0.0..=1.0).contains(&map.mvs_coverage)
                })
        {
            return Err("PatchMatch MVS evidence does not cover the DA3 frame set exactly");
        }
    }
    Ok(())
}

fn da3_tsdf_product_identity(
    calibrated: bool,
    mvs_guided: bool,
) -> Result<(&'static str, &'static str, &'static str), &'static str> {
    if calibrated && mvs_guided {
        return Err("choose either --calibrated or --mvs-guided, not both");
    }
    Ok(if mvs_guided {
        (
            "da3-mvs-guided-colmap-surfel",
            "colmap-mvs-geometric-plus-da3-local-guidance",
            "da3-mvs-guided-colmap-tsdf",
        )
    } else if calibrated {
        (
            "da3-pose-conditioned-colmap-calibrated-surfel",
            "da3-base-pose-conditioned-colmap-calibrated",
            "da3-pose-conditioned-colmap-calibrated-tsdf",
        )
    } else {
        (
            "da3-pose-conditioned-colmap-surfel",
            "da3-base-pose-conditioned-colmap",
            "da3-pose-conditioned-colmap-tsdf",
        )
    })
}

#[derive(Debug, Deserialize)]
struct Da3PoseConditionedPly {
    schema: String,
    file: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct Da3PoseConditionedDepthFrames {
    schema: String,
    directory: String,
    width: usize,
    height: usize,
    frames: Vec<Da3PoseConditionedDepthFrame>,
}

#[derive(Debug, Deserialize)]
struct Da3PoseConditionedDepthFrame {
    frame_index: usize,
    file: String,
    sha256: String,
}

fn execute(command: Command) -> Result<(), Box<dyn std::error::Error>> {
    match command {
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
        Command::PoseImportColmapModel {
            scene,
            cameras_txt,
            images_txt,
            points3d_txt,
            provider_version,
            settings_fingerprint,
        } => {
            let bundle = SceneBundle::open(scene)?;
            let raster = bundle.read_raster_manifest()?;
            let solution = vestra_core::parse_colmap_global_model(
                &std::fs::read_to_string(cameras_txt)?,
                &std::fs::read_to_string(images_txt)?,
                &std::fs::read_to_string(points3d_txt)?,
                &raster,
                vestra_core::PoseProvider {
                    kind: "colmap".to_owned(),
                    version: provider_version,
                    settings_fingerprint,
                },
            )?;
            let track_count = solution
                .global_trajectory
                .as_ref()
                .map_or(0, |evidence| evidence.tracks.len());
            let hash = bundle.write_pose_solution(&solution)?;
            println!(
                "{}",
                serde_json::json!({
                    "schema": "vestra.pose-import-global-model/v1",
                    "pose_solution": hash,
                    "registered_frames": solution.diagnostics.registered_frames,
                    "sparse_tracks": track_count,
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
        Command::InspectColmapFrameGlobal {
            scene,
            pose_solution,
        } => {
            let bundle = SceneBundle::open(scene)?;
            let reports = vestra_core::frame_global_reports(
                &bundle,
                &pose_solution,
                vestra_core::FrameGlobalFusionSettings::default(),
            )?;
            println!(
                "{}",
                serde_json::json!({
                    "schema": "vestra.colmap-frame-global-inspect/v1",
                    "pose_solution": pose_solution,
                    "frames": reports.iter().map(|report| serde_json::json!({
                        "frame_index": report.frame_index,
                        "registered": report.registered,
                        "scale_samples": report.scale_samples,
                        "held_out_samples": report.held_out_samples,
                        "scale": report.scale,
                        "held_out_median_log_error": report.held_out_median_log_error,
                    })).collect::<Vec<_>>(),
                })
            );
        }
        Command::FuseColmapFrameGlobal {
            scene,
            pose_solution,
            raw_surfels,
        } => {
            let bundle = SceneBundle::open(scene)?;
            let fusion = vestra_core::fuse_scene_bundle_frame_global(
                &bundle,
                &pose_solution,
                vestra_core::FrameGlobalFusionSettings {
                    tsdf: (!raw_surfels).then(TsdfSettings::default),
                    ..vestra_core::FrameGlobalFusionSettings::default()
                },
            )?;
            println!(
                "{}",
                serde_json::json!({
                    "schema": "vestra.colmap-frame-global-fusion/v1",
                    "bundle": bundle.root(),
                    "pose_solution": pose_solution,
                    "fused_chunk": fusion.chunk_hash,
                    "fused_frames": fusion.fused_frames,
                    "omitted_frames": fusion.omitted_frames,
                    "fused_points": fusion.points,
                    "surface": if raw_surfels { "surfel" } else { "tsdf" },
                })
            );
        }
        Command::ImportColmapMvs {
            scene,
            ply,
            photometric,
            pose_solution,
        } => {
            let bundle = SceneBundle::open(scene)?;
            let solution = bundle.read_pose_solution(&pose_solution)?;
            let source_frames = solution
                .frames
                .iter()
                .filter(|frame| frame.registered)
                .map(|frame| frame.frame_index)
                .collect::<Vec<_>>();
            if source_frames.is_empty() || solution.global_trajectory.is_none() {
                return Err(
                    "COLMAP MVS import requires a calibrated pose solution with registered frames"
                        .into(),
                );
            }
            let cloud = import_colmap_fused_ply(&ply)?;
            let (id, pose_authority) = if photometric {
                (
                    "colmap-mvs-photometric-control",
                    "colmap-dense-mvs-photometric",
                )
            } else {
                ("colmap-mvs-geometric", "colmap-dense-mvs-geometric")
            };
            let chunk_hash = bundle.write_fused_scene_as(
                &cloud,
                id,
                pose_authority,
                "surfel",
                Some(pose_solution.clone()),
            )?;
            bundle.set_world_product_source_frames(id, &source_frames)?;
            println!(
                "{}",
                serde_json::json!({
                    "schema": "vestra.colmap-mvs-import/v1",
                    "bundle": bundle.root(),
                    "input": ply,
                    "pose_solution": pose_solution,
                    "source_frames": source_frames.len(),
                    "fused_chunk": chunk_hash,
                    "fused_points": cloud.points.len(),
                    "surface": "surfel",
                    "consistency": if photometric { "photometric-only" } else { "geometric" },
                })
            );
        }
        Command::ImportDa3PoseConditioned {
            scene,
            artifact,
            pose_solution,
        } => {
            let bundle = SceneBundle::open(scene)?;
            let raster = bundle.read_raster_manifest()?;
            let sidecar: Da3PoseConditionedArtifact = serde_json::from_reader(BufReader::new(
                File::open(artifact.join("manifest.json"))?,
            ))?;
            let calibrated = sidecar.schema == "vestra.da3-pose-conditioned-calibration/v2";
            let mvs_hybrid = sidecar.schema == "vestra.da3-mvs-hybrid/v1";
            let mvs_guided = mvs_hybrid
                && sidecar.hybrid.as_ref().is_some_and(|hybrid| {
                    hybrid.pixel_policy == "mvs-zbuffer-plus-coarse-local-ratio/v1"
                });
            let mvs_patchmatch = mvs_hybrid
                && sidecar.hybrid.as_ref().is_some_and(|hybrid| {
                    hybrid.pixel_policy == "colmap-patchmatch-geometric-resample-else-da3/v1"
                });
            let verified_derivative = calibrated || mvs_hybrid;
            if (sidecar.schema != "vestra.da3-pose-conditioned/v1" && !verified_derivative)
                || sidecar.raster_fingerprint != raster.raster_fingerprint
                || sidecar.pose_solution_hash != pose_solution
                || !sidecar.align_to_input_ext_scale
            {
                return Err("DA3 artifact does not bind this raster, COLMAP pose solution, and external pose scale".into());
            }
            if verified_derivative && sidecar.decision.as_deref() != Some("accepted") {
                return Err(
                    "verified DA3 derivative was not accepted by its evidence contract".into(),
                );
            }
            if verified_derivative {
                validate_calibrated_da3_contract(&sidecar)?;
            }
            if mvs_hybrid {
                validate_mvs_hybrid_evidence(&sidecar)?;
            }
            if sidecar.ply.schema != "vestra.da3-pose-conditioned-ply/v1"
                || Path::new(&sidecar.ply.file)
                    .file_name()
                    .and_then(|name| name.to_str())
                    != Some(sidecar.ply.file.as_str())
                || sha256_file(&artifact.join(&sidecar.ply.file))? != sidecar.ply.sha256
            {
                return Err(
                    "DA3 artifact PLY is missing, unsafe, or does not match its recorded SHA-256"
                        .into(),
                );
            }
            if sidecar.depth_frames.schema != "vestra.da3-pose-conditioned-depth-frames/v1"
                || Path::new(&sidecar.depth_frames.directory)
                    .file_name()
                    .and_then(|name| name.to_str())
                    != Some(sidecar.depth_frames.directory.as_str())
                || sidecar.depth_frames.width != 504
                || sidecar.depth_frames.height != 336
            {
                return Err("DA3 artifact has no supported 504x336 depth-preview contract".into());
            }
            let solution = bundle.read_pose_solution(&pose_solution)?;
            let registered = solution
                .frames
                .iter()
                .filter(|frame| frame.registered)
                .map(|frame| frame.frame_index)
                .collect::<Vec<_>>();
            if sidecar.frames != registered || registered.is_empty() {
                return Err(
                    "DA3 artifact frame ownership differs from the registered COLMAP trajectory"
                        .into(),
                );
            }
            let published_frames = sidecar.published_frames.as_deref().unwrap_or(&registered);
            if published_frames.is_empty()
                || !published_frames
                    .iter()
                    .all(|frame| registered.binary_search(frame).is_ok())
                || published_frames.windows(2).any(|pair| pair[0] >= pair[1])
                || published_frames.len() * 100 < registered.len() * 85
            {
                return Err(
                    "DA3 artifact published frames are not an ordered, sufficiently covered subset of the registered COLMAP trajectory".into(),
                );
            }
            let depth_indices = sidecar
                .depth_frames
                .frames
                .iter()
                .map(|frame| frame.frame_index)
                .collect::<Vec<_>>();
            if depth_indices != published_frames {
                return Err(
                    "DA3 depth-preview frames differ from the published COLMAP frame subset".into(),
                );
            }
            for frame in &sidecar.depth_frames.frames {
                if Path::new(&frame.file)
                    .file_name()
                    .and_then(|name| name.to_str())
                    != Some(frame.file.as_str())
                    || frame.sha256.len() != 64
                    || sha256_file(
                        &artifact
                            .join(&sidecar.depth_frames.directory)
                            .join(&frame.file),
                    )? != frame.sha256
                {
                    return Err("DA3 depth-preview asset is missing, unsafe, or does not match its recorded SHA-256".into());
                }
            }
            let cloud = import_colmap_fused_ply(artifact.join(&sidecar.ply.file))?;
            let id = if mvs_patchmatch {
                "da3-mvs-patchmatch-colmap-surfel"
            } else if mvs_guided {
                "da3-mvs-guided-colmap-surfel"
            } else if mvs_hybrid {
                "da3-mvs-hybrid-colmap-surfel"
            } else if calibrated {
                "da3-pose-conditioned-colmap-calibrated-surfel"
            } else {
                "da3-pose-conditioned-colmap-surfel"
            };
            let depth_target = bundle.root().join("depth").join(id);
            if depth_target.exists() {
                fs::remove_dir_all(&depth_target)?;
            }
            fs::create_dir_all(&depth_target)?;
            for frame in &sidecar.depth_frames.frames {
                fs::copy(
                    artifact
                        .join(&sidecar.depth_frames.directory)
                        .join(&frame.file),
                    depth_target.join(&frame.file),
                )?;
            }
            let authority = if mvs_patchmatch {
                "colmap-patchmatch-geometric-plus-da3"
            } else if mvs_guided {
                "colmap-mvs-geometric-plus-da3-local-guidance"
            } else if mvs_hybrid {
                "colmap-mvs-geometric-plus-da3"
            } else if calibrated {
                "da3-base-pose-conditioned-colmap-calibrated"
            } else {
                "da3-base-pose-conditioned-colmap"
            };
            let chunk_hash = bundle.write_fused_scene_as(
                &cloud,
                id,
                authority,
                "surfel",
                Some(pose_solution.clone()),
            )?;
            bundle.set_world_product_source_frames(id, published_frames)?;
            bundle.set_world_product_depth_frame_count(id, sidecar.depth_frames.frames.len())?;
            println!(
                "{}",
                serde_json::json!({
                    "schema": if mvs_hybrid { "vestra.da3-mvs-hybrid-import/v1" } else if calibrated { "vestra.da3-pose-conditioned-calibrated-import/v2" } else { "vestra.da3-pose-conditioned-import/v1" },
                    "bundle": bundle.root(),
                    "artifact": artifact,
                    "pose_solution": pose_solution,
                    "source_frames": published_frames.len(),
                    "depth_preview_frames": sidecar.depth_frames.frames.len(),
                    "fused_chunk": chunk_hash,
                    "fused_points": cloud.points.len(),
                    "surface": "surfel",
                    "authority": authority,
                })
            );
        }
        Command::FuseDa3PoseConditionedTsdf {
            scene,
            pose_solution,
            maximum_observations,
            calibrated,
            mvs_guided,
        } => {
            if maximum_observations == 0 {
                return Err("maximum observations must be positive".into());
            }
            let bundle = SceneBundle::open(scene)?;
            let manifest = bundle.manifest()?;
            let (source_id, expected_authority, id) =
                da3_tsdf_product_identity(calibrated, mvs_guided)?;
            let raw = manifest
                .world_products
                .iter()
                .find(|product| product.id == source_id)
                .ok_or("requested pose-conditioned DA3 surfel product has not been imported")?;
            if raw.pose_authority != expected_authority
                || raw.pose_solution_hash.as_deref() != Some(pose_solution.as_str())
            {
                return Err("pose-conditioned DA3 surfel product does not bind the requested COLMAP pose solution".into());
            }
            let source_frames = raw.source_frame_indices.clone();
            let raw_cloud = bundle.read_fused_scene(&raw.fused_chunk_hash)?;
            let solution = bundle.read_pose_solution(&pose_solution)?;
            let cameras = solution
                .frames
                .iter()
                .filter(|frame| frame.registered)
                .filter_map(|frame| colmap_camera_centre(frame.world_to_camera))
                .collect::<Vec<_>>();
            if cameras.len() < 3 {
                return Err(
                    "TSDF derivative requires at least three valid global camera centres".into(),
                );
            }
            let stride = raw_cloud.points.len().div_ceil(maximum_observations).max(1);
            let observations = raw_cloud
                .points
                .iter()
                .step_by(stride)
                .map(|point| TsdfObservation {
                    position: point.position,
                    color_srgb: point.color_srgb,
                    confidence: point.confidence,
                    radius: point.radius,
                    frame_index: point.first_observing_frame,
                })
                .collect::<Vec<_>>();
            let surfels = fuse_normal_space_tsdf(&observations, &cameras, TsdfSettings::default());
            let tsdf_cloud = FusedSceneChunk {
                alignments: Vec::new(),
                pose_graph_edges: Vec::new(),
                pose_graph: None,
                window_poses: Vec::new(),
                voxel_size: 0.0,
                points: surfels
                    .into_iter()
                    .map(|surfel| FusedPoint {
                        position: surfel.position,
                        normal: surfel.normal,
                        color_srgb: surfel.color_srgb,
                        confidence: 1.0,
                        radius: surfel.radius,
                        first_observing_frame: surfel.first_observing_frame,
                        contributors: surfel.contributors,
                    })
                    .collect(),
            };
            let chunk_hash = bundle.write_fused_scene_as(
                &tsdf_cloud,
                id,
                expected_authority,
                "tsdf",
                Some(pose_solution.clone()),
            )?;
            bundle.set_world_product_source_frames(id, &source_frames)?;
            println!(
                "{}",
                serde_json::json!({
                    "schema": "vestra.da3-pose-conditioned-tsdf/v1",
                    "bundle": bundle.root(),
                    "pose_solution": pose_solution,
                    "source_product": source_id,
                    "fused_chunk": chunk_hash,
                    "source_points": raw_cloud.points.len(),
                    "tsdf_observations": observations.len(),
                    "fused_points": tsdf_cloud.points.len(),
                    "surface": "tsdf",
                })
            );
        }
        Command::ExtractArchitecture {
            scene,
            source_product,
            maximum_source_points,
            maximum_planes,
        } => {
            if maximum_source_points == 0 || maximum_planes == 0 {
                return Err("architecture extraction limits must be positive".into());
            }
            let bundle = SceneBundle::open(scene)?;
            let manifest = bundle.manifest()?;
            let source = manifest
                .world_products
                .iter()
                .find(|product| product.id == source_product)
                .cloned()
                .ok_or("requested source world product has not been published")?;
            let source_cloud = bundle.read_fused_scene(&source.fused_chunk_hash)?;
            let extraction = vestra_core::extract_architectural_planes(
                &source_cloud.points,
                ArchitectureSettings {
                    maximum_source_points,
                    maximum_planes,
                    ..ArchitectureSettings::default()
                },
            );
            if extraction.planes.is_empty() || extraction.points.is_empty() {
                return Err("no sufficiently supported architectural planes were found; retain the source surfel world".into());
            }
            let product_id = format!("{}-architecture", source.id);
            let architecture_cloud = FusedSceneChunk {
                alignments: Vec::new(),
                pose_graph_edges: Vec::new(),
                pose_graph: None,
                window_poses: Vec::new(),
                voxel_size: 0.0,
                points: extraction.points,
            };
            let chunk_hash = bundle.write_fused_scene_as(
                &architecture_cloud,
                &product_id,
                &format!("{}+supported-planes", source.pose_authority),
                "architectural-plane-support",
                source.pose_solution_hash.clone(),
            )?;
            let mesh =
                vestra_core::architecture_mesh_from_support_points(&architecture_cloud.points);
            let mesh_hash = bundle.set_world_product_architecture_mesh(&product_id, &mesh)?;
            bundle.set_world_product_source_frames(&product_id, &source.source_frame_indices)?;
            bundle.set_world_product_depth_frame_count(&product_id, source.depth_frame_count)?;
            println!(
                "{}",
                serde_json::json!({
                    "schema": "vestra.architecture-plane-support/v1",
                    "bundle": bundle.root(),
                    "source_product": source.id,
                    "product": product_id,
                    "fused_chunk": chunk_hash,
                    "planes": extraction.planes,
                    "surface_points": architecture_cloud.points.len(),
                    "mesh_vertices": mesh.vertices.len(),
                    "mesh_triangles": mesh.indices.len() / 3,
                    "mesh_chunk": mesh_hash,
                    "policy": "supported-planar-cells-only; openings-remain-unsupported",
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
        Command::Demo { scene, port } => {
            let _bundle = SceneBundle::open(&scene)?;
            eprintln!("Vestra demo is listening at http://127.0.0.1:{port}");
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

/// Recovers a global camera centre from a row-major COLMAP W2C matrix.
/// Keeping it here avoids treating any local window transform as a camera
/// authority for the pose-conditioned TSDF derivative.
fn colmap_camera_centre(world_to_camera: [f64; 12]) -> Option<[f32; 3]> {
    if !world_to_camera.iter().all(|value| value.is_finite()) {
        return None;
    }
    let translation = [world_to_camera[3], world_to_camera[7], world_to_camera[11]];
    let centre = [
        -(world_to_camera[0] * translation[0]
            + world_to_camera[4] * translation[1]
            + world_to_camera[8] * translation[2]),
        -(world_to_camera[1] * translation[0]
            + world_to_camera[5] * translation[1]
            + world_to_camera[9] * translation[2]),
        -(world_to_camera[2] * translation[0]
            + world_to_camera[6] * translation[1]
            + world_to_camera[10] * translation[2]),
    ];
    centre.iter().all(|value| value.is_finite()).then_some([
        centre[0] as f32,
        centre[1] as f32,
        centre[2] as f32,
    ])
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
    video: &Path,
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
        "video={video_hash};candidate_fps={candidate_fps:?};hard_max_frames={hard_max_frames};geometry_keyframe_selection=v3;geometry_minimum_gap_seconds=0.4;geometry_minimum_novelty=0.015;geometry_maximum_gap_seconds=0.6;geometry_minimum_sharpness=0.012;width={width};height={height};chunk={chunk_size};overlap={overlap};minimum_confidence={minimum_confidence:?};pixel_stride={pixel_stride};cpp_pr2_relative={cpp_pr2_relative}"
    );
    Ok(Sha256::digest(settings.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn fusion_settings(tsdf: bool) -> StitchSettings {
    StitchSettings {
        surface_fusion: if tsdf {
            SurfaceFusion::NormalSpaceTsdf(TsdfSettings::default())
        } else {
            SurfaceFusion::Voxel
        },
        ..StitchSettings::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::error::ErrorKind;
    use vestra_core::{AlignmentReport, FusedPoint, FusedSceneChunk};

    #[test]
    fn product_help_has_exact_curated_command_surface() {
        let command = product_command();
        let names = command
            .get_subcommands()
            .map(ClapCommand::get_name)
            .collect::<Vec<_>>();
        assert_eq!(names, PRODUCT_COMMANDS.as_slice());
        assert!(
            command
                .get_subcommands()
                .all(|item| item.get_all_aliases().next().is_none())
        );
    }

    #[test]
    fn lab_help_has_exact_engineering_command_surface() {
        let command = lab_command();
        let names = command
            .get_subcommands()
            .map(ClapCommand::get_name)
            .collect::<Vec<_>>();
        assert_eq!(names, LAB_COMMANDS.as_slice());
        assert!(
            command
                .get_subcommands()
                .all(|item| item.get_all_aliases().next().is_none())
        );
    }

    #[test]
    fn product_rejects_every_lab_command() {
        for command in LAB_COMMANDS {
            let error = try_parse_product_from(["vestra", command]).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::InvalidSubcommand, "{command}");
        }
    }

    #[test]
    fn lab_rejects_every_product_command() {
        for command in PRODUCT_COMMANDS {
            let error = try_parse_lab_from(["vestra-lab", command]).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::InvalidSubcommand, "{command}");
        }
    }

    #[test]
    fn curated_product_parser_reuses_reconstruction_arguments() {
        let command = try_parse_product_from([
            "vestra",
            "reconstruct",
            "--video",
            "capture.mov",
            "--model",
            "model.gguf",
            "--output",
            "world.vestra",
            "--resume",
        ])
        .unwrap();
        let Command::Reconstruct {
            video,
            model,
            output,
            resume,
            ..
        } = command
        else {
            panic!("expected reconstruct command");
        };
        assert_eq!(video, PathBuf::from("capture.mov"));
        assert_eq!(model, PathBuf::from("model.gguf"));
        assert_eq!(output, PathBuf::from("world.vestra"));
        assert!(resume);
    }

    #[test]
    fn demo_accepts_only_a_precomputed_scene_and_port() {
        let command = try_parse_product_from([
            "vestra",
            "demo",
            "--scene",
            "world.vestra",
            "--port",
            "9000",
        ])
        .unwrap();
        let Command::Demo { scene, port } = command else {
            panic!("expected demo command");
        };
        assert_eq!(scene, PathBuf::from("world.vestra"));
        assert_eq!(port, 9000);
    }

    #[test]
    fn new_browser_jobs_default_to_bounded_product_geometry() {
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
            cpp_pr2_relative,
            pixel_stride,
            ..
        } = cli.command
        else {
            panic!("expected app command");
        };
        assert!(!cpp_pr2_relative);
        assert_eq!(pixel_stride, 2);
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
    fn calibrated_da3_contract_requires_immutable_source_and_safe_profile() {
        let sidecar = Da3PoseConditionedArtifact {
            schema: "vestra.da3-pose-conditioned-calibration/v2".into(),
            raster_fingerprint: "raster".into(),
            pose_solution_hash: "pose".into(),
            align_to_input_ext_scale: true,
            frames: vec![1],
            published_frames: Some(vec![1]),
            decision: Some("accepted".into()),
            source: Some(Da3CalibrationSource {
                raw_manifest_sha256: "a".repeat(64),
                raster_fingerprint: "raster".into(),
                pose_solution_hash: "pose".into(),
                batch_files: vec![Da3CalibrationBatch {
                    file: "batch-0000.npz".into(),
                    sha256: "b".repeat(64),
                }],
            }),
            contract: Some(Da3CalibrationContract {
                minimum_accepted_frame_fraction: 0.85,
                pixel_mapping: "pixel-center-resize/v1".into(),
                track_split: "sha256-track-id-fold/v1".into(),
            }),
            hybrid: None,
            ply: Da3PoseConditionedPly {
                schema: "vestra.da3-pose-conditioned-ply/v1".into(),
                file: "world.ply".into(),
                sha256: "c".repeat(64),
            },
            depth_frames: Da3PoseConditionedDepthFrames {
                schema: "vestra.da3-pose-conditioned-depth-frames/v1".into(),
                directory: "depth-frames".into(),
                width: 504,
                height: 336,
                frames: Vec::new(),
            },
        };
        assert!(validate_calibrated_da3_contract(&sidecar).is_ok());
        let mut unsafe_sidecar = sidecar;
        unsafe_sidecar.contract.as_mut().unwrap().track_split = "random".into();
        assert!(validate_calibrated_da3_contract(&unsafe_sidecar).is_err());
    }

    #[test]
    fn mvs_hybrid_contract_requires_real_dense_evidence() {
        let sidecar = Da3PoseConditionedArtifact {
            schema: "vestra.da3-mvs-hybrid/v1".into(),
            raster_fingerprint: "raster".into(),
            pose_solution_hash: "pose".into(),
            align_to_input_ext_scale: true,
            frames: vec![1],
            published_frames: Some(vec![1]),
            decision: Some("accepted".into()),
            source: None,
            contract: None,
            hybrid: Some(Da3MvsHybridEvidence {
                mvs_ply_sha256: Some("d".repeat(64)),
                mvs_vertices: Some(1),
                mvs_depth_map_index_sha256: None,
                mvs_depth_map_count: None,
                per_frame: Vec::new(),
                pixel_policy: "mvs-zbuffer-where-observed-else-da3/v1".into(),
                median_mvs_coverage: 0.2,
            }),
            ply: Da3PoseConditionedPly {
                schema: "vestra.da3-pose-conditioned-ply/v1".into(),
                file: "world.ply".into(),
                sha256: "c".repeat(64),
            },
            depth_frames: Da3PoseConditionedDepthFrames {
                schema: "vestra.da3-pose-conditioned-depth-frames/v1".into(),
                directory: "depth-frames".into(),
                width: 504,
                height: 336,
                frames: Vec::new(),
            },
        };
        assert!(validate_mvs_hybrid_evidence(&sidecar).is_ok());
        let mut guided = sidecar;
        guided.hybrid.as_mut().unwrap().pixel_policy =
            "mvs-zbuffer-plus-coarse-local-ratio/v1".into();
        assert!(validate_mvs_hybrid_evidence(&guided).is_ok());
        let mut patchmatch = guided;
        let evidence = patchmatch.hybrid.as_mut().unwrap();
        evidence.mvs_ply_sha256 = None;
        evidence.mvs_vertices = None;
        evidence.mvs_depth_map_index_sha256 = Some("e".repeat(64));
        evidence.mvs_depth_map_count = Some(1);
        evidence.per_frame = vec![Da3MvsPatchMatchFrame {
            frame_index: 1,
            file: "frame-000002.ppm.geometric.bin".into(),
            sha256: "f".repeat(64),
            mvs_coverage: 0.5,
        }];
        evidence.pixel_policy = "colmap-patchmatch-geometric-resample-else-da3/v1".into();
        assert!(validate_mvs_hybrid_evidence(&patchmatch).is_ok());
        let mut mismatched_frames = patchmatch;
        mismatched_frames.hybrid.as_mut().unwrap().per_frame[0].frame_index = 2;
        assert!(validate_mvs_hybrid_evidence(&mismatched_frames).is_err());
        let mut incomplete = mismatched_frames;
        incomplete.hybrid.as_mut().unwrap().per_frame[0].frame_index = 1;
        incomplete.hybrid.as_mut().unwrap().median_mvs_coverage = 0.0;
        assert!(validate_mvs_hybrid_evidence(&incomplete).is_err());
    }

    #[test]
    fn guided_tsdf_identity_is_separate_and_cannot_be_combined_with_calibration() {
        assert_eq!(
            da3_tsdf_product_identity(false, true),
            Ok((
                "da3-mvs-guided-colmap-surfel",
                "colmap-mvs-geometric-plus-da3-local-guidance",
                "da3-mvs-guided-colmap-tsdf",
            ))
        );
        assert!(da3_tsdf_product_identity(true, true).is_err());
    }

    #[test]
    fn colmap_w2c_camera_centre_preserves_global_translation() {
        assert_eq!(
            colmap_camera_centre([1.0, 0.0, 0.0, -4.0, 0.0, 1.0, 0.0, 2.5, 0.0, 0.0, 1.0, -7.0,]),
            Some([4.0, -2.5, 7.0])
        );
        assert_eq!(colmap_camera_centre([f64::NAN; 12]), None);
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
