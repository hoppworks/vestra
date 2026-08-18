//! Local-only HTTP host for the dependency-free Vestra browser studio.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex, OnceLock},
    thread,
    time::Duration,
};

use vestra_core::{
    CameraCalibration, MeasuredPoint, SceneBundle, SceneManifest, SimilarityTransform,
    WindowMeasuredChunk, camera_centre_direction,
};

const INDEX_HTML: &str = include_str!("index.html");
const INTAKE_HTML: &str = include_str!("intake.html");
const CAMERA_CONTROLS_JS: &str = include_str!("camera-controls.js");
const REPLAY_MAGIC: [u8; 4] = *b"VRPL";
const REPLAY_VERSION: u16 = 2;

/// One measured window is sufficient for sequential replay: a source frame is
/// owned by its earliest window, and the browser requests frames in video
/// order. Keeping only that window prevents a replay from repeatedly parsing
/// a 23 MiB immutable evidence chunk for every animation tick.
#[derive(Debug)]
struct ReplayCache {
    root: PathBuf,
    bundle: SceneBundle,
    frame_hashes: BTreeMap<usize, String>,
    current_hash: String,
    current_window: WindowMeasuredChunk,
}

static REPLAY_CACHE: OnceLock<Mutex<Option<ReplayCache>>> = OnceLock::new();

/// Local process-launch configuration for the browser intake. The server never
/// accepts a destination or executable from HTTP; both are fixed by the CLI
/// that started it.
#[derive(Debug, Clone)]
pub struct IntakeConfig {
    pub executable: PathBuf,
    pub model: PathBuf,
    pub jobs_root: PathBuf,
    pub port: u16,
    pub candidate_fps: f64,
    pub hard_max_frames: usize,
    pub width: usize,
    pub height: usize,
    pub chunk_size: usize,
    pub overlap: usize,
    pub minimum_confidence: f32,
    pub pixel_stride: usize,
    /// Produce an additional normal-space TSDF surfel derivative. Raw
    /// measured observations remain immutable and independently viewable.
    pub tsdf: bool,
    /// Opt-in dense capture preserving the PR #2 emission evidence profile.
    pub cpp_pr2_relative: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum StudioError {
    #[error("could not bind Vestra Studio: {0}")]
    Bind(#[from] std::io::Error),
    #[error("scene manifest is missing at {0}")]
    MissingManifest(PathBuf),
    #[error("Vestra Studio intake configuration is invalid: {0}")]
    IntakeConfig(&'static str),
}

/// Serves one scene only on localhost. This intentionally has no remote bind,
/// authentication surface, upload endpoint, or directory listing.
pub fn serve(scene_root: impl Into<PathBuf>, port: u16) -> Result<(), StudioError> {
    let scene_root = scene_root.into();
    if !scene_root.join("manifest.json").is_file() {
        return Err(StudioError::MissingManifest(
            scene_root.join("manifest.json"),
        ));
    }
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    for stream in listener.incoming().flatten() {
        // A surfel chunk can be much larger than the manifest. Serving it on
        // the accept loop would make one slow WebGL upload freeze the entire
        // Studio, including the replay button and its depth assets.
        let root = scene_root.clone();
        thread::spawn(move || {
            let _ = handle(stream, &root);
        });
    }
    Ok(())
}

/// Serves a localhost-only intake page. A selected video is streamed to a
/// job-owned directory, then the same executable launches `reconstruct`; the
/// browser only polls immutable local job state and links to a second
/// localhost-only Studio viewer after success.
pub fn serve_intake(config: IntakeConfig) -> Result<(), StudioError> {
    if !config.executable.is_file() {
        return Err(StudioError::IntakeConfig(
            "the Vestra executable is missing",
        ));
    }
    if !config.model.is_file() {
        return Err(StudioError::IntakeConfig("the model file is missing"));
    }
    if config.hard_max_frames == 0
        || !config.candidate_fps.is_finite()
        || config.candidate_fps <= 0.0
        || config.width == 0
        || config.height == 0
    {
        return Err(StudioError::IntakeConfig(
            "candidate rate, frame ceiling, and raster dimensions must be positive",
        ));
    }
    fs::create_dir_all(&config.jobs_root)?;
    let next_job = next_job_id(&config.jobs_root)?;
    let active = recover_latest_job(&config)?;
    let state = Arc::new(Mutex::new(IntakeState {
        config,
        next_job,
        active,
    }));
    let port = state.lock().expect("intake state lock").config.port;
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    for stream in listener.incoming().flatten() {
        // A browser may open a speculative connection and never send a full
        // request. Keep it from blocking the local intake server (and video
        // upload/polling) by isolating each connection with a bounded read.
        let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
        let state = Arc::clone(&state);
        thread::spawn(move || {
            let _ = handle_intake(stream, &state);
        });
    }
    Ok(())
}

#[derive(Debug)]
struct IntakeState {
    config: IntakeConfig,
    next_job: u64,
    active: Option<IntakeJob>,
}

#[derive(Debug)]
struct IntakeJob {
    id: u64,
    root: PathBuf,
    video: PathBuf,
    scene: PathBuf,
    log: PathBuf,
    settings: IntakeSettings,
    child: Option<Child>,
    outcome: IntakeOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum IntakeOutcome {
    Running,
    CancelRequested,
    Cancelled,
    Interrupted,
    Complete,
    Failed,
}

/// The reconstruction settings are persisted with each browser job. Resuming
/// never adopts values from a later `vestra app` invocation, because that
/// would violate the scene provenance contract checked by `reconstruct`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct IntakeSettings {
    #[serde(default = "default_candidate_fps")]
    candidate_fps: f64,
    #[serde(default = "default_hard_max_frames")]
    hard_max_frames: usize,
    width: usize,
    height: usize,
    chunk_size: usize,
    overlap: usize,
    minimum_confidence: f32,
    pixel_stride: usize,
    #[serde(default)]
    tsdf: bool,
    #[serde(default)]
    cpp_pr2_relative: bool,
}

const fn default_candidate_fps() -> f64 {
    8.0
}

const fn default_hard_max_frames() -> usize {
    1800
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct IntakeRecord {
    schema: String,
    id: u64,
    video_name: String,
    settings: IntakeSettings,
    outcome: IntakeOutcome,
}

const INTAKE_RECORD_FILE: &str = "job.json";
const INTAKE_RECORD_SCHEMA: &str = "vestra.intake-job/v1";

fn handle_intake(mut stream: TcpStream, state: &Arc<Mutex<IntakeState>>) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request = String::new();
    reader.read_line(&mut request)?;
    let mut fields = request.split_whitespace();
    let method = fields.next().unwrap_or("");
    let path = fields.next().unwrap_or("/");
    let mut content_length = None;
    let mut file_name = None;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if line == "\r\n" || line.is_empty() {
            break;
        }
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        match name.trim().to_ascii_lowercase().as_str() {
            "content-length" => content_length = value.trim().parse::<u64>().ok(),
            "x-vestra-file-name" => file_name = Some(value.trim().to_owned()),
            _ => {}
        }
    }
    if method == "GET" {
        if path == "/input-video" {
            let video = state.lock().ok().and_then(|guard| {
                guard.active.as_ref().and_then(|job| {
                    (job.outcome == IntakeOutcome::Complete && job.video.is_file())
                        .then(|| job.video.clone())
                })
            });
            return match video {
                Some(video) => {
                    let (status, content_type, body) = read_file(video, "video/mp4");
                    write_response(&mut stream, status, content_type, &body)
                }
                None => write_response(
                    &mut stream,
                    "404 Not Found",
                    "text/plain; charset=utf-8",
                    b"completed input video not found",
                ),
            };
        }
        // Route using only the path component, but forward the complete
        // request into Studio so `?product=…` selects the same derivative
        // product in the primary intake `/world/` flow.
        let scene_path = intake_scene_request_path(path);
        let scene = state.lock().ok().and_then(|guard| {
            guard.active.as_ref().and_then(|job| {
                (job.outcome == IntakeOutcome::Complete
                    && job.scene.join("manifest.json").is_file())
                .then(|| job.scene.clone())
            })
        });
        if let Some(scene_path) = scene_path {
            return match scene {
                Some(scene) => handle_scene_path(&mut stream, &scene, &scene_path),
                None => write_response(
                    &mut stream,
                    "404 Not Found",
                    "text/plain; charset=utf-8",
                    b"completed local world not found",
                ),
            };
        }
    }
    match (method, path) {
        ("GET", "/") | ("GET", "/index.html") => write_response(
            &mut stream,
            "200 OK",
            "text/html; charset=utf-8",
            INTAKE_HTML.as_bytes(),
        ),
        ("GET", "/api/job") => {
            let payload = intake_status(state);
            write_response(&mut stream, "200 OK", "application/json", &payload)
        }
        ("POST", "/api/job/cancel") => match cancel_intake_job(state) {
            Ok(payload) => {
                write_response(&mut stream, "202 Accepted", "application/json", &payload)
            }
            Err(message) => write_response(
                &mut stream,
                "409 Conflict",
                "text/plain; charset=utf-8",
                message.as_bytes(),
            ),
        },
        ("POST", "/api/job/resume") => match resume_intake_job(state) {
            Ok(payload) => {
                write_response(&mut stream, "202 Accepted", "application/json", &payload)
            }
            Err(message) => write_response(
                &mut stream,
                "409 Conflict",
                "text/plain; charset=utf-8",
                message.as_bytes(),
            ),
        },
        ("POST", "/api/job") => {
            let Some(length) = content_length else {
                return write_response(
                    &mut stream,
                    "411 Length Required",
                    "text/plain; charset=utf-8",
                    b"content-length required",
                );
            };
            let Some(name) = file_name.and_then(|name| safe_video_name(&name)) else {
                return write_response(
                    &mut stream,
                    "400 Bad Request",
                    "text/plain; charset=utf-8",
                    b"supported video filename required",
                );
            };
            if length == 0 || length > 8 * 1024 * 1024 * 1024 {
                return write_response(
                    &mut stream,
                    "413 Payload Too Large",
                    "text/plain; charset=utf-8",
                    b"video must be between 1 byte and 8 GiB",
                );
            }
            match start_intake_job(state, &mut reader, length, &name) {
                Ok(payload) => {
                    write_response(&mut stream, "202 Accepted", "application/json", &payload)
                }
                Err(message) => write_response(
                    &mut stream,
                    "409 Conflict",
                    "text/plain; charset=utf-8",
                    message.as_bytes(),
                ),
            }
        }
        _ => write_response(
            &mut stream,
            "404 Not Found",
            "text/plain; charset=utf-8",
            b"not found",
        ),
    }
}

fn safe_video_name(raw: &str) -> Option<String> {
    let name = Path::new(raw).file_name()?.to_str()?.to_owned();
    if name != raw {
        return None;
    }
    let extension = Path::new(&name).extension()?.to_str()?.to_ascii_lowercase();
    matches!(extension.as_str(), "mov" | "mp4" | "m4v" | "avi").then_some(name)
}

fn start_intake_job(
    state: &Arc<Mutex<IntakeState>>,
    reader: &mut BufReader<TcpStream>,
    length: u64,
    file_name: &str,
) -> Result<Vec<u8>, String> {
    let (config, id) = {
        let mut guard = state
            .lock()
            .map_err(|_| "intake state is unavailable".to_owned())?;
        if guard.active.as_ref().is_some_and(job_is_in_progress) {
            return Err(
                "this local intake accepts one job; restart it before selecting another video"
                    .into(),
            );
        }
        let id = guard.next_job;
        guard.next_job = guard.next_job.checked_add(1).ok_or("job id overflow")?;
        (guard.config.clone(), id)
    };
    let root = config.jobs_root.join(format!("job-{id:06}"));
    fs::create_dir_all(&root)
        .map_err(|error| format!("could not create job directory: {error}"))?;
    let video = root.join(file_name);
    let mut output = fs::File::create(&video)
        .map_err(|error| format!("could not create local video: {error}"))?;
    let copied = std::io::copy(&mut reader.take(length), &mut output)
        .map_err(|error| format!("could not write local video: {error}"))?;
    if copied != length {
        let _ = fs::remove_file(&video);
        return Err("upload ended before its declared content length".into());
    }
    let settings = IntakeSettings::from(&config);
    let scene = root.join("world.vestra");
    let log = root.join("reconstruct.log");
    let mut job = IntakeJob {
        id,
        root,
        video,
        scene,
        log,
        settings,
        outcome: IntakeOutcome::Running,
        child: None,
    };
    write_intake_record(&job).map_err(|error| format!("could not persist job state: {error}"))?;
    job.child = Some(spawn_reconstruction(&config, &job, false)?);
    state
        .lock()
        .map_err(|_| "intake state is unavailable".to_owned())?
        .active = Some(job);
    Ok(
        serde_json::to_vec(&serde_json::json!({"job": id, "state": "running"}))
            .expect("fixed intake response serializes"),
    )
}

fn job_is_in_progress(job: &IntakeJob) -> bool {
    matches!(
        job.outcome,
        IntakeOutcome::Running | IntakeOutcome::CancelRequested
    )
}

impl From<&IntakeConfig> for IntakeSettings {
    fn from(config: &IntakeConfig) -> Self {
        Self {
            candidate_fps: config.candidate_fps,
            hard_max_frames: config.hard_max_frames,
            width: config.width,
            height: config.height,
            chunk_size: config.chunk_size,
            overlap: config.overlap,
            minimum_confidence: config.minimum_confidence,
            pixel_stride: config.pixel_stride,
            tsdf: config.tsdf,
            cpp_pr2_relative: config.cpp_pr2_relative,
        }
    }
}

fn record_path(root: &Path) -> PathBuf {
    root.join(INTAKE_RECORD_FILE)
}

fn write_intake_record(job: &IntakeJob) -> std::io::Result<()> {
    let record = IntakeRecord {
        schema: INTAKE_RECORD_SCHEMA.into(),
        id: job.id,
        video_name: job
            .video
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .into(),
        settings: job.settings.clone(),
        outcome: job.outcome,
    };
    let path = record_path(&job.root);
    let temporary = job.root.join(format!(".{INTAKE_RECORD_FILE}.tmp"));
    fs::write(&temporary, serde_json::to_vec_pretty(&record)?)?;
    fs::rename(temporary, path)
}

fn next_job_id(jobs_root: &Path) -> std::io::Result<u64> {
    let mut maximum = 0_u64;
    for entry in fs::read_dir(jobs_root)? {
        let entry = entry?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let Some(id) = name
            .strip_prefix("job-")
            .and_then(|value| value.parse::<u64>().ok())
        else {
            continue;
        };
        maximum = maximum.max(id);
    }
    maximum
        .checked_add(1)
        .ok_or_else(|| std::io::Error::other("job id overflow"))
}

fn recover_latest_job(config: &IntakeConfig) -> std::io::Result<Option<IntakeJob>> {
    let mut latest = None::<(u64, PathBuf, IntakeRecord)>;
    for entry in fs::read_dir(&config.jobs_root)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let root = entry.path();
        let Ok(payload) = fs::read(record_path(&root)) else {
            continue;
        };
        let Ok(record) = serde_json::from_slice::<IntakeRecord>(&payload) else {
            continue;
        };
        if record.schema != INTAKE_RECORD_SCHEMA || safe_video_name(&record.video_name).is_none() {
            continue;
        }
        if latest.as_ref().is_none_or(|(id, _, _)| record.id > *id) {
            latest = Some((record.id, root, record));
        }
    }
    let Some((id, root, record)) = latest else {
        return Ok(None);
    };
    let mut job = IntakeJob {
        id,
        video: root.join(&record.video_name),
        scene: root.join("world.vestra"),
        log: root.join("reconstruct.log"),
        root,
        settings: record.settings,
        child: None,
        outcome: record.outcome,
    };
    if job.outcome == IntakeOutcome::Running || job.outcome == IntakeOutcome::CancelRequested {
        job.outcome = IntakeOutcome::Interrupted;
        write_intake_record(&job)?;
    }
    Ok(Some(job))
}

fn spawn_reconstruction(
    config: &IntakeConfig,
    job: &IntakeJob,
    resume: bool,
) -> Result<Child, String> {
    let log_file = fs::OpenOptions::new()
        .create(true)
        .append(resume)
        .write(true)
        .truncate(!resume)
        .open(&job.log)
        .map_err(|error| format!("could not create job log: {error}"))?;
    let mut command = Command::new(&config.executable);
    command
        .args([
            "reconstruct",
            "--video",
            job.video.to_string_lossy().as_ref(),
            "--model",
            config.model.to_string_lossy().as_ref(),
            "--output",
            job.scene.to_string_lossy().as_ref(),
            "--candidate-fps",
            &job.settings.candidate_fps.to_string(),
            "--hard-max-frames",
            &job.settings.hard_max_frames.to_string(),
            "--width",
            &job.settings.width.to_string(),
            "--height",
            &job.settings.height.to_string(),
            "--chunk-size",
            &job.settings.chunk_size.to_string(),
            "--overlap",
            &job.settings.overlap.to_string(),
            "--minimum-confidence",
            &job.settings.minimum_confidence.to_string(),
            "--pixel-stride",
            &job.settings.pixel_stride.to_string(),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::from(
            log_file
                .try_clone()
                .map_err(|error| format!("could not clone job log: {error}"))?,
        ))
        .stderr(Stdio::from(log_file));
    if job.settings.cpp_pr2_relative {
        command.arg("--cpp-pr2-relative");
    }
    if job.settings.tsdf {
        command.arg("--tsdf");
    }
    if resume {
        command.arg("--resume");
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    command
        .spawn()
        .map_err(|error| format!("could not start reconstruction: {error}"))
}

fn cancel_intake_job(state: &Arc<Mutex<IntakeState>>) -> Result<Vec<u8>, String> {
    let mut guard = state
        .lock()
        .map_err(|_| "intake state is unavailable".to_owned())?;
    let Some(job) = guard.active.as_mut() else {
        return Err("there is no active job to cancel".into());
    };
    if job.outcome != IntakeOutcome::Running {
        return Err("only a running job can be canceled".into());
    }
    let child = job
        .child
        .as_mut()
        .ok_or_else(|| "the running job process is unavailable".to_owned())?;
    request_interrupt(child)
        .map_err(|error| format!("could not interrupt reconstruction: {error}"))?;
    job.outcome = IntakeOutcome::CancelRequested;
    write_intake_record(job).map_err(|error| format!("could not persist cancellation: {error}"))?;
    serde_json::to_vec(&serde_json::json!({"job": job.id, "state": "cancel_requested"}))
        .map_err(|error| error.to_string())
}

fn resume_intake_job(state: &Arc<Mutex<IntakeState>>) -> Result<Vec<u8>, String> {
    let mut guard = state
        .lock()
        .map_err(|_| "intake state is unavailable".to_owned())?;
    let config = guard.config.clone();
    let Some(job) = guard.active.as_mut() else {
        return Err("there is no interrupted job to resume".into());
    };
    if !matches!(
        job.outcome,
        IntakeOutcome::Cancelled | IntakeOutcome::Interrupted | IntakeOutcome::Failed
    ) {
        return Err("only a canceled, interrupted, or failed job can be resumed".into());
    }
    if !job.video.is_file() {
        return Err("the persisted video for this job is missing".into());
    }
    job.child = Some(spawn_reconstruction(&config, job, true)?);
    job.outcome = IntakeOutcome::Running;
    write_intake_record(job).map_err(|error| format!("could not persist resumed job: {error}"))?;
    serde_json::to_vec(&serde_json::json!({"job": job.id, "state": "running", "resumed": true}))
        .map_err(|error| error.to_string())
}

#[cfg(unix)]
fn request_interrupt(child: &mut Child) -> std::io::Result<()> {
    unsafe extern "C" {
        fn kill(pid: i32, signal: i32) -> i32;
    }
    const SIGINT: i32 = 2;
    let pid = i32::try_from(child.id()).map_err(|_| std::io::Error::other("child pid overflow"))?;
    // The child starts a process group, so FFmpeg descendants receive the same
    // graceful interrupt as the reconstruction process.
    if unsafe { kill(-pid, SIGINT) } == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(not(unix))]
fn request_interrupt(child: &mut Child) -> std::io::Result<()> {
    child.kill()
}

fn intake_status(state: &Arc<Mutex<IntakeState>>) -> Vec<u8> {
    let mut guard = match state.lock() {
        Ok(guard) => guard,
        Err(_) => return br#"{"state":"unavailable"}"#.to_vec(),
    };
    let Some(job) = guard.active.as_mut() else {
        return br#"{"state":"idle"}"#.to_vec();
    };
    if matches!(
        job.outcome,
        IntakeOutcome::Running | IntakeOutcome::CancelRequested
    ) {
        let Some(child) = job.child.as_mut() else {
            job.outcome = IntakeOutcome::Interrupted;
            let _ = write_intake_record(job);
            return intake_payload(job);
        };
        match child.try_wait() {
            Ok(Some(status)) if status.success() => {
                job.child = None;
                job.outcome = if job.outcome == IntakeOutcome::CancelRequested {
                    IntakeOutcome::Cancelled
                } else {
                    IntakeOutcome::Complete
                };
            }
            Ok(Some(_)) | Err(_) => {
                job.child = None;
                job.outcome = if job.outcome == IntakeOutcome::CancelRequested {
                    IntakeOutcome::Cancelled
                } else {
                    IntakeOutcome::Failed
                };
            }
            Ok(None) => {}
        }
        let _ = write_intake_record(job);
    }
    intake_payload(job)
}

fn intake_payload(job: &IntakeJob) -> Vec<u8> {
    let log_tail = fs::read_to_string(&job.log).ok().map(|log| {
        log.chars()
            .rev()
            .take(2_000)
            .collect::<String>()
            .chars()
            .rev()
            .collect::<String>()
    });
    let state = match job.outcome {
        IntakeOutcome::Running => "running",
        IntakeOutcome::CancelRequested => "cancel_requested",
        IntakeOutcome::Cancelled => "cancelled",
        IntakeOutcome::Interrupted => "interrupted",
        IntakeOutcome::Complete => "complete",
        IntakeOutcome::Failed => "failed",
    };
    serde_json::to_vec(&serde_json::json!({
        "job": job.id,
        "state": state,
        "viewer": if job.outcome == IntakeOutcome::Complete && job.scene.join("manifest.json").is_file() { Some("/world/") } else { None },
        "can_cancel": job.outcome == IntakeOutcome::Running,
        "can_resume": matches!(job.outcome, IntakeOutcome::Cancelled | IntakeOutcome::Interrupted | IntakeOutcome::Failed),
        "log_tail": log_tail,
    }))
    .expect("fixed intake status serializes")
}

/// The Studio bundle uses root-relative asset paths. Keep those assets on the
/// same localhost origin as the intake page so a single SSH tunnel can show a
/// completed world without exposing or guessing a second port.
fn intake_world_path(path: &str) -> Option<&str> {
    if path == "/world" || path.starts_with("/world/") {
        let remainder = path.strip_prefix("/world").expect("checked world prefix");
        return Some(match remainder {
            "" | "/" => "/",
            _ => remainder,
        });
    }
    matches!(
        path,
        "/manifest.json" | "/evidence.json" | "/camera-controls.js" | "/input-video"
    )
    .then_some(path)
    .or_else(|| {
        (path.starts_with("/chunks/")
            || path.starts_with("/sources/")
            || path.starts_with("/replay/frames/")
            || path.starts_with("/depth/frames/"))
        .then_some(path)
    })
}

/// Normalizes an intake `/world/` asset path while retaining its query for
/// Studio-level product selection. The routing allowlist deliberately sees
/// only the path component.
fn intake_scene_request_path(request_path: &str) -> Option<String> {
    let (route_path, query) = request_path
        .split_once('?')
        .map_or((request_path, ""), |(route_path, query)| {
            (route_path, query)
        });
    intake_world_path(route_path).map(|scene_path| {
        if query.is_empty() {
            scene_path.to_owned()
        } else {
            format!("{scene_path}?{query}")
        }
    })
}

fn write_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)
}

fn handle(mut stream: TcpStream, root: &Path) -> std::io::Result<()> {
    let mut request = String::new();
    let mut reader = BufReader::new(stream.try_clone()?);
    reader.read_line(&mut request)?;
    let path = request.split_whitespace().nth(1).unwrap_or("/");
    handle_scene_path(&mut stream, root, path)
}

fn handle_scene_path(
    stream: &mut TcpStream,
    root: &Path,
    requested_path: &str,
) -> std::io::Result<()> {
    let (path, query) = requested_path
        .split_once('?')
        .map_or((requested_path, ""), |(path, query)| (path, query));
    let product = query_product_id(query);
    let (status, content_type, body) = match path {
        "/" | "/index.html" => (
            "200 OK",
            "text/html; charset=utf-8",
            INDEX_HTML.as_bytes().to_vec(),
        ),
        "/camera-controls.js" => (
            "200 OK",
            "text/javascript; charset=utf-8",
            CAMERA_CONTROLS_JS.as_bytes().to_vec(),
        ),
        "/manifest.json" => selected_manifest(root, product),
        "/evidence.json" => evidence(root, product),
        "/input-video" => input_video(root),
        _ if path.starts_with("/replay/frames/") && path.ends_with(".bin") => {
            replay_frame(root, path)
        }
        _ if path.starts_with("/depth/frames/") && path.ends_with(".bmp") => {
            product_depth_frame(root, product, path)
        }
        _ if path.starts_with("/chunks/") && path.ends_with(".json") && safe_chunk_path(path) => {
            read_file(root.join(&path[1..]), "application/json")
        }
        _ if path.starts_with("/chunks/") && path.ends_with(".bin") && safe_chunk_path(path) => {
            read_file(root.join(&path[1..]), "application/octet-stream")
        }
        _ if path.starts_with("/sources/") && path.ends_with(".bmp") => {
            source_thumbnail(root, path)
        }
        _ => (
            "404 Not Found",
            "text/plain; charset=utf-8",
            b"not found".to_vec(),
        ),
    };
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(&body)
}

/// Serves the immutable source capture retained alongside a completed world.
/// The handler accepts no path from the browser: only the scene-local copy is
/// eligible, so a Studio URL cannot become a local file reader.
fn input_video(root: &Path) -> (&'static str, &'static str, Vec<u8>) {
    for (name, mime) in [
        ("input-video.mp4", "video/mp4"),
        ("input-video.mov", "video/quicktime"),
    ] {
        let path = root.join(name);
        if path.is_file() {
            return read_file(path, mime);
        }
    }
    not_found()
}

/// Product IDs are generated by Vestra and intentionally constrained to a
/// small URL-safe alphabet.  Invalid values are ignored rather than becoming
/// a filesystem path or an implicit selection.
fn query_product_id(query: &str) -> Option<&str> {
    let id = query
        .split('&')
        .find_map(|part| part.strip_prefix("product="))?;
    (!id.is_empty()
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-'))
    .then_some(id)
}

fn selected_manifest(
    root: &Path,
    product_id: Option<&str>,
) -> (&'static str, &'static str, Vec<u8>) {
    let payload = (|| -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let bundle = SceneBundle::open(root)?;
        let manifest = manifest_for_product(bundle.manifest()?, product_id)?;
        Ok(serde_json::to_vec(&manifest)?)
    })();
    match payload {
        Ok(body) => ("200 OK", "application/json", body),
        Err(_) => (
            "404 Not Found",
            "text/plain; charset=utf-8",
            b"requested world product unavailable".to_vec(),
        ),
    }
}

fn manifest_for_product(
    mut manifest: SceneManifest,
    product_id: Option<&str>,
) -> Result<SceneManifest, Box<dyn std::error::Error>> {
    // Studio has one intentional 3D-world view, not a hidden model picker.
    // Prefer the dense geometric MVS surfel product whenever it is present;
    // architectural-plane derivatives remain export/evidence artifacts and
    // must not replace the user's coloured capture-derived world.
    let preferred = manifest
        .world_products
        .iter()
        .find(|product| product.id == "colmap-mvs-geometric")
        .or_else(|| {
            manifest
                .world_products
                .iter()
                .find(|product| product.surface_mode == "surfel")
        })
        .map(|product| product.id.as_str());
    let id = product_id
        .or(preferred)
        .or(manifest.selected_world_product.as_deref());
    let Some(id) = id else {
        return Ok(manifest);
    };
    let product = manifest
        .world_products
        .iter()
        .find(|product| product.id == id)
        .cloned()
        .ok_or_else(|| format!("unknown world product {id:?}"))?;
    manifest.fused_chunk_hash = Some(product.fused_chunk_hash);
    // World products are served through their compact binary surfel assets.
    // Never leak legacy JSON chunk aliases from the previously selected
    // product into the returned manifest.
    manifest.fused_point_chunk_hashes.clear();
    manifest.fused_point_binary_chunk_hashes = product.point_binary_chunk_hashes;
    manifest.fused_preview_point_binary_chunk_hashes = product.preview_point_binary_chunk_hashes;
    manifest.fused_summary = Some(product.summary);
    manifest.selected_world_product = Some(product.id);
    Ok(manifest)
}

/// Emits only compact diagnostic evidence for Studio. The raw camera W2C
/// matrices stay in the immutable chunks; this endpoint derives camera rays
/// in the fused relative frame, never inventing metric coordinates.
fn evidence(root: &Path, product_id: Option<&str>) -> (&'static str, &'static str, Vec<u8>) {
    let payload = (|| -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let bundle = SceneBundle::open(root)?;
        let manifest = manifest_for_product(bundle.manifest()?, product_id)?;
        let selected_product = manifest.selected_world_product.as_deref().and_then(|id| {
            manifest
                .world_products
                .iter()
                .find(|product| product.id == id)
        });
        // Independent products can carry a calibrated pose solution. Resolve
        // their compact source-camera proof without scanning local DA3
        // windows, which are neither necessary nor valid evidence for them.
        if let Some(product) = selected_product.filter(|product| {
            product.pose_authority != "local-pr2-relative" && product.pose_solution_hash.is_some()
        }) {
            return frame_global_evidence(root, &bundle, product);
        }
        // A dense MVS control cloud is deliberately independent from Vestra's
        // measured windows. Reading the latter is both incorrect evidence and
        // can traverse many gigabytes only to produce no camera rays.
        if selected_product.is_some_and(|product| product.pose_authority != "local-pr2-relative") {
            return Ok(serde_json::to_vec(&serde_json::json!({
                "scale": "provider-defined",
                "camera_rays": [],
                "source_frames": [],
                "diagnostic_links": [],
            }))?);
        }
        let Some(hash) = manifest.fused_chunk_hash else {
            return Ok(serde_json::to_vec(&serde_json::json!({"camera_rays": []}))?);
        };
        let fused = bundle.read_fused_scene(&hash)?;
        let mut camera_rays = Vec::new();
        let mut source_frames = BTreeSet::new();
        let mut window_anchors = BTreeMap::new();
        for measured_hash in manifest.measured_chunk_hashes {
            let window = bundle.read_measured_window(&measured_hash)?;
            let Some(pose) = fused
                .window_poses
                .iter()
                .find(|pose| pose.window_index == window.window.index)
            else {
                continue;
            };
            for view in window.views {
                if source_frame_path(root, view.frame_index).is_file() {
                    source_frames.insert(view.frame_index);
                }
                let Some(camera) = camera_centre_direction(view.frame_index, view.camera) else {
                    continue;
                };
                let direction = rotate(pose.local_to_world, camera.forward_local);
                if !direction.iter().all(|value| value.is_finite()) {
                    continue;
                }
                let origin = pose.local_to_world.apply(camera.centre_local);
                if origin.iter().all(|value| value.is_finite()) {
                    window_anchors.entry(window.window.index).or_insert(origin);
                }
                let corners = camera_frustum_directions(view.camera)
                    .map(|directions| {
                        directions.map(|direction| rotate(pose.local_to_world, direction))
                    })
                    .filter(|directions| {
                        directions.iter().flatten().all(|value| value.is_finite())
                    });
                camera_rays.push(serde_json::json!({
                    "window_index": window.window.index,
                    "frame_index": view.frame_index,
                    "origin": origin,
                    "forward": direction,
                    "corners": corners,
                }));
            }
        }
        let seam_links = diagnostic_links(
            &fused.window_poses,
            &fused.alignments,
            &fused.pose_graph_edges,
            &window_anchors,
        );
        Ok(serde_json::to_vec(&serde_json::json!({
            "scale": "relative",
            "camera_rays": camera_rays,
            "source_frames": source_frames.into_iter().collect::<Vec<_>>(),
            "diagnostic_links": seam_links,
        }))?)
    })();
    match payload {
        Ok(body) => ("200 OK", "application/json", body),
        Err(_) => (
            "404 Not Found",
            "text/plain; charset=utf-8",
            b"scene evidence unavailable".to_vec(),
        ),
    }
}

fn frame_global_evidence(
    root: &Path,
    bundle: &SceneBundle,
    product: &vestra_core::WorldProduct,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let pose_hash = product
        .pose_solution_hash
        .as_deref()
        .ok_or("frame-global product has no pose solution")?;
    let solution = bundle.read_pose_solution(pose_hash)?;
    let trajectory = solution
        .global_trajectory
        .as_ref()
        .ok_or("frame-global pose solution has no calibrated trajectory")?;
    let accepted = product
        .source_frame_indices
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut camera_rays = Vec::new();
    let mut source_frames = Vec::new();
    for frame in solution
        .frames
        .iter()
        .filter(|frame| frame.registered && accepted.contains(&frame.frame_index))
    {
        let Some(camera_id) = trajectory.frame_camera_ids.get(&frame.frame_index) else {
            continue;
        };
        let Some(camera) = trajectory
            .camera_models
            .iter()
            .find(|camera| camera.camera_id == *camera_id)
        else {
            continue;
        };
        let Some((origin, forward, corners, right, up, vertical_fov_radians, aspect_ratio)) =
            colmap_camera_ray(frame.world_to_camera, camera)
        else {
            continue;
        };
        if source_frame_path(root, frame.frame_index).is_file() {
            source_frames.push(frame.frame_index);
        }
        camera_rays.push(serde_json::json!({
            "window_index": null,
            "frame_index": frame.frame_index,
            "origin": origin,
            "forward": forward,
            "corners": corners,
            "right": right,
            "up": up,
            // This is not cosmetic metadata: Studio uses it only for the
            // explicit "match 3D camera" mode, where the projection must
            // agree with the calibrated source raster rather than the browser
            // window's aspect ratio.
            "vertical_fov_radians": vertical_fov_radians,
            "aspect_ratio": aspect_ratio,
        }));
    }
    Ok(serde_json::to_vec(&serde_json::json!({
        "scale": "relative",
        "camera_rays": camera_rays,
        "source_frames": source_frames,
        "diagnostic_links": [],
    }))?)
}

fn colmap_camera_ray(
    pose: [f64; 12],
    camera: &vestra_core::ColmapCameraModel,
) -> Option<(
    [f32; 3],
    [f32; 3],
    [[f32; 3]; 4],
    [f32; 3],
    [f32; 3],
    f32,
    f32,
)> {
    let [focal, cx, cy, radial] = *<&[f64; 4]>::try_from(camera.parameters.as_slice()).ok()?;
    if !(focal.is_finite() && focal > 0.0 && cx.is_finite() && cy.is_finite() && radial.is_finite())
    {
        return None;
    }
    let centre = |camera_point: [f64; 3]| {
        let shifted = [
            camera_point[0] - pose[3],
            camera_point[1] - pose[7],
            camera_point[2] - pose[11],
        ];
        let world = [
            pose[0] * shifted[0] + pose[4] * shifted[1] + pose[8] * shifted[2],
            pose[1] * shifted[0] + pose[5] * shifted[1] + pose[9] * shifted[2],
            pose[2] * shifted[0] + pose[6] * shifted[1] + pose[10] * shifted[2],
        ];
        world
            .iter()
            .all(|value| value.is_finite())
            .then_some(world.map(|value| value as f32))
    };
    let camera_axis = |axis: [f64; 3]| {
        let world = [
            pose[0] * axis[0] + pose[4] * axis[1] + pose[8] * axis[2],
            pose[1] * axis[0] + pose[5] * axis[1] + pose[9] * axis[2],
            pose[2] * axis[0] + pose[6] * axis[1] + pose[10] * axis[2],
        ];
        let length = world.iter().map(|value| value * value).sum::<f64>().sqrt();
        (length.is_finite() && length > 1e-9).then_some(world.map(|value| (value / length) as f32))
    };
    let direction = |image: [f64; 2]| {
        let (xd, yd) = ((image[0] - cx) / focal, (image[1] - cy) / focal);
        let (mut x, mut y) = (xd, yd);
        for _ in 0..8 {
            let scale = 1.0 + radial * (x * x + y * y);
            if !scale.is_finite() || scale.abs() < 1e-8 {
                return None;
            }
            x = xd / scale;
            y = yd / scale;
        }
        camera_axis([x, y, 1.0])
    };
    let origin = centre([0.0, 0.0, 0.0])?;
    let forward = camera_axis([0.0, 0.0, 1.0])?;
    let right = camera_axis([1.0, 0.0, 0.0])?;
    // COLMAP camera +Y points down in the source raster. Studio needs the
    // image's visual-up direction, not an arbitrary global up vector.
    let up = camera_axis([0.0, -1.0, 0.0])?;
    let corners = [
        direction([0.0, 0.0])?,
        direction([camera.width as f64 - 1.0, 0.0])?,
        direction([camera.width as f64 - 1.0, camera.height as f64 - 1.0])?,
        direction([0.0, camera.height as f64 - 1.0])?,
    ];
    let vertical_fov_radians = 2.0 * ((camera.height as f64) / (2.0 * focal)).atan();
    let aspect_ratio = camera.width as f64 / camera.height as f64;
    (vertical_fov_radians.is_finite()
        && vertical_fov_radians > 0.0
        && aspect_ratio.is_finite()
        && aspect_ratio > 0.0)
        .then_some((
            origin,
            forward,
            corners,
            right,
            up,
            vertical_fov_radians as f32,
            aspect_ratio as f32,
        ))
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
struct DiagnosticLink {
    from: [f32; 3],
    to: [f32; 3],
    kind: &'static str,
}

/// Converts persisted alignment provenance into visible links. Sequential
/// `AlignmentReport`s are ordered by adjacent windows. Explicit loop edges
/// are retained by new fused bundles; legacy bundles still show their seams.
fn diagnostic_links(
    poses: &[vestra_core::FusedWindowPose],
    alignments: &[vestra_core::AlignmentReport],
    pose_graph_edges: &[vestra_core::PoseGraphEdge],
    anchors: &BTreeMap<usize, [f32; 3]>,
) -> Vec<DiagnosticLink> {
    let anchor_for_node = |node: usize| {
        poses
            .get(node)
            .and_then(|pose| anchors.get(&pose.window_index))
            .copied()
    };
    let mut links = alignments
        .iter()
        .enumerate()
        .filter_map(|(index, _)| {
            Some(DiagnosticLink {
                from: anchor_for_node(index)?,
                to: anchor_for_node(index + 1)?,
                kind: "seam",
            })
        })
        .collect::<Vec<_>>();
    links.extend(
        pose_graph_edges
            .iter()
            .filter(|edge| edge.loop_closure)
            .filter_map(|edge| {
                Some(DiagnosticLink {
                    from: anchor_for_node(edge.from)?,
                    to: anchor_for_node(edge.to)?,
                    kind: "loop",
                })
            }),
    );
    links
}

/// Four normalized corner directions for a diagnostic image-plane frustum.
/// `CameraCalibration` is W2C, so camera-space rays become local-world rays
/// by multiplication with `Rᵀ`. The visualized frustum length is selected by
/// Studio from the active relative-world extent, not from a metric claim.
fn camera_frustum_directions(calibration: CameraCalibration) -> Option<[[f32; 3]; 4]> {
    let matrix = calibration.world_to_camera;
    let intrinsics = calibration.intrinsics;
    let fx = intrinsics[0];
    let fy = intrinsics[4];
    let cx = intrinsics[2];
    let cy = intrinsics[5];
    if !matrix.iter().all(|value| value.is_finite())
        || !intrinsics.iter().all(|value| value.is_finite())
        || fx <= 0.0
        || fy <= 0.0
        || cx < 0.0
        || cy < 0.0
    {
        return None;
    }
    let corners = [
        [0.0, 0.0],
        [2.0 * cx, 0.0],
        [2.0 * cx, 2.0 * cy],
        [0.0, 2.0 * cy],
    ];
    Some(corners.map(|[u, v]| {
        normalize_direction([
            matrix[0] * ((u - cx) / fx) + matrix[4] * ((v - cy) / fy) + matrix[8],
            matrix[1] * ((u - cx) / fx) + matrix[5] * ((v - cy) / fy) + matrix[9],
            matrix[2] * ((u - cx) / fx) + matrix[6] * ((v - cy) / fy) + matrix[10],
        ])
    }))
}

fn normalize_direction(direction: [f32; 3]) -> [f32; 3] {
    let length = direction
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    [
        direction[0] / length,
        direction[1] / length,
        direction[2] / length,
    ]
}

/// Converts one decoded RGB24 source frame into a browser-readable BMP only
/// when the local Studio asks for an integer frame index. The decode cache is
/// already part of a local reconstruction bundle; this endpoint neither lists
/// files nor accepts arbitrary paths.
fn source_thumbnail(root: &Path, request_path: &str) -> (&'static str, &'static str, Vec<u8>) {
    let Some(index) = source_frame_index(request_path) else {
        return not_found();
    };
    match ppm_to_bmp(&source_frame_path(root, index)) {
        Ok(body) => ("200 OK", "image/bmp", body),
        Err(_) => not_found(),
    }
}

fn source_frame_index(request_path: &str) -> Option<usize> {
    request_path
        .strip_prefix("/sources/")?
        .strip_suffix(".bmp")?
        .parse::<usize>()
        .ok()
}

fn source_frame_path(root: &Path, frame_index: usize) -> PathBuf {
    root.join("decoded").join(format!(
        "frame-{:06}.ppm",
        frame_index.checked_add(1).unwrap_or(usize::MAX)
    ))
}

/// Returns the real, sparse depth samples for one reconstructed source frame.
/// These are not synthesized from the input video: every returned pixel/color
/// pair originates from `MeasuredPoint` evidence emitted by the engine.
fn replay_frame(root: &Path, request_path: &str) -> (&'static str, &'static str, Vec<u8>) {
    let Some(frame_index) = replay_frame_index(request_path) else {
        return not_found();
    };
    let payload = (|| -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let cache = REPLAY_CACHE.get_or_init(|| Mutex::new(None));
        let mut cache = cache.lock().map_err(|_| "replay cache unavailable")?;
        if cache.as_ref().is_none_or(|entry| entry.root != root) {
            let bundle = SceneBundle::open(root)?;
            let manifest = bundle.manifest()?;
            let mut owners = BTreeMap::new();
            let mut first = None;
            for hash in &manifest.measured_chunk_hashes {
                let window = bundle.read_measured_window(hash)?;
                for view in &window.views {
                    // Chunks are content-addressed and therefore hash-sorted,
                    // not capture-ordered. Select the lowest window index for
                    // an overlap frame so replay exactly follows first-owner
                    // reconstruction semantics.
                    match owners.entry(view.frame_index) {
                        std::collections::btree_map::Entry::Vacant(entry) => {
                            entry.insert((window.window.index, hash.clone()));
                        }
                        std::collections::btree_map::Entry::Occupied(mut entry)
                            if window.window.index < entry.get().0 =>
                        {
                            entry.insert((window.window.index, hash.clone()));
                        }
                        std::collections::btree_map::Entry::Occupied(_) => {}
                    }
                }
                match &first {
                    Some((index, _, _)) if *index <= window.window.index => {}
                    _ => first = Some((window.window.index, hash.clone(), window)),
                }
            }
            let (_, first_hash, first_window) = first.ok_or("scene has no measured evidence")?;
            let frame_hashes = owners
                .into_iter()
                .map(|(frame, (_, hash))| (frame, hash))
                .collect();
            *cache = Some(ReplayCache {
                root: root.to_path_buf(),
                bundle,
                frame_hashes,
                current_hash: first_hash,
                current_window: first_window,
            });
        }
        let entry = cache.as_mut().expect("replay cache initialized");
        let hash = entry
            .frame_hashes
            .get(&frame_index)
            .ok_or("source frame is outside the reconstruction")?;
        if entry.current_hash != *hash {
            entry.current_window = entry.bundle.read_measured_window(hash)?;
            entry.current_hash = hash.clone();
        }
        let view = entry
            .current_window
            .views
            .iter()
            .find(|view| view.frame_index == frame_index)
            .ok_or("source frame has no measured depth samples")?;
        Ok(encode_replay_points(&view.points, view.camera))
    })();
    match payload {
        Ok(body) => ("200 OK", "application/octet-stream", body),
        Err(_) => not_found(),
    }
}

fn replay_frame_index(request_path: &str) -> Option<usize> {
    request_path
        .strip_prefix("/replay/frames/")?
        .strip_suffix(".bin")?
        .parse::<usize>()
        .ok()
}

/// Serves a display-only, colourized DA3 depth raster retained with an
/// independent world product. The browser never synthesizes this image from
/// surfels, so it remains a truthful per-camera depth result.
fn product_depth_frame(
    root: &Path,
    product_id: Option<&str>,
    request_path: &str,
) -> (&'static str, &'static str, Vec<u8>) {
    let Some(frame_index) = request_path
        .strip_prefix("/depth/frames/")
        .and_then(|value| value.strip_suffix(".bmp"))
        .and_then(|value| value.parse::<usize>().ok())
    else {
        return not_found();
    };
    let result = (|| -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let bundle = SceneBundle::open(root)?;
        let manifest = bundle.manifest()?;
        let selected = product_id.or(manifest.selected_world_product.as_deref());
        let product = manifest
            .world_products
            .iter()
            .find(|candidate| Some(candidate.id.as_str()) == selected)
            .ok_or("depth request has no selected world product")?;
        if product.depth_frame_count == 0
            || !product
                .source_frame_indices
                .binary_search(&frame_index)
                .is_ok()
        {
            return Err("selected product has no retained depth frame".into());
        }
        ppm_to_bmp(
            &root
                .join("depth")
                .join(&product.id)
                .join(format!("frame-{frame_index:06}.ppm")),
        )
    })();
    match result {
        Ok(body) => ("200 OK", "image/bmp", body),
        Err(_) => not_found(),
    }
}

/// `VRPL` v2 is a camera-space point-cloud payload, not an image raster.
/// It contains magic/version/reserved/count, `fx/fy/cx/cy`, then
/// `[camera_x:f32, camera_y:f32, camera_z:f32, r:u8, g:u8, b:u8]` per real
/// measured point. The Studio deliberately renders it with a small camera
/// offset so depth changes are visible instead of reconstructing the input
/// image as a regular dot grid.
fn encode_replay_points(points: &[MeasuredPoint], camera: CameraCalibration) -> Vec<u8> {
    let samples = points
        .iter()
        .filter_map(|point| {
            let [r00, r01, r02, tx, r10, r11, r12, ty, r20, r21, r22, tz] = camera.world_to_camera;
            let [x, y, z] = point.position;
            let camera_position = [
                r00 * x + r01 * y + r02 * z + tx,
                r10 * x + r11 * y + r12 * z + ty,
                r20 * x + r21 * y + r22 * z + tz,
            ];
            (camera_position.iter().all(|value| value.is_finite()) && camera_position[2] > 0.0)
                .then_some((camera_position, point.color_srgb))
        })
        .collect::<Vec<_>>();
    let count = u32::try_from(samples.len()).expect("bounded measured point payload");
    let mut payload = Vec::with_capacity(28 + samples.len() * 15);
    payload.extend_from_slice(&REPLAY_MAGIC);
    payload.extend_from_slice(&REPLAY_VERSION.to_le_bytes());
    payload.extend_from_slice(&0_u16.to_le_bytes());
    payload.extend_from_slice(&count.to_le_bytes());
    for value in [
        camera.intrinsics[0],
        camera.intrinsics[4],
        camera.intrinsics[2],
        camera.intrinsics[5],
    ] {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    for (position, color) in samples {
        for value in position {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        payload.extend_from_slice(&color);
    }
    payload
}

fn ppm_to_bmp(path: &Path) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let payload = fs::read(path)?;
    let mut offset = 0;
    if ppm_token(&payload, &mut offset).as_deref() != Some(b"P6".as_slice()) {
        return Err("source frame is not binary RGB PPM".into());
    }
    let width = ppm_usize(&payload, &mut offset, "width")?;
    let height = ppm_usize(&payload, &mut offset, "height")?;
    if ppm_usize(&payload, &mut offset, "maximum component")? != 255 {
        return Err("source PPM must be RGB24".into());
    }
    if !payload.get(offset).is_some_and(u8::is_ascii_whitespace) {
        return Err("source PPM header has no pixel delimiter".into());
    }
    offset += 1;
    if payload.get(offset - 1) == Some(&b'\r') && payload.get(offset) == Some(&b'\n') {
        offset += 1;
    }
    let pixel_bytes = width
        .checked_mul(height)
        .and_then(|value| value.checked_mul(3))
        .ok_or("source PPM dimensions overflow")?;
    let pixels = payload
        .get(offset..)
        .filter(|pixels| pixels.len() == pixel_bytes)
        .ok_or("source PPM has an invalid RGB payload")?;
    let row_bytes = width.checked_mul(3).ok_or("source BMP row overflows")?;
    let row_stride = row_bytes.checked_add(3).ok_or("source BMP row overflows")? & !3;
    let image_bytes = row_stride
        .checked_mul(height)
        .ok_or("source BMP dimensions overflow")?;
    let file_size = 54usize
        .checked_add(image_bytes)
        .ok_or("source BMP dimensions overflow")?;
    let width_u32 = u32::try_from(width)?;
    let height_u32 = u32::try_from(height)?;
    let file_size_u32 = u32::try_from(file_size)?;
    let image_bytes_u32 = u32::try_from(image_bytes)?;

    let mut bmp = Vec::with_capacity(file_size);
    bmp.extend_from_slice(b"BM");
    bmp.extend_from_slice(&file_size_u32.to_le_bytes());
    bmp.extend_from_slice(&[0; 4]);
    bmp.extend_from_slice(&54_u32.to_le_bytes());
    bmp.extend_from_slice(&40_u32.to_le_bytes());
    bmp.extend_from_slice(&width_u32.to_le_bytes());
    bmp.extend_from_slice(&height_u32.to_le_bytes());
    bmp.extend_from_slice(&1_u16.to_le_bytes());
    bmp.extend_from_slice(&24_u16.to_le_bytes());
    bmp.extend_from_slice(&0_u32.to_le_bytes());
    bmp.extend_from_slice(&image_bytes_u32.to_le_bytes());
    bmp.extend_from_slice(&[0; 16]);
    for row in (0..height).rev() {
        let source = &pixels[row * row_bytes..(row + 1) * row_bytes];
        for rgb in source.chunks_exact(3) {
            bmp.extend_from_slice(&[rgb[2], rgb[1], rgb[0]]);
        }
        bmp.resize(bmp.len() + row_stride - row_bytes, 0);
    }
    Ok(bmp)
}

fn ppm_token<'a>(payload: &'a [u8], offset: &mut usize) -> Option<&'a [u8]> {
    loop {
        while payload.get(*offset).is_some_and(u8::is_ascii_whitespace) {
            *offset += 1;
        }
        if payload.get(*offset) != Some(&b'#') {
            break;
        }
        while payload.get(*offset).is_some_and(|byte| *byte != b'\n') {
            *offset += 1;
        }
    }
    let start = *offset;
    while payload
        .get(*offset)
        .is_some_and(|byte| !byte.is_ascii_whitespace())
    {
        *offset += 1;
    }
    (start < *offset).then_some(&payload[start..*offset])
}

fn ppm_usize(
    payload: &[u8],
    offset: &mut usize,
    name: &str,
) -> Result<usize, Box<dyn std::error::Error>> {
    std::str::from_utf8(ppm_token(payload, offset).ok_or("source PPM header is incomplete")?)
        .map_err(|_| format!("source PPM {name} is not UTF-8"))?
        .parse::<usize>()
        .map_err(|_| format!("source PPM {name} is invalid").into())
}

fn rotate(transform: SimilarityTransform, point: [f32; 3]) -> [f32; 3] {
    let r = transform.rotation;
    [
        r[0] * point[0] + r[1] * point[1] + r[2] * point[2],
        r[3] * point[0] + r[4] * point[1] + r[5] * point[2],
        r[6] * point[0] + r[7] * point[1] + r[8] * point[2],
    ]
}

fn read_file(path: PathBuf, content_type: &'static str) -> (&'static str, &'static str, Vec<u8>) {
    match fs::read(path) {
        Ok(body) => ("200 OK", content_type, body),
        Err(_) => (
            "404 Not Found",
            "text/plain; charset=utf-8",
            b"not found".to_vec(),
        ),
    }
}

fn not_found() -> (&'static str, &'static str, Vec<u8>) {
    (
        "404 Not Found",
        "text/plain; charset=utf-8",
        b"not found".to_vec(),
    )
}

fn safe_chunk_path(path: &str) -> bool {
    let Some(name) = path.strip_prefix("/chunks/").and_then(|value| {
        value
            .strip_suffix(".json")
            .or_else(|| value.strip_suffix(".bin"))
    }) else {
        return false;
    };
    let hash = name
        .strip_prefix("fused-")
        .or_else(|| name.strip_prefix("points-"))
        .unwrap_or(name);
    hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{Read, Write},
        thread,
    };

    #[test]
    fn intake_accepts_only_plain_supported_video_filenames() {
        assert_eq!(
            safe_video_name("walkthrough.MOV"),
            Some("walkthrough.MOV".into())
        );
        assert!(safe_video_name("nested/room.mp4").is_none());
        assert!(safe_video_name("room.mkv").is_none());
        assert!(safe_video_name("../../scene.mov").is_none());
    }

    #[test]
    fn intake_routes_completed_world_and_its_root_relative_assets() {
        assert_eq!(intake_world_path("/world"), Some("/"));
        assert_eq!(intake_world_path("/world/"), Some("/"));
        assert_eq!(
            intake_world_path("/world/manifest.json"),
            Some("/manifest.json")
        );
        assert_eq!(
            intake_scene_request_path("/world/manifest.json?product=tsdf-active"),
            Some("/manifest.json?product=tsdf-active".to_owned())
        );
        assert_eq!(intake_world_path("/manifest.json"), Some("/manifest.json"));
        assert_eq!(
            intake_world_path("/camera-controls.js"),
            Some("/camera-controls.js")
        );
        assert_eq!(
            intake_world_path("/chunks/000001.bin"),
            Some("/chunks/000001.bin")
        );
        assert_eq!(
            intake_world_path("/sources/frame-000001.bmp"),
            Some("/sources/frame-000001.bmp")
        );
        assert_eq!(intake_world_path("/api/job"), None);
        assert_eq!(intake_world_path("/unknown"), None);
    }

    #[test]
    fn completed_jobs_do_not_block_a_new_local_capture() {
        let root = std::env::temp_dir().join(format!("vestra-intake-state-{}", std::process::id()));
        let job = IntakeJob {
            id: 1,
            root: root.clone(),
            video: root.join("room.mov"),
            scene: root.join("world.vestra"),
            log: root.join("reconstruct.log"),
            settings: IntakeSettings {
                candidate_fps: 1.0,
                hard_max_frames: 1,
                width: 1,
                height: 1,
                chunk_size: 1,
                overlap: 0,
                minimum_confidence: 1.0,
                pixel_stride: 1,
                tsdf: false,
                cpp_pr2_relative: false,
            },
            child: None,
            outcome: IntakeOutcome::Complete,
        };
        assert!(!job_is_in_progress(&job));
        let mut running = job;
        running.outcome = IntakeOutcome::Running;
        assert!(job_is_in_progress(&running));
    }

    #[test]
    fn studio_offers_a_non_mutating_return_to_capture_link() {
        assert!(INDEX_HTML.contains("href=\"/\">← back to capture</a>"));
    }

    #[test]
    fn completed_scene_serves_its_retained_input_video() {
        let root =
            std::env::temp_dir().join(format!("vestra-studio-input-video-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("input-video.mov"), b"capture").unwrap();
        let (status, content_type, body) = input_video(&root);
        assert_eq!(status, "200 OK");
        assert_eq!(content_type, "video/quicktime");
        assert_eq!(body, b"capture");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn studio_uses_a_depth_buffer_to_hide_occluded_surfels() {
        assert!(INDEX_HTML.contains("depth:true"));
        assert!(INDEX_HTML.contains("gl.enable(gl.DEPTH_TEST)"));
        assert!(INDEX_HTML.contains("const MAX_SURFEL_PIXELS=12"));
        assert!(INDEX_HTML.contains("gl_PointSize=clamp"));
        assert!(INDEX_HTML.contains("gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT)"));
        assert!(INDEX_HTML.contains("gl_FragColor=vec4(v,1.)"));
        assert!(INDEX_HTML.contains("VestraCameraControls.viewMatrix(eye,orientation)"));
        assert!(!INDEX_HTML.contains("e=center.map"));
        assert!(INDEX_HTML.contains("architectureProduct"));
        assert!(INDEX_HTML.contains("adding verified planar support"));
        assert!(INDEX_HTML.contains("batch.overlay ? .32 : 1"));
    }

    #[test]
    fn product_query_accepts_only_generated_world_product_ids() {
        assert_eq!(
            query_product_id("product=local-active"),
            Some("local-active")
        );
        assert_eq!(
            query_product_id("ignored=1&product=droid-slam-global-active"),
            Some("droid-slam-global-active")
        );
        assert_eq!(query_product_id("product=../../manifest"), None);
        assert_eq!(query_product_id("product=bad%2Fid"), None);
    }

    #[test]
    fn studio_caps_a_dense_binary_world_to_a_responsive_even_preview() {
        assert!(INDEX_HTML.contains("const MAX_VISUAL_CHUNKS=24"));
        assert!(INDEX_HTML.contains("function selectVisualChunkUrls(urls,binary)"));
        assert!(INDEX_HTML.contains("await loadLayer(visualUrls,Boolean(binary),hasFused)"));
        assert!(
            INDEX_HTML.contains(
                "const visualUrls=preview?urls:selectVisualChunkUrls(urls,Boolean(binary))"
            )
        );
        assert!(INDEX_HTML.contains("fused_preview_point_binary_chunk_hashes"));
        assert!(INDEX_HTML.contains("attribute float a"));
        assert!(INDEX_HTML.contains("bind(batch.a)"));
        assert!(INDEX_HTML.contains("a:make(radii,'a',1)"));
        assert!(INDEX_HTML.contains("attribute.size||3"));
        assert!(INDEX_HTML.contains("fetch('manifest.json'+productSuffix"));
        assert!(!INDEX_HTML.contains("fetch('/manifest.json"));
    }

    #[test]
    fn studio_matches_a_calibrated_source_camera_with_its_own_lens() {
        assert!(INDEX_HTML.contains("vertical_fov_radians"));
        assert!(INDEX_HTML.contains("aspect_ratio"));
        assert!(INDEX_HTML.contains("matchedCamera"));
        assert!(INDEX_HTML.contains("matched calibrated source camera and lens"));
        assert!(INDEX_HTML.contains("gl.viewport(left,bottom,width,height)"));
    }

    #[test]
    fn studio_presents_only_the_world_and_depth_replay_modes() {
        assert!(INDEX_HTML.contains("id=\"replay-toggle\">open video + depth"));
        assert!(INDEX_HTML.contains("id=\"replay-close\">open 3D world"));
        assert!(INDEX_HTML.contains("id=\"input-video\" src=\"input-video\" controls muted"));
        assert!(!INDEX_HTML.contains("id=\"products\""));
        assert!(!INDEX_HTML.contains("id=\"layers\""));
        assert!(!INDEX_HTML.contains("id=\"replay-parallax\""));
        assert!(!INDEX_HTML.contains("canvas.onpointerdown"));
        assert!(!INDEX_HTML.contains("canvas.addEventListener('wheel'"));
        assert!(INDEX_HTML.contains("arrow keys move · W A S D look · R reset"));
        assert!(INDEX_HTML.contains("command.startsWith('look')"));
        assert!(INDEX_HTML.contains("official DA3 depth map · matched COLMAP camera"));
    }

    #[test]
    fn studio_replay_uses_the_completed_local_input_video() {
        assert_eq!(intake_world_path("/input-video"), Some("/input-video"));
        assert_eq!(
            intake_world_path("/replay/frames/42.bin"),
            Some("/replay/frames/42.bin")
        );
        assert!(INDEX_HTML.contains("id=\"input-video\""));
        assert!(INDEX_HTML.contains("id=\"replay-points\""));
        assert!(INDEX_HTML.contains("src=\"input-video\""));
        assert!(INDEX_HTML.contains("fetch('replay/frames/"));
        assert!(INDEX_HTML.contains("function updateReplay()"));
        assert!(INDEX_HTML.contains("original capture · 3:2 crop"));
        assert!(INDEX_HTML.contains("depth map · matched camera"));
        assert!(INDEX_HTML.contains("function drawReplayDepthMap("));
        assert!(INDEX_HTML.contains("depth samples · "));
        assert!(INDEX_HTML.contains("depth map"));
        assert!(INDEX_HTML.contains("object-fit:cover"));
        assert!(INDEX_HTML.contains("replay-landscape"));
        assert!(INDEX_HTML.contains("function arrangeReplay()"));
        assert!(INDEX_HTML.contains("function setReplay(open)"));
    }

    #[test]
    fn replay_payload_contains_only_real_measured_sample_pixels() {
        let camera = CameraCalibration {
            world_to_camera: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            intrinsics: [504.0, 0.0, 252.0, 0.0, 336.0, 168.0, 0.0, 0.0, 1.0],
        };
        let payload = encode_replay_points(
            &[
                MeasuredPoint {
                    position: [1.0, 2.0, 4.0],
                    normal: [0.0, 0.0, 1.0],
                    color_srgb: [7, 8, 9],
                    confidence: 1.0,
                    radius: 0.1,
                    source_pixel: [4, 2],
                },
                MeasuredPoint {
                    position: [2.0, 3.0, 5.0],
                    normal: [0.0, 0.0, 1.0],
                    color_srgb: [10, 11, 12],
                    confidence: 1.0,
                    radius: 0.1,
                    source_pixel: [500, 332],
                },
            ],
            camera,
        );
        assert_eq!(&payload[..4], b"VRPL");
        assert_eq!(u16::from_le_bytes(payload[4..6].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(payload[8..12].try_into().unwrap()), 2);
        assert_eq!(
            f32::from_le_bytes(payload[12..16].try_into().unwrap()),
            504.0
        );
        assert_eq!(
            f32::from_le_bytes(payload[16..20].try_into().unwrap()),
            336.0
        );
        assert_eq!(f32::from_le_bytes(payload[28..32].try_into().unwrap()), 1.0);
        assert_eq!(f32::from_le_bytes(payload[36..40].try_into().unwrap()), 4.0);
        assert_eq!(&payload[40..43], &[7, 8, 9]);
        assert_eq!(replay_frame_index("/replay/frames/42.bin"), Some(42));
        assert_eq!(replay_frame_index("/replay/frames/42.json"), None);
    }

    #[test]
    fn studio_transforms_diagnostic_geometry_into_the_point_cloud_coordinate_system() {
        assert!(INDEX_HTML.contains("cameraToViewer(rawRay.origin)"));
        assert!(INDEX_HTML.contains("cameraToViewer(rawRay.forward)"));
        assert!(INDEX_HTML.contains("cameraToViewer(link.from)"));
        assert!(INDEX_HTML.contains("cameraToViewer(link.to)"));
    }

    #[test]
    fn recovered_running_job_is_marked_interrupted_without_losing_settings() {
        let root = std::env::temp_dir().join(format!(
            "vestra-intake-recovery-test-{}",
            std::process::id()
        ));
        let job_root = root.join("job-000007");
        fs::create_dir_all(&job_root).unwrap();
        let settings = IntakeSettings {
            candidate_fps: 8.0,
            hard_max_frames: 24,
            width: 504,
            height: 336,
            chunk_size: 12,
            overlap: 3,
            minimum_confidence: 0.5,
            pixel_stride: 6,
            tsdf: true,
            cpp_pr2_relative: true,
        };
        let job = IntakeJob {
            id: 7,
            root: job_root.clone(),
            video: job_root.join("room.mov"),
            scene: job_root.join("world.vestra"),
            log: job_root.join("reconstruct.log"),
            settings: settings.clone(),
            child: None,
            outcome: IntakeOutcome::Running,
        };
        fs::write(&job.video, b"fixture").unwrap();
        write_intake_record(&job).unwrap();
        let config = IntakeConfig {
            executable: root.join("vestra"),
            model: root.join("model.gguf"),
            jobs_root: root.clone(),
            port: 4317,
            candidate_fps: 1.0,
            hard_max_frames: 1,
            width: 1,
            height: 1,
            chunk_size: 1,
            overlap: 0,
            minimum_confidence: 1.0,
            pixel_stride: 1,
            tsdf: false,
            cpp_pr2_relative: false,
        };
        let recovered = recover_latest_job(&config).unwrap().unwrap();
        assert_eq!(recovered.id, 7);
        assert_eq!(recovered.outcome, IntakeOutcome::Interrupted);
        assert_eq!(recovered.settings.candidate_fps, settings.candidate_fps);
        assert_eq!(recovered.settings.hard_max_frames, settings.hard_max_frames);
        assert_eq!(recovered.settings.pixel_stride, settings.pixel_stride);
        assert_eq!(recovered.settings.tsdf, settings.tsdf);
        assert_eq!(
            recovered.settings.cpp_pr2_relative,
            settings.cpp_pr2_relative
        );
        let persisted: IntakeRecord =
            serde_json::from_slice(&fs::read(record_path(&job_root)).unwrap()).unwrap();
        assert_eq!(persisted.outcome, IntakeOutcome::Interrupted);
        assert_eq!(next_job_id(&root).unwrap(), 8);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn only_sha256_chunk_paths_are_servable() {
        let hash = "a".repeat(64);
        assert!(safe_chunk_path(&format!("/chunks/{hash}.json")));
        assert!(safe_chunk_path(&format!("/chunks/fused-{hash}.json")));
        assert!(safe_chunk_path(&format!("/chunks/points-{hash}.json")));
        assert!(safe_chunk_path(&format!("/chunks/points-{hash}.bin")));
        assert!(!safe_chunk_path("/chunks/../../manifest.json"));
        assert!(!safe_chunk_path("/chunks/xyz.json"));
    }

    #[test]
    fn local_host_serves_a_bundle_manifest() {
        let root = std::env::temp_dir().join(format!("vestra-studio-test-{}", std::process::id()));
        if root.exists() {
            fs::remove_dir_all(&root).unwrap();
        }
        SceneBundle::create(
            &root,
            vestra_core::SceneProvenance {
                engine_revision: "test".into(),
                kernel_revision: "test".into(),
                model_fingerprint: "test".into(),
                settings_fingerprint: "test".into(),
            },
        )
        .unwrap();
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let root_for_server = root.clone();
        let worker = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            handle(stream, &root_for_server).unwrap();
        });
        let mut client = TcpStream::connect(address).unwrap();
        client
            .write_all(b"GET /manifest.json HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .unwrap();
        let mut response = String::new();
        client.read_to_string(&mut response).unwrap();
        worker.join().unwrap();
        assert!(response.starts_with("HTTP/1.1 200 OK"));
        assert!(response.contains("\"schema\":\"vestra.scene/v1\""));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rotation_preserves_identity_camera_direction() {
        assert_eq!(
            rotate(SimilarityTransform::IDENTITY, [0.0, 0.0, 1.0]),
            [0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn calibration_frustum_uses_w2c_transpose_and_intrinsic_image_corners() {
        let corners = camera_frustum_directions(CameraCalibration {
            world_to_camera: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            intrinsics: [2.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 0.0, 1.0],
        })
        .unwrap();
        let expected = 1.0 / 6.0_f32.sqrt();
        assert!((corners[0][0] + expected).abs() < 1e-6);
        assert!((corners[0][1] + expected).abs() < 1e-6);
        assert!((corners[0][2] - 2.0 * expected).abs() < 1e-6);
        assert!((corners[2][0] - expected).abs() < 1e-6);
        assert!((corners[2][1] - expected).abs() < 1e-6);
        assert!((corners[2][2] - 2.0 * expected).abs() < 1e-6);
    }

    #[test]
    fn colmap_global_camera_evidence_has_a_finite_origin_and_frustum() {
        let camera = vestra_core::ColmapCameraModel {
            camera_id: 7,
            model: "SIMPLE_RADIAL".to_owned(),
            width: 1620,
            height: 1080,
            parameters: vec![810.0, 810.0, 540.0, 0.0],
        };
        let (origin, forward, corners, right, up, vertical_fov_radians, aspect_ratio) =
            colmap_camera_ray(
                [1.0, 0.0, 0.0, -2.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, -3.0],
                &camera,
            )
            .expect("valid calibrated COLMAP camera");
        assert_eq!(origin, [2.0, -1.0, 3.0]);
        assert_eq!(right, [1.0, 0.0, 0.0]);
        assert_eq!(up, [0.0, -1.0, 0.0]);
        assert!(forward.iter().all(|value| value.is_finite()));
        assert!(corners.iter().flatten().all(|value| value.is_finite()));
        assert!((vertical_fov_radians - 2.0 * (2.0_f32 / 3.0).atan()).abs() < 1e-6);
        assert!((aspect_ratio - 1.5).abs() < 1e-6);
    }

    #[test]
    fn diagnostic_links_preserve_seams_and_only_verified_loop_edges() {
        let poses = [
            vestra_core::FusedWindowPose {
                window_index: 10,
                local_to_world: SimilarityTransform::IDENTITY,
            },
            vestra_core::FusedWindowPose {
                window_index: 11,
                local_to_world: SimilarityTransform::IDENTITY,
            },
            vestra_core::FusedWindowPose {
                window_index: 12,
                local_to_world: SimilarityTransform::IDENTITY,
            },
        ];
        let anchors = BTreeMap::from([
            (10, [0.0, 0.0, 0.0]),
            (11, [1.0, 0.0, 0.0]),
            (12, [2.0, 0.0, 0.0]),
        ]);
        let alignment = vestra_core::AlignmentReport {
            transform: SimilarityTransform::IDENTITY,
            correspondence_count: 100,
            inlier_count: 100,
            rms_residual: 0.0,
            normalized_rms_residual: 0.0,
        };
        let edges = [
            vestra_core::PoseGraphEdge {
                from: 0,
                to: 1,
                measurement: SimilarityTransform::IDENTITY,
                information: 1.0,
                loop_closure: false,
            },
            vestra_core::PoseGraphEdge {
                from: 2,
                to: 0,
                measurement: SimilarityTransform::IDENTITY,
                information: 1.0,
                loop_closure: true,
            },
        ];

        let links = diagnostic_links(&poses, &[alignment.clone(), alignment], &edges, &anchors);
        assert_eq!(links.len(), 3);
        assert_eq!(links[0].kind, "seam");
        assert_eq!(links[1].from, [1.0, 0.0, 0.0]);
        assert_eq!(links[2].kind, "loop");
        assert_eq!(links[2].from, [2.0, 0.0, 0.0]);
        assert_eq!(links[2].to, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn decoded_rgb24_source_is_served_as_a_bottom_up_bgr_bmp() {
        let root =
            std::env::temp_dir().join(format!("vestra-studio-source-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("decoded")).unwrap();
        fs::write(
            source_frame_path(&root, 0),
            [b"P6\n2 1\n255\n".as_slice(), &[1, 2, 3, 4, 5, 6]].concat(),
        )
        .unwrap();

        let (status, content_type, body) = source_thumbnail(&root, "/sources/0.bmp");
        assert_eq!(status, "200 OK");
        assert_eq!(content_type, "image/bmp");
        assert_eq!(&body[..2], b"BM");
        assert_eq!(u32::from_le_bytes(body[18..22].try_into().unwrap()), 2);
        assert_eq!(u32::from_le_bytes(body[22..26].try_into().unwrap()), 1);
        assert_eq!(&body[54..60], &[3, 2, 1, 6, 5, 4]);
        assert_eq!(
            source_thumbnail(&root, "/sources/../../manifest.bmp").0,
            "404 Not Found"
        );
        fs::remove_dir_all(root).unwrap();
    }
}
