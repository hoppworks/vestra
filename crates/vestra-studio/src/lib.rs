//! Local-only HTTP host for the dependency-free Vestra browser studio.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
};

use vestra_core::{CameraCalibration, SceneBundle, SimilarityTransform, camera_centre_direction};

const INDEX_HTML: &str = include_str!("index.html");
const INTAKE_HTML: &str = include_str!("intake.html");

/// Local process-launch configuration for the browser intake. The server never
/// accepts a destination or executable from HTTP; both are fixed by the CLI
/// that started it.
#[derive(Debug, Clone)]
pub struct IntakeConfig {
    pub executable: PathBuf,
    pub model: PathBuf,
    pub jobs_root: PathBuf,
    pub port: u16,
    pub frames: usize,
    pub width: usize,
    pub height: usize,
    pub chunk_size: usize,
    pub overlap: usize,
    pub minimum_confidence: f32,
    pub pixel_stride: usize,
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
        let _ = handle(stream, &scene_root);
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
    if config.frames == 0 || config.width == 0 || config.height == 0 {
        return Err(StudioError::IntakeConfig(
            "frame and raster dimensions must be positive",
        ));
    }
    fs::create_dir_all(&config.jobs_root)?;
    let state = Arc::new(Mutex::new(IntakeState {
        config,
        next_job: 1,
        active: None,
    }));
    let port = state.lock().expect("intake state lock").config.port;
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    for stream in listener.incoming().flatten() {
        let _ = handle_intake(stream, &state);
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
    scene: PathBuf,
    log: PathBuf,
    child: Child,
    viewer: Option<Child>,
    outcome: IntakeOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IntakeOutcome {
    Running,
    Complete,
    Failed,
}

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
        if guard.active.is_some() {
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
    let scene = root.join("world.vestra");
    let log = root.join("reconstruct.log");
    let log_file =
        fs::File::create(&log).map_err(|error| format!("could not create job log: {error}"))?;
    let child = Command::new(&config.executable)
        .args([
            "reconstruct",
            "--video",
            video.to_string_lossy().as_ref(),
            "--model",
            config.model.to_string_lossy().as_ref(),
            "--output",
            scene.to_string_lossy().as_ref(),
            "--frames",
            &config.frames.to_string(),
            "--width",
            &config.width.to_string(),
            "--height",
            &config.height.to_string(),
            "--chunk-size",
            &config.chunk_size.to_string(),
            "--overlap",
            &config.overlap.to_string(),
            "--minimum-confidence",
            &config.minimum_confidence.to_string(),
            "--pixel-stride",
            &config.pixel_stride.to_string(),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::from(
            log_file
                .try_clone()
                .map_err(|error| format!("could not clone job log: {error}"))?,
        ))
        .stderr(Stdio::from(log_file))
        .spawn()
        .map_err(|error| format!("could not start reconstruction: {error}"))?;
    state
        .lock()
        .map_err(|_| "intake state is unavailable".to_owned())?
        .active = Some(IntakeJob {
        id,
        scene,
        log,
        child,
        viewer: None,
        outcome: IntakeOutcome::Running,
    });
    Ok(
        serde_json::to_vec(&serde_json::json!({"job": id, "state": "running"}))
            .expect("fixed intake response serializes"),
    )
}

fn intake_status(state: &Arc<Mutex<IntakeState>>) -> Vec<u8> {
    let mut guard = match state.lock() {
        Ok(guard) => guard,
        Err(_) => return br#"{"state":"unavailable"}"#.to_vec(),
    };
    let config = guard.config.clone();
    let Some(job) = guard.active.as_mut() else {
        return br#"{"state":"idle"}"#.to_vec();
    };
    if job.outcome == IntakeOutcome::Running {
        match job.child.try_wait() {
            Ok(Some(status)) if status.success() => {
                let viewer_port = config.port.saturating_add(1);
                job.viewer = Command::new(&config.executable)
                    .args([
                        "serve",
                        "--scene",
                        job.scene.to_string_lossy().as_ref(),
                        "--port",
                        &viewer_port.to_string(),
                    ])
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn()
                    .ok();
                job.outcome = IntakeOutcome::Complete;
            }
            Ok(Some(_)) | Err(_) => job.outcome = IntakeOutcome::Failed,
            Ok(None) => {}
        }
    }
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
        IntakeOutcome::Complete => "complete",
        IntakeOutcome::Failed => "failed",
    };
    serde_json::to_vec(&serde_json::json!({
        "job": job.id,
        "state": state,
        "viewer": if job.outcome == IntakeOutcome::Complete { Some(format!("http://127.0.0.1:{}", config.port.saturating_add(1))) } else { None },
        "log_tail": log_tail,
    }))
    .expect("fixed intake status serializes")
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
    let (status, content_type, body) = match path {
        "/" | "/index.html" => (
            "200 OK",
            "text/html; charset=utf-8",
            INDEX_HTML.as_bytes().to_vec(),
        ),
        "/manifest.json" => read_file(root.join("manifest.json"), "application/json"),
        "/evidence.json" => evidence(root),
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

/// Emits only compact diagnostic evidence for Studio. The raw camera W2C
/// matrices stay in the immutable chunks; this endpoint derives camera rays
/// in the fused relative frame, never inventing metric coordinates.
fn evidence(root: &Path) -> (&'static str, &'static str, Vec<u8>) {
    let payload = (|| -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let bundle = SceneBundle::open(root)?;
        let manifest = bundle.manifest()?;
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
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("manifest.json"),
            b"{\"schema\":\"vestra.scene/v1\"}",
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
        assert!(response.ends_with("{\"schema\":\"vestra.scene/v1\"}"));
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
