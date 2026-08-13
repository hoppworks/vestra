//! Local-only HTTP host for the dependency-free Vestra browser studio.

use std::{
    collections::BTreeSet,
    fs,
    io::{BufRead, BufReader, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
};

use vestra_core::{CameraCalibration, SceneBundle, SimilarityTransform, camera_centre_direction};

const INDEX_HTML: &str = include_str!("index.html");

#[derive(Debug, thiserror::Error)]
pub enum StudioError {
    #[error("could not bind Vestra Studio: {0}")]
    Bind(#[from] std::io::Error),
    #[error("scene manifest is missing at {0}")]
    MissingManifest(PathBuf),
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
                    "origin": pose.local_to_world.apply(camera.centre_local),
                    "forward": direction,
                    "corners": corners,
                }));
            }
        }
        Ok(serde_json::to_vec(&serde_json::json!({
            "scale": "relative",
            "camera_rays": camera_rays,
            "source_frames": source_frames.into_iter().collect::<Vec<_>>(),
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
