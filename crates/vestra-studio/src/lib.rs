//! Local-only HTTP host for the dependency-free Vestra browser studio.

use std::{
    fs,
    io::{BufRead, BufReader, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
};

use vestra_core::{SceneBundle, SimilarityTransform, camera_centre_direction};

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
                let Some(camera) = camera_centre_direction(view.frame_index, view.camera) else {
                    continue;
                };
                let direction = rotate(pose.local_to_world, camera.forward_local);
                if !direction.iter().all(|value| value.is_finite()) {
                    continue;
                }
                camera_rays.push(serde_json::json!({
                    "window_index": window.window.index,
                    "frame_index": view.frame_index,
                    "origin": pose.local_to_world.apply(camera.centre_local),
                    "forward": direction,
                }));
            }
        }
        Ok(serde_json::to_vec(&serde_json::json!({
            "scale": "relative",
            "camera_rays": camera_rays,
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
}
