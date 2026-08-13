//! Local FFmpeg-backed video ingestion.
//!
//! Commands are passed as argument vectors; no user-controlled text is ever
//! interpreted by a shell. Decoded PPM frames keep the product's RGB contract
//! explicit and avoid a second opaque image-decoder dependency.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use crate::OwnedFrame;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoExtractionSettings {
    pub width: usize,
    pub height: usize,
    pub max_frames: usize,
}

impl Default for VideoExtractionSettings {
    fn default() -> Self {
        Self {
            width: 504,
            height: 336,
            max_frames: 120,
        }
    }
}

#[derive(Debug)]
pub struct VideoFrames {
    pub duration_seconds: f64,
    pub frames: Vec<OwnedFrame>,
    pub decoded_directory: PathBuf,
}

#[derive(Debug, thiserror::Error)]
pub enum VideoInputError {
    #[error("video dimensions and max frame count must be positive")]
    InvalidSettings,
    #[error("ffprobe is unavailable or failed: {0}")]
    Probe(String),
    #[error("video duration must be finite and positive, got {0:?}")]
    InvalidDuration(Option<String>),
    #[error("ffmpeg is unavailable or failed: {0}")]
    Decode(String),
    #[error("decoded frame directory already exists at {0}")]
    ExistingOutput(PathBuf),
    #[error("decoded PPM is invalid: {0}")]
    Ppm(String),
    #[error("video I/O failed: {0}")]
    Io(#[from] std::io::Error),
}

/// Decodes a deterministic, uniformly sampled RGB frame set.
///
/// `work_directory` becomes an owned decode cache. It must not exist yet,
/// which makes accidental reuse of an unrelated video impossible.
pub fn extract_video_frames(
    video: &Path,
    work_directory: impl Into<PathBuf>,
    settings: VideoExtractionSettings,
) -> Result<VideoFrames, VideoInputError> {
    if settings.width == 0 || settings.height == 0 || settings.max_frames == 0 {
        return Err(VideoInputError::InvalidSettings);
    }
    let duration_stdout = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(video)
        .output()
        .map_err(|error| VideoInputError::Probe(error.to_string()))?;
    if !duration_stdout.status.success() {
        return Err(VideoInputError::Probe(
            String::from_utf8_lossy(&duration_stdout.stderr)
                .trim()
                .to_owned(),
        ));
    }
    let duration_text = String::from_utf8_lossy(&duration_stdout.stdout)
        .trim()
        .to_owned();
    let duration = duration_text.parse::<f64>().ok();
    let Some(duration_seconds) = duration.filter(|value| value.is_finite() && *value > 0.0) else {
        return Err(VideoInputError::InvalidDuration(Some(duration_text)));
    };

    let decoded_directory = work_directory.into();
    if decoded_directory.exists() {
        return Err(VideoInputError::ExistingOutput(decoded_directory));
    }
    fs::create_dir_all(&decoded_directory)?;
    let frame_pattern = decoded_directory.join("frame-%06d.ppm");
    let filter = format!(
        "fps={}/{duration_seconds:.9},scale={}:{}:flags=lanczos",
        settings.max_frames, settings.width, settings.height
    );
    let decode = Command::new("ffmpeg")
        .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(video)
        .args(["-vf", &filter, "-frames:v"])
        .arg(settings.max_frames.to_string())
        .args(["-pix_fmt", "rgb24"])
        .arg(&frame_pattern)
        .output()
        .map_err(|error| VideoInputError::Decode(error.to_string()))?;
    if !decode.status.success() {
        return Err(VideoInputError::Decode(
            String::from_utf8_lossy(&decode.stderr).trim().to_owned(),
        ));
    }

    let mut paths = fs::read_dir(&decoded_directory)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "ppm"))
        .collect::<Vec<_>>();
    paths.sort_unstable();
    let frames = paths
        .iter()
        .map(|path| read_ppm_rgb(path))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(VideoFrames {
        duration_seconds,
        frames,
        decoded_directory,
    })
}

fn read_ppm_rgb(path: &Path) -> Result<OwnedFrame, VideoInputError> {
    let bytes = fs::read(path)?;
    let mut cursor = 0;
    let magic = ppm_token(&bytes, &mut cursor)?;
    if magic != b"P6" {
        return Err(VideoInputError::Ppm(
            "only binary P6 PPM is supported".to_owned(),
        ));
    }
    let width = ppm_number(&bytes, &mut cursor, "width")?;
    let height = ppm_number(&bytes, &mut cursor, "height")?;
    let max_value = ppm_number(&bytes, &mut cursor, "max value")?;
    if width == 0 || height == 0 || max_value != 255 {
        return Err(VideoInputError::Ppm(
            "PPM dimensions must be positive and max value must be 255".to_owned(),
        ));
    }
    if cursor >= bytes.len() || !bytes[cursor].is_ascii_whitespace() {
        return Err(VideoInputError::Ppm(
            "PPM header lacks a raster delimiter".to_owned(),
        ));
    }
    cursor += 1;
    let expected = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| VideoInputError::Ppm("PPM dimensions overflow".to_owned()))?;
    if bytes.len() != cursor + expected {
        return Err(VideoInputError::Ppm(
            "PPM raster length does not match header".to_owned(),
        ));
    }
    Ok(OwnedFrame {
        rgb_hwc_u8: bytes[cursor..].to_vec(),
        width,
        height,
    })
}

fn ppm_number(bytes: &[u8], cursor: &mut usize, name: &str) -> Result<usize, VideoInputError> {
    let token = ppm_token(bytes, cursor)?;
    std::str::from_utf8(token)
        .ok()
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| VideoInputError::Ppm(format!("PPM {name} is invalid")))
}

fn ppm_token<'a>(bytes: &'a [u8], cursor: &mut usize) -> Result<&'a [u8], VideoInputError> {
    while *cursor < bytes.len() {
        if bytes[*cursor].is_ascii_whitespace() {
            *cursor += 1;
        } else if bytes[*cursor] == b'#' {
            while *cursor < bytes.len() && bytes[*cursor] != b'\n' {
                *cursor += 1;
            }
        } else {
            break;
        }
    }
    let start = *cursor;
    while *cursor < bytes.len() && !bytes[*cursor].is_ascii_whitespace() {
        *cursor += 1;
    }
    if start == *cursor {
        return Err(VideoInputError::Ppm("PPM header is truncated".to_owned()));
    }
    Ok(&bytes[start..*cursor])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ppm_reader_accepts_comments_and_preserves_rgb_bytes() {
        let root = std::env::temp_dir().join(format!("vestra-ppm-test-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let file = root.join("frame.ppm");
        fs::write(&file, b"P6\n# capture\n2 1\n255\n\x01\x02\x03\x04\x05\x06").unwrap();
        let frame = read_ppm_rgb(&file).unwrap();
        assert_eq!(frame.width, 2);
        assert_eq!(frame.height, 1);
        assert_eq!(frame.rgb_hwc_u8, vec![1, 2, 3, 4, 5, 6]);
        fs::remove_dir_all(root).unwrap();
    }
}
