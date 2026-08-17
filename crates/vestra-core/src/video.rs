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

use crate::{OwnedFrame, RasterCrop};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureDisposition {
    Ready,
    Review,
    Recapture,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CaptureQuality {
    pub disposition: CaptureDisposition,
    pub frame_count: usize,
    /// Mean absolute luma change between neighbouring selected frames, in
    /// normalized `[0, 1]` units. It is a capture-risk indicator, not a
    /// geometric quality claim.
    pub mean_adjacent_luma_delta: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoExtractionSettings {
    pub width: usize,
    pub height: usize,
    /// Candidate frames per second before geometry selection. This is a rate,
    /// not a duration-dependent total, so a long capture does not silently
    /// become temporally sparse.
    pub candidate_fps: f64,
    /// Safety ceiling for a single local capture. It prevents an accidental
    /// multi-hour upload from exhausting the host while retaining a quality-
    /// first candidate rate for normal room walkthroughs.
    pub max_frames: usize,
}

impl Default for VideoExtractionSettings {
    fn default() -> Self {
        Self {
            width: 504,
            height: 336,
            candidate_fps: 8.0,
            max_frames: 1800,
        }
    }
}

#[derive(Debug)]
pub struct VideoFrames {
    pub duration_seconds: f64,
    pub frames: Vec<OwnedFrame>,
    pub decoded_directory: PathBuf,
    pub capture_quality: CaptureQuality,
}

/// Exact source and crop geometry of the canonical decoded raster.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoRasterMetadata {
    pub duration_seconds: f64,
    pub source_width: usize,
    pub source_height: usize,
    pub crop: RasterCrop,
}

#[derive(Debug, thiserror::Error)]
pub enum VideoInputError {
    #[error("video dimensions and max frame count must be positive")]
    InvalidSettings,
    #[error("ffprobe is unavailable or failed: {0}")]
    Probe(String),
    #[error("video duration must be finite and positive, got {0:?}")]
    InvalidDuration(Option<String>),
    #[error(
        "portrait capture {width}x{height} cannot be reconstructed with Vestra's locked 3:2 landscape raster; record in landscape"
    )]
    PortraitCapture { width: usize, height: usize },
    #[error("ffmpeg is unavailable or failed: {0}")]
    Decode(String),
    #[error("decoded frame directory already exists at {0}")]
    ExistingOutput(PathBuf),
    #[error(
        "decoded frame cache at {path} does not match the locked reconstruction raster/frame contract: {reason}"
    )]
    InvalidCache { path: PathBuf, reason: String },
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
    if settings.width == 0
        || settings.height == 0
        || settings.max_frames == 0
        || !settings.candidate_fps.is_finite()
        || settings.candidate_fps <= 0.0
    {
        return Err(VideoInputError::InvalidSettings);
    }
    let duration_seconds = probe_duration(video)?;
    let geometry = probe_geometry(video)?;
    if geometry.width < geometry.height {
        return Err(VideoInputError::PortraitCapture {
            width: geometry.width,
            height: geometry.height,
        });
    }

    let decoded_directory = work_directory.into();
    if decoded_directory.exists() {
        return Err(VideoInputError::ExistingOutput(decoded_directory));
    }
    fs::create_dir_all(&decoded_directory)?;
    let frame_pattern = decoded_directory.join("frame-%06d.ppm");
    let filter = center_crop_filter(duration_seconds, settings, geometry)?;
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

    load_decoded_frame_cache_with_duration(&decoded_directory, settings, duration_seconds)
}

/// Probes the immutable video-to-raster contract without decoding pixels. The
/// caller persists it beside the decoded cache before invoking a pose provider.
pub fn video_raster_metadata(
    video: &Path,
    settings: VideoExtractionSettings,
) -> Result<VideoRasterMetadata, VideoInputError> {
    if settings.width == 0 || settings.height == 0 || settings.max_frames == 0 {
        return Err(VideoInputError::InvalidSettings);
    }
    let duration_seconds = probe_duration(video)?;
    let geometry = probe_geometry(video)?;
    if geometry.width < geometry.height {
        return Err(VideoInputError::PortraitCapture {
            width: geometry.width,
            height: geometry.height,
        });
    }
    Ok(VideoRasterMetadata {
        duration_seconds,
        source_width: geometry.width,
        source_height: geometry.height,
        crop: crop_geometry(settings, geometry)?,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VideoGeometry {
    width: usize,
    height: usize,
}

/// Builds Vestra's locked image transform: crop the source around its centre
/// to the reconstruction aspect ratio, then resize. This avoids the geometric
/// distortion caused by a direct non-uniform scale.
fn center_crop_filter(
    _duration_seconds: f64,
    settings: VideoExtractionSettings,
    geometry: VideoGeometry,
) -> Result<String, VideoInputError> {
    let crop = crop_geometry(settings, geometry)?;
    Ok(format!(
        "fps={:.9},crop={}:{}:{}:{},scale={}:{}:flags=lanczos",
        settings.candidate_fps,
        crop.width,
        crop.height,
        crop.x,
        crop.y,
        settings.width,
        settings.height,
    ))
}

fn crop_geometry(
    settings: VideoExtractionSettings,
    geometry: VideoGeometry,
) -> Result<RasterCrop, VideoInputError> {
    let (width, height) = if geometry.width * settings.height >= geometry.height * settings.width {
        (
            geometry.height * settings.width / settings.height,
            geometry.height,
        )
    } else {
        (
            geometry.width,
            geometry.width * settings.height / settings.width,
        )
    };
    if width == 0 || height == 0 {
        return Err(VideoInputError::InvalidSettings);
    }
    Ok(RasterCrop {
        x: (geometry.width - width) / 2,
        y: (geometry.height - height) / 2,
        width,
        height,
    })
}

/// Loads the deterministic decode cache produced by [`extract_video_frames`].
/// This is deliberately strict: callers use it only after locking video and
/// settings provenance, and every cached image must match the requested raster.
pub fn load_decoded_frame_cache(
    video: &Path,
    decoded_directory: impl Into<PathBuf>,
    settings: VideoExtractionSettings,
) -> Result<VideoFrames, VideoInputError> {
    if settings.width == 0 || settings.height == 0 || settings.max_frames == 0 {
        return Err(VideoInputError::InvalidSettings);
    }
    let duration_seconds = probe_duration(video)?;
    load_decoded_frame_cache_with_duration(&decoded_directory.into(), settings, duration_seconds)
}

fn probe_duration(video: &Path) -> Result<f64, VideoInputError> {
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
    Ok(duration_seconds)
}

fn probe_geometry(video: &Path) -> Result<VideoGeometry, VideoInputError> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
        ])
        .arg(video)
        .output()
        .map_err(|error| VideoInputError::Probe(error.to_string()))?;
    if !output.status.success() {
        return Err(VideoInputError::Probe(
            String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        ));
    }
    let dimensions = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let Some((width, height)) = dimensions.split_once('x') else {
        return Err(VideoInputError::Probe(format!(
            "ffprobe did not return video dimensions: {dimensions:?}"
        )));
    };
    let width = width.parse::<usize>().ok();
    let height = height.parse::<usize>().ok();
    match (
        width.filter(|value| *value > 0),
        height.filter(|value| *value > 0),
    ) {
        (Some(width), Some(height)) => Ok(VideoGeometry { width, height }),
        _ => Err(VideoInputError::Probe(format!(
            "ffprobe returned invalid video dimensions: {dimensions:?}"
        ))),
    }
}

fn load_decoded_frame_cache_with_duration(
    decoded_directory: &Path,
    settings: VideoExtractionSettings,
    duration_seconds: f64,
) -> Result<VideoFrames, VideoInputError> {
    let frames = load_decoded_rgb24_cache(decoded_directory, settings)?;
    let capture_quality = assess_capture_quality(&frames);
    Ok(VideoFrames {
        duration_seconds,
        frames,
        decoded_directory: decoded_directory.to_path_buf(),
        capture_quality,
    })
}

/// Loads and validates canonical RGB24 PPM frames without probing the original
/// video. This supports deterministic diagnostic replay after a capture file is
/// unavailable; normal reconstruction still records the source duration.
pub fn load_decoded_rgb24_cache(
    decoded_directory: &Path,
    settings: VideoExtractionSettings,
) -> Result<Vec<OwnedFrame>, VideoInputError> {
    if !decoded_directory.is_dir() {
        return Err(VideoInputError::InvalidCache {
            path: decoded_directory.to_path_buf(),
            reason: "directory is missing".to_owned(),
        });
    }
    let mut paths = fs::read_dir(decoded_directory)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "ppm"))
        .collect::<Vec<_>>();
    paths.sort_unstable();
    let frames = paths
        .iter()
        .map(|path| read_ppm_rgb(path))
        .collect::<Result<Vec<_>, _>>()?;
    if frames.is_empty() || frames.len() > settings.max_frames {
        return Err(VideoInputError::InvalidCache {
            path: decoded_directory.to_path_buf(),
            reason: format!(
                "expected 1..={} frames, found {}",
                settings.max_frames,
                frames.len()
            ),
        });
    }
    if frames
        .iter()
        .any(|frame| frame.width != settings.width || frame.height != settings.height)
    {
        return Err(VideoInputError::InvalidCache {
            path: decoded_directory.to_path_buf(),
            reason: format!("expected {}x{} RGB frames", settings.width, settings.height),
        });
    }
    Ok(frames)
}

/// Computes a deterministic low-cost warning signal before expensive model
/// work. Static footage is insufficient for a coherent walk-through world;
/// the product reports it but leaves deliberate diagnostic runs possible.
pub fn assess_capture_quality(frames: &[OwnedFrame]) -> CaptureQuality {
    if frames.len() < 2 {
        return CaptureQuality {
            disposition: CaptureDisposition::Recapture,
            frame_count: frames.len(),
            mean_adjacent_luma_delta: 0.0,
        };
    }
    let mut deltas = Vec::with_capacity(frames.len() - 1);
    for pair in frames.windows(2) {
        let left = &pair[0];
        let right = &pair[1];
        if left.width != right.width || left.height != right.height {
            return CaptureQuality {
                disposition: CaptureDisposition::Recapture,
                frame_count: frames.len(),
                mean_adjacent_luma_delta: 0.0,
            };
        }
        let mut total = 0.0;
        for (a, b) in left
            .rgb_hwc_u8
            .chunks_exact(3)
            .zip(right.rgb_hwc_u8.chunks_exact(3))
        {
            let luma = |rgb: &[u8]| {
                (0.2126 * f32::from(rgb[0])
                    + 0.7152 * f32::from(rgb[1])
                    + 0.0722 * f32::from(rgb[2]))
                    / 255.0
            };
            total += (luma(a) - luma(b)).abs();
        }
        deltas.push(total / (left.width * left.height) as f32);
    }
    let mean = deltas.iter().sum::<f32>() / deltas.len() as f32;
    let disposition = if mean < 0.002 {
        CaptureDisposition::Recapture
    } else if mean < 0.01 {
        CaptureDisposition::Review
    } else {
        CaptureDisposition::Ready
    };
    CaptureQuality {
        disposition,
        frame_count: frames.len(),
        mean_adjacent_luma_delta: mean,
    }
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

    #[test]
    fn decode_cache_requires_the_locked_raster_and_is_reusable() {
        let root =
            std::env::temp_dir().join(format!("vestra-decode-cache-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("frame-000001.ppm"),
            b"P6\n2 1\n255\n\x01\x02\x03\x04\x05\x06",
        )
        .unwrap();
        let settings = VideoExtractionSettings {
            width: 2,
            height: 1,
            candidate_fps: 1.0,
            max_frames: 2,
        };
        let cached = load_decoded_frame_cache_with_duration(&root, settings, 3.0).unwrap();
        assert_eq!(cached.frames.len(), 1);
        assert_eq!(cached.decoded_directory, root);
        assert!(matches!(
            load_decoded_frame_cache_with_duration(
                &root,
                VideoExtractionSettings {
                    width: 1,
                    ..settings
                },
                3.0,
            ),
            Err(VideoInputError::InvalidCache { .. })
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn static_capture_is_marked_for_recapture() {
        let frame = OwnedFrame {
            rgb_hwc_u8: vec![10, 20, 30],
            width: 1,
            height: 1,
        };
        assert_eq!(
            assess_capture_quality(&[frame.clone(), frame]).disposition,
            CaptureDisposition::Recapture
        );
    }

    #[test]
    fn changing_capture_is_ready_for_processing() {
        let dark = OwnedFrame {
            rgb_hwc_u8: vec![0, 0, 0],
            width: 1,
            height: 1,
        };
        let bright = OwnedFrame {
            rgb_hwc_u8: vec![255, 255, 255],
            width: 1,
            height: 1,
        };
        assert_eq!(
            assess_capture_quality(&[dark, bright]).disposition,
            CaptureDisposition::Ready
        );
    }

    #[test]
    fn landscape_16_by_9_is_center_cropped_to_three_by_two_before_resize() {
        let filter = center_crop_filter(
            12.0,
            VideoExtractionSettings::default(),
            VideoGeometry {
                width: 1920,
                height: 1080,
            },
        )
        .unwrap();
        assert_eq!(
            filter,
            "fps=8.000000000,crop=1620:1080:150:0,scale=504:336:flags=lanczos"
        );
    }

    #[test]
    fn crop_filter_keeps_an_already_three_by_two_source_unchanged() {
        let filter = center_crop_filter(
            1.0,
            VideoExtractionSettings::default(),
            VideoGeometry {
                width: 1500,
                height: 1000,
            },
        )
        .unwrap();
        assert!(filter.contains("crop=1500:1000:0:0"));
    }
}
