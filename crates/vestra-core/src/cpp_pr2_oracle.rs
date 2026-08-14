//! Strict binary interchange for the pinned C++ PR #2 streaming oracle.
//!
//! The production pipeline does not depend on the C++ project. These types are
//! deliberately diagnostic-only: they make an exact pre-voxel stitch comparison
//! possible without weakening Vestra's scene format or quality policy.

use std::io::{Read, Write};

use crate::WindowSettings;

const FIXTURE_MAGIC: [u8; 4] = *b"VPS1";
const OUTPUT_MAGIC: [u8; 4] = *b"VPO1";
const CAPI_STREAM_MAGIC: [u8; 4] = *b"CPS1";
const CAPI_STREAM_VERSION: u32 = 1;
const MULTIVIEW_MAGIC: [u8; 4] = *b"MVO1";
const MULTIVIEW_VERSION: u32 = 1;
/// Version 2 established the base sequential streaming oracle. Version 3 adds
/// explicit opt-in geometry branches while retaining a reader for the durable
/// V2 evidence artifacts.
const FIXTURE_VERSION: u32 = 3;
const LEGACY_FIXTURE_VERSION: u32 = 2;
const OUTPUT_VERSION: u32 = 1;
const BRANCH_ICP_REFINE: u32 = 1 << 0;
const BRANCH_LOOP_CLOSE: u32 = 1 << 1;
const SUPPORTED_BRANCHES: u32 = BRANCH_ICP_REFINE | BRANCH_LOOP_CLOSE;

/// Optional C++ PR #2 geometry phases to include in one oracle run.
///
/// Metric scale and model-dependent branches deliberately remain absent: this
/// interchange isolates the relative-scale streaming geometry that can be
/// reproduced from recorded model outputs alone.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CppPr2StreamBranches {
    pub icp_refine: bool,
    pub loop_close: bool,
}

impl CppPr2StreamBranches {
    const fn bits(self) -> u32 {
        (self.icp_refine as u32) * BRANCH_ICP_REFINE | (self.loop_close as u32) * BRANCH_LOOP_CLOSE
    }

    fn from_bits(bits: u32) -> Result<Self, CppPr2OracleError> {
        if bits & !SUPPORTED_BRANCHES != 0 {
            return Err(CppPr2OracleError::InvalidHeader);
        }
        Ok(Self {
            icp_refine: bits & BRANCH_ICP_REFINE != 0,
            loop_close: bits & BRANCH_LOOP_CLOSE != 0,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CppPr2Frame {
    pub intrinsics: [f32; 9],
    /// Row-major world-to-camera 3×4 matrix.
    pub world_to_camera: [f32; 12],
    pub depth: Vec<f32>,
    pub confidence: Vec<f32>,
    pub rgb_hwc_u8: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CppPr2Fixture {
    pub frame_count: usize,
    pub width: usize,
    pub height: usize,
    pub windows: WindowSettings,
    pub confidence_percentile: f64,
    pub point_size: f32,
    pub minimum_overlap_points: usize,
    /// Optional host-geometry phases executed by the pinned C++ stream oracle.
    /// V2 artifacts decode as [`CppPr2StreamBranches::default`].
    pub branches: CppPr2StreamBranches,
    /// One ordered inference result set per multi-view window. Overlap views
    /// are intentionally repeated because DA3 re-infers them per window.
    pub window_views: Vec<Vec<CppPr2Frame>>,
}

/// Raw `da::StreamCloud` output from the exact C++ stitcher, before optional
/// server-side voxel/TSDF processing.
#[derive(Debug, Clone, PartialEq)]
pub struct CppPr2StreamOutput {
    pub frame_count: usize,
    pub width: usize,
    pub height: usize,
    pub warnings: i32,
    pub loops_found: i32,
    pub metric_scale: f32,
    pub xyz: Vec<f32>,
    pub rgb: Vec<u8>,
    pub radius: Vec<f32>,
    pub counts: Vec<i32>,
    pub window_pos: Vec<f32>,
    pub window_mid_frame: Vec<i32>,
    pub frame_pos: Vec<f32>,
    pub frame_fwd: Vec<f32>,
}

/// Output from Vestra's small C++ C-API stream harness.
///
/// Unlike `VPO1`, this intentionally contains only the public C-API cloud and
/// per-frame camera trajectory. It is the end-to-end differential boundary
/// for a real DA3 model run over the same decoded RGB frames.
#[derive(Debug, Clone, PartialEq)]
pub struct CppPr2CapiStreamOutput {
    pub frame_count: usize,
    pub xyz: Vec<f32>,
    pub rgb: Vec<u8>,
    pub radius: Vec<f32>,
    pub counts: Vec<i32>,
    pub frame_pos: Vec<f32>,
    pub frame_fwd: Vec<f32>,
}

/// One C++ `Engine::depth_pose_multi` result before any geometry work.
#[derive(Debug, Clone, PartialEq)]
pub struct CppPr2MultiViewView {
    pub depth: Vec<f32>,
    pub confidence: Vec<f32>,
    pub world_to_camera: [f32; 12],
    pub intrinsics: [f32; 9],
}

/// Exact C++ multi-view model boundary recorded by `MVO1`.
#[derive(Debug, Clone, PartialEq)]
pub struct CppPr2MultiViewOutput {
    pub frame_count: usize,
    pub windows: WindowSettings,
    pub views: Vec<Vec<CppPr2MultiViewView>>,
    pub width: usize,
    pub height: usize,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CppPr2OracleError {
    #[error("oracle I/O failed: {0}")]
    Io(String),
    #[error("unexpected oracle magic")]
    Magic,
    #[error("unsupported oracle version {0}")]
    Version(u32),
    #[error("invalid oracle dimensions or parameter values")]
    InvalidHeader,
    #[error("oracle payload length is inconsistent")]
    InconsistentPayload,
    #[error("oracle input contains trailing bytes")]
    TrailingBytes,
}

impl CppPr2Fixture {
    /// Reads a complete window-scoped VPS1 fixture. It refuses extra bytes so
    /// the exact model evidence cannot be confused with a partial append.
    pub fn read_vps1(reader: &mut impl Read) -> Result<Self, CppPr2OracleError> {
        let mut magic = [0; 4];
        read_exact(reader, &mut magic)?;
        if magic != FIXTURE_MAGIC {
            return Err(CppPr2OracleError::Magic);
        }
        let version = read_u32(reader)?;
        if version != FIXTURE_VERSION && version != LEGACY_FIXTURE_VERSION {
            return Err(CppPr2OracleError::Version(version));
        }
        let frame_count = read_u32(reader)? as usize;
        let height = read_u32(reader)? as usize;
        let width = read_u32(reader)? as usize;
        let chunk_size = read_u32(reader)? as usize;
        let overlap = read_u32(reader)? as usize;
        let confidence_percentile = read_f64(reader)?;
        let point_size = read_f32(reader)?;
        let minimum_overlap_points = read_u32(reader)? as usize;
        let branches = if version == FIXTURE_VERSION {
            CppPr2StreamBranches::from_bits(read_u32(reader)?)?
        } else {
            CppPr2StreamBranches::default()
        };
        let window_count = read_u32(reader)? as usize;
        let windows = WindowSettings {
            chunk_size,
            overlap,
        };
        let expected_lengths = expected_window_lengths(frame_count, windows);
        if frame_count == 0 || width == 0 || height == 0 || expected_lengths.len() != window_count {
            return Err(CppPr2OracleError::InvalidHeader);
        }
        let plane = width
            .checked_mul(height)
            .ok_or(CppPr2OracleError::InvalidHeader)?;
        let rgb_len = plane
            .checked_mul(3)
            .ok_or(CppPr2OracleError::InvalidHeader)?;
        let mut window_views = Vec::with_capacity(window_count);
        for expected in expected_lengths {
            let view_count = read_u32(reader)? as usize;
            if view_count != expected {
                return Err(CppPr2OracleError::InconsistentPayload);
            }
            let mut views = Vec::with_capacity(view_count);
            for _ in 0..view_count {
                let mut intrinsics = [0.0; 9];
                let mut world_to_camera = [0.0; 12];
                for value in &mut intrinsics {
                    *value = read_f32(reader)?;
                }
                for value in &mut world_to_camera {
                    *value = read_f32(reader)?;
                }
                views.push(CppPr2Frame {
                    intrinsics,
                    world_to_camera,
                    depth: read_f32_vec(reader, plane)?,
                    confidence: read_f32_vec(reader, plane)?,
                    rgb_hwc_u8: read_u8_vec(reader, rgb_len)?,
                });
            }
            window_views.push(views);
        }
        let mut extra = [0; 1];
        if reader.read(&mut extra).map_err(io_error)? != 0 {
            return Err(CppPr2OracleError::TrailingBytes);
        }
        let fixture = Self {
            frame_count,
            width,
            height,
            windows,
            confidence_percentile,
            point_size,
            minimum_overlap_points,
            branches,
            window_views,
        };
        fixture.validate()?;
        Ok(fixture)
    }

    /// Serializes the VPS1 format consumed by `vestra_cpp_stream_fixture_dump`.
    /// Optional C++ branches are intentionally not represented by this tier.
    pub fn write_vps1(&self, writer: &mut impl Write) -> Result<(), CppPr2OracleError> {
        self.validate()?;
        writer
            .write_all(&FIXTURE_MAGIC)
            .and_then(|()| write_u32(writer, FIXTURE_VERSION))
            .and_then(|()| write_u32(writer, self.frame_count as u32))
            .and_then(|()| write_u32(writer, self.height as u32))
            .and_then(|()| write_u32(writer, self.width as u32))
            .and_then(|()| write_u32(writer, self.windows.chunk_size as u32))
            .and_then(|()| write_u32(writer, self.windows.overlap as u32))
            .and_then(|()| write_f64(writer, self.confidence_percentile))
            .and_then(|()| write_f32(writer, self.point_size))
            .and_then(|()| write_u32(writer, self.minimum_overlap_points as u32))
            .and_then(|()| write_u32(writer, self.branches.bits()))
            .and_then(|()| write_u32(writer, self.window_views.len() as u32))
            .map_err(io_error)?;
        for views in &self.window_views {
            write_u32(writer, views.len() as u32).map_err(io_error)?;
            for frame in views {
                for value in frame.intrinsics {
                    write_f32(writer, value).map_err(io_error)?;
                }
                for value in frame.world_to_camera {
                    write_f32(writer, value).map_err(io_error)?;
                }
                write_f32s(writer, &frame.depth)?;
                write_f32s(writer, &frame.confidence)?;
                writer.write_all(&frame.rgb_hwc_u8).map_err(io_error)?;
            }
        }
        Ok(())
    }

    fn validate(&self) -> Result<(), CppPr2OracleError> {
        let Some(plane) = self.width.checked_mul(self.height) else {
            return Err(CppPr2OracleError::InvalidHeader);
        };
        let expected_window_lengths = expected_window_lengths(self.frame_count, self.windows);
        if self.frame_count == 0
            || self.windows.chunk_size < 2
            || self.windows.overlap >= self.windows.chunk_size
            || !self.confidence_percentile.is_finite()
            || !(0.0..=100.0).contains(&self.confidence_percentile)
            || !self.point_size.is_finite()
            || self.point_size <= 0.0
            || self.minimum_overlap_points > u32::MAX as usize
            || self.width > u32::MAX as usize
            || self.height > u32::MAX as usize
            || self.frame_count > u32::MAX as usize
            || self.window_views.len() > u32::MAX as usize
            || self.windows.chunk_size > u32::MAX as usize
            || self.windows.overlap > u32::MAX as usize
            || plane.checked_mul(3).is_none()
        {
            return Err(CppPr2OracleError::InvalidHeader);
        }
        if self.window_views.len() != expected_window_lengths.len()
            || self
                .window_views
                .iter()
                .zip(expected_window_lengths)
                .any(|(views, expected)| views.len() != expected)
            || self.window_views.iter().flatten().any(|frame| {
                frame.depth.len() != plane
                    || frame.confidence.len() != plane
                    || frame.rgb_hwc_u8.len() != plane * 3
                    || !frame.intrinsics.iter().all(|value| value.is_finite())
                    || !frame.world_to_camera.iter().all(|value| value.is_finite())
                    || !frame.depth.iter().all(|value| value.is_finite())
                    || !frame.confidence.iter().all(|value| value.is_finite())
            })
        {
            return Err(CppPr2OracleError::InconsistentPayload);
        }
        Ok(())
    }
}

fn expected_window_lengths(frame_count: usize, settings: WindowSettings) -> Vec<usize> {
    if frame_count == 0 || settings.chunk_size < 2 || settings.overlap >= settings.chunk_size {
        return Vec::new();
    }
    let step = settings.chunk_size - settings.overlap;
    let mut lengths = Vec::new();
    for start in (0..frame_count).step_by(step) {
        lengths.push((frame_count - start).min(settings.chunk_size));
        if start + settings.chunk_size >= frame_count {
            break;
        }
    }
    lengths
}

impl CppPr2StreamOutput {
    /// Reads only a complete VPO1 artifact. Trailing bytes are rejected so an
    /// interrupted or mismatched oracle result cannot silently look valid.
    pub fn read_vpo1(reader: &mut impl Read) -> Result<Self, CppPr2OracleError> {
        let mut magic = [0; 4];
        read_exact(reader, &mut magic)?;
        if magic != OUTPUT_MAGIC {
            return Err(CppPr2OracleError::Magic);
        }
        let version = read_u32(reader)?;
        if version != OUTPUT_VERSION {
            return Err(CppPr2OracleError::Version(version));
        }
        let frame_count = read_u32(reader)? as usize;
        let height = read_u32(reader)? as usize;
        let width = read_u32(reader)? as usize;
        let point_count = read_u32(reader)? as usize;
        let window_count = read_u32(reader)? as usize;
        let warnings = read_i32(reader)?;
        let loops_found = read_i32(reader)?;
        let metric_scale = read_f32(reader)?;
        let valid_plane = width.checked_mul(height).is_some();
        if frame_count == 0 || width == 0 || height == 0 || !valid_plane {
            return Err(CppPr2OracleError::InvalidHeader);
        }
        let xyz = read_f32_vec(
            reader,
            point_count
                .checked_mul(3)
                .ok_or(CppPr2OracleError::InvalidHeader)?,
        )?;
        let rgb = read_u8_vec(
            reader,
            point_count
                .checked_mul(3)
                .ok_or(CppPr2OracleError::InvalidHeader)?,
        )?;
        let radius = read_f32_vec(reader, point_count)?;
        let counts = read_i32_vec(reader, frame_count)?;
        let window_pos = read_f32_vec(
            reader,
            window_count
                .checked_mul(3)
                .ok_or(CppPr2OracleError::InvalidHeader)?,
        )?;
        let window_mid_frame = read_i32_vec(reader, window_count)?;
        let frame_pos = read_f32_vec(
            reader,
            frame_count
                .checked_mul(3)
                .ok_or(CppPr2OracleError::InvalidHeader)?,
        )?;
        let frame_fwd = read_f32_vec(
            reader,
            frame_count
                .checked_mul(3)
                .ok_or(CppPr2OracleError::InvalidHeader)?,
        )?;
        let mut extra = [0; 1];
        if reader.read(&mut extra).map_err(io_error)? != 0 {
            return Err(CppPr2OracleError::TrailingBytes);
        }
        Ok(Self {
            frame_count,
            width,
            height,
            warnings,
            loops_found,
            metric_scale,
            xyz,
            rgb,
            radius,
            counts,
            window_pos,
            window_mid_frame,
            frame_pos,
            frame_fwd,
        })
    }
}

impl CppPr2CapiStreamOutput {
    /// Reads a complete `CPS1` stream-harness artifact and rejects trailing
    /// bytes. The format is deliberately tiny and independent of C++ headers.
    pub fn read_cps1(reader: &mut impl Read) -> Result<Self, CppPr2OracleError> {
        let mut magic = [0; 4];
        read_exact(reader, &mut magic)?;
        if magic != CAPI_STREAM_MAGIC {
            return Err(CppPr2OracleError::Magic);
        }
        let version = read_u32(reader)?;
        if version != CAPI_STREAM_VERSION {
            return Err(CppPr2OracleError::Version(version));
        }
        let frame_count = read_u32(reader)? as usize;
        let point_count = read_u32(reader)? as usize;
        let pose_frames = read_u32(reader)? as usize;
        if frame_count == 0 || pose_frames > frame_count {
            return Err(CppPr2OracleError::InvalidHeader);
        }
        let counts = read_i32_vec(reader, frame_count)?;
        if counts.iter().any(|count| *count < 0)
            || counts.iter().map(|count| *count as usize).sum::<usize>() != point_count
        {
            return Err(CppPr2OracleError::InconsistentPayload);
        }
        let xyz = read_f32_vec(
            reader,
            point_count
                .checked_mul(3)
                .ok_or(CppPr2OracleError::InvalidHeader)?,
        )?;
        let rgb = read_u8_vec(
            reader,
            point_count
                .checked_mul(3)
                .ok_or(CppPr2OracleError::InvalidHeader)?,
        )?;
        let radius = read_f32_vec(reader, point_count)?;
        let frame_pos = read_f32_vec(
            reader,
            pose_frames
                .checked_mul(3)
                .ok_or(CppPr2OracleError::InvalidHeader)?,
        )?;
        let frame_fwd = read_f32_vec(
            reader,
            pose_frames
                .checked_mul(3)
                .ok_or(CppPr2OracleError::InvalidHeader)?,
        )?;
        let mut extra = [0; 1];
        if reader.read(&mut extra).map_err(io_error)? != 0 {
            return Err(CppPr2OracleError::TrailingBytes);
        }
        Ok(Self {
            frame_count,
            xyz,
            rgb,
            radius,
            counts,
            frame_pos,
            frame_fwd,
        })
    }
}

impl CppPr2MultiViewOutput {
    /// Reads a complete C++ `Engine::depth_pose_multi` dump. The reader is
    /// intentionally strict: a mismatched schedule or tensor shape is proof
    /// that the two runtime arms did not perform the same model workload.
    pub fn read_mvo1(reader: &mut impl Read) -> Result<Self, CppPr2OracleError> {
        let mut magic = [0; 4];
        read_exact(reader, &mut magic)?;
        if magic != MULTIVIEW_MAGIC {
            return Err(CppPr2OracleError::Magic);
        }
        let version = read_u32(reader)?;
        if version != MULTIVIEW_VERSION {
            return Err(CppPr2OracleError::Version(version));
        }
        let frame_count = read_u32(reader)? as usize;
        let chunk_size = read_u32(reader)? as usize;
        let overlap = read_u32(reader)? as usize;
        let window_count = read_u32(reader)? as usize;
        if frame_count == 0 || chunk_size < 2 || overlap >= chunk_size || window_count == 0 {
            return Err(CppPr2OracleError::InvalidHeader);
        }
        let expected = window_lengths(
            frame_count,
            WindowSettings {
                chunk_size,
                overlap,
            },
        );
        if expected.len() != window_count {
            return Err(CppPr2OracleError::InconsistentPayload);
        }
        let mut views = Vec::with_capacity(window_count);
        let mut dimensions = None;
        for (window_index, expected_views) in expected.into_iter().enumerate() {
            let start = read_u32(reader)? as usize;
            let view_count = read_u32(reader)? as usize;
            let height = read_u32(reader)? as usize;
            let width = read_u32(reader)? as usize;
            let expected_start = window_index * (chunk_size - overlap);
            let pixels = height
                .checked_mul(width)
                .ok_or(CppPr2OracleError::InvalidHeader)?;
            if start != expected_start || view_count != expected_views || pixels == 0 {
                return Err(CppPr2OracleError::InconsistentPayload);
            }
            match dimensions {
                Some(existing) if existing != (width, height) => {
                    return Err(CppPr2OracleError::InconsistentPayload);
                }
                None => dimensions = Some((width, height)),
                _ => {}
            }
            let mut window_views = Vec::with_capacity(view_count);
            for _ in 0..view_count {
                let depth = read_f32_vec(reader, pixels)?;
                let confidence = read_f32_vec(reader, pixels)?;
                let world_to_camera = read_f32_array::<12>(reader)?;
                let intrinsics = read_f32_array::<9>(reader)?;
                window_views.push(CppPr2MultiViewView {
                    depth,
                    confidence,
                    world_to_camera,
                    intrinsics,
                });
            }
            views.push(window_views);
        }
        let mut extra = [0; 1];
        if reader.read(&mut extra).map_err(io_error)? != 0 {
            return Err(CppPr2OracleError::TrailingBytes);
        }
        let Some((width, height)) = dimensions else {
            return Err(CppPr2OracleError::InvalidHeader);
        };
        Ok(Self {
            frame_count,
            windows: WindowSettings {
                chunk_size,
                overlap,
            },
            views,
            width,
            height,
        })
    }
}

fn io_error(error: std::io::Error) -> CppPr2OracleError {
    CppPr2OracleError::Io(error.to_string())
}

fn read_exact(reader: &mut impl Read, bytes: &mut [u8]) -> Result<(), CppPr2OracleError> {
    reader.read_exact(bytes).map_err(io_error)
}

fn read_u32(reader: &mut impl Read) -> Result<u32, CppPr2OracleError> {
    let mut bytes = [0; 4];
    read_exact(reader, &mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}
fn read_i32(reader: &mut impl Read) -> Result<i32, CppPr2OracleError> {
    let mut bytes = [0; 4];
    read_exact(reader, &mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}
fn read_f32(reader: &mut impl Read) -> Result<f32, CppPr2OracleError> {
    let mut bytes = [0; 4];
    read_exact(reader, &mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}
fn read_f64(reader: &mut impl Read) -> Result<f64, CppPr2OracleError> {
    let mut bytes = [0; 8];
    read_exact(reader, &mut bytes)?;
    Ok(f64::from_le_bytes(bytes))
}
fn write_u32(writer: &mut impl Write, value: u32) -> std::io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}
fn write_f32(writer: &mut impl Write, value: f32) -> std::io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}
fn write_f64(writer: &mut impl Write, value: f64) -> std::io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}
fn write_f32s(writer: &mut impl Write, values: &[f32]) -> Result<(), CppPr2OracleError> {
    for value in values {
        write_f32(writer, *value).map_err(io_error)?;
    }
    Ok(())
}
fn read_u8_vec(reader: &mut impl Read, count: usize) -> Result<Vec<u8>, CppPr2OracleError> {
    let mut values = vec![0; count];
    read_exact(reader, &mut values)?;
    Ok(values)
}
fn read_f32_vec(reader: &mut impl Read, count: usize) -> Result<Vec<f32>, CppPr2OracleError> {
    (0..count).map(|_| read_f32(reader)).collect()
}
fn read_f32_array<const N: usize>(reader: &mut impl Read) -> Result<[f32; N], CppPr2OracleError> {
    let values = read_f32_vec(reader, N)?;
    values
        .try_into()
        .map_err(|_| CppPr2OracleError::InconsistentPayload)
}

fn window_lengths(frame_count: usize, windows: WindowSettings) -> Vec<usize> {
    let step = windows.chunk_size - windows.overlap;
    let mut lengths = Vec::new();
    for start in (0..frame_count).step_by(step) {
        lengths.push((frame_count - start).min(windows.chunk_size));
        if start + windows.chunk_size >= frame_count {
            break;
        }
    }
    lengths
}
fn read_i32_vec(reader: &mut impl Read, count: usize) -> Result<Vec<i32>, CppPr2OracleError> {
    (0..count).map(|_| read_i32(reader)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> CppPr2Fixture {
        let frame = CppPr2Frame {
            intrinsics: [1.0, 0.0, 0.5, 0.0, 1.0, 0.5, 0.0, 0.0, 1.0],
            world_to_camera: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            depth: vec![2.0; 4],
            confidence: vec![1.0; 4],
            rgb_hwc_u8: vec![2; 12],
        };
        CppPr2Fixture {
            frame_count: 3,
            width: 2,
            height: 2,
            windows: WindowSettings {
                chunk_size: 2,
                overlap: 1,
            },
            confidence_percentile: 55.0,
            point_size: 1.2,
            minimum_overlap_points: 3,
            branches: CppPr2StreamBranches::default(),
            window_views: vec![
                vec![frame.clone(), frame.clone()],
                vec![frame.clone(), frame],
            ],
        }
    }

    #[test]
    fn vps1_is_little_endian_and_rejects_invalid_pixels() {
        let mut bytes = Vec::new();
        let original = fixture();
        original.write_vps1(&mut bytes).unwrap();
        assert_eq!(&bytes[..4], b"VPS1");
        assert_eq!(&bytes[4..8], &FIXTURE_VERSION.to_le_bytes());
        assert_eq!(
            CppPr2Fixture::read_vps1(&mut bytes.as_slice()).unwrap(),
            original
        );
        let mut invalid = fixture();
        invalid.window_views[0][0].depth.pop();
        assert_eq!(
            invalid.write_vps1(&mut Vec::new()),
            Err(CppPr2OracleError::InconsistentPayload)
        );
    }

    #[test]
    fn vps3_preserves_opt_in_geometry_branches() {
        let mut fixture = fixture();
        fixture.branches = CppPr2StreamBranches {
            icp_refine: true,
            loop_close: true,
        };
        let mut bytes = Vec::new();
        fixture.write_vps1(&mut bytes).unwrap();
        assert_eq!(&bytes[4..8], &FIXTURE_VERSION.to_le_bytes());
        assert_eq!(
            CppPr2Fixture::read_vps1(&mut bytes.as_slice())
                .unwrap()
                .branches,
            fixture.branches
        );
    }

    #[test]
    fn cps1_reader_preserves_capi_cloud_and_rejects_count_mismatch() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&CAPI_STREAM_MAGIC);
        bytes.extend_from_slice(&CAPI_STREAM_VERSION.to_le_bytes());
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        for count in [1_i32, 1] {
            bytes.extend_from_slice(&count.to_le_bytes());
        }
        for value in [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&[7, 8, 9, 10, 11, 12]);
        for value in [0.1_f32, 0.2] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let parsed = CppPr2CapiStreamOutput::read_cps1(&mut bytes.as_slice()).unwrap();
        assert_eq!(parsed.counts, vec![1, 1]);
        assert_eq!(parsed.xyz, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(parsed.rgb, vec![7, 8, 9, 10, 11, 12]);
        assert_eq!(parsed.radius, vec![0.1, 0.2]);
        let mut invalid = bytes;
        invalid[20..24].copy_from_slice(&2_i32.to_le_bytes());
        assert_eq!(
            CppPr2CapiStreamOutput::read_cps1(&mut invalid.as_slice()),
            Err(CppPr2OracleError::InconsistentPayload)
        );
    }

    #[test]
    fn vps2_decodes_without_optional_geometry_branches() {
        let fixture = fixture();
        let mut v3 = Vec::new();
        fixture.write_vps1(&mut v3).unwrap();
        let mut v2 = Vec::with_capacity(v3.len() - 4);
        v2.extend_from_slice(&v3[..4]);
        v2.extend_from_slice(&LEGACY_FIXTURE_VERSION.to_le_bytes());
        // The V3-only branch bitmap follows the common 44-byte header.
        v2.extend_from_slice(&v3[8..44]);
        v2.extend_from_slice(&v3[48..]);
        let decoded = CppPr2Fixture::read_vps1(&mut v2.as_slice()).unwrap();
        assert_eq!(decoded.branches, CppPr2StreamBranches::default());
        assert_eq!(decoded.window_views, fixture.window_views);
    }

    #[test]
    fn vpo1_reader_rejects_trailing_bytes() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"VPO1");
        for value in [OUTPUT_VERSION, 1, 1, 1, 0, 0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&0_i32.to_le_bytes());
        bytes.extend_from_slice(&0_i32.to_le_bytes());
        bytes.extend_from_slice(&0.0_f32.to_le_bytes());
        // `counts`, `frame_pos`, and `frame_fwd` for the one declared input frame.
        bytes.extend_from_slice(&[0_u8; 28]);
        bytes.extend_from_slice(&[9]);
        assert_eq!(
            CppPr2StreamOutput::read_vpo1(&mut bytes.as_slice()),
            Err(CppPr2OracleError::TrailingBytes)
        );
    }
}
