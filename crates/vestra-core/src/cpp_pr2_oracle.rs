//! Strict binary interchange for the pinned C++ PR #2 streaming oracle.
//!
//! The production pipeline does not depend on the C++ project. These types are
//! deliberately diagnostic-only: they make an exact pre-voxel stitch comparison
//! possible without weakening Vestra's scene format or quality policy.

use std::io::{Read, Write};

use crate::WindowSettings;

const FIXTURE_MAGIC: [u8; 4] = *b"VPS1";
const OUTPUT_MAGIC: [u8; 4] = *b"VPO1";
const VERSION: u32 = 1;

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
    pub width: usize,
    pub height: usize,
    pub windows: WindowSettings,
    pub confidence_percentile: f64,
    pub point_size: f32,
    pub minimum_overlap_points: usize,
    pub frames: Vec<CppPr2Frame>,
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
    /// Serializes the VPS1 format consumed by `vestra_cpp_stream_fixture_dump`.
    /// Optional C++ branches are intentionally not represented by this tier.
    pub fn write_vps1(&self, writer: &mut impl Write) -> Result<(), CppPr2OracleError> {
        self.validate()?;
        writer
            .write_all(&FIXTURE_MAGIC)
            .and_then(|()| write_u32(writer, VERSION))
            .and_then(|()| write_u32(writer, self.frames.len() as u32))
            .and_then(|()| write_u32(writer, self.height as u32))
            .and_then(|()| write_u32(writer, self.width as u32))
            .and_then(|()| write_u32(writer, self.windows.chunk_size as u32))
            .and_then(|()| write_u32(writer, self.windows.overlap as u32))
            .and_then(|()| write_f64(writer, self.confidence_percentile))
            .and_then(|()| write_f32(writer, self.point_size))
            .and_then(|()| write_u32(writer, self.minimum_overlap_points as u32))
            .map_err(io_error)?;
        for frame in &self.frames {
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
        Ok(())
    }

    fn validate(&self) -> Result<(), CppPr2OracleError> {
        let Some(plane) = self.width.checked_mul(self.height) else {
            return Err(CppPr2OracleError::InvalidHeader);
        };
        if self.frames.is_empty()
            || self.windows.chunk_size < 2
            || self.windows.overlap >= self.windows.chunk_size
            || !self.confidence_percentile.is_finite()
            || !(0.0..=100.0).contains(&self.confidence_percentile)
            || !self.point_size.is_finite()
            || self.point_size <= 0.0
            || self.minimum_overlap_points > u32::MAX as usize
            || self.width > u32::MAX as usize
            || self.height > u32::MAX as usize
            || self.frames.len() > u32::MAX as usize
            || self.windows.chunk_size > u32::MAX as usize
            || self.windows.overlap > u32::MAX as usize
            || plane.checked_mul(3).is_none()
        {
            return Err(CppPr2OracleError::InvalidHeader);
        }
        if self.frames.iter().any(|frame| {
            frame.depth.len() != plane
                || frame.confidence.len() != plane
                || frame.rgb_hwc_u8.len() != plane * 3
                || !frame.intrinsics.iter().all(|value| value.is_finite())
                || !frame.world_to_camera.iter().all(|value| value.is_finite())
                || !frame.depth.iter().all(|value| value.is_finite())
                || !frame.confidence.iter().all(|value| value.is_finite())
        }) {
            return Err(CppPr2OracleError::InconsistentPayload);
        }
        Ok(())
    }
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
        if version != VERSION {
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
            width: 2,
            height: 2,
            windows: WindowSettings {
                chunk_size: 2,
                overlap: 1,
            },
            confidence_percentile: 55.0,
            point_size: 1.2,
            minimum_overlap_points: 3,
            frames: vec![frame.clone(), frame],
        }
    }

    #[test]
    fn vps1_is_little_endian_and_rejects_invalid_pixels() {
        let mut bytes = Vec::new();
        fixture().write_vps1(&mut bytes).unwrap();
        assert_eq!(&bytes[..4], b"VPS1");
        assert_eq!(&bytes[4..8], &1_u32.to_le_bytes());
        let mut invalid = fixture();
        invalid.frames[0].depth.pop();
        assert_eq!(
            invalid.write_vps1(&mut Vec::new()),
            Err(CppPr2OracleError::InconsistentPayload)
        );
    }

    #[test]
    fn vpo1_reader_rejects_trailing_bytes() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"VPO1");
        for value in [VERSION, 1, 1, 1, 0, 0] {
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
