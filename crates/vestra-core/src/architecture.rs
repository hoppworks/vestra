//! Conservative architectural-plane extraction for an already global Vestra world.
//!
//! This module deliberately creates a separate **fused** interpretation.  It
//! never mutates the measured/MVS surfel layer and it never fills an arbitrary
//! rectangle between plane extrema: every emitted surface cell needs direct
//! planar support.  Consequently a door-sized absence in a wall remains an
//! opening rather than becoming invented geometry.

use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::Path,
};

use serde::{Deserialize, Serialize};

use crate::FusedPoint;
use crate::{ColmapCameraModel, PoseSolution, RasterManifest};

/// Semantic categories that can select architectural geometry.  The integer
/// representation is deliberately stable because masks are compact evidence
/// assets rather than a Rust-only in-memory detail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(u8)]
pub enum ArchitectureClass {
    Unknown = 0,
    Floor = 1,
    Wall = 2,
    CeilingOrRoof = 3,
    DoorOrOpening = 4,
    Window = 5,
    NonArchitectural = 6,
}

impl ArchitectureClass {
    fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Unknown),
            1 => Some(Self::Floor),
            2 => Some(Self::Wall),
            3 => Some(Self::CeilingOrRoof),
            4 => Some(Self::DoorOrOpening),
            5 => Some(Self::Window),
            6 => Some(Self::NonArchitectural),
            _ => None,
        }
    }
    #[must_use]
    pub const fn supports_surface(self) -> bool {
        matches!(self, Self::Floor | Self::Wall | Self::CeilingOrRoof)
    }

    #[must_use]
    pub const fn is_opening(self) -> bool {
        matches!(self, Self::DoorOrOpening | Self::Window)
    }
}

/// One decoded source-frame mask. `classes` and `confidences` are dense,
/// row-major rasters with exactly `width * height` values.  They are evidence
/// that can select measured geometry; they are never geometry by themselves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArchitectureSemanticFrame {
    pub frame_index: usize,
    pub width: usize,
    pub height: usize,
    pub classes: Vec<ArchitectureClass>,
    pub confidences: Vec<f32>,
}

impl ArchitectureSemanticFrame {
    /// Returns the class and confidence for an exact decoded-raster pixel.
    #[must_use]
    pub fn sample(&self, x: usize, y: usize) -> Option<(ArchitectureClass, f32)> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let index = y.checked_mul(self.width)?.checked_add(x)?;
        let class = *self.classes.get(index)?;
        let confidence = *self.confidences.get(index)?;
        (confidence.is_finite() && (0.0..=1.0).contains(&confidence)).then_some((class, confidence))
    }

    #[must_use]
    pub fn is_well_formed(&self) -> bool {
        self.width > 0
            && self.height > 0
            && self.classes.len() == self.width.saturating_mul(self.height)
            && self.confidences.len() == self.classes.len()
            && self
                .confidences
                .iter()
                .all(|confidence| confidence.is_finite() && (0.0..=1.0).contains(confidence))
    }
}

/// Versioned, model-provenanced semantic evidence sidecar.  It intentionally
/// records the licence string supplied by the runner so a checkpoint cannot
/// silently become a product dependency.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArchitectureSemanticEvidence {
    pub schema: String,
    pub runner: String,
    pub model_id: String,
    pub model_revision: String,
    pub model_license: String,
    pub frames: Vec<ArchitectureSemanticFrame>,
}

impl ArchitectureSemanticEvidence {
    pub const SCHEMA: &'static str = "vestra.architecture-semantics/v1";

    /// Validates the sidecar independently from any reconstruction so a bad
    /// mask cannot be mistaken for evidence of a wall or opening.
    pub fn validate(&self) -> Result<(), ArchitectureEvidenceError> {
        if self.schema != Self::SCHEMA {
            return Err(ArchitectureEvidenceError::UnsupportedSchema {
                actual: self.schema.clone(),
            });
        }
        if self.runner.trim().is_empty()
            || self.model_id.trim().is_empty()
            || self.model_revision.trim().is_empty()
            || self.model_license.trim().is_empty()
        {
            return Err(ArchitectureEvidenceError::MissingProvenance);
        }
        let mut previous = None;
        for frame in &self.frames {
            if !frame.is_well_formed() {
                return Err(ArchitectureEvidenceError::MalformedFrame {
                    frame_index: frame.frame_index,
                });
            }
            if previous.is_some_and(|index| index >= frame.frame_index) {
                return Err(ArchitectureEvidenceError::UnorderedFrames);
            }
            previous = Some(frame.frame_index);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ArchitectureEvidenceError {
    #[error("unsupported architecture semantic-evidence schema `{actual}`")]
    UnsupportedSchema { actual: String },
    #[error("semantic evidence is missing runner, model, revision, or licence provenance")]
    MissingProvenance,
    #[error("semantic evidence frame {frame_index} has incompatible raster data")]
    MalformedFrame { frame_index: usize },
    #[error("semantic evidence frames must be strictly ordered by frame index")]
    UnorderedFrames,
    #[error("failed to read semantic raster payload: {0}")]
    Io(String),
    #[error("semantic raster payload is malformed: {0}")]
    InvalidBinary(String),
}

/// Compact, dependency-free raster payload emitted by
/// `run_architecture_semantics.py`.  This is intentionally separate from the
/// JSON manifest: loading thousands of dense labels must not require parsing a
/// gigantic JSON document.
#[derive(Debug, Clone, PartialEq)]
pub struct ArchitectureSemanticVolume {
    frame_indices: Vec<usize>,
    width: usize,
    height: usize,
    classes: Vec<u8>,
    confidences: Vec<u8>,
}

impl ArchitectureSemanticVolume {
    const MAGIC: &'static [u8] = b"VSEM1";

    pub fn read(path: impl AsRef<Path>) -> Result<Self, ArchitectureEvidenceError> {
        let bytes =
            fs::read(path).map_err(|error| ArchitectureEvidenceError::Io(error.to_string()))?;
        if !bytes.starts_with(Self::MAGIC) || bytes.len() < 17 {
            return Err(ArchitectureEvidenceError::InvalidBinary(
                "missing VSEM1 header".to_owned(),
            ));
        }
        let u32_at = |offset: usize| {
            bytes
                .get(offset..offset + 4)
                .and_then(|slice| <&[u8; 4]>::try_from(slice).ok())
                .map(|slice| u32::from_le_bytes(*slice) as usize)
        };
        let frame_count = u32_at(5).ok_or_else(|| {
            ArchitectureEvidenceError::InvalidBinary("truncated frame count".to_owned())
        })?;
        let width = u32_at(9).ok_or_else(|| {
            ArchitectureEvidenceError::InvalidBinary("truncated width".to_owned())
        })?;
        let height = u32_at(13).ok_or_else(|| {
            ArchitectureEvidenceError::InvalidBinary("truncated height".to_owned())
        })?;
        let pixels = width.checked_mul(height).ok_or_else(|| {
            ArchitectureEvidenceError::InvalidBinary("raster dimensions overflow".to_owned())
        })?;
        let index_bytes = frame_count.checked_mul(4).ok_or_else(|| {
            ArchitectureEvidenceError::InvalidBinary("frame-index count overflow".to_owned())
        })?;
        let data_bytes = frame_count.checked_mul(pixels).ok_or_else(|| {
            ArchitectureEvidenceError::InvalidBinary("raster count overflow".to_owned())
        })?;
        let index_start = 17_usize;
        let classes_start = index_start.checked_add(index_bytes).ok_or_else(|| {
            ArchitectureEvidenceError::InvalidBinary("frame-index offset overflow".to_owned())
        })?;
        let confidence_start = classes_start.checked_add(data_bytes).ok_or_else(|| {
            ArchitectureEvidenceError::InvalidBinary("class offset overflow".to_owned())
        })?;
        let end = confidence_start.checked_add(data_bytes).ok_or_else(|| {
            ArchitectureEvidenceError::InvalidBinary("confidence offset overflow".to_owned())
        })?;
        if width == 0 || height == 0 || bytes.len() != end {
            return Err(ArchitectureEvidenceError::InvalidBinary(
                "unexpected binary length or zero raster dimension".to_owned(),
            ));
        }
        let mut frame_indices = Vec::with_capacity(frame_count);
        for offset in (index_start..classes_start).step_by(4) {
            frame_indices.push(u32_at(offset).ok_or_else(|| {
                ArchitectureEvidenceError::InvalidBinary("truncated frame index".to_owned())
            })?);
        }
        if frame_indices.windows(2).any(|pair| pair[0] >= pair[1])
            || bytes[classes_start..confidence_start]
                .iter()
                .any(|code| ArchitectureClass::from_code(*code).is_none())
        {
            return Err(ArchitectureEvidenceError::InvalidBinary(
                "unordered frame indices or unknown class code".to_owned(),
            ));
        }
        Ok(Self {
            frame_indices,
            width,
            height,
            classes: bytes[classes_start..confidence_start].to_vec(),
            confidences: bytes[confidence_start..end].to_vec(),
        })
    }

    #[must_use]
    pub fn sample(
        &self,
        frame_index: usize,
        x: usize,
        y: usize,
    ) -> Option<(ArchitectureClass, f32)> {
        let frame = self.frame_indices.binary_search(&frame_index).ok()?;
        let pixel = y.checked_mul(self.width)?.checked_add(x)?;
        if x >= self.width || y >= self.height {
            return None;
        }
        let index = frame
            .checked_mul(self.width.checked_mul(self.height)?)?
            .checked_add(pixel)?;
        Some((
            ArchitectureClass::from_code(*self.classes.get(index)?)?,
            f32::from(*self.confidences.get(index)?) / 255.0,
        ))
    }

    #[must_use]
    pub const fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }
}

/// Settings are expressed as fractions of the scene diagonal because a
/// Vestra world is relative-scale until it has an independently verified
/// metric anchor.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ArchitectureSettings {
    /// Bound the source evidence examined by the deterministic extractor.
    pub maximum_source_points: usize,
    /// Maximum planes emitted into the architecture layer.
    pub maximum_planes: usize,
    /// Plane inlier distance as a fraction of the scene diagonal.
    pub plane_distance_fraction: f32,
    /// Minimum absolute cosine between a source normal and a fitted plane.
    pub minimum_normal_alignment: f32,
    /// Minimum independently sampled support points for a visible plane.
    pub minimum_plane_support: usize,
    /// Deterministic RANSAC hypotheses evaluated for each dominant plane.
    pub ransac_trials_per_plane: usize,
    /// Minimum support samples in a generated surface cell.  Empty cells are
    /// intentionally preserved as holes/openings.
    pub minimum_cell_support: usize,
    /// Grid-cell edge as a fraction of the scene diagonal.
    pub cell_size_fraction: f32,
}

impl Default for ArchitectureSettings {
    fn default() -> Self {
        Self {
            maximum_source_points: 250_000,
            maximum_planes: 12,
            plane_distance_fraction: 0.003,
            minimum_normal_alignment: 0.94,
            minimum_plane_support: 900,
            ransac_trials_per_plane: 256,
            minimum_cell_support: 2,
            cell_size_fraction: 0.0035,
        }
    }
}

/// Evidence-backed description of one extracted planar surface.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArchitecturalPlane {
    pub normal: [f32; 3],
    pub offset: f32,
    pub support_points: usize,
    pub emitted_surface_cells: usize,
}

/// A conservative architecture layer. `points` is renderable by the existing
/// surfel Studio; `planes` is provenance for a later mesh/export layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArchitectureExtraction {
    pub planes: Vec<ArchitecturalPlane>,
    pub points: Vec<FusedPoint>,
}

/// A conservative, render-ready surface layer derived solely from the
/// supported cells of [`ArchitectureExtraction`].  It deliberately does not
/// join cells across an unsupported gap: a doorway or window remains a hole
/// in the triangle set rather than becoming a plausible-looking invention.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArchitectureMesh {
    pub vertices: Vec<ArchitectureMeshVertex>,
    pub indices: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ArchitectureMeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color_srgb: [u8; 3],
}

/// A floor/wall-only selection made by reprojecting supported global plane
/// cells into the registered semantic rasters. Ceiling and roof classes are
/// deliberately absent: sloped beams and incomplete ceiling capture are not
/// safe architectural surfaces for the current product.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifiedArchitectureSelection {
    pub floor_points: Vec<FusedPoint>,
    pub wall_points: Vec<FusedPoint>,
    pub rejected_points: usize,
    pub minimum_agreeing_views: usize,
}

/// Selects directly supported planar cells as either floor or wall only when
/// registered cameras agree with the semantic evidence. This does not fill a
/// single cell; it merely filters cells emitted by the geometric extractor.
#[must_use]
pub fn select_verified_floor_and_walls(
    support_points: &[FusedPoint],
    semantic: &ArchitectureSemanticVolume,
    solution: &PoseSolution,
    raster: &RasterManifest,
    minimum_confidence: f32,
    minimum_agreeing_views: usize,
) -> VerifiedArchitectureSelection {
    if !minimum_confidence.is_finite()
        || !(0.0..=1.0).contains(&minimum_confidence)
        || minimum_agreeing_views == 0
        || semantic.dimensions() != (raster.output_width, raster.output_height)
    {
        return VerifiedArchitectureSelection {
            floor_points: Vec::new(),
            wall_points: Vec::new(),
            rejected_points: support_points.len(),
            minimum_agreeing_views,
        };
    }
    let Some(trajectory) = solution.global_trajectory.as_ref() else {
        return VerifiedArchitectureSelection {
            floor_points: Vec::new(),
            wall_points: Vec::new(),
            rejected_points: support_points.len(),
            minimum_agreeing_views,
        };
    };
    let views = solution
        .frames
        .iter()
        .filter(|frame| frame.registered)
        .filter_map(|frame| {
            let camera_id = trajectory.frame_camera_ids.get(&frame.frame_index)?;
            let camera = trajectory
                .camera_models
                .iter()
                .find(|camera| camera.camera_id == *camera_id)?;
            Some((frame.frame_index, frame.world_to_camera, camera))
        })
        .collect::<Vec<_>>();
    let mut floor_points = Vec::new();
    let mut wall_points = Vec::new();
    let mut rejected_points = 0;
    for point in support_points {
        let mut floor_votes = 0_usize;
        let mut wall_votes = 0_usize;
        let mut opening_votes = 0_usize;
        for (frame_index, pose, camera) in &views {
            let Some((x, y)) = project_to_semantic_raster(point.position, *pose, camera, raster)
            else {
                continue;
            };
            let Some((class, confidence)) = semantic.sample(*frame_index, x, y) else {
                continue;
            };
            if confidence < minimum_confidence {
                continue;
            }
            match class {
                ArchitectureClass::Floor => floor_votes += 1,
                ArchitectureClass::Wall => wall_votes += 1,
                ArchitectureClass::DoorOrOpening | ArchitectureClass::Window => opening_votes += 1,
                _ => {}
            }
        }
        // An opening can only veto a surface when it has independent support;
        // a single confused pixel does not erase a whole observed cell.
        if opening_votes >= minimum_agreeing_views
            || floor_votes.max(wall_votes) < minimum_agreeing_views
        {
            rejected_points += 1;
        } else if floor_votes > wall_votes {
            let mut floor = point.clone();
            floor.color_srgb = [164, 142, 108];
            floor_points.push(floor);
        } else if wall_votes > floor_votes {
            let mut wall = point.clone();
            wall.color_srgb = [122, 143, 157];
            wall_points.push(wall);
        } else {
            rejected_points += 1;
        }
    }
    VerifiedArchitectureSelection {
        floor_points,
        wall_points,
        rejected_points,
        minimum_agreeing_views,
    }
}

/// Extracts dominant planar support and turns only occupied planar cells into
/// neutral, radius-aware surfels. It is deterministic for a fixed point order.
#[must_use]
pub fn extract_architectural_planes(
    points: &[FusedPoint],
    settings: ArchitectureSettings,
) -> ArchitectureExtraction {
    if !settings_are_valid(settings) || points.is_empty() {
        return ArchitectureExtraction {
            planes: Vec::new(),
            points: Vec::new(),
        };
    }
    let finite = points
        .iter()
        .filter(|point| valid_point(point))
        .collect::<Vec<_>>();
    let Some((low, high)) = bounds(&finite) else {
        return ArchitectureExtraction {
            planes: Vec::new(),
            points: Vec::new(),
        };
    };
    let diagonal = length(subtract(high, low));
    if !diagonal.is_finite() || diagonal <= f32::EPSILON {
        return ArchitectureExtraction {
            planes: Vec::new(),
            points: Vec::new(),
        };
    }
    let tolerance = (diagonal * settings.plane_distance_fraction).max(1e-5);
    let cell_edge = (diagonal * settings.cell_size_fraction).max(tolerance * 1.5);
    let sampled = deterministic_sample(&finite, settings.maximum_source_points);
    let mut remaining = (0..sampled.len()).collect::<Vec<_>>();
    let mut planes = Vec::new();
    let mut output = Vec::new();

    while planes.len() < settings.maximum_planes {
        let Some(seed) = ransac_plane_seed(
            &sampled,
            &remaining,
            tolerance,
            settings.ransac_trials_per_plane,
        ) else {
            break;
        };
        let support = remaining
            .iter()
            .copied()
            .filter(|index| (dot(seed.0, sampled[*index].position) - seed.1).abs() <= tolerance)
            .map(|index| (index, sampled[index]))
            .collect::<Vec<_>>();
        let Some((normal, offset)) = fit_plane(
            &support
                .iter()
                .map(|(_, point)| point.position)
                .collect::<Vec<_>>(),
        ) else {
            break;
        };
        let refined = remaining
            .iter()
            .copied()
            .filter(|index| (dot(normal, sampled[*index].position) - offset).abs() <= tolerance)
            .map(|index| (index, sampled[index]))
            .collect::<Vec<_>>();
        if refined.len() < settings.minimum_plane_support {
            break;
        }
        let plane_points = emit_supported_plane_cells(
            &refined,
            normal,
            offset,
            cell_edge,
            settings.minimum_cell_support,
        );
        if plane_points.is_empty() {
            continue;
        }
        let assigned = refined
            .iter()
            .map(|(index, _)| *index)
            .collect::<HashSet<_>>();
        remaining.retain(|index| !assigned.contains(index));
        planes.push(ArchitecturalPlane {
            normal,
            offset,
            support_points: refined.len(),
            emitted_surface_cells: plane_points.len(),
        });
        output.extend(plane_points);
    }
    ArchitectureExtraction {
        planes,
        points: output,
    }
}

/// Turns the existing evidence-backed support cells into planar quad tiles.
///
/// The support-cell radius is `0.78 * cell_edge`; reconstructing the cell
/// edge from that stable extractor contract makes adjacent cells meet while
/// retaining every unsupported cell as a true mesh hole.  Tiles are kept
/// separate instead of greedily bridging neighbours so that a small semantic
/// or geometric gap can never be silently closed.
#[must_use]
pub fn architecture_mesh_from_support_points(points: &[FusedPoint]) -> ArchitectureMesh {
    let mut vertices = Vec::with_capacity(points.len().saturating_mul(4));
    let mut indices = Vec::with_capacity(points.len().saturating_mul(6));
    for point in points {
        if !valid_point(point) || !point.radius.is_finite() || point.radius <= 0.0 {
            continue;
        }
        let Some(normal) = normalize(point.normal) else {
            continue;
        };
        let (u, v) = plane_basis(normal);
        // `emit_supported_plane_cells` stores radius = 0.78 * cell_edge.
        let half_edge = point.radius / 1.56;
        if !half_edge.is_finite() || half_edge <= 0.0 {
            continue;
        }
        let corners = [
            add(
                point.position,
                add(scale(u, -half_edge), scale(v, -half_edge)),
            ),
            add(
                point.position,
                add(scale(u, half_edge), scale(v, -half_edge)),
            ),
            add(
                point.position,
                add(scale(u, half_edge), scale(v, half_edge)),
            ),
            add(
                point.position,
                add(scale(u, -half_edge), scale(v, half_edge)),
            ),
        ];
        let Ok(base) = u32::try_from(vertices.len()) else {
            break;
        };
        vertices.extend(corners.into_iter().map(|position| ArchitectureMeshVertex {
            position,
            normal,
            color_srgb: point.color_srgb,
        }));
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    ArchitectureMesh { vertices, indices }
}

fn ransac_plane_seed(
    points: &[&FusedPoint],
    remaining: &[usize],
    tolerance: f32,
    trials: usize,
) -> Option<([f32; 3], f32)> {
    if remaining.len() < 3 || trials == 0 {
        return None;
    }
    let mut best = None;
    let mut best_support = 0_usize;
    for trial in 0..trials {
        let first = remaining[deterministic_index(trial as u64 * 3 + 1, remaining.len())];
        let second = remaining[deterministic_index(trial as u64 * 3 + 2, remaining.len())];
        let third = remaining[deterministic_index(trial as u64 * 3 + 3, remaining.len())];
        if first == second || first == third || second == third {
            continue;
        }
        let a = points[first].position;
        let b = points[second].position;
        let c = points[third].position;
        let Some(normal) = normalize(cross(subtract(b, a), subtract(c, a))) else {
            continue;
        };
        let normal = canonical_normal(normal);
        let offset = dot(normal, a);
        let support = remaining
            .iter()
            .filter(|index| (dot(normal, points[**index].position) - offset).abs() <= tolerance)
            .count();
        if support > best_support {
            best_support = support;
            best = Some((normal, offset));
        }
    }
    best
}

fn deterministic_index(seed: u64, bound: usize) -> usize {
    let mut value = seed ^ 0x9e37_79b9_7f4a_7c15;
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    (value % bound as u64) as usize
}

fn emit_supported_plane_cells(
    support: &[(usize, &FusedPoint)],
    normal: [f32; 3],
    offset: f32,
    cell_edge: f32,
    minimum_cell_support: usize,
) -> Vec<FusedPoint> {
    let (u, v) = plane_basis(normal);
    #[derive(Clone, Copy)]
    struct Cell {
        count: usize,
        color: [f64; 3],
        confidence: f64,
        first_frame: i32,
    }
    let mut cells = BTreeMap::<(i32, i32), Cell>::new();
    for (_, point) in support {
        let up = dot(u, point.position);
        let vp = dot(v, point.position);
        let key = (
            (up / cell_edge).floor() as i32,
            (vp / cell_edge).floor() as i32,
        );
        let cell = cells.entry(key).or_insert(Cell {
            count: 0,
            color: [0.0; 3],
            confidence: 0.0,
            first_frame: point.first_observing_frame,
        });
        cell.count += 1;
        for axis in 0..3 {
            cell.color[axis] += f64::from(point.color_srgb[axis]);
        }
        cell.confidence += f64::from(point.confidence.max(0.0));
        cell.first_frame = match (cell.first_frame, point.first_observing_frame) {
            (-1, frame) | (frame, -1) => frame,
            (left, right) => left.min(right),
        };
    }
    cells
        .into_iter()
        .filter(|(_, cell)| cell.count >= minimum_cell_support)
        .map(|(key, cell)| {
            let count = cell.count as f64;
            // Grid-aligned centres make adjacent verified cells share their
            // boundaries exactly in the later triangle mesh. The source
            // samples still decide whether a cell exists at all.
            let average_u = (key.0 as f32 + 0.5) * cell_edge;
            let average_v = (key.1 as f32 + 0.5) * cell_edge;
            // n * offset is the closest point on the plane to the origin.
            let position = add(
                scale(normal, offset),
                add(scale(u, average_u), scale(v, average_v)),
            );
            // A neutral architectural palette is intentionally independent of
            // source textures; the original surfel product retains the RGB truth.
            let brightness = ((cell.color[0] + cell.color[1] + cell.color[2]) / (3.0 * count))
                .clamp(70.0, 225.0) as u8;
            FusedPoint {
                position,
                normal,
                color_srgb: [
                    brightness.saturating_sub(16),
                    brightness,
                    brightness.saturating_sub(24),
                ],
                confidence: (cell.confidence / count) as f32,
                radius: cell_edge * 0.78,
                first_observing_frame: cell.first_frame,
                contributors: cell.count as u32,
            }
        })
        .collect()
}

fn project_to_semantic_raster(
    position: [f32; 3],
    pose: [f64; 12],
    camera: &ColmapCameraModel,
    raster: &RasterManifest,
) -> Option<(usize, usize)> {
    let [focal, cx, cy, radial] = *<&[f64; 4]>::try_from(camera.parameters.as_slice()).ok()?;
    if camera.model != "SIMPLE_RADIAL" || camera.width == 0 || camera.height == 0 {
        return None;
    }
    let point = position.map(f64::from);
    let camera_point = [
        pose[0] * point[0] + pose[1] * point[1] + pose[2] * point[2] + pose[3],
        pose[4] * point[0] + pose[5] * point[1] + pose[6] * point[2] + pose[7],
        pose[8] * point[0] + pose[9] * point[1] + pose[10] * point[2] + pose[11],
    ];
    if !camera_point[2].is_finite() || camera_point[2] <= 0.0 {
        return None;
    }
    let x = camera_point[0] / camera_point[2];
    let y = camera_point[1] / camera_point[2];
    let radial_scale = 1.0 + radial * (x * x + y * y);
    let image_x = focal * x * radial_scale + cx;
    let image_y = focal * y * radial_scale + cy;
    let x = ((image_x + 0.5) * raster.output_width as f64 / camera.width as f64 - 0.5).round();
    let y = ((image_y + 0.5) * raster.output_height as f64 / camera.height as f64 - 0.5).round();
    (x.is_finite()
        && y.is_finite()
        && x >= 0.0
        && y >= 0.0
        && x < raster.output_width as f64
        && y < raster.output_height as f64)
        .then_some((x as usize, y as usize))
}

fn fit_plane(points: &[[f32; 3]]) -> Option<([f32; 3], f32)> {
    if points.len() < 3 {
        return None;
    }
    let mut mean = [0.0_f64; 3];
    for point in points {
        for axis in 0..3 {
            mean[axis] += f64::from(point[axis]);
        }
    }
    for value in &mut mean {
        *value /= points.len() as f64;
    }
    let mut covariance = [[0.0_f64; 3]; 3];
    for point in points {
        let delta = [
            f64::from(point[0]) - mean[0],
            f64::from(point[1]) - mean[1],
            f64::from(point[2]) - mean[2],
        ];
        for row in 0..3 {
            for column in 0..3 {
                covariance[row][column] += delta[row] * delta[column];
            }
        }
    }
    let (values, vectors) = jacobi_eigen_3(covariance);
    let minimum = (0..3).min_by(|&a, &b| values[a].total_cmp(&values[b]))?;
    let normal = canonical_normal(normalize([
        vectors[0][minimum] as f32,
        vectors[1][minimum] as f32,
        vectors[2][minimum] as f32,
    ])?);
    let centre = mean.map(|value| value as f32);
    Some((normal, dot(normal, centre)))
}

fn jacobi_eigen_3(mut matrix: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut vectors = [[0.0; 3]; 3];
    for axis in 0..3 {
        vectors[axis][axis] = 1.0;
    }
    for _ in 0..32 {
        let mut pair = (0, 1);
        let mut greatest = matrix[0][1].abs();
        for row in 0..3 {
            for column in row + 1..3 {
                if matrix[row][column].abs() > greatest {
                    greatest = matrix[row][column].abs();
                    pair = (row, column);
                }
            }
        }
        if greatest <= 1e-12 {
            break;
        }
        let (left, right) = pair;
        let theta =
            0.5 * (2.0 * matrix[left][right]).atan2(matrix[right][right] - matrix[left][left]);
        let cosine = theta.cos();
        let sine = theta.sin();
        for axis in 0..3 {
            let l = matrix[axis][left];
            let r = matrix[axis][right];
            matrix[axis][left] = cosine * l - sine * r;
            matrix[axis][right] = sine * l + cosine * r;
        }
        for axis in 0..3 {
            let l = matrix[left][axis];
            let r = matrix[right][axis];
            matrix[left][axis] = cosine * l - sine * r;
            matrix[right][axis] = sine * l + cosine * r;
        }
        for axis in 0..3 {
            let l = vectors[axis][left];
            let r = vectors[axis][right];
            vectors[axis][left] = cosine * l - sine * r;
            vectors[axis][right] = sine * l + cosine * r;
        }
    }
    ([matrix[0][0], matrix[1][1], matrix[2][2]], vectors)
}

fn deterministic_sample<'a>(points: &[&'a FusedPoint], maximum: usize) -> Vec<&'a FusedPoint> {
    if points.len() <= maximum {
        return points.to_vec();
    }
    let stride = points.len().div_ceil(maximum);
    points
        .iter()
        .step_by(stride)
        .copied()
        .take(maximum)
        .collect()
}

fn bounds(points: &[&FusedPoint]) -> Option<([f32; 3], [f32; 3])> {
    let mut low = [f32::INFINITY; 3];
    let mut high = [f32::NEG_INFINITY; 3];
    for point in points {
        for axis in 0..3 {
            low[axis] = low[axis].min(point.position[axis]);
            high[axis] = high[axis].max(point.position[axis]);
        }
    }
    low.iter()
        .chain(high.iter())
        .all(|value| value.is_finite())
        .then_some((low, high))
}

fn settings_are_valid(settings: ArchitectureSettings) -> bool {
    settings.maximum_source_points > 0
        && settings.maximum_planes > 0
        && settings.plane_distance_fraction.is_finite()
        && settings.plane_distance_fraction > 0.0
        && settings.minimum_normal_alignment.is_finite()
        && (0.0..=1.0).contains(&settings.minimum_normal_alignment)
        && settings.minimum_plane_support >= 3
        && settings.ransac_trials_per_plane > 0
        && settings.minimum_cell_support > 0
        && settings.cell_size_fraction.is_finite()
        && settings.cell_size_fraction > 0.0
}

fn valid_point(point: &FusedPoint) -> bool {
    point.position.iter().all(|value| value.is_finite())
        && point.normal.iter().all(|value| value.is_finite())
}

fn canonical_normal(normal: [f32; 3]) -> [f32; 3] {
    let normal = normalize(normal).unwrap_or([0.0, 0.0, 1.0]);
    let axis = normal
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
        .map(|(axis, _)| axis)
        .unwrap_or(2);
    if normal[axis] < 0.0 {
        scale(normal, -1.0)
    } else {
        normal
    }
}

fn plane_basis(normal: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let reference = if normal[2].abs() < 0.9 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let u = normalize(cross(reference, normal)).unwrap_or([1.0, 0.0, 0.0]);
    let v = normalize(cross(normal, u)).unwrap_or([0.0, 1.0, 0.0]);
    (u, v)
}

fn dot(left: [f32; 3], right: [f32; 3]) -> f32 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}
fn length(value: [f32; 3]) -> f32 {
    dot(value, value).sqrt()
}
fn normalize(value: [f32; 3]) -> Option<[f32; 3]> {
    let length = length(value);
    (length.is_finite() && length > f32::EPSILON).then(|| scale(value, 1.0 / length))
}
fn scale(value: [f32; 3], factor: f32) -> [f32; 3] {
    [value[0] * factor, value[1] * factor, value[2] * factor]
}
fn add(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}
fn subtract(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}
fn cross(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(position: [f32; 3]) -> FusedPoint {
        FusedPoint {
            position,
            normal: [0.0, 0.0, 1.0],
            color_srgb: [180, 170, 160],
            confidence: 1.0,
            radius: 0.01,
            first_observing_frame: 4,
            contributors: 1,
        }
    }

    #[test]
    fn supported_plane_emits_only_observed_cells_and_preserves_door_hole() {
        let mut points = Vec::new();
        for y in 0..80 {
            for x in 0..100 {
                // This central unsupported rectangle represents a doorway.
                if (30..70).contains(&x) && (0..50).contains(&y) {
                    continue;
                }
                points.push(point([x as f32 * 0.02, y as f32 * 0.02, 0.0]));
                points.push(point([x as f32 * 0.02 + 0.002, y as f32 * 0.02, 0.0]));
            }
        }
        let extraction = extract_architectural_planes(
            &points,
            ArchitectureSettings {
                minimum_plane_support: 500,
                cell_size_fraction: 0.02,
                ..ArchitectureSettings::default()
            },
        );
        assert_eq!(extraction.planes.len(), 1);
        assert!(extraction.points.len() > 300);
        assert!(
            extraction
                .points
                .iter()
                .all(|point| point.position[0] < 0.75
                    || point.position[0] > 1.25
                    || point.position[1] > 0.85)
        );
    }

    #[test]
    fn supported_cells_become_triangles_without_closing_a_missing_cell() {
        let mut support = vec![point([0.0, 0.0, 0.0]), point([0.1, 0.0, 0.0])];
        for cell in &mut support {
            cell.radius = 0.078;
        }
        let mesh = architecture_mesh_from_support_points(&support);
        assert_eq!(mesh.vertices.len(), 8);
        assert_eq!(mesh.indices.len(), 12);
        // Two independent support cells produce exactly two quads. There is no
        // bridge triangle over the absent neighbour that could become a door.
        assert!(mesh.indices.chunks_exact(3).all(|triangle| {
            let min = *triangle.iter().min().unwrap();
            let max = *triangle.iter().max().unwrap();
            max - min < 4
        }));
    }

    #[test]
    fn semantic_selection_requires_two_floor_or_wall_views_and_rejects_openings() {
        let semantic = ArchitectureSemanticVolume {
            frame_indices: vec![0, 1, 2, 3],
            width: 2,
            height: 2,
            classes: vec![
                ArchitectureClass::Floor as u8,
                ArchitectureClass::Floor as u8,
                ArchitectureClass::Floor as u8,
                ArchitectureClass::Floor as u8,
                ArchitectureClass::Floor as u8,
                ArchitectureClass::Floor as u8,
                ArchitectureClass::Floor as u8,
                ArchitectureClass::Floor as u8,
                ArchitectureClass::DoorOrOpening as u8,
                ArchitectureClass::DoorOrOpening as u8,
                ArchitectureClass::DoorOrOpening as u8,
                ArchitectureClass::DoorOrOpening as u8,
                ArchitectureClass::DoorOrOpening as u8,
                ArchitectureClass::DoorOrOpening as u8,
                ArchitectureClass::DoorOrOpening as u8,
                ArchitectureClass::DoorOrOpening as u8,
            ],
            confidences: vec![255; 16],
        };
        let camera = ColmapCameraModel {
            camera_id: 1,
            model: "SIMPLE_RADIAL".to_owned(),
            width: 2,
            height: 2,
            parameters: vec![1.0, 0.0, 0.0, 0.0],
        };
        let solution = PoseSolution {
            schema: "vestra.pose-solution/v1".to_owned(),
            provider: crate::PoseProvider {
                kind: "colmap".to_owned(),
                version: "test".to_owned(),
                settings_fingerprint: "test".to_owned(),
            },
            raster_fingerprint: "test".to_owned(),
            coordinate_convention: "COLMAP world; W2C row-major 3x4 f64".to_owned(),
            frames: (0..4)
                .map(|frame_index| crate::PoseFrame {
                    frame_index,
                    image_name: format!("frame-{frame_index:06}.ppm"),
                    registered: true,
                    world_to_camera: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                })
                .collect(),
            diagnostics: crate::PoseDiagnostics::default(),
            global_trajectory: Some(crate::GlobalTrajectoryEvidence {
                camera_models: vec![camera],
                frame_camera_ids: [(0, 1), (1, 1), (2, 1), (3, 1)].into_iter().collect(),
                tracks: Vec::new(),
            }),
        };
        let raster = RasterManifest {
            schema: "vestra.raster/v1".to_owned(),
            source_sha256: "test".to_owned(),
            duration_seconds: 1.0,
            source_width: 2,
            source_height: 2,
            crop: crate::RasterCrop {
                x: 0,
                y: 0,
                width: 2,
                height: 2,
            },
            output_width: 2,
            output_height: 2,
            frames: Vec::new(),
            raster_fingerprint: "test".to_owned(),
        };
        let selected = select_verified_floor_and_walls(
            &[point([0.0, 0.0, 1.0])],
            &semantic,
            &solution,
            &raster,
            0.65,
            2,
        );
        // Two floor observations are enough, but the third independent door
        // observation vetoes publication of that candidate cell.
        assert!(selected.floor_points.is_empty());
        assert_eq!(selected.rejected_points, 1);
    }

    #[test]
    fn invalid_or_normal_free_points_do_not_create_architecture() {
        let mut points = vec![point([0.0, 0.0, 0.0]); 2000];
        for point in &mut points {
            point.normal = [0.0; 3];
        }
        assert!(
            extract_architectural_planes(&points, ArchitectureSettings::default())
                .points
                .is_empty()
        );
    }

    #[test]
    fn semantic_evidence_requires_exact_rasters_and_model_provenance() {
        let evidence = ArchitectureSemanticEvidence {
            schema: ArchitectureSemanticEvidence::SCHEMA.to_owned(),
            runner: "vestra-semantics/0.1".to_owned(),
            model_id: "local/indoor-scene-parser".to_owned(),
            model_revision: "0123456789abcdef".to_owned(),
            model_license: "research-only".to_owned(),
            frames: vec![ArchitectureSemanticFrame {
                frame_index: 7,
                width: 2,
                height: 2,
                classes: vec![
                    ArchitectureClass::Wall,
                    ArchitectureClass::DoorOrOpening,
                    ArchitectureClass::Floor,
                    ArchitectureClass::CeilingOrRoof,
                ],
                confidences: vec![0.99, 0.95, 0.9, 0.9],
            }],
        };
        evidence.validate().unwrap();
        assert_eq!(
            evidence.frames[0].sample(1, 0),
            Some((ArchitectureClass::DoorOrOpening, 0.95))
        );
        assert!(ArchitectureClass::Wall.supports_surface());
        assert!(ArchitectureClass::DoorOrOpening.is_opening());

        let mut malformed = evidence;
        malformed.frames[0].confidences.pop();
        assert!(matches!(
            malformed.validate(),
            Err(ArchitectureEvidenceError::MalformedFrame { frame_index: 7 })
        ));
    }

    #[test]
    fn semantic_volume_reads_exact_frame_pixel_labels() {
        let path = std::env::temp_dir().join(format!("vestra-vsem-{}.bin", std::process::id()));
        let mut bytes = b"VSEM1".to_vec();
        for value in [2_u32, 2, 1, 3, 9] {
            bytes.extend(value.to_le_bytes());
        }
        bytes.extend([
            ArchitectureClass::Wall as u8,
            ArchitectureClass::DoorOrOpening as u8,
            ArchitectureClass::Floor as u8,
            ArchitectureClass::CeilingOrRoof as u8,
        ]);
        bytes.extend([255, 128, 64, 0]);
        fs::write(&path, bytes).unwrap();
        let volume = ArchitectureSemanticVolume::read(&path).unwrap();
        fs::remove_file(path).unwrap();
        assert_eq!(volume.dimensions(), (2, 1));
        assert_eq!(
            volume.sample(3, 1, 0),
            Some((ArchitectureClass::DoorOrOpening, 128.0 / 255.0))
        );
        assert_eq!(
            volume.sample(9, 0, 0),
            Some((ArchitectureClass::Floor, 64.0 / 255.0))
        );
        assert_eq!(volume.sample(3, 2, 0), None);
    }
}
