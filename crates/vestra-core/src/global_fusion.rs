//! Frame-global dense-depth rebasing.
//!
//! This is intentionally distinct from the legacy window-Sim(3) path. A
//! globally bundle-adjusted COLMAP trajectory owns every output camera; DA3
//! contributes dense relative depth for the source frame only. Sparse COLMAP
//! tracks calibrate that depth in the global coordinate system.

use std::collections::BTreeMap;

use crate::{
    CameraCalibration, FusedPoint, FusedSceneChunk, MeasuredFrameChunk, PoseSolution, SceneBundle,
    SceneBundleError, TsdfObservation, TsdfSettings, WindowMeasuredChunk, fuse_normal_space_tsdf,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameGlobalFusionSettings {
    /// Minimum inlier depth/track observations used to calibrate one source
    /// frame. A frame below this is omitted rather than interpolated.
    pub minimum_scale_samples: usize,
    /// Deterministic held-out residual limit in log-depth units.
    pub maximum_held_out_median_log_error: f32,
    /// Reject sparse tracks whose final global-BA reprojection error is worse.
    pub maximum_track_reprojection_error_px: f64,
    /// A frame-global product needs broad trajectory coverage. Frames that
    /// fail the held-out scale gate are omitted, never interpolated, but a
    /// product with only isolated good fragments must not be published.
    pub minimum_fused_frame_fraction: f32,
    /// Robust radius for temporal log-depth-scale smoothing.  Only adjacent
    /// frames that independently pass the held-out gate participate; gaps in
    /// global registration or scale evidence are never bridged.
    pub temporal_scale_smoothing_radius: usize,
    /// Maximum log-depth disagreement when confirming an observation through
    /// an adjacent globally calibrated source frame.  This gate applies only
    /// to the TSDF derivative; the raw surfel product remains immutable dense
    /// evidence for inspection.
    pub maximum_neighbor_depth_log_error: Option<f32>,
    /// Number of independent adjacent depth maps required by the TSDF gate.
    pub minimum_neighbor_depth_matches: usize,
    /// Maximum source-frame-index separation considered adjacent for the
    /// depth-consistency gate.  Gaps in registration are never interpolated.
    pub neighbor_frame_radius: usize,
    /// Bounded, deterministic evidence set for normal-space TSDF. The raw
    /// surfel mode remains complete; this prevents a redundant dense raster
    /// from turning a browser-oriented surface product into an hours-long PCA
    /// job.
    pub maximum_tsdf_observations: Option<usize>,
    /// Build a second, explicit TSDF surface rather than raw rebased surfels.
    pub tsdf: Option<TsdfSettings>,
}

impl Default for FrameGlobalFusionSettings {
    fn default() -> Self {
        Self {
            minimum_scale_samples: 12,
            maximum_held_out_median_log_error: 0.20,
            maximum_track_reprojection_error_px: 2.5,
            minimum_fused_frame_fraction: 0.85,
            temporal_scale_smoothing_radius: 2,
            maximum_neighbor_depth_log_error: Some(0.20),
            minimum_neighbor_depth_matches: 1,
            neighbor_frame_radius: 2,
            maximum_tsdf_observations: Some(6_000_000),
            tsdf: Some(TsdfSettings::default()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrameGlobalReport {
    pub frame_index: usize,
    pub registered: bool,
    pub scale_samples: usize,
    pub held_out_samples: usize,
    pub scale: Option<f32>,
    pub held_out_median_log_error: Option<f32>,
}

#[derive(Debug, thiserror::Error)]
pub enum FrameGlobalFusionError {
    #[error("scene persistence failed: {0}")]
    Scene(#[from] SceneBundleError),
    #[error("pose solution has no globally bundle-adjusted sparse trajectory evidence")]
    MissingTrajectoryEvidence,
    #[error(
        "frame {frame_index} has {actual} trusted depth-scale samples; need at least {minimum}"
    )]
    InsufficientDepthScaleEvidence {
        frame_index: usize,
        actual: usize,
        minimum: usize,
    },
    #[error("frame {frame_index} held-out depth-scale error {actual:.4} exceeds {maximum:.4}")]
    DepthScaleQuality {
        frame_index: usize,
        actual: f32,
        maximum: f32,
    },
    #[error("frame-global settings are invalid")]
    InvalidSettings,
    #[error(
        "only {actual}/{total} frames passed frame-global gates; need at least {minimum_fraction:.0}% coverage"
    )]
    InsufficientFrameCoverage {
        actual: usize,
        total: usize,
        minimum_fraction: f32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameGlobalFusionProgress {
    pub chunk_hash: String,
    pub fused_frames: usize,
    pub omitted_frames: usize,
    pub points: usize,
}

/// Diagnoses all canonical DA3 frames against a full COLMAP model without
/// writing a world product.
pub fn frame_global_reports(
    bundle: &SceneBundle,
    pose_solution_hash: &str,
    settings: FrameGlobalFusionSettings,
) -> Result<Vec<FrameGlobalReport>, FrameGlobalFusionError> {
    validate_settings(settings)?;
    let solution = bundle.read_pose_solution(pose_solution_hash)?;
    let evidence = solution
        .global_trajectory
        .as_ref()
        .ok_or(FrameGlobalFusionError::MissingTrajectoryEvidence)?;
    let raster = bundle.read_raster_manifest()?;
    let (_, reports) = select_canonical_views(bundle, &solution, evidence, &raster, settings)?;
    Ok(reports)
}

/// Emits a separate global-BA world. Every accepted point is reprojected from
/// its own source frame through the COLMAP camera; no window transform is
/// fitted or composed.
pub fn fuse_scene_bundle_frame_global(
    bundle: &SceneBundle,
    pose_solution_hash: &str,
    settings: FrameGlobalFusionSettings,
) -> Result<FrameGlobalFusionProgress, FrameGlobalFusionError> {
    validate_settings(settings)?;
    let solution = bundle.read_pose_solution(pose_solution_hash)?;
    let evidence = solution
        .global_trajectory
        .as_ref()
        .ok_or(FrameGlobalFusionError::MissingTrajectoryEvidence)?;
    let raster = bundle.read_raster_manifest()?;
    let (views, reports) = select_canonical_views(bundle, &solution, evidence, &raster, settings)?;
    let reports =
        temporally_smooth_reports(reports, &views, &solution, evidence, &raster, settings)?;
    let accepted = reports
        .iter()
        .map(|report| {
            report.scale.is_some()
                && report
                    .held_out_median_log_error
                    .is_some_and(|error| error <= settings.maximum_held_out_median_log_error)
                && global_pose_and_camera(report.frame_index, &solution, evidence).is_some()
        })
        .collect::<Vec<_>>();
    let candidate_observations = views
        .iter()
        .zip(&accepted)
        .filter(|(_, accepted)| **accepted)
        .map(|((_, view), _)| view.points.len())
        .sum();
    let tsdf_sample_threshold = settings
        .maximum_tsdf_observations
        .map(|budget| observation_sample_threshold(candidate_observations, budget));
    let mut observations = Vec::with_capacity(
        settings
            .maximum_tsdf_observations
            .unwrap_or(candidate_observations)
            .min(candidate_observations),
    );
    let mut raw_points = Vec::new();
    let mut cameras = Vec::new();
    let mut fused_frames = 0;
    let mut accepted_frame_indices = Vec::new();
    let mut omitted_frames = 0;
    let mut global_frames = Vec::new();
    for ((frame_index, view), (report, accepted)) in
        views.into_iter().zip(reports.into_iter().zip(accepted))
    {
        if !accepted {
            omitted_frames += 1;
            continue;
        }
        let Some(scale) = report.scale else {
            omitted_frames += 1;
            continue;
        };
        let held_out = report.held_out_median_log_error.ok_or(
            FrameGlobalFusionError::InsufficientDepthScaleEvidence {
                frame_index,
                actual: report.held_out_samples,
                minimum: 1,
            },
        )?;
        if held_out > settings.maximum_held_out_median_log_error {
            omitted_frames += 1;
            continue;
        }
        let Some((pose, camera)) = global_pose_and_camera(frame_index, &solution, evidence) else {
            omitted_frames += 1;
            continue;
        };
        global_frames.push(GlobalFrame {
            frame_index,
            view,
            scale,
            pose,
            camera: camera.clone(),
        });
    }
    let depth_maps = settings
        .maximum_neighbor_depth_log_error
        .map(|_| global_depth_maps(&global_frames, &raster));
    for frame in &global_frames {
        fused_frames += 1;
        accepted_frame_indices.push(frame.frame_index);
        cameras.push(camera_centre(frame.pose));
        for point in &frame.view.points {
            let Some((position, normal, radius)) = rebase_point(
                point,
                frame.view.camera,
                frame.pose,
                &frame.camera,
                &raster,
                frame.scale,
            ) else {
                continue;
            };
            if settings.tsdf.is_some() {
                if tsdf_sample_threshold.is_some_and(|threshold| {
                    !keep_tsdf_observation(frame.frame_index, point.source_pixel, threshold)
                }) {
                    continue;
                }
                if let (Some(maximum_error), Some(depth_maps)) = (
                    settings.maximum_neighbor_depth_log_error,
                    depth_maps.as_deref(),
                ) && !has_neighbor_depth_agreement(
                    position,
                    frame.frame_index,
                    depth_maps,
                    &raster,
                    settings.neighbor_frame_radius,
                    settings.minimum_neighbor_depth_matches,
                    maximum_error,
                ) {
                    continue;
                }
                observations.push(TsdfObservation {
                    position,
                    color_srgb: point.color_srgb,
                    confidence: point.confidence,
                    radius,
                    frame_index: frame.frame_index as i32,
                });
            } else {
                raw_points.push(FusedPoint {
                    position,
                    normal,
                    color_srgb: point.color_srgb,
                    confidence: point.confidence,
                    radius,
                    first_observing_frame: frame.frame_index as i32,
                    contributors: 1,
                });
            }
        }
    }
    let total_candidate_frames = (fused_frames + omitted_frames).max(1);
    if !has_required_coverage(
        fused_frames,
        total_candidate_frames,
        settings.minimum_fused_frame_fraction,
    ) {
        return Err(FrameGlobalFusionError::InsufficientFrameCoverage {
            actual: fused_frames,
            total: total_candidate_frames,
            minimum_fraction: settings.minimum_fused_frame_fraction,
        });
    }
    let (points, voxel_size) = if let Some(tsdf) = settings.tsdf {
        let points = fuse_normal_space_tsdf(&observations, &cameras, tsdf)
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
            .collect::<Vec<_>>();
        let voxel = points.first().map_or(0.0, |point| point.radius / 0.6);
        (points, voxel)
    } else {
        (raw_points, 0.0)
    };
    let count = points.len();
    let fused = FusedSceneChunk {
        alignments: Vec::new(),
        pose_graph_edges: Vec::new(),
        pose_graph: None,
        window_poses: Vec::new(),
        voxel_size,
        points,
    };
    let product_id = if settings.tsdf.is_some() {
        "colmap-ba-frame-global-tsdf"
    } else {
        "colmap-ba-frame-global-surfel"
    };
    let chunk_hash = bundle.write_fused_scene_as(
        &fused,
        product_id,
        "colmap-ba-frame-global",
        if settings.tsdf.is_some() {
            "tsdf"
        } else {
            "surfel"
        },
        Some(pose_solution_hash.to_owned()),
    )?;
    bundle.set_world_product_source_frames(product_id, &accepted_frame_indices)?;
    Ok(FrameGlobalFusionProgress {
        chunk_hash,
        fused_frames,
        omitted_frames,
        points: count,
    })
}

fn validate_settings(settings: FrameGlobalFusionSettings) -> Result<(), FrameGlobalFusionError> {
    if settings.minimum_scale_samples == 0
        || !settings.maximum_held_out_median_log_error.is_finite()
        || settings.maximum_held_out_median_log_error < 0.0
        || !settings.maximum_track_reprojection_error_px.is_finite()
        || settings.maximum_track_reprojection_error_px <= 0.0
        || !settings.minimum_fused_frame_fraction.is_finite()
        || !(0.0..=1.0).contains(&settings.minimum_fused_frame_fraction)
        || settings
            .maximum_neighbor_depth_log_error
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        || (settings.maximum_neighbor_depth_log_error.is_some()
            && (settings.minimum_neighbor_depth_matches == 0
                || settings.neighbor_frame_radius == 0))
        || settings
            .maximum_tsdf_observations
            .is_some_and(|budget| budget == 0)
    {
        return Err(FrameGlobalFusionError::InvalidSettings);
    }
    Ok(())
}

struct GlobalFrame {
    frame_index: usize,
    view: MeasuredFrameChunk,
    scale: f32,
    pose: [f64; 12],
    camera: crate::ColmapCameraModel,
}

struct GlobalDepthMap {
    frame_index: usize,
    pose: [f64; 12],
    camera: crate::ColmapCameraModel,
    depth: Vec<f32>,
}

fn global_depth_maps(
    frames: &[GlobalFrame],
    raster: &crate::RasterManifest,
) -> Vec<GlobalDepthMap> {
    frames
        .iter()
        .map(|frame| {
            let mut depth = vec![f32::NAN; raster.output_width * raster.output_height];
            for point in &frame.view.points {
                let x = point.source_pixel[0] as usize;
                let y = point.source_pixel[1] as usize;
                if x >= raster.output_width || y >= raster.output_height {
                    continue;
                }
                if let Some(value) = local_camera_depth(point.position, frame.view.camera)
                    .map(|value| value * frame.scale)
                    .filter(|value| value.is_finite() && *value > 0.0)
                {
                    depth[y * raster.output_width + x] = value;
                }
            }
            GlobalDepthMap {
                frame_index: frame.frame_index,
                pose: frame.pose,
                camera: frame.camera.clone(),
                depth,
            }
        })
        .collect()
}

fn has_neighbor_depth_agreement(
    position: [f32; 3],
    source_frame: usize,
    maps: &[GlobalDepthMap],
    raster: &crate::RasterManifest,
    neighbor_radius: usize,
    minimum_matches: usize,
    maximum_log_error: f32,
) -> bool {
    let mut matches = 0;
    for map in maps {
        if map.frame_index == source_frame
            || map.frame_index.abs_diff(source_frame) > neighbor_radius
        {
            continue;
        }
        let Some((pixel, projected_depth)) =
            project_world_to_raster(position, map.pose, &map.camera, raster)
        else {
            continue;
        };
        let Some(observed_depth) =
            bilinear_depth(&map.depth, raster.output_width, raster.output_height, pixel)
        else {
            continue;
        };
        let error = (projected_depth.ln() - observed_depth.ln()).abs();
        if error.is_finite() && error <= maximum_log_error {
            matches += 1;
            if matches >= minimum_matches {
                return true;
            }
        }
    }
    false
}

fn project_world_to_raster(
    position: [f32; 3],
    pose: [f64; 12],
    camera: &crate::ColmapCameraModel,
    raster: &crate::RasterManifest,
) -> Option<([f32; 2], f32)> {
    let [focal, cx, cy, radial] = *<&[f64; 4]>::try_from(camera.parameters.as_slice()).ok()?;
    let point = position.map(f64::from);
    let camera_point = [
        pose[0] * point[0] + pose[1] * point[1] + pose[2] * point[2] + pose[3],
        pose[4] * point[0] + pose[5] * point[1] + pose[6] * point[2] + pose[7],
        pose[8] * point[0] + pose[9] * point[1] + pose[10] * point[2] + pose[11],
    ];
    if !(camera_point[2].is_finite() && camera_point[2] > 0.0) {
        return None;
    }
    let (x, y) = (
        camera_point[0] / camera_point[2],
        camera_point[1] / camera_point[2],
    );
    let radial_scale = 1.0 + radial * (x * x + y * y);
    let image = [focal * x * radial_scale + cx, focal * y * radial_scale + cy];
    let pixel = [
        ((image[0] + 0.5) * raster.output_width as f64 / camera.width as f64 - 0.5) as f32,
        ((image[1] + 0.5) * raster.output_height as f64 / camera.height as f64 - 0.5) as f32,
    ];
    (pixel.iter().all(|value| value.is_finite())
        && pixel[0] >= 0.0
        && pixel[1] >= 0.0
        && pixel[0] < raster.output_width.saturating_sub(1) as f32
        && pixel[1] < raster.output_height.saturating_sub(1) as f32)
        .then_some((pixel, camera_point[2] as f32))
}

fn bilinear_depth(depth: &[f32], width: usize, height: usize, pixel: [f32; 2]) -> Option<f32> {
    let x0 = pixel[0].floor() as usize;
    let y0 = pixel[1].floor() as usize;
    let (x1, y1) = (x0.checked_add(1)?, y0.checked_add(1)?);
    if x1 >= width || y1 >= height {
        return None;
    }
    let tx = pixel[0] - x0 as f32;
    let ty = pixel[1] - y0 as f32;
    let values = [
        depth[y0 * width + x0],
        depth[y0 * width + x1],
        depth[y1 * width + x0],
        depth[y1 * width + x1],
    ];
    values
        .iter()
        .all(|value| value.is_finite() && *value > 0.0)
        .then_some(
            (values[0] * (1.0 - tx) + values[1] * tx) * (1.0 - ty)
                + (values[2] * (1.0 - tx) + values[3] * tx) * ty,
        )
}

fn observation_sample_threshold(candidate_observations: usize, budget: usize) -> u64 {
    if candidate_observations <= budget {
        return u64::MAX;
    }
    ((budget as f64 / candidate_observations as f64) * u64::MAX as f64) as u64
}

fn keep_tsdf_observation(frame_index: usize, source_pixel: [u32; 2], threshold: u64) -> bool {
    let mut hash = frame_index as u64 ^ 0x9e37_79b9_7f4a_7c15;
    hash ^= u64::from(source_pixel[0]).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    hash = hash.rotate_left(27).wrapping_mul(0x94d0_49bb_1331_11eb);
    hash ^= u64::from(source_pixel[1]).wrapping_mul(0x369d_ea0f_31a5_3f85);
    hash ^= hash >> 29;
    hash <= threshold
}

fn has_required_coverage(fused_frames: usize, total_frames: usize, minimum_fraction: f32) -> bool {
    fused_frames as f32 / total_frames.max(1) as f32 >= minimum_fraction
}

/// Chooses one measured DA3 view per source frame using sparse global evidence.
/// A frame occurs in overlapping DA3 windows; first-owner selection is stable
/// but can select a locally inconsistent depth map.  This selector ranks every
/// available candidate by the same held-out COLMAP-track contract used for
/// publication, then moves only the winning point array into the global pass.
fn select_canonical_views(
    bundle: &SceneBundle,
    solution: &PoseSolution,
    evidence: &crate::GlobalTrajectoryEvidence,
    raster: &crate::RasterManifest,
    settings: FrameGlobalFusionSettings,
) -> Result<(BTreeMap<usize, MeasuredFrameChunk>, Vec<FrameGlobalReport>), FrameGlobalFusionError> {
    let mut windows = bundle
        .manifest()?
        .measured_chunk_hashes
        .iter()
        .map(|hash| bundle.read_measured_window(hash))
        .collect::<Result<Vec<WindowMeasuredChunk>, _>>()?;
    windows.sort_by_key(|window| window.window.index);
    let mut selected = BTreeMap::<usize, (usize, usize, FrameGlobalReport)>::new();
    for window in &windows {
        for (view_index, view) in window.views.iter().enumerate() {
            let report =
                report_for_frame(view.frame_index, view, solution, evidence, raster, settings)?;
            let replace = selected
                .get(&view.frame_index)
                .is_none_or(|(_, _, current)| {
                    candidate_report_is_better(&report, current, settings)
                });
            if replace {
                selected.insert(view.frame_index, (window.window.index, view_index, report));
            }
        }
    }
    let mut views = BTreeMap::new();
    for window in windows {
        for (view_index, view) in window.views.into_iter().enumerate() {
            let Some((winner_window, winner_view, _)) = selected.get(&view.frame_index) else {
                continue;
            };
            if *winner_window == window.window.index && *winner_view == view_index {
                views.insert(view.frame_index, view);
            }
        }
    }
    let reports = selected
        .into_iter()
        .map(|(_, (_, _, report))| report)
        .collect();
    Ok((views, reports))
}

fn candidate_report_is_better(
    candidate: &FrameGlobalReport,
    current: &FrameGlobalReport,
    settings: FrameGlobalFusionSettings,
) -> bool {
    let accepted = |report: &FrameGlobalReport| {
        report.scale.is_some()
            && report
                .held_out_median_log_error
                .is_some_and(|error| error <= settings.maximum_held_out_median_log_error)
    };
    match (accepted(candidate), accepted(current)) {
        (true, false) => return true,
        (false, true) => return false,
        _ => {}
    }
    let candidate_error = candidate.held_out_median_log_error.unwrap_or(f32::INFINITY);
    let current_error = current.held_out_median_log_error.unwrap_or(f32::INFINITY);
    candidate_error.total_cmp(&current_error).is_lt()
        || (candidate_error == current_error && candidate.scale_samples > current.scale_samples)
}

fn report_for_frame(
    frame_index: usize,
    view: &MeasuredFrameChunk,
    solution: &PoseSolution,
    evidence: &crate::GlobalTrajectoryEvidence,
    raster: &crate::RasterManifest,
    settings: FrameGlobalFusionSettings,
) -> Result<FrameGlobalReport, FrameGlobalFusionError> {
    if global_pose_and_camera(frame_index, solution, evidence).is_none() {
        return Ok(FrameGlobalReport {
            frame_index,
            registered: false,
            scale_samples: 0,
            held_out_samples: 0,
            scale: None,
            held_out_median_log_error: None,
        });
    }
    let evidence =
        scale_evidence_for_frame(frame_index, view, solution, evidence, raster, settings)?;
    if evidence.train.len() < settings.minimum_scale_samples || evidence.held_out.is_empty() {
        return Ok(FrameGlobalReport {
            frame_index,
            registered: true,
            scale_samples: evidence.train.len(),
            held_out_samples: evidence.held_out.len(),
            scale: None,
            held_out_median_log_error: None,
        });
    }
    let mut train = evidence.train;
    let log_scale = median(&mut train);
    let scale = log_scale.exp();
    let held_out_median_log_error = held_out_error(&evidence.held_out, log_scale);
    Ok(FrameGlobalReport {
        frame_index,
        registered: true,
        scale_samples: train.len(),
        held_out_samples: evidence.held_out.len(),
        scale: Some(scale),
        held_out_median_log_error: Some(held_out_median_log_error),
    })
}

#[derive(Debug)]
struct ScaleEvidence {
    train: Vec<f32>,
    held_out: Vec<f32>,
}

fn scale_evidence_for_frame(
    frame_index: usize,
    view: &MeasuredFrameChunk,
    solution: &PoseSolution,
    evidence: &crate::GlobalTrajectoryEvidence,
    raster: &crate::RasterManifest,
    settings: FrameGlobalFusionSettings,
) -> Result<ScaleEvidence, FrameGlobalFusionError> {
    let Some((global_pose, camera)) = global_pose_and_camera(frame_index, solution, evidence)
    else {
        return Ok(ScaleEvidence {
            train: Vec::new(),
            held_out: Vec::new(),
        });
    };
    let points = view
        .points
        .iter()
        .map(|point| (point.source_pixel, point))
        .collect::<BTreeMap<_, _>>();
    let mut train = Vec::new();
    let mut held_out = Vec::new();
    for track in &evidence.tracks {
        if track.reprojection_error_px > settings.maximum_track_reprojection_error_px {
            continue;
        }
        let Some(observation) = track
            .observations
            .iter()
            .find(|observation| observation.frame_index == frame_index)
        else {
            continue;
        };
        let Some(pixel) = raster_pixel(observation.image_xy, camera, raster) else {
            continue;
        };
        let Some(point) = points.get(&pixel) else {
            continue;
        };
        let Some(depth) = local_camera_depth(point.position, view.camera) else {
            continue;
        };
        let sparse_depth = camera_depth(track.position, global_pose);
        if !sparse_depth.is_finite() || sparse_depth <= 0.0 {
            continue;
        }
        let ratio = sparse_depth / f64::from(depth);
        if !ratio.is_finite() || ratio <= 0.0 {
            continue;
        }
        if track.point_id % 5 == 0 {
            held_out.push(ratio.ln() as f32);
        } else {
            train.push(ratio.ln() as f32);
        }
    }
    Ok(ScaleEvidence { train, held_out })
}

fn held_out_error(held_out: &[f32], log_scale: f32) -> f32 {
    let mut residuals = held_out
        .iter()
        .map(|value| (*value - log_scale).abs())
        .collect::<Vec<_>>();
    median(&mut residuals)
}

fn temporally_smooth_reports(
    mut reports: Vec<FrameGlobalReport>,
    views: &BTreeMap<usize, MeasuredFrameChunk>,
    solution: &PoseSolution,
    evidence: &crate::GlobalTrajectoryEvidence,
    raster: &crate::RasterManifest,
    settings: FrameGlobalFusionSettings,
) -> Result<Vec<FrameGlobalReport>, FrameGlobalFusionError> {
    if settings.temporal_scale_smoothing_radius == 0 {
        return Ok(reports);
    }
    let eligible = reports
        .iter()
        .map(|report| {
            report.scale.is_some_and(|_| {
                report
                    .held_out_median_log_error
                    .is_some_and(|error| error <= settings.maximum_held_out_median_log_error)
            })
        })
        .collect::<Vec<_>>();
    let raw_logs = reports
        .iter()
        .map(|report| report.scale.map(f32::ln))
        .collect::<Vec<_>>();
    for index in 0..reports.len() {
        if !eligible[index] {
            continue;
        }
        let start = index.saturating_sub(settings.temporal_scale_smoothing_radius);
        let end = (index + settings.temporal_scale_smoothing_radius + 1).min(reports.len());
        let mut logs = Vec::new();
        for neighbor in start..end {
            if !eligible[neighbor]
                || reports[neighbor]
                    .frame_index
                    .abs_diff(reports[index].frame_index)
                    > settings.temporal_scale_smoothing_radius
            {
                continue;
            }
            // Never bridge a rejected/missing frame within the smoothing span.
            let lower = neighbor.min(index);
            let upper = neighbor.max(index);
            if (lower..=upper).any(|position| !eligible[position]) {
                continue;
            }
            logs.push(raw_logs[neighbor].expect("eligible report has scale"));
        }
        if logs.len() <= 1 {
            continue;
        }
        let smoothed_log = median(&mut logs);
        let view = views
            .get(&reports[index].frame_index)
            .expect("report has canonical view");
        let samples = scale_evidence_for_frame(
            reports[index].frame_index,
            view,
            solution,
            evidence,
            raster,
            settings,
        )?;
        let error = held_out_error(&samples.held_out, smoothed_log);
        if error <= settings.maximum_held_out_median_log_error {
            reports[index].scale = Some(smoothed_log.exp());
            reports[index].held_out_median_log_error = Some(error);
        }
    }
    Ok(reports)
}

fn global_pose_and_camera<'a>(
    frame_index: usize,
    solution: &'a PoseSolution,
    evidence: &'a crate::GlobalTrajectoryEvidence,
) -> Option<([f64; 12], &'a crate::ColmapCameraModel)> {
    let pose = solution
        .frames
        .iter()
        .find(|frame| frame.frame_index == frame_index && frame.registered)?;
    let camera_id = evidence.frame_camera_ids.get(&frame_index)?;
    let camera = evidence
        .camera_models
        .iter()
        .find(|camera| camera.camera_id == *camera_id)?;
    Some((pose.world_to_camera, camera))
}

fn raster_pixel(
    image_xy: [f64; 2],
    camera: &crate::ColmapCameraModel,
    raster: &crate::RasterManifest,
) -> Option<[u32; 2]> {
    if camera.width == 0
        || camera.height == 0
        || camera.width * raster.output_height != camera.height * raster.output_width
    {
        return None;
    }
    let project = |value: f64, source: usize, destination: usize| {
        ((value + 0.5) * destination as f64 / source as f64 - 0.5).round()
    };
    let x = project(image_xy[0], camera.width, raster.output_width);
    let y = project(image_xy[1], camera.height, raster.output_height);
    (x >= 0.0 && y >= 0.0 && x < raster.output_width as f64 && y < raster.output_height as f64)
        .then_some([x as u32, y as u32])
}

fn local_camera_depth(position: [f32; 3], camera: CameraCalibration) -> Option<f32> {
    let matrix = camera.world_to_camera;
    let depth =
        matrix[8] * position[0] + matrix[9] * position[1] + matrix[10] * position[2] + matrix[11];
    (depth.is_finite() && depth > 0.0).then_some(depth)
}

fn camera_depth(position: [f64; 3], pose: [f64; 12]) -> f64 {
    pose[8] * position[0] + pose[9] * position[1] + pose[10] * position[2] + pose[11]
}

fn rebase_point(
    point: &crate::MeasuredPoint,
    local_camera: CameraCalibration,
    global_pose: [f64; 12],
    global_camera: &crate::ColmapCameraModel,
    raster: &crate::RasterManifest,
    scale: f32,
) -> Option<([f32; 3], [f32; 3], f32)> {
    if !point.position.iter().all(|value| value.is_finite())
        || !point.normal.iter().all(|value| value.is_finite())
        || !point.radius.is_finite()
        || point.radius <= 0.0
        || !point.confidence.is_finite()
        || point.confidence <= 0.0
        || !scale.is_finite()
        || scale <= 0.0
    {
        return None;
    }
    let depth = local_camera_depth(point.position, local_camera)? * scale;
    let pixel = [
        f64::from(point.source_pixel[0]),
        f64::from(point.source_pixel[1]),
    ];
    let image_xy = [
        (pixel[0] + 0.5) * global_camera.width as f64 / raster.output_width as f64 - 0.5,
        (pixel[1] + 0.5) * global_camera.height as f64 / raster.output_height as f64 - 0.5,
    ];
    let ray = simple_radial_ray(image_xy, global_camera)?;
    let camera_point = [
        ray[0] * f64::from(depth),
        ray[1] * f64::from(depth),
        f64::from(depth),
    ];
    let position = world_from_camera(camera_point, global_pose)?;
    let local_normal_camera = rotate_f32(local_camera.world_to_camera, point.normal);
    let normal = normalize(rotate_transpose_f64(global_pose, local_normal_camera))?;
    let local_depth = local_camera_depth(point.position, local_camera)?;
    let pixel_radius =
        point.radius * local_camera.intrinsics[0].min(local_camera.intrinsics[4]) / local_depth;
    let radius = pixel_radius * depth / global_camera.parameters[0] as f32;
    (radius.is_finite() && radius > 0.0).then_some((position, normal, radius))
}

fn simple_radial_ray(image_xy: [f64; 2], camera: &crate::ColmapCameraModel) -> Option<[f64; 2]> {
    let [focal, cx, cy, radial] = camera.parameters.as_slice() else {
        return None;
    };
    if *focal <= 0.0 {
        return None;
    }
    let (xd, yd) = ((image_xy[0] - cx) / focal, (image_xy[1] - cy) / focal);
    let (mut x, mut y) = (xd, yd);
    for _ in 0..8 {
        let scale = 1.0 + radial * (x * x + y * y);
        if !scale.is_finite() || scale.abs() < 1e-8 {
            return None;
        }
        x = xd / scale;
        y = yd / scale;
    }
    (x.is_finite() && y.is_finite()).then_some([x, y])
}

fn world_from_camera(camera: [f64; 3], pose: [f64; 12]) -> Option<[f32; 3]> {
    let d = [
        camera[0] - pose[3],
        camera[1] - pose[7],
        camera[2] - pose[11],
    ];
    let world = [
        pose[0] * d[0] + pose[4] * d[1] + pose[8] * d[2],
        pose[1] * d[0] + pose[5] * d[1] + pose[9] * d[2],
        pose[2] * d[0] + pose[6] * d[1] + pose[10] * d[2],
    ];
    world.iter().all(|value| value.is_finite()).then_some([
        world[0] as f32,
        world[1] as f32,
        world[2] as f32,
    ])
}

fn camera_centre(pose: [f64; 12]) -> [f32; 3] {
    world_from_camera([0.0, 0.0, 0.0], pose).unwrap_or([0.0; 3])
}

fn rotate_f32(matrix: [f32; 12], vector: [f32; 3]) -> [f32; 3] {
    [
        matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
        matrix[4] * vector[0] + matrix[5] * vector[1] + matrix[6] * vector[2],
        matrix[8] * vector[0] + matrix[9] * vector[1] + matrix[10] * vector[2],
    ]
}

fn rotate_transpose_f64(matrix: [f64; 12], vector: [f32; 3]) -> [f32; 3] {
    [
        (matrix[0] * f64::from(vector[0])
            + matrix[4] * f64::from(vector[1])
            + matrix[8] * f64::from(vector[2])) as f32,
        (matrix[1] * f64::from(vector[0])
            + matrix[5] * f64::from(vector[1])
            + matrix[9] * f64::from(vector[2])) as f32,
        (matrix[2] * f64::from(vector[0])
            + matrix[6] * f64::from(vector[1])
            + matrix[10] * f64::from(vector[2])) as f32,
    ]
}

fn normalize(vector: [f32; 3]) -> Option<[f32; 3]> {
    let length = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    (length.is_finite() && length > 1e-6).then_some([
        vector[0] / length,
        vector[1] / length,
        vector[2] / length,
    ])
}

fn median(values: &mut [f32]) -> f32 {
    values.sort_by(f32::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebasing_uses_global_frame_pose_not_local_window_transform() {
        let raster = crate::finalized_raster_manifest(crate::RasterManifest {
            schema: String::new(),
            source_sha256: "x".to_owned(),
            duration_seconds: 1.0,
            source_width: 1620,
            source_height: 1080,
            crop: crate::RasterCrop {
                x: 0,
                y: 0,
                width: 1620,
                height: 1080,
            },
            output_width: 504,
            output_height: 336,
            frames: vec![crate::RasterFrame {
                frame_index: 0,
                file_name: "frame-000001.ppm".to_owned(),
                sha256: "x".to_owned(),
                timestamp_millis: 0,
            }],
            raster_fingerprint: String::new(),
        });
        let point = crate::MeasuredPoint {
            position: [0.0, 0.0, 2.0],
            normal: [0.0, 0.0, -1.0],
            color_srgb: [1, 2, 3],
            confidence: 1.0,
            radius: 0.01,
            source_pixel: [252, 168],
        };
        let local = CameraCalibration {
            world_to_camera: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            intrinsics: [252.0, 0.0, 252.0, 0.0, 252.0, 168.0, 0.0, 0.0, 1.0],
        };
        let global = crate::ColmapCameraModel {
            camera_id: 1,
            model: "SIMPLE_RADIAL".to_owned(),
            width: 1620,
            height: 1080,
            parameters: vec![810.0, 810.0, 540.0, 0.0],
        };
        let rebased = rebase_point(
            &point,
            local,
            [1.0, 0.0, 0.0, -5.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            &global,
            &raster,
            2.0,
        )
        .unwrap();
        // Integer raster pixels map through the half-pixel resize contract,
        // so the centre-adjacent sample has a small but expected ray offset.
        assert!((rebased.0[0] - 5.0).abs() < 0.01);
        assert!((rebased.0[2] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn raster_pixel_keeps_half_pixel_resize_contract() {
        let raster = crate::finalized_raster_manifest(crate::RasterManifest {
            schema: String::new(),
            source_sha256: "x".to_owned(),
            duration_seconds: 1.0,
            source_width: 1620,
            source_height: 1080,
            crop: crate::RasterCrop {
                x: 0,
                y: 0,
                width: 1620,
                height: 1080,
            },
            output_width: 504,
            output_height: 336,
            frames: vec![crate::RasterFrame {
                frame_index: 0,
                file_name: "frame-000001.ppm".to_owned(),
                sha256: "x".to_owned(),
                timestamp_millis: 0,
            }],
            raster_fingerprint: String::new(),
        });
        let camera = crate::ColmapCameraModel {
            camera_id: 1,
            model: "SIMPLE_RADIAL".to_owned(),
            width: 1620,
            height: 1080,
            parameters: vec![810.0, 810.0, 540.0, 0.0],
        };
        assert_eq!(
            raster_pixel([811.107142857, 541.107142857], &camera, &raster),
            Some([252, 168])
        );
    }

    #[test]
    fn frame_global_coverage_rejects_isolated_good_fragments() {
        assert!(has_required_coverage(200, 230, 0.85));
        assert!(!has_required_coverage(195, 230, 0.85));
        assert!(!has_required_coverage(0, 0, 0.85));
    }

    #[test]
    fn tsdf_sampling_is_deterministic_and_bounded() {
        let threshold = observation_sample_threshold(55_000_000, 6_000_000);
        assert!(threshold < u64::MAX);
        assert_eq!(
            keep_tsdf_observation(17, [42, 99], threshold),
            keep_tsdf_observation(17, [42, 99], threshold)
        );
        assert_eq!(observation_sample_threshold(12, 12), u64::MAX);
    }

    #[test]
    fn canonical_view_selection_prefers_held_out_verified_depth() {
        let settings = FrameGlobalFusionSettings::default();
        let weak = FrameGlobalReport {
            frame_index: 7,
            registered: true,
            scale_samples: 80,
            held_out_samples: 10,
            scale: Some(2.0),
            held_out_median_log_error: Some(0.31),
        };
        let verified = FrameGlobalReport {
            frame_index: 7,
            registered: true,
            scale_samples: 12,
            held_out_samples: 3,
            scale: Some(2.2),
            held_out_median_log_error: Some(0.04),
        };
        assert!(candidate_report_is_better(&verified, &weak, settings));
        assert!(!candidate_report_is_better(&weak, &verified, settings));
    }

    #[test]
    fn tsdf_neighbor_gate_accepts_only_reprojected_depth_agreement() {
        let raster = crate::finalized_raster_manifest(crate::RasterManifest {
            schema: String::new(),
            source_sha256: "x".into(),
            duration_seconds: 1.0,
            source_width: 4,
            source_height: 4,
            crop: crate::RasterCrop {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
            },
            output_width: 4,
            output_height: 4,
            frames: vec![],
            raster_fingerprint: String::new(),
        });
        let map = GlobalDepthMap {
            frame_index: 1,
            pose: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            camera: crate::ColmapCameraModel {
                camera_id: 1,
                model: "SIMPLE_RADIAL".into(),
                width: 4,
                height: 4,
                parameters: vec![2.0, 1.5, 1.5, 0.0],
            },
            depth: vec![2.0; 16],
        };
        assert!(has_neighbor_depth_agreement(
            [0.0, 0.0, 2.0],
            0,
            &[map],
            &raster,
            2,
            1,
            0.02
        ));
        let inconsistent = GlobalDepthMap {
            depth: vec![1.0; 16],
            ..GlobalDepthMap {
                frame_index: 1,
                pose: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                camera: crate::ColmapCameraModel {
                    camera_id: 1,
                    model: "SIMPLE_RADIAL".into(),
                    width: 4,
                    height: 4,
                    parameters: vec![2.0, 1.5, 1.5, 0.0],
                },
                depth: vec![2.0; 16],
            }
        };
        assert!(!has_neighbor_depth_agreement(
            [0.0, 0.0, 2.0],
            0,
            &[inconsistent],
            &raster,
            2,
            1,
            0.02
        ));
    }
}
