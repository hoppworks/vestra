#!/usr/bin/env python3
"""Measure dominant planar support in a published Vestra point cloud.

This is a geometry diagnostic, not a floor detector and not a quality verdict.
It extracts several deterministic RANSAC planes from a bounded uniform point
sample, reports their support and orthogonal residuals normalized by the cloud
diagonal, and leaves the source PLY untouched. Comparing the same report
between products makes global bending visible without assuming a metric scale
or an up axis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from inspect_colmap_mvs import read_vertices


def percentile(values: Any, q: float, np: Any) -> float | None:
    if len(values) == 0:
        return None
    return float(np.percentile(values, q))


def fit_plane(points: Any, np: Any) -> tuple[Any, float] | None:
    centre = points.mean(axis=0)
    if not np.isfinite(centre).all():
        return None
    # Three collinear RANSAC samples do not define a plane. Reject them before
    # SVD rather than accepting an arbitrary null-space vector.
    if np.linalg.matrix_rank(points - centre) < 2:
        return None
    _, _, vectors = np.linalg.svd(points - centre, full_matrices=False)
    normal = vectors[-1]
    length = float(np.linalg.norm(normal))
    if not np.isfinite(normal).all() or not np.isfinite(length) or length <= 0:
        return None
    normal = normal / length
    offset = -float(normal @ centre)
    return (normal, offset) if np.isfinite(offset) else None


def dominant_planes(
    points: Any,
    plane_count: int,
    trials: int,
    threshold: float,
    seed: int,
    np: Any,
) -> list[dict[str, Any]]:
    """Extract disjoint planes; all random choices are deterministic."""
    if len(points) < 3:
        return []
    remaining = np.ones(len(points), dtype=bool)
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    for _ in range(plane_count):
        indices = np.flatnonzero(remaining)
        if len(indices) < 3:
            break
        candidate = points[indices]
        best_mask = None
        best_count = 0
        for _ in range(trials):
            sample = candidate[rng.choice(len(candidate), size=3, replace=False)]
            fit = fit_plane(sample, np)
            if fit is None:
                continue
            normal, offset = fit
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                residual = np.abs(candidate @ normal + offset)
            mask = np.isfinite(residual) & (residual <= threshold)
            count = int(mask.sum())
            if count > best_count:
                best_mask, best_count = mask, count
        if best_mask is None or best_count < 3:
            break
        chosen_indices = indices[best_mask]
        fit = fit_plane(points[chosen_indices], np)
        if fit is None:
            break
        normal, offset = fit
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            residual = np.abs(points[chosen_indices] @ normal + offset)
        if not np.isfinite(residual).all():
            continue
        centre = points[chosen_indices].mean(axis=0)
        output.append({
            "sample_points": int(len(chosen_indices)),
            "sample_fraction": float(len(chosen_indices) / len(points)),
            "normal": [float(value) for value in normal],
            "centroid": [float(value) for value in centre],
            "residual_median": percentile(residual, 50, np),
            "residual_p95": percentile(residual, 95, np),
        })
        remaining[chosen_indices] = False
    return output


def inspect_positions(
    positions: Any,
    plane_count: int = 3,
    trials: int = 256,
    threshold_fraction: float = 0.002,
    seed: int = 20260817,
    np: Any = None,
) -> dict[str, Any]:
    if np is None:
        import numpy as np
    positions = np.asarray(positions, dtype=np.float64)
    positions = positions[np.isfinite(positions).all(axis=1)]
    if len(positions) < 3:
        raise ValueError("need at least three finite positions")
    lower, upper = positions.min(axis=0), positions.max(axis=0)
    diagonal = float(np.linalg.norm(upper - lower))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("cloud bounds have no extent")
    threshold = diagonal * threshold_fraction
    planes = dominant_planes(positions, plane_count, trials, threshold, seed, np)
    for plane in planes:
        plane["residual_median_fraction_of_diagonal"] = plane["residual_median"] / diagonal
        plane["residual_p95_fraction_of_diagonal"] = plane["residual_p95"] / diagonal
    return {
        "schema": "vestra.world-plane-inspection/v1",
        "sample_points": int(len(positions)),
        "bounds": {"min": [float(value) for value in lower], "max": [float(value) for value in upper]},
        "diagonal": diagonal,
        "ransac": {"plane_count": plane_count, "trials": trials, "threshold_fraction": threshold_fraction, "threshold": threshold, "seed": seed},
        "planes": planes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-points", type=int, default=250_000)
    parser.add_argument("--planes", type=int, default=3)
    parser.add_argument("--trials", type=int, default=256)
    parser.add_argument("--threshold-fraction", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()
    if args.maximum_points <= 0 or args.planes <= 0 or args.trials <= 0 or not 0 < args.threshold_fraction < 1:
        parser.error("all counts must be positive and threshold fraction must be in (0, 1)")
    import numpy as np
    vertices = read_vertices(args.ply, args.maximum_points)
    positions = np.array([[vertex.x, vertex.y, vertex.z] for vertex in vertices], dtype=np.float64)
    report = inspect_positions(positions, args.planes, args.trials, args.threshold_fraction, args.seed, np)
    report["ply"] = str(args.ply)
    report["vertices_selected"] = len(vertices)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
