"""Convert raw method output into the input-volume frame.

The static transform follows DGNS's `Experimentations/best_align.py::raw_to_gt_frame`. The pipeline then estimates a
translation against the input volume surface; it never fits a reconstruction to its GT mesh.

Each method declares how its raw output relates to the GT in its registry entry:

    "alignment": {
        "frame": "voxel",                 # "voxel": raw mesh in voxel index units; "gt": already in the GT frame
        "swap_xz": false,                 # swap X and Z axes (flips the faces to keep the orientation)
        "scale_divisor": 1.0,             # extra divisor, relative to the grid size (e.g. 2.0 for a 2x finer grid)
        "offset_voxels": 0.0,             # residual constant offset, in voxels of the base grid
        "offset_voxels_by_grid": {"128": -0.5}   # optional per-grid-dimension override of offset_voxels
    }
"""
from __future__ import annotations

from typing import Any

import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial import cKDTree

DEFAULT_ALIGNMENT: dict[str, Any] = {
    "frame": "voxel",
    "swap_xz": False,
    "scale_divisor": 1.0,
    "offset_voxels": 0.0,
}


def resolve_alignment(spec: dict[str, Any] | None) -> dict[str, Any]:
    merged = {**DEFAULT_ALIGNMENT, **(spec or {})}
    if merged["frame"] not in {"voxel", "gt"}:
        raise ValueError(f"alignment.frame must be 'voxel' or 'gt', got {merged['frame']!r}")
    return merged


def offset_voxels(alignment: dict[str, Any], grid_dim: int) -> float:
    by_grid = alignment.get("offset_voxels_by_grid") or {}
    return float(by_grid.get(str(int(grid_dim)), alignment["offset_voxels"]))


def align_mesh(mesh: trimesh.Trimesh, spec: dict[str, Any] | None, grid_dim: int) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    alignment = resolve_alignment(spec)
    meta: dict[str, Any] = {"grid_dim": int(grid_dim), **alignment}
    if alignment["frame"] == "gt":
        return mesh.copy(), meta

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    if alignment["swap_xz"]:
        verts = verts[:, [2, 1, 0]]
        faces = faces[:, ::-1]
    divisor = float(grid_dim) * float(alignment["scale_divisor"])
    verts = verts / divisor - 0.5 + offset_voxels(alignment, grid_dim) / float(grid_dim)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False), meta


def recenter_mesh_to_reference(
    mesh: trimesh.Trimesh,
    reference: trimesh.Trimesh,
    grid_dim: int,
    *,
    sample_count: int = 12000,
    search_radius_voxels: float = 4.0,
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Estimate a bounded translation that best matches the reference surface."""
    if grid_dim < 1:
        raise ValueError(f"grid_dim must be positive, got {grid_dim}")
    if sample_count < 1:
        raise ValueError(f"sample_count must be positive, got {sample_count}")
    if search_radius_voxels <= 0:
        raise ValueError(f"search_radius_voxels must be positive, got {search_radius_voxels}")
    if len(mesh.faces) == 0 or len(reference.faces) == 0:
        raise ValueError("Both meshes must have faces to estimate a recentering offset")

    source_points, _ = trimesh.sample.sample_surface(mesh, sample_count, seed=17)
    reference_points, _ = trimesh.sample.sample_surface(reference, sample_count, seed=23)
    source_tree = cKDTree(source_points)
    reference_tree = cKDTree(reference_points)

    def symmetric_chamfer(translation: np.ndarray) -> float:
        source_to_reference = reference_tree.query(source_points + translation, workers=1)[0]
        reference_to_source = source_tree.query(reference_points - translation, workers=1)[0]
        return float(source_to_reference.mean() + reference_to_source.mean())

    before = symmetric_chamfer(np.zeros(3, dtype=np.float64))
    radius = float(search_radius_voxels) / float(grid_dim)
    result = minimize(
        symmetric_chamfer,
        np.zeros(3, dtype=np.float64),
        method="Powell",
        bounds=[(-radius, radius)] * 3,
        options={"xtol": 1e-7, "ftol": 1e-8, "maxiter": 150},
    )
    if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError(f"Volume-based recentering failed: {result.message}")

    translation = np.asarray(result.x, dtype=np.float64)
    after = float(result.fun)
    applied = after < before
    if not applied:
        translation = np.zeros(3, dtype=np.float64)
        after = before

    recentered = mesh.copy()
    recentered.apply_translation(translation)
    boundary_tolerance = radius * 1e-3
    return recentered, {
        "reference": "shape.vol exposed voxel faces",
        "method": "bounded symmetric surface Chamfer translation",
        "translation": translation.tolist(),
        "translation_voxels": (translation * float(grid_dim)).tolist(),
        "sample_count": int(sample_count),
        "search_radius_voxels": float(search_radius_voxels),
        "search_bound_hit": bool(np.any(np.abs(translation) >= radius - boundary_tolerance)),
        "distance_before": before,
        "distance_after": after,
        "applied": applied,
    }
