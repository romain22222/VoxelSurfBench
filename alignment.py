"""Deterministic conversion of a raw method output to the GT frame ([-0.5, 0.5]^3). Nothing is fitted to the GT.

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
