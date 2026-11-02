"""Polyscope helpers shared by visualize_overlays.py and adjust_camera.py, plus the camera.json format.

camera.json (written by adjust_camera.py, stored in <shape>/gt/):

    {
      "version": 1,
      "view": { ...polyscope view as returned by ps.get_view_as_json()... },
      "scene": {"up_dir", "ground_plane_mode", "ground_plane_height_factor", "background_color"},
      "structures": {"gt": {color, transparency, smooth_shade, edge_width, edge_color, material}, "vol": {...}}
    }
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from common import read_json, write_json

GROUND_PLANE_MODES = ["none", "tile", "tile_reflection", "shadow_only"]
GT_COLOR = (0.84, 0.15, 0.16)
VOL_COLOR = (0.50, 0.50, 0.50)
METHOD_COLORS = [
    (0.12, 0.47, 0.71), (1.00, 0.50, 0.05), (0.17, 0.63, 0.17), (0.58, 0.40, 0.74), (0.55, 0.34, 0.29),
    (0.89, 0.47, 0.76), (0.74, 0.74, 0.13), (0.09, 0.75, 0.81), (0.68, 0.78, 0.91), (1.00, 0.73, 0.47),
]

DEFAULT_SCENE: dict[str, Any] = {
    "up_dir": "z_up",
    "ground_plane_mode": "none",
    "ground_plane_height_factor": 0.0,
    "background_color": [1.0, 1.0, 1.0, 1.0],
}
STYLE_KEYS = ("color", "transparency", "smooth_shade", "edge_width", "edge_color", "material")


def default_style(label: str, index: int = 0, *, alpha: float = 0.38, gt_alpha: float = 0.8) -> dict[str, Any]:
    """Look of a structure when camera.json does not say otherwise."""
    if label == "gt":
        return {"color": GT_COLOR, "transparency": gt_alpha, "edge_width": 1.0}
    if label == "vol":
        return {"color": VOL_COLOR, "transparency": 1.0, "edge_width": 1.0}
    return {"color": METHOD_COLORS[index % len(METHOD_COLORS)], "transparency": 1.0 - alpha, "edge_width": 1.0}


def apply_scene(ps: Any, scene: dict[str, Any] | None) -> dict[str, Any]:
    scene = {**DEFAULT_SCENE, **(scene or {})}
    ps.set_up_dir(scene["up_dir"])
    ps.set_ground_plane_mode(scene["ground_plane_mode"])
    ps.set_ground_plane_height_factor(float(scene["ground_plane_height_factor"]), True)
    ps.set_background_color(tuple(scene["background_color"]))
    return scene


def register_voxels(ps: Any, name: str, grid: np.ndarray, style: dict[str, Any]):
    """Delegate .vol conversion and visualization to its format-specific handler."""
    from common import get_handler

    return get_handler(".vol").register_polyscope(ps, name, grid, style)


def apply_style(handle: Any, style: dict[str, Any]) -> None:
    if "color" in style:
        handle.set_color(tuple(style["color"]))
    if "transparency" in style:
        handle.set_transparency(float(style["transparency"]))
    if "smooth_shade" in style:
        handle.set_smooth_shade(bool(style["smooth_shade"]))
    if "edge_width" in style:
        handle.set_edge_width(float(style["edge_width"]))
    if "edge_color" in style:
        handle.set_edge_color(tuple(style["edge_color"]))
    if "material" in style:
        handle.set_material(style["material"])


def read_style(handle: Any) -> dict[str, Any]:
    return {
        "color": [float(c) for c in handle.get_color()],
        "transparency": float(handle.get_transparency()),
        "smooth_shade": bool(handle.get_smooth_shade()),
        "edge_width": float(handle.get_edge_width()),
        "edge_color": [float(c) for c in handle.get_edge_color()],
        "material": str(handle.get_material()),
    }


def load_camera(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = read_json(path)
    return data if isinstance(data, dict) and "view" in data else None


def apply_view(ps: Any, camera: dict[str, Any] | None) -> None:
    if camera and camera.get("view"):
        ps.set_view_from_json(json.dumps(camera["view"]))


def save_camera(path: Path, ps: Any, scene: dict[str, Any], structures: dict[str, Any]) -> None:
    write_json(
        path,
        {
            "version": 1,
            "view": json.loads(ps.get_view_as_json()),
            "scene": {**scene, "background_color": [float(c) for c in ps.get_background_color()]},
            "structures": {name: read_style(handle) for name, handle in structures.items()},
        },
    )
