from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

_base = importlib.import_module("3dObjectHandlers.3dObject")


class PlyHandler(_base.Object3D):
    """PLY files may hold a mesh or a bare point cloud (no faces): the latter is returned as a `trimesh.PointCloud`."""

    extensions = (".ply",)

    def load(self, path: Path):
        loaded = trimesh.load(str(path), file_type="ply", process=False)
        if isinstance(loaded, trimesh.Scene):
            loaded = loaded.dump(concatenate=True)
        if isinstance(loaded, trimesh.PointCloud):
            return loaded
        if not isinstance(loaded, trimesh.Trimesh):
            raise TypeError(f"Could not load a mesh or point cloud from {path}")
        if len(loaded.faces) == 0:
            return trimesh.PointCloud(np.asarray(loaded.vertices))
        return loaded

    def save(self, geometry, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        geometry.export(str(path), file_type="ply")

    def register_polyscope(self, ps: Any, name: str, geometry, style: dict[str, Any] | None = None):
        s = _base.merged_style(style)
        vertices = np.asarray(geometry.vertices, dtype=np.float64)
        if not hasattr(geometry, "faces"):
            handle = ps.register_point_cloud(name, vertices, enabled=s["enabled"])
            handle.set_color(tuple(s["color"]))
            handle.set_radius(s["point_radius"])
            return handle
        handle = ps.register_surface_mesh(name, vertices, np.asarray(geometry.faces, dtype=np.int64), enabled=s["enabled"])
        handle.set_color(tuple(s["color"]))
        handle.set_transparency(float(s["transparency"]))
        handle.set_smooth_shade(bool(s["smooth_shade"]))
        handle.set_edge_width(float(s["edge_width"]))
        handle.set_edge_color(tuple(s["edge_color"]))
        handle.set_material(s["material"])
        return handle
