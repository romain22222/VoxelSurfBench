"""Template for 3D object formats. Every supported format (.obj, .off, .ply, ...) has its own file in this folder
containing a subclass of `Object3D`; no behavior is inherited, each subclass has to implement the three methods below.

To support a new format: copy any handler (e.g. obj.py), set `extensions`, rewrite `load`, `save` and
`register_polyscope`. It is discovered automatically.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

Geometry = Any

# Visual parameters understood by `register_polyscope`.
DEFAULT_STYLE: dict[str, Any] = {
    "enabled": True,
    "color": (0.55, 0.55, 0.55),
    "transparency": 1.0,  # 1 = opaque
    "smooth_shade": False,
    "edge_width": 0.0,
    "edge_color": (0.0, 0.0, 0.0),
    "material": "clay",
    "point_radius": 0.0015,
}


def merged_style(style: dict[str, Any] | None) -> dict[str, Any]:
    return {**DEFAULT_STYLE, **(style or {})}


class Object3D(ABC):
    #: lowercase extensions handled by the subclass, with the dot (e.g. (".obj",)); the first one is the default.
    extensions: tuple[str, ...] = ()

    @abstractmethod
    def load(self, path: Path) -> Geometry:
        """Read `path` into a format-specific geometry (mesh, point cloud, voxel grid, etc.)."""

    @abstractmethod
    def save(self, geometry: Geometry, path: Path) -> None:
        """Write `geometry` to `path` (create the parent folder if needed)."""

    @abstractmethod
    def register_polyscope(self, ps: Any, name: str, geometry: Geometry, style: dict[str, Any] | None = None) -> Any:
        """Add `geometry` to the current polyscope scene as structure `name`, using `merged_style(style)`.
        Return the polyscope structure handle."""
