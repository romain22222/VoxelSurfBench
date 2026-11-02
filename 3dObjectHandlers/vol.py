"""Handler for compressed DGtal .vol voxel grids."""
from __future__ import annotations

import importlib
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_base = importlib.import_module("3dObjectHandlers.3dObject")


@dataclass(frozen=True)
class VolGrid:
    values: np.ndarray


class VolHandler(_base.Object3D):
    extensions = (".vol",)
    is_volume = True

    def load(self, path: Path) -> VolGrid:
        from common import load_grid_from_vol

        return VolGrid(load_grid_from_vol(path))

    def save(self, geometry: VolGrid, path: Path) -> None:
        grid = np.where(np.asarray(geometry.values) > 0, 255, 0).astype(np.uint8)
        if grid.ndim != 3:
            raise ValueError(f"Expected a 3D voxel grid, got {grid.shape}")
        x, y, z = (int(dim) for dim in grid.shape)
        header = (
            f"Center-X: {x // 2}\nCenter-Y: {y // 2}\nCenter-Z: {z // 2}\n"
            f"X: {x}\nY: {y}\nZ: {z}\nVoxel-Size: 1\nAlpha-Color: 0\n"
            "Voxel-Endian: 0\nInt-Endian: 0123\nVersion: 3\n"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(header.encode("latin-1") + b".\n" + zlib.compress(grid.tobytes(), level=6))

    def register_polyscope(
        self, ps: Any, name: str, geometry: VolGrid | np.ndarray, style: dict[str, Any] | None = None
    ):
        from common import voxel_quads
        from scene import apply_style

        settings = _base.merged_style(style)
        grid = geometry.values if isinstance(geometry, VolGrid) else np.asarray(geometry)
        vertices, quads = voxel_quads(grid)
        faces = np.concatenate((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]), axis=0)
        handle = ps.register_surface_mesh(name, vertices, faces, enabled=settings["enabled"])
        apply_style(handle, settings)
        return handle
