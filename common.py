"""Shared helpers: paths, JSON, method registry, ground-truth folder layout, .vol I/O."""
from __future__ import annotations

import importlib
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
METHODS_FILE = ROOT / "methods.json"
GT_DIRNAME = "gt"

# Fixed file names inside <shape>/gt/ (the gt mesh extension depends on the handler).
VOL_NAME = "shape.vol"
BINVOX_NAME = "shape.binvox"
VISNORMALS_NAME = "visnormals.xyz"
SOURCE_NAME = "source.json"
CAMERA_NAME = "camera.json"
GT_MESH_STEM = "gt"

# Fixed file names inside <shape>/<method>/.
RESULTS_NAME = "results.json"
LOG_NAME = "method.log"
BASE_STEM = "base"
ALIGNED_STEM = "aligned"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ----------------------------------------------------------------------------- 3D object handlers

def get_handler(path_or_ext: str | Path):
    """Handler instance for a file (or a bare extension such as '.obj')."""
    handlers = importlib.import_module("3dObjectHandlers")
    return handlers.get_handler(path_or_ext)


def supported_extensions() -> list[str]:
    return importlib.import_module("3dObjectHandlers").supported_extensions()


def mesh_extensions() -> list[str]:
    return importlib.import_module("3dObjectHandlers").mesh_extensions()


def load_mesh(path: Path):
    return get_handler(path).load(path)


# ----------------------------------------------------------------------------- registry

def load_registry(methods_file: Path = METHODS_FILE) -> dict[str, Any]:
    if not methods_file.exists():
        return {}
    raw = read_json(methods_file)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid method registry: {methods_file}")
    return raw


def save_registry(registry: dict[str, Any], methods_file: Path = METHODS_FILE) -> None:
    write_json(methods_file, registry)


# ----------------------------------------------------------------------------- shape folders

@dataclass(frozen=True)
class GtInputs:
    shape_dir: Path
    name: str
    gt_dir: Path
    vol: Path
    binvox: Path
    visnormals: Path
    source_json: Path
    points: Path
    gt_mesh: Path
    camera_json: Path  # may not exist yet (see adjust_camera.py)


def gt_dir_of(shape_dir: Path) -> Path:
    return shape_dir / GT_DIRNAME


def find_gt_mesh(gt_dir: Path) -> Path | None:
    for ext in mesh_extensions():
        candidate = gt_dir / f"{GT_MESH_STEM}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_gt_inputs(shape_dir: Path, *, require_sidecars: bool = True) -> GtInputs:
    shape_dir = Path(shape_dir).resolve()
    gt_dir = gt_dir_of(shape_dir)
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Missing '{GT_DIRNAME}' folder in {shape_dir}")
    gt_mesh = find_gt_mesh(gt_dir)
    if gt_mesh is None:
        raise FileNotFoundError(f"No {GT_MESH_STEM}.<ext> ({', '.join(mesh_extensions())}) in {gt_dir}")
    inputs = GtInputs(
        shape_dir=shape_dir,
        name=shape_dir.name,
        gt_dir=gt_dir,
        vol=gt_dir / VOL_NAME,
        binvox=gt_dir / BINVOX_NAME,
        visnormals=gt_dir / VISNORMALS_NAME,
        source_json=gt_dir / SOURCE_NAME,
        points=gt_dir / f"{shape_dir.name}.ply",
        gt_mesh=gt_mesh,
        camera_json=gt_dir / CAMERA_NAME,
    )
    required = [inputs.vol, inputs.binvox, inputs.visnormals, inputs.source_json] if require_sidecars else [inputs.vol]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(str(p) for p in missing))
    return inputs


def method_dir(shape_dir: Path, method: str) -> Path:
    return Path(shape_dir) / method


def list_shape_dirs(shapes_root: Path) -> list[Path]:
    root = Path(shapes_root)
    if not root.is_dir():
        raise NotADirectoryError(f"Shapes root not found: {root}")
    return sorted(p.parent for p in root.rglob(GT_DIRNAME) if p.is_dir())


# ----------------------------------------------------------------------------- .vol

def load_grid_from_vol(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    sep = b".\n"
    idx = raw.find(sep)
    if idx == -1:
        raise ValueError(f"VOL header terminator not found in {path}")
    header: dict[str, str] = {}
    for line in raw[:idx].decode("latin-1").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()
    nx, ny, nz = int(header["X"]), int(header["Y"]), int(header["Z"])
    data = zlib.decompress(raw[idx + len(sep):])
    return np.frombuffer(data, dtype=np.uint8).reshape((nx, ny, nz))


def grid_dim_from_vol(vol_path: Path) -> int:
    return int(max(load_grid_from_vol(vol_path).shape))


def voxel_quads(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exposed faces of the occupied voxels as quads, in the unit cube [-0.5, 0.5]^3 (same frame as the GT)."""
    grid = grid > 0
    if not grid.any():
        raise ValueError("Empty volume")
    padded = np.pad(grid, 1)
    # Per direction: (axis, step, 4 corner offsets in CCW order seen from outside).
    face_defs = [
        (0, 1, [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)]),
        (0, -1, [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)]),
        (1, 1, [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)]),
        (1, -1, [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]),
        (2, 1, [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]),
        (2, -1, [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)]),
    ]
    corners: list[np.ndarray] = []
    for axis, step, offsets in face_defs:
        exposed = padded & ~np.roll(padded, -step, axis=axis)
        idx = np.argwhere(exposed[1:-1, 1:-1, 1:-1])
        if len(idx):
            corners.append(np.stack([idx + np.asarray(o) for o in offsets], axis=1))
    quads = np.concatenate(corners, axis=0)
    uniq, inverse = np.unique(quads.reshape(-1, 3), axis=0, return_inverse=True)
    verts = -0.5 + uniq.astype(np.float64) / float(max(grid.shape))
    return verts, inverse.reshape(-1, 4)
