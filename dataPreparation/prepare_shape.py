from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shlex
import sys
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.command_runner import run_command


def _wsl_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    if len(value) >= 2 and value[1] == ":":
        return f"/mnt/{value[0].lower()}{value[2:]}"
    return value


def _write_binvox(grid: np.ndarray, path: Path) -> None:
    dimensions = grid.shape
    flattened = np.transpose(grid > 0, (0, 2, 1)).astype(np.uint8).ravel()
    encoded = bytearray()
    value = int(flattened[0])
    count = 0
    for item in flattened:
        item = int(item)
        if item == value and count < 255:
            count += 1
            continue
        encoded.extend((value, count))
        value = item
        count = 1
    encoded.extend((value, count))

    header = (
        f"#binvox 1\n"
        f"dim {dimensions[0]} {dimensions[1]} {dimensions[2]}\n"
        "translate -0.5 -0.5 -0.5\n"
        "scale 1\n"
        "data\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header.encode("ascii") + encoded)


def _run_visibility_normals(
    visibility_tool: str,
    volume_path: Path,
    normals_path: Path,
    gridstep: float,
    radius: int,
) -> None:
    arguments = [
        visibility_tool,
        "-i",
        _wsl_path(volume_path) if os.name == "nt" else str(volume_path),
        "-g",
        str(gridstep),
        "-r",
        str(radius),
        "--noInterface",
        "--saveNormals",
        _wsl_path(normals_path) if os.name == "nt" else str(normals_path),
    ]
    if os.name == "nt":
        if not visibility_tool.startswith("/"):
            arguments[0] = _wsl_path(Path(visibility_tool))
        command = shlex.join(arguments)
        run_command("wsl", ["bash", "-lc", command])
    else:
        run_command(arguments[0], arguments[1:])

    if not normals_path.is_file() or normals_path.stat().st_size == 0:
        raise FileNotFoundError(f"DGtal did not produce visibility normals: {normals_path}")
    normals = np.loadtxt(normals_path)
    if normals.ndim != 2 or normals.shape[1] not in (3, 6) or not np.isfinite(normals).all():
        raise ValueError(f"Invalid visibility-normal data in {normals_path}: {normals.shape}")


def _normalize_mesh(mesh_path: Path) -> tuple[trimesh.Trimesh, np.ndarray, float]:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"Expected a triangle mesh with vertices and faces: {mesh_path}")

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center = bounds.mean(axis=0)
    scale = float(np.ptp(bounds, axis=0).max())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Cannot normalize a mesh with degenerate bounds: {mesh_path}")
    mesh.apply_translation(-center)
    mesh.apply_scale(1.0 / scale)
    return mesh, center, scale


def prepare_shape(args: argparse.Namespace) -> Path:
    mesh_path = args.mesh.resolve()
    shape_dir = args.shape_dir.resolve()
    gt_dir = shape_dir / "gt"
    shape_name = shape_dir.name
    outputs = (
        gt_dir / "source.stl",
        gt_dir / "gt.ply",
        gt_dir / f"{shape_name}.ply",
        gt_dir / "shape.vol",
        gt_dir / "shape.binvox",
        gt_dir / "visnormals.xyz",
        gt_dir / "source.json",
    )
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Preparation outputs already exist; pass --overwrite to replace them:\n"
            + "\n".join(str(path) for path in existing)
        )

    mesh, center, scale = _normalize_mesh(mesh_path)
    gt_dir.mkdir(parents=True, exist_ok=True)
    source_path = gt_dir / "source.stl"
    if mesh_path != source_path:
        source_path.write_bytes(mesh_path.read_bytes())
    mesh.export(gt_dir / "gt.ply")

    voxelized = mesh.voxelized(pitch=1.0 / args.resolution).fill()
    voxel_centers = np.asarray(voxelized.points)
    indices = np.floor((voxel_centers + 0.5) * args.resolution).astype(np.int64)
    inside = np.all((indices >= 0) & (indices < args.resolution), axis=1)
    grid = np.zeros((args.resolution,) * 3, dtype=np.uint8)
    grid[tuple(indices[inside].T)] = 255
    if not grid.any():
        raise ValueError("Voxelization is empty; check the input mesh and resolution.")

    vol_handler = importlib.import_module("3dObjectHandlers.vol")
    from common import get_handler, load_grid_from_vol, voxel_quads

    volume_path = gt_dir / "shape.vol"
    get_handler(".vol").save(vol_handler.VolGrid(grid), volume_path)
    _write_binvox(grid, gt_dir / "shape.binvox")
    voxel_vertices, voxel_quads_faces = voxel_quads(load_grid_from_vol(volume_path))
    voxel_faces = np.concatenate(
        (voxel_quads_faces[:, [0, 1, 2]], voxel_quads_faces[:, [0, 2, 3]]),
        axis=0,
    )
    voxel_surface = trimesh.Trimesh(vertices=voxel_vertices, faces=voxel_faces, process=False)
    points, _ = trimesh.sample.sample_surface(voxel_surface, args.point_count, seed=args.seed)
    trimesh.PointCloud(vertices=points).export(gt_dir / f"{shape_name}.ply")
    _run_visibility_normals(
        args.visibility_tool,
        volume_path,
        gt_dir / "visnormals.xyz",
        args.gridstep,
        args.radius,
    )

    manifest = {
        "source_mesh": source_path.name,
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "resolution": args.resolution,
        "normalization": {
            "center": center.tolist(),
            "scale": scale,
            "target_bounds": [-0.5, 0.5],
        },
        "point_sampling": {
            "source": "shape.vol",
            "method": "uniform area sampling over exposed voxel faces",
            "count": args.point_count,
            "seed": args.seed,
        },
        "voxelization": {
            "method": "trimesh.voxelized(pitch=1/resolution).fill()",
            "occupied_voxels": int(np.count_nonzero(grid)),
        },
        "visibility_normals": {
            "method": "DGtal visibilityLattices",
            "gridstep": args.gridstep,
            "radius": args.radius,
        },
    }
    (gt_dir / "source.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return shape_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create VoxelSurfBench inputs from an original triangle mesh."
    )
    parser.add_argument("--mesh", type=Path, required=True, help="Original source mesh; existing voxels are not used.")
    parser.add_argument("--shape-dir", type=Path, required=True, help="Output shape directory containing gt/.")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--point-count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--visibility-tool", required=True, help="DGtal visibilityLattices executable.")
    parser.add_argument("--gridstep", type=float, default=1.0)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.resolution < 2:
        parser.error("--resolution must be at least 2.")
    if args.point_count < 1:
        parser.error("--point-count must be positive.")
    if args.gridstep <= 0:
        parser.error("--gridstep must be positive.")
    if args.radius < 1:
        parser.error("--radius must be positive.")
    return args


if __name__ == "__main__":
    print(prepare_shape(parse_args()))
