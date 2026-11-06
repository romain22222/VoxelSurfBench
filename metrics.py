"""Python orchestration for the C++ metrics executables in metrics/."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import trimesh

from common import ROOT, get_handler
from methods.command_runner import run_command

METRICS_BIN_DIR = Path(os.environ.get("VOXELSURFBENCH_METRICS_BIN_DIR", ROOT / "metrics" / "build" / "bin"))

SURFACE_KEYS = {
    "chamfer_l1",
    "chamfer_l2",
    "chamfer_rmse",
    "hausdorff",
    "mean_distance_pred_to_gt",
    "mean_distance_gt_to_pred",
    "max_distance_pred_to_gt",
    "max_distance_gt_to_pred",
    "normal_angle_gt_to_pred_deg",
    "normal_angle_pred_to_gt_deg",
    "normal_angle_mean_deg",
}
TOPOLOGY_KEYS = {
    "vertex_count",
    "face_count",
    "component_count",
    "watertight",
    "winding_consistent",
    "euler_number",
    "surface_area",
    "volume",
}
BBOX_KEYS = {
    "bbox_center_distance",
    "bbox_extent_distance",
    "bbox_extent_ratio",
    "bbox_iou",
}
NORMAL_ANGLE_KEYS = {
    "normal_angle_gt_to_pred_deg",
    "normal_angle_pred_to_gt_deg",
    "normal_angle_mean_deg",
}
TRIM90_KEYS = {
    "chamfer_l1_trim90",
    "chamfer_l2_trim90",
    "chamfer_rmse_trim90",
    "hausdorff_trim90",
    "mean_distance_pred_to_gt_trim90",
    "mean_distance_gt_to_pred_trim90",
    "max_distance_pred_to_gt_trim90",
    "max_distance_gt_to_pred_trim90",
    "normal_angle_pred_to_gt_trim90_deg",
    "normal_angle_gt_to_pred_trim90_deg",
    "normal_angle_mean_trim90_deg",
}
ALL_METRIC_KEYS = SURFACE_KEYS | TOPOLOGY_KEYS | BBOX_KEYS | TRIM90_KEYS


def _executable(name: str) -> Path:
    candidates = [METRICS_BIN_DIR / name, METRICS_BIN_DIR / f"{name}.exe"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Missing C++ metric executable '{name}'. Build them with "
        f"`cmake -S metrics -B metrics/build && cmake --build metrics/build --config Release`, "
        f"or set VOXELSURFBENCH_METRICS_BIN_DIR (looked in {METRICS_BIN_DIR})."
    )


def _run_json(name: str, *args: str) -> dict[str, Any]:
    proc = run_command(
        _executable(name),
        args,
        check=False,
        echo=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed ({proc.returncode}): {proc.stdout[-500:]} {proc.stderr[-500:]}")
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{name} did not return valid JSON: {proc.stdout[-500:]}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{name} returned a non-object JSON result")
    return result


def _export_obj(mesh: trimesh.Trimesh, path: Path) -> None:
    get_handler(".obj").save(mesh, path)


def to_voxel_scale(mesh: trimesh.Trimesh, grid_dim: int) -> trimesh.Trimesh:
    """Map a mesh in the GT frame ([-0.5, 0.5]^3) to voxel units ([0, grid_dim]^3)."""
    vertices = (mesh.vertices + 0.5) * float(grid_dim)
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)


def surface_metrics(pred: trimesh.Trimesh, gt: trimesh.Trimesh, *, n_samples: int = 100000) -> dict[str, float]:
    if n_samples <= 0:
        raise ValueError("surface sample count must be greater than zero")
    with tempfile.TemporaryDirectory(prefix="voxelsurfbench_metrics_") as tmp:
        pred_path, gt_path = Path(tmp) / "pred.obj", Path(tmp) / "gt.obj"
        _export_obj(pred, pred_path)
        _export_obj(gt, gt_path)
        data = _run_json("objCompare", str(pred_path), str(gt_path), "-n", str(int(n_samples)), "--json")
    keys = SURFACE_KEYS | TRIM90_KEYS
    missing = keys.difference(data)
    if missing:
        raise ValueError(f"objCompare is missing metric keys: {', '.join(sorted(missing))}")
    return {key: data[key] for key in sorted(keys)}


def topology_metrics(mesh: trimesh.Trimesh) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="voxelsurfbench_metrics_") as tmp:
        path = Path(tmp) / "mesh.obj"
        _export_obj(mesh, path)
        return _run_json("objTopology", str(path))


def bbox_metrics(pred: trimesh.Trimesh, gt: trimesh.Trimesh) -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix="voxelsurfbench_metrics_") as tmp:
        pred_path, gt_path = Path(tmp) / "pred.obj", Path(tmp) / "gt.obj"
        _export_obj(pred, pred_path)
        _export_obj(gt, gt_path)
        return _run_json("objBbox", str(pred_path), str(gt_path))


def voxel_scale_metrics(
    pred: trimesh.Trimesh, gt: trimesh.Trimesh, grid_dim: int, *, n_samples: int = 100000
) -> dict[str, Any]:
    """All metrics (surface, topology, bbox) measured in voxel units via C++ programs."""
    pred_v, gt_v = to_voxel_scale(pred, grid_dim), to_voxel_scale(gt, grid_dim)
    metrics: dict[str, Any] = surface_metrics(pred_v, gt_v, n_samples=n_samples)
    metrics.update(topology_metrics(pred_v))
    metrics.update(bbox_metrics(pred_v, gt_v))
    metrics["unit"] = "voxel"
    metrics["grid_dim"] = int(grid_dim)
    return metrics
