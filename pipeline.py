"""Run one registered reconstruction method on one shape.

Registry entry (methods.json), placeholders `$NAME` are expanded in `path`, `cwd`, `args`, `python`, `output_*`:

    "my_method": {
        "path": "$ROOT/adapters/run_my_method.py",   # .py files are run with Python, anything else as an executable
        "cwd": "$ROOT",                              # optional
        "args": ["--vol", "$VOL", "--output", "$OUTPUT_MESH"],
        "python": "...", "python_candidates": [...], # optional interpreter(s) for .py paths
        "steps": [ {path, cwd, args}, ... ],         # optional: several commands run in sequence instead of path/args
        "output_ext": "obj",                         # format of base.<ext> / aligned.<ext>
        "output_path": "...", "output_glob": "...",  # optional: where to find the result if not written to $OUTPUT_MESH
        "alignment": { ... }                         # see alignment.py
    }

Placeholders: ROOT, SHAPE_DIR, SHAPE_NAME, GT_DIR, METHOD, METHOD_DIR, WORKDIR (temporary), OUTPUT_MESH,
VOL, BINVOX, VISNORMS, GT_MESH, SOURCE_JSON, CAMERA_JSON.

Output layout: <shape>/<method>/{results.json, aligned.<ext>, base.<ext>, method.log}.
"""
from __future__ import annotations

import glob as globmod
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from alignment import align_mesh
from common import (
    ALIGNED_STEM,
    BASE_STEM,
    LOG_NAME,
    METHODS_FILE,
    RESULTS_NAME,
    ROOT,
    get_handler,
    grid_dim_from_vol,
    load_gt_inputs,
    load_mesh,
    load_registry,
    mesh_extensions,
    method_dir,
    read_json,
    write_json,
)
from metrics import voxel_scale_metrics


def _expand(value: Any, context: dict[str, str]) -> Any:
    if isinstance(value, str):
        for key in sorted(context, key=len, reverse=True):
            value = value.replace(f"${key}", context[key])
        return value
    if isinstance(value, list):
        return [_expand(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item, context) for key, item in value.items()}
    return value


def _resolve_python(step: dict[str, Any], context: dict[str, str], python_executable: str | None) -> str:
    candidates = [python_executable] if python_executable else []
    if not candidates:
        if "python" in step:
            candidates.append(str(_expand(step["python"], context)))
        candidates += [str(_expand(v, context)) for v in step.get("python_candidates", [])]
    for candidate in candidates:
        if Path(candidate).exists():
            return str(candidate)
        found = shutil.which(candidate)
        if found:
            return found
        if python_executable:
            raise FileNotFoundError(f"Python executable not found: {candidate}")
    return sys.executable


def _command_from_step(step: dict[str, Any], context: dict[str, str], python_executable: str | None) -> tuple[list[str], Path | None]:
    raw_path = str(_expand(step["path"], context))
    args = [str(arg) for arg in _expand(step.get("args", []), context)]
    cwd = Path(_expand(step["cwd"], context)).resolve() if step.get("cwd") is not None else None
    if Path(raw_path).suffix.lower() == ".py":
        return [_resolve_python(step, context, python_executable), raw_path] + args, cwd
    return [raw_path] + args, cwd


def _collect_output(spec: dict[str, Any], context: dict[str, str], work_dir: Path, output_mesh: Path) -> Path:
    if output_mesh.exists():
        return output_mesh
    if "output_path" in spec:
        path = Path(_expand(spec["output_path"], context))
        if path.exists():
            return path
    if "output_glob" in spec:
        matches = [Path(p) for p in globmod.glob(str(_expand(spec["output_glob"], context)), recursive=True)]
        matches = [p for p in matches if p.is_file()]
        if matches:
            return max(matches, key=lambda p: p.stat().st_mtime)
    exts = set(mesh_extensions())
    candidates = [p for p in work_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(f"Could not locate the output of method {context['METHOD']}")


def _find_output(directory: Path, stem: str) -> Path | None:
    for ext in mesh_extensions():
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def is_done(shape_dir: Path, method: str) -> bool:
    mdir = method_dir(shape_dir, method)
    results = mdir / RESULTS_NAME
    if (
        not results.exists()
        or _find_output(mdir, ALIGNED_STEM) is None
        or _find_output(mdir, BASE_STEM) is None
        or not (mdir / LOG_NAME).is_file()
    ):
        return False
    try:
        return read_json(results).get("status") == "success"
    except (OSError, ValueError):
        return False


def execute_registered_method(
    *,
    shape_dir: Path,
    method: str,
    methods_file: Path = METHODS_FILE,
    dry_run: bool = False,
    surface_samples: int = 100000,
    python_executable: str | None = None,
) -> dict[str, Any]:
    inputs = load_gt_inputs(shape_dir)
    registry = load_registry(methods_file)
    if method not in registry:
        raise KeyError(f"Unknown method '{method}' in {methods_file}")
    spec = registry[method]
    output_ext = "." + str(spec.get("output_ext", "obj")).lstrip(".").lower()
    handler = get_handler(output_ext)

    mdir = method_dir(inputs.shape_dir, method)
    base_path = mdir / f"{BASE_STEM}{output_ext}"
    aligned_path = mdir / f"{ALIGNED_STEM}{output_ext}"
    results_path = mdir / RESULTS_NAME
    log_path = mdir / LOG_NAME

    work_dir = Path(tempfile.mkdtemp(prefix=f"voxelsurfbench_{inputs.name}_{method}_"))
    try:
        context = {
            "ROOT": str(ROOT),
            "SHAPE_DIR": str(inputs.shape_dir),
            "SHAPE_NAME": inputs.name,
            "GT_DIR": str(inputs.gt_dir),
            "METHOD": method,
            "METHOD_DIR": str(mdir),
            "WORKDIR": str(work_dir),
            "OUTPUT_MESH": str(base_path),
            "VOL": str(inputs.vol),
            "BINVOX": str(inputs.binvox),
            "VISNORMS": str(inputs.visnormals),
            "GT_MESH": str(inputs.gt_mesh),
            "SOURCE_JSON": str(inputs.source_json),
            "CAMERA_JSON": str(inputs.camera_json),
        }
        steps = spec["steps"] if "steps" in spec else [spec]
        plan = [_command_from_step(step, context, python_executable) for step in steps]
        run_info: dict[str, Any] = {
            "shape": inputs.name,
            "method": method,
            "status": "pending",
            "commands": [cmd for cmd, _ in plan],
        }
        if dry_run:
            run_info["status"] = "dry-run"
            return run_info

        mdir.mkdir(parents=True, exist_ok=True)
        for stem in (BASE_STEM, ALIGNED_STEM):
            for ext in mesh_extensions():
                old_output = mdir / f"{stem}{ext}"
                if old_output.exists():
                    old_output.unlink()
        for old_file in (results_path, log_path):
            if old_file.exists():
                old_file.unlink()

        started = time.time()
        try:
            with log_path.open("w", encoding="utf-8") as log:
                for command, cwd in plan:
                    log.write("+ " + " ".join(command) + "\n")
                    log.flush()
                    ret = subprocess.run(command, cwd=str(cwd) if cwd else None, stdout=log, stderr=subprocess.STDOUT, text=True).returncode
                    if ret != 0:
                        raise RuntimeError(f"Method '{method}' failed with exit code {ret}. See {log_path}")
            run_info["elapsed_sec"] = time.time() - started

            collected = _collect_output(spec, context, work_dir, base_path)
            if collected.resolve() != base_path.resolve():
                if collected.suffix.lower() == output_ext:
                    shutil.copy2(collected, base_path)
                else:
                    handler.save(get_handler(collected).load(collected), base_path)

            gt_mesh = load_mesh(inputs.gt_mesh)
            pred = handler.load(base_path)
            if not hasattr(pred, "faces"):
                raise ValueError("Method output is a point cloud; a triangle mesh is required")
            grid_dim = grid_dim_from_vol(inputs.vol)
            aligned, alignment = align_mesh(pred, spec.get("alignment"), grid_dim)
            handler.save(aligned, aligned_path)
            run_info["metrics"] = voxel_scale_metrics(aligned, gt_mesh, grid_dim, n_samples=surface_samples)
            run_info["alignment"] = alignment
            run_info["status"] = "success"
        except Exception as exc:
            run_info["status"] = "failed"
            run_info["error"] = str(exc)
            write_json(results_path, run_info)
            raise
        run_info["files"] = {
            "base": base_path.name,
            "aligned": aligned_path.name,
            "log": log_path.name,
        }
        write_json(results_path, run_info)
        return run_info
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def run_if_needed(
    shape_dir: Path,
    method: str,
    *,
    methods_file: Path = METHODS_FILE,
    overwrite: bool = False,
    dry_run: bool = False,
    surface_samples: int = 100000,
    python_executable: str | None = None,
) -> str:
    """Run `method` on `shape_dir` unless already done. Returns 'skipped', 'success', 'dry-run' or 'failed'."""
    if not overwrite and not dry_run and is_done(shape_dir, method):
        return "skipped"
    try:
        info = execute_registered_method(
            shape_dir=shape_dir,
            method=method,
            methods_file=methods_file,
            dry_run=dry_run,
            surface_samples=surface_samples,
            python_executable=python_executable,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {shape_dir.name}/{method}: {exc}")
        return "failed"
    return info["status"]
