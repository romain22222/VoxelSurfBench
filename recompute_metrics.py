"""Recompute selected C++ metrics for successful runs.

Examples:
    python recompute_metrics.py --measure all --shapes-root experiments
    python recompute_metrics.py --measure trim90 --methods mc,dc --workers 4
    python recompute_metrics.py --measure chamfer_l1 --since 2026-01-01T00:00:00+00:00 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    ALIGNED_STEM,
    RESULTS_NAME,
    grid_dim_from_vol,
    list_shape_dirs,
    load_gt_inputs,
    load_mesh,
    mesh_extensions,
    read_json,
    write_json,
)
from metrics import ALL_METRIC_KEYS, NORMAL_ANGLE_KEYS, TRIM90_KEYS, voxel_scale_metrics

DEFAULT_SHAPES_ROOT = Path("experiments")


def _ignore_sigint() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute selected metrics for all successful method runs.")
    parser.add_argument("--measure", required=True, help="all, trim90, normal_angles, new, or one metric key.")
    parser.add_argument("--shapes-root", type=Path, default=DEFAULT_SHAPES_ROOT)
    parser.add_argument("--methods", type=str, default="all", help="Comma-separated method names or 'all'.")
    parser.add_argument("--exclude-methods", type=str, default="", help="Comma-separated method names to skip.")
    parser.add_argument("--surface-samples", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--since", type=str, default="", help="Skip results recomputed at/after this ISO UTC time.")
    parser.add_argument("--dry-run", action="store_true", help="Compute metrics but don't write results.json.")
    return parser.parse_args()


def resolve_methods_arg(value: str) -> set[str] | None:
    if value.strip().lower() == "all":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def find_mesh(directory: Path, stem: str) -> Path | None:
    for ext in mesh_extensions():
        candidate = directory / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def compute_all(pred_path: Path, gt_path: Path, vol_path: Path, surface_samples: int) -> dict[str, Any]:
    return voxel_scale_metrics(
        load_mesh(pred_path),
        load_mesh(gt_path),
        grid_dim_from_vol(vol_path),
        n_samples=surface_samples,
    )


def collect_jobs(args: argparse.Namespace) -> tuple[list[tuple[Path, dict[str, Any], Path, Path, Path]], int]:
    selected = resolve_methods_arg(args.methods)
    excluded = {m.strip() for m in args.exclude_methods.split(",") if m.strip()}
    jobs = []
    skipped = 0

    for shape_dir in list_shape_dirs(args.shapes_root.resolve()):
        try:
            inputs = load_gt_inputs(shape_dir, require_sidecars=False)
        except (OSError, ValueError) as exc:
            print(f"[skip] {shape_dir.name}: {exc}")
            skipped += 1
            continue
        if not inputs.vol.is_file():
            skipped += 1
            continue
        for method_dir in sorted(p for p in shape_dir.iterdir() if p.is_dir() and p.name != "gt"):
            results_path = method_dir / RESULTS_NAME
            if not results_path.is_file():
                continue
            try:
                payload = read_json(results_path)
            except (OSError, ValueError) as exc:
                print(f"[skip] {results_path}: invalid JSON ({exc})")
                skipped += 1
                continue
            if not isinstance(payload, dict):
                print(f"[skip] {results_path}: results must be a JSON object")
                skipped += 1
                continue
            method = str(payload.get("method", method_dir.name))
            if (
                method in excluded
                or payload.get("status") != "success"
                or (selected is not None and method not in selected)
                or (args.since and str(payload.get("metrics_recomputed_at", "")) >= args.since)
            ):
                skipped += 1
                continue
            pred_path = find_mesh(method_dir, ALIGNED_STEM)
            if pred_path is None:
                skipped += 1
                continue
            jobs.append((results_path, payload, pred_path, inputs.gt_mesh, inputs.vol))
    return jobs, skipped


def apply_measure(payload: dict[str, Any], computed: dict[str, Any], measure: str) -> None:
    current = payload.get("metrics")
    if not isinstance(current, dict):
        current = {}
        payload["metrics"] = current
    if measure == "all":
        current.update(computed)
    elif measure == "trim90":
        current.update({key: computed[key] for key in TRIM90_KEYS})
    elif measure == "new":
        for key, value in computed.items():
            if current.get(key) is None:
                current[key] = value
    elif measure == "normal_angles":
        current.update({key: computed[key] for key in NORMAL_ANGLE_KEYS})
    else:
        current[measure] = computed[measure]


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.surface_samples < 1:
        raise ValueError("--surface-samples must be at least 1")
    if args.measure not in {"all", "normal_angles", "trim90", "new"} and args.measure not in ALL_METRIC_KEYS:
        raise KeyError(f"Unknown metric key: {args.measure}")

    jobs, skipped = collect_jobs(args)
    if not jobs:
        print(f"[done] updated=0 skipped={skipped} dry_run={args.dry_run}")
        return 0

    changed = 0
    pool = ProcessPoolExecutor(max_workers=args.workers, initializer=_ignore_sigint)
    try:
        futures = {
            pool.submit(compute_all, pred_path, gt_path, vol_path, args.surface_samples): (results_path, payload)
            for results_path, payload, pred_path, gt_path, vol_path in jobs
        }
        for future in as_completed(futures):
            results_path, payload = futures[future]
            try:
                computed = future.result()
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                print(f"[err] {results_path}: {exc}", flush=True)
                continue
            apply_measure(payload, computed, args.measure)
            payload["metrics_recomputed_at"] = datetime.now(timezone.utc).isoformat()
            payload["metrics_recomputed_key"] = args.measure
            if not args.dry_run:
                write_json(results_path, payload)
            changed += 1
            print(f"[ok] {results_path} ({changed}/{len(jobs)})", flush=True)
    except KeyboardInterrupt:
        print(f"\n[interrupted] updated={changed}/{len(jobs)}; stopping workers...", flush=True)
        pool.shutdown(wait=False, cancel_futures=True)
        for process in list(getattr(pool, "_processes", {}).values()):
            process.terminate()
        return 130
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    print(f"[done] updated={changed} skipped={skipped} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
