"""Run the pipeline for every method (or a subset) on one shape.

    python run_shape.py --shape-dir experiments/shapes/barrel_stove
    python run_shape.py --shape-dir experiments/shapes/barrel_stove --methods mc,dc --overwrite
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import METHODS_FILE, load_registry
from pipeline import run_if_needed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape-dir", type=Path, required=True, help="Shape folder (contains the 'gt' folder).")
    parser.add_argument("--methods", type=str, default="all", help="Comma-separated list or 'all'.")
    parser.add_argument("--methods-file", type=Path, default=METHODS_FILE)
    parser.add_argument("--python", type=str, default=None, help="Python executable forced for Python-based methods.")
    parser.add_argument("--surface-samples", type=int, default=100000)
    parser.add_argument("--overwrite", action="store_true", help="Re-run methods even if their results already exist.")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Resolve the commands without running them.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_registry(args.methods_file)
    if args.methods.strip().lower() == "all":
        methods = list(registry)
    else:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        unknown = [m for m in methods if m not in registry]
        if unknown:
            raise SystemExit(f"Unknown method(s): {', '.join(unknown)}")
    if not methods:
        raise SystemExit(f"No method registered in {args.methods_file} (see add_method.py).")

    shape_dir = args.shape_dir.resolve()
    statuses: dict[str, str] = {}
    for method in methods:
        print(f"[run] {shape_dir.name} / {method}")
        statuses[method] = run_if_needed(
            shape_dir,
            method,
            methods_file=args.methods_file,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            surface_samples=args.surface_samples,
            python_executable=args.python,
        )
        print(f"[{statuses[method]}] {method}")
        if statuses[method] == "failed" and args.stop_on_error:
            break

    print("\n" + "\n".join(f"{m:30s} {s}" for m, s in statuses.items()))
    return 1 if "failed" in statuses.values() else 0


if __name__ == "__main__":
    sys.exit(main())
