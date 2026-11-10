"""Run the pipeline for one method on every shape of a shapes root (folders containing a 'gt' folder).

    python run_method.py --method mc --shapes-root shapes
    python run_method.py --method mc --shapes-root shapes --shapes barrel_stove,modular_pipes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import METHODS_FILE, list_shape_dirs, load_registry
from pipeline import recenter_existing_method, run_if_needed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True)
    parser.add_argument("--shapes-root", type=Path, default=Path("experiments"))
    parser.add_argument(
        "--shapes",
        type=str,
        default="all",
        help="Comma-separated shape names or paths relative to --shapes-root, or 'all'.",
    )
    parser.add_argument("--methods-file", type=Path, default=METHODS_FILE)
    parser.add_argument("--python", type=str, default=None, help="Python executable forced for Python-based methods.")
    parser.add_argument("--surface-samples", type=int, default=100000)
    parser.add_argument("--overwrite", action="store_true", help="Re-run shapes even if their results already exist.")
    parser.add_argument(
        "--recenter-only",
        action="store_true",
        help="Recompute volume-based recentering and metrics from existing base meshes.",
    )
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Resolve the commands without running them.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.recenter_only and (args.overwrite or args.dry_run):
        raise SystemExit("--recenter-only cannot be combined with --overwrite or --dry-run.")
    if args.method not in load_registry(args.methods_file):
        raise SystemExit(f"Unknown method '{args.method}' in {args.methods_file} (see add_method.py).")

    shape_dirs = list_shape_dirs(args.shapes_root.resolve())
    if args.shapes.strip().lower() != "all":
        wanted = [s.strip() for s in args.shapes.split(",") if s.strip()]
        root = args.shapes_root.resolve()
        by_relative = {str(p.relative_to(root)): p for p in shape_dirs}
        by_name: dict[str, list[Path]] = {}
        for path in shape_dirs:
            by_name.setdefault(path.name, []).append(path)
        selected: list[Path] = []
        unknown: list[str] = []
        ambiguous: list[str] = []
        for name in wanted:
            normalized = name.replace("/", "\\")
            if normalized in by_relative:
                selected.append(by_relative[normalized])
            elif name in by_name and len(by_name[name]) == 1:
                selected.append(by_name[name][0])
            elif name in by_name:
                ambiguous.append(name)
            else:
                unknown.append(name)
        if unknown:
            raise SystemExit(f"Unknown shape(s): {', '.join(unknown)}")
        if ambiguous:
            raise SystemExit(
                f"Ambiguous shape name(s): {', '.join(ambiguous)}; use paths relative to --shapes-root, "
                "e.g. shapes_128/barrel_stove"
            )
        shape_dirs = selected
    if not shape_dirs:
        raise SystemExit(f"No shape folder (with a 'gt' folder) found in {args.shapes_root}")

    statuses: dict[str, str] = {}
    for shape_dir in shape_dirs:
        print(f"[run] {args.method} / {shape_dir.name}")
        if args.recenter_only:
            try:
                recenter_existing_method(
                    shape_dir,
                    args.method,
                    methods_file=args.methods_file,
                    surface_samples=args.surface_samples,
                )
                statuses[shape_dir.name] = "success"
            except Exception as exc:
                print(f"[error] {shape_dir.name}/{args.method}: {exc}")
                statuses[shape_dir.name] = "failed"
        else:
            statuses[shape_dir.name] = run_if_needed(
                shape_dir,
                args.method,
                methods_file=args.methods_file,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                surface_samples=args.surface_samples,
                python_executable=args.python,
            )
        print(f"[{statuses[shape_dir.name]}] {shape_dir.name}")
        if statuses[shape_dir.name] == "failed" and args.stop_on_error:
            break

    print("\n" + "\n".join(f"{s:30s} {st}" for s, st in statuses.items()))
    return 1 if "failed" in statuses.values() else 0


if __name__ == "__main__":
    sys.exit(main())
