"""Register a method in methods.json.

Command line (every information is a parameter; quote the arguments so your shell does not expand `$VOL`...):

    python add_method.py --name mc --path '$ROOT/adapters/run_mc.py' \
        --args '--vol $VOL --output $OUTPUT_MESH' --output-ext obj \
        --swap-xz --offset-voxels 1.0 --offset-voxels-grid 128=-0.5

Interactive (no parameter, or -i): the script asks for each piece of information.

Placeholders usable in path/cwd/args: $ROOT $SHAPE_DIR $SHAPE_NAME $GT_DIR $METHOD $METHOD_DIR $WORKDIR
$OUTPUT_MESH $VOL $BINVOX $VISNORMS $GT_MESH $SOURCE_JSON $CAMERA_JSON (see pipeline.py, alignment.py).
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any

from alignment import DEFAULT_ALIGNMENT
from common import GT_DIRNAME, METHODS_FILE, load_registry, mesh_extensions, save_registry

NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--interactive", action="store_true", help="Ask for the information interactively.")
    parser.add_argument("--methods-file", type=Path, default=METHODS_FILE)
    parser.add_argument("--name", help="Method name (also the result folder name).")
    parser.add_argument("--path", help="Script (.py, run with Python) or executable.")
    parser.add_argument("--args", default="", help="Arguments of the command, as one string (shell syntax).")
    parser.add_argument("--cwd", help="Working directory.")
    parser.add_argument("--python", help="Python interpreter for a .py path.")
    parser.add_argument("--steps-json", help="JSON list (or file) of steps {path, args, cwd, python}; replaces --path/--args.")
    parser.add_argument("--output-ext", default="obj", help="Format of the method output / aligned mesh.")
    parser.add_argument("--output-path", help="Where the method writes its result, if not $OUTPUT_MESH.")
    parser.add_argument("--output-glob", help="Glob locating the method result, if not $OUTPUT_MESH.")
    parser.add_argument("--frame", choices=["voxel", "gt"], default=DEFAULT_ALIGNMENT["frame"],
                        help="'voxel': raw output in voxel index units; 'gt': already in the GT frame [-0.5, 0.5]^3.")
    parser.add_argument("--swap-xz", action="store_true", help="Raw output has X and Z swapped.")
    parser.add_argument("--scale-divisor", type=float, default=DEFAULT_ALIGNMENT["scale_divisor"],
                        help="Extra scale divisor (e.g. 2 for a 2x finer grid).")
    parser.add_argument("--offset-voxels", type=float, default=DEFAULT_ALIGNMENT["offset_voxels"])
    parser.add_argument("--offset-voxels-grid", action="append", default=[], metavar="DIM=OFFSET",
                        help="Offset for a specific grid dimension, e.g. 128=-0.5 (repeatable).")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing method of the same name.")
    return parser.parse_args()


def make_step(path: str, args: str = "", cwd: str | None = None, python: str | None = None) -> dict[str, Any]:
    step: dict[str, Any] = {"path": path}
    if cwd:
        step["cwd"] = cwd
    if python:
        step["python"] = python
    step["args"] = shlex.split(args) if args else []
    return step


def build_spec(
    *,
    steps: list[dict[str, Any]],
    output_ext: str,
    output_path: str | None,
    output_glob: str | None,
    frame: str,
    swap_xz: bool,
    scale_divisor: float,
    offset_voxels: float,
    offset_by_grid: dict[str, float],
) -> dict[str, Any]:
    if not steps:
        raise ValueError("A path (or steps) is required")
    spec: dict[str, Any] = dict(steps[0]) if len(steps) == 1 else {"steps": steps}
    ext = output_ext.lstrip(".").lower()
    if "." + ext not in mesh_extensions():
        raise ValueError(f"Unsupported output format '{ext}' (supported: {', '.join(mesh_extensions())})")
    spec["output_ext"] = ext
    if output_path:
        spec["output_path"] = output_path
    if output_glob:
        spec["output_glob"] = output_glob
    alignment: dict[str, Any] = {
        "frame": frame,
        "swap_xz": swap_xz,
        "scale_divisor": scale_divisor,
        "offset_voxels": offset_voxels,
    }
    if offset_by_grid:
        alignment["offset_voxels_by_grid"] = offset_by_grid
    spec["alignment"] = alignment
    return spec


def check_name(name: str) -> None:
    if not NAME_PATTERN.match(name):
        raise ValueError("Method name must match [A-Za-z0-9][A-Za-z0-9_.-]*")
    if name == GT_DIRNAME:
        raise ValueError(f"'{GT_DIRNAME}' is reserved")


def parse_grid_offsets(items: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items:
        dim, _, offset = item.partition("=")
        if not dim.strip().isdigit() or not offset:
            raise ValueError(f"Invalid --offset-voxels-grid '{item}', expected DIM=OFFSET")
        out[dim.strip()] = float(offset)
    return out


def load_steps(value: str) -> list[dict[str, Any]]:
    text = Path(value).read_text(encoding="utf-8") if Path(value).is_file() else value
    steps = json.loads(text)
    if not isinstance(steps, list) or not all(isinstance(s, dict) and "path" in s for s in steps):
        raise ValueError("--steps-json must be a list of objects with at least a 'path'")
    return steps


# ----------------------------------------------------------------------------- interactive

def ask(prompt: str, default: str | None = None) -> str:
    """Empty answer returns `default`; with default None the question is repeated until answered."""
    suffix = f" [{default}]" if default else ""
    while True:
        answer = input(f"{prompt}{suffix}: ").strip()
        if answer:
            return answer
        if default is not None:
            return default
        print("  A value is required.")


def ask_yes_no(prompt: str, default: bool) -> bool:
    return ask(f"{prompt} (y/n)", "y" if default else "n").lower().startswith("y")


def ask_float(prompt: str, default: float) -> float:
    while True:
        try:
            return float(ask(prompt, str(default)))
        except ValueError:
            print("  Not a number.")


def interactive(registry: dict[str, Any]) -> tuple[str, dict[str, Any], bool]:
    print("Register a new method. Placeholders: $ROOT $SHAPE_DIR $SHAPE_NAME $GT_DIR $METHOD $METHOD_DIR $WORKDIR")
    print("$OUTPUT_MESH $VOL $BINVOX $VISNORMS $GT_MESH $SOURCE_JSON $CAMERA_JSON\n")
    while True:
        name = ask("Method name")
        try:
            check_name(name)
            break
        except ValueError as exc:
            print(f"  {exc}")
    force = False
    if name in registry:
        force = ask_yes_no(f"'{name}' already exists, overwrite?", False)
        if not force:
            raise SystemExit("Aborted.")

    steps: list[dict[str, Any]] = []
    while True:
        label = f"step {len(steps) + 1}" if steps else "command"
        path = ask(f"Path of the script/executable ({label})")
        args = ask("Arguments (one line, shell syntax; empty = none)", "")
        cwd = ask("Working directory (empty = none)", "")
        python = ask("Python interpreter (empty = current one)", "") if path.lower().endswith(".py") else ""
        steps.append(make_step(path, args, cwd, python))
        if not ask_yes_no("Add another command to run afterwards?", False):
            break

    ext = ask(f"Output format ({', '.join(mesh_extensions())})", "obj")
    output_path = ask("Output path if not $OUTPUT_MESH (empty = $OUTPUT_MESH)", "")
    output_glob = "" if output_path else ask("Output glob if not $OUTPUT_MESH (empty = none)", "")

    print("\nAlignment of the raw output to the GT frame [-0.5, 0.5]^3")
    frame = "gt" if ask_yes_no("Is the raw output already in the GT frame?", False) else "voxel"
    swap_xz, scale_divisor, offset = False, DEFAULT_ALIGNMENT["scale_divisor"], DEFAULT_ALIGNMENT["offset_voxels"]
    by_grid: dict[str, float] = {}
    if frame == "voxel":
        swap_xz = ask_yes_no("Are the X and Z axes swapped?", False)
        scale_divisor = ask_float("Extra scale divisor (2 for a 2x finer grid)", scale_divisor)
        offset = ask_float("Constant offset in voxels", offset)
        while ask_yes_no("Add an offset specific to one grid dimension?", False):
            dim = ask("Grid dimension (e.g. 128)")
            if dim.isdigit():
                by_grid[dim] = ask_float(f"Offset in voxels for {dim}^3", offset)
            else:
                print("  Not an integer.")

    spec = build_spec(
        steps=steps,
        output_ext=ext,
        output_path=output_path or None,
        output_glob=output_glob or None,
        frame=frame,
        swap_xz=swap_xz,
        scale_divisor=scale_divisor,
        offset_voxels=offset,
        offset_by_grid=by_grid,
    )
    return name, spec, force


def main() -> int:
    args = parse_args()
    registry = load_registry(args.methods_file)
    try:
        if args.interactive or args.name is None:
            name, spec, force = interactive(registry)
        else:
            name, force = args.name, args.force
            check_name(name)
            steps = load_steps(args.steps_json) if args.steps_json else (
                [make_step(args.path, args.args, args.cwd, args.python)] if args.path else []
            )
            spec = build_spec(
                steps=steps,
                output_ext=args.output_ext,
                output_path=args.output_path,
                output_glob=args.output_glob,
                frame=args.frame,
                swap_xz=args.swap_xz,
                scale_divisor=args.scale_divisor,
                offset_voxels=args.offset_voxels,
                offset_by_grid=parse_grid_offsets(args.offset_voxels_grid),
            )
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    if name in registry and not force:
        print(f"[error] method '{name}' already exists (use --force to overwrite)", file=sys.stderr)
        return 1
    registry[name] = spec
    save_registry(registry, args.methods_file)
    print(f"[done] '{name}' written to {args.methods_file}")
    print(json.dumps({name: spec}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
