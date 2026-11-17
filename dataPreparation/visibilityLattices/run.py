from __future__ import annotations

import argparse
import os
import shlex
import shutil
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
REPO_DIR = THIS_DIR / "visibilityLattices"
EXECUTABLE = REPO_DIR / "build" / "visibilityLattices"
sys.path.insert(0, str(ROOT))

from methods.command_runner import run_command


def wsl_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    if len(value) >= 2 and value[1] == ":":
        return f"/mnt/{value[0].lower()}{value[2:]}"
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute DGtal visibility normals for a VOL file.")
    parser.add_argument("--vol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gridstep", type=float, default=1.0)
    parser.add_argument("--radius", type=int, default=5)
    args = parser.parse_args()
    if args.gridstep <= 0:
        parser.error("--gridstep must be positive.")
    if args.radius < 1:
        parser.error("--radius must be positive.")
    return args


def main() -> int:
    args = parse_args()
    volume = args.vol.resolve()
    output = args.output.resolve()
    executable = EXECUTABLE.resolve()
    if not volume.is_file():
        raise FileNotFoundError(f"VOL input not found: {volume}")
    if not executable.is_file():
        raise FileNotFoundError(
            f"Missing {executable}; build visibilityLattices first with "
            "python dataPreparation/visibilityLattices/build.py"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    arguments = [
        "-i", wsl_path(volume) if os.name == "nt" else str(volume),
        "-g", str(args.gridstep),
        "-r", str(args.radius),
        "--noInterface",
        "--saveNormals",
        wsl_path(output) if os.name == "nt" else str(output),
    ]
    if os.name == "nt":
        if not shutil.which("wsl"):
            raise FileNotFoundError("WSL2 is required to run visibilityLattices on Windows.")
        script = shlex.join([wsl_path(executable), *arguments])
        run_command("wsl", ["bash", "-lc", script])
    else:
        run_command(executable, arguments, cwd=REPO_DIR)
    if not output.is_file() or output.stat().st_size == 0:
        raise FileNotFoundError(f"visibilityLattices did not produce normals: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

