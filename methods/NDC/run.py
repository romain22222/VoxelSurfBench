from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from python_env import python_executable


METHODS = {"ndc", "ndcx", "undc"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NDC on one BINVOX input.")
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--binvox", type=Path, required=True)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-mesh", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    binvox = args.binvox.resolve()
    work_dir = args.work_dir.resolve()
    output_mesh = args.output_mesh.resolve()
    python = python_executable()
    checkpoint_dir = repo_dir / "pretrained" / args.method

    if not (repo_dir / "main.py").is_file():
        raise FileNotFoundError(f"NDC repository not found: {repo_dir}")
    if not python.is_file():
        raise FileNotFoundError(f"Shared Python environment not found: {python}. Run methods/NDC/build.py first.")
    if not binvox.is_file():
        raise FileNotFoundError(f"BINVOX input not found: {binvox}")
    required = [
        checkpoint_dir / f"weights_{args.method}_voxel_bool.pth",
        checkpoint_dir / f"weights_{args.method}_voxel_float.pth",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "NDC pretrained weights are missing. Run methods/NDC/build.py or provide the official weights archive:\n"
            + "\n".join(str(path) for path in missing)
        )

    sample_dir = work_dir / f"{args.method}_run"
    sample_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(python), "main.py",
        "--test_input", str(binvox),
        "--input_type", "voxel",
        "--method", args.method,
        "--checkpoint_dir", str(checkpoint_dir),
        "--sample_dir", str(sample_dir),
        "--gpu", args.gpu,
    ]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(repo_dir), check=True)

    generated = sample_dir / f"quicktest_{args.method}_voxel.obj"
    if not generated.is_file():
        raise FileNotFoundError(f"NDC did not produce the expected mesh: {generated}")
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(generated, output_mesh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
