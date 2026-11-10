from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from command_runner import run_command
from python_env import python_executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeuralPull with optional visibility normals.")
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--visnormals", type=Path)
    parser.add_argument("--shape-name", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-mesh", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def create_config(template: Path, destination: Path, data_dir: Path, output_root: Path) -> None:
    replacements = {
        "data_dir": data_dir.resolve().as_posix().rstrip("/") + "/",
        "base_exp_dir": output_root.resolve().as_posix().rstrip("/") + "/",
    }
    found = set()
    lines: list[str] = []
    for line in template.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^(\s*)(data_dir|base_exp_dir)\s*=.*$", line)
        if match:
            key = match.group(2)
            value = replacements[key]
            lines.append(f'{match.group(1)}{key} = "{value}"')
            found.add(key)
        else:
            lines.append(line)
    missing = set(replacements) - found
    if missing:
        raise ValueError(f"NPN config is missing settings: {', '.join(sorted(missing))}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    points = args.points.resolve()
    visnormals = args.visnormals.resolve() if args.visnormals else None
    work_dir = args.work_dir.resolve()
    output_mesh = args.output_mesh.resolve()
    python = python_executable()

    if not (repo_dir / "run.py").is_file():
        raise FileNotFoundError(f"NPN repository not found: {repo_dir}. Run methods/NPN/build.py first.")
    if not python.is_file():
        raise FileNotFoundError(f"Shared Python environment not found: {python}. Run methods/NPN/build.py first.")
    if not points.is_file():
        raise FileNotFoundError(f"NPN point cloud not found: {points}")
    if visnormals is not None and not visnormals.is_file():
        raise FileNotFoundError(f"NPN visibility normals not found: {visnormals}")

    data_dir = work_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(points, data_dir / f"{args.shape_name}.ply")
    visnormals_data = data_dir / f"{args.shape_name}_visnormals.xyz"
    if visnormals is not None:
        shutil.copy2(visnormals, visnormals_data)
    elif visnormals_data.exists():
        visnormals_data.unlink()

    run_name = "reconstruction"
    output_root = work_dir / "outs"
    config = work_dir / f"{args.shape_name}_npull.conf"
    create_config(repo_dir / "confs" / "npull.conf", config, data_dir, output_root)

    check_cuda = (
        "import torch\n"
        "if not torch.cuda.is_available():\n"
        "    raise SystemExit('NPN requires a CUDA-capable GPU and CUDA-enabled PyTorch.')"
    )
    run_command(python, ["-c", check_cuda])
    arguments = [
        repo_dir / "run.py",
        "--gpu", str(args.gpu),
        "--conf", str(config),
        "--dataname", args.shape_name,
        "--dir", run_name,
    ]
    run_command(python, arguments, cwd=repo_dir)

    output_dir = output_root / run_name / "outputs"
    candidates = sorted(output_dir.glob("*.ply"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"NPN did not produce a mesh in {output_dir}")
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidates[-1], output_mesh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
