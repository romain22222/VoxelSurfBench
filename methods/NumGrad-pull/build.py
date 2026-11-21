from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from command_runner import run_command
from python_env import ensure_venv, python_executable


REPO_URL = "https://github.com/CuiRuikai/NumGrad-Pull.git"
REPO_DIR = THIS_DIR / "NumGrad-Pull"


def ensure_repo() -> None:
    if REPO_DIR.exists():
        required = (
            REPO_DIR / ".git",
            REPO_DIR / "run.py",
            REPO_DIR / "confs" / "ngpull.conf",
            REPO_DIR / "models" / "dataset.py",
            REPO_DIR / "models" / "fields.py",
        )
        if not all(path.exists() for path in required):
            raise RuntimeError(f"{REPO_DIR} exists but is not a NumGrad-Pull checkout.")
        return

    run_command("git", ["clone", "--depth", "1", REPO_URL, REPO_DIR])
    print(f"[info] Cloned NumGrad-Pull to {REPO_DIR}")


def install_environment() -> None:
    python = ensure_venv()
    torch_probe = run_command(
        python,
        ["-c", "import torch; print(torch.version.cuda or '')"],
        check=False,
        echo=False,
        capture_output=True,
        text=True,
    )
    if torch_probe.returncode != 0 or not torch_probe.stdout.strip():
        run_command(python, [
            "-m", "pip", "install", "--disable-pip-version-check",
            "--upgrade", "--force-reinstall",
            "--index-url", "https://download.pytorch.org/whl/cu121", "torch",
        ])
    # run.py imports mcubes (PyMCubes), pyhocon, trimesh, tqdm, scipy.
    run_command(python, [
        "-m", "pip", "install", "--disable-pip-version-check",
        "numpy<1.24", "scipy", "trimesh", "tqdm", "pyhocon", "PyMCubes",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clone NumGrad-Pull and install its CUDA dependencies in the shared environment."
    )
    parser.add_argument("--clone-only", action="store_true", help="Only clone the NumGrad-Pull repository.")
    args = parser.parse_args()

    ensure_repo()
    if args.clone_only:
        print(f"[done] NumGrad-Pull repository cloned at {REPO_DIR}")
        return 0
    install_environment()
    print(f"[done] Shared Python environment: {python_executable().parent.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
