from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from python_env import ensure_venv, python_executable


REPO_URL = "https://github.com/sbreuils/discreteGeometry-NeuralSurface.git"
SOURCE_PATH = "NeuralPull-normals"
REPO_DIR = THIS_DIR / "NPN"
SOURCE_FILES = (
    "run.py",
    "confs/npull.conf",
    "models/dataset.py",
    "models/fields.py",
    "models/utils.py",
)


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def ensure_repo() -> None:
    if REPO_DIR.exists():
        if not (REPO_DIR / ".git").exists() or not (REPO_DIR / "run.py").is_file():
            raise RuntimeError(f"{REPO_DIR} exists but is not the standalone NPN repository.")
        return

    with tempfile.TemporaryDirectory(prefix="npn-source-") as temp:
        upstream = Path(temp) / "upstream"
        run([
            "git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1",
            REPO_URL, str(upstream),
        ])
        run(["git", "-C", str(upstream), "sparse-checkout", "init", "--no-cone"])
        patterns = [f"/{SOURCE_PATH}/{name}" for name in SOURCE_FILES]
        run(["git", "-C", str(upstream), "sparse-checkout", "set", "--no-cone", *patterns])
        run(["git", "-C", str(upstream), "checkout", "--detach"])

        source_root = upstream / SOURCE_PATH
        REPO_DIR.mkdir(parents=True)
        for relative in SOURCE_FILES:
            source = source_root / relative
            if not source.is_file():
                raise FileNotFoundError(f"Required NPN source file is missing: {source}")
            destination = REPO_DIR / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    run(["git", "init", str(REPO_DIR)])
    print("[info] Created a standalone NPN repository from only the method source files.")


def install_environment() -> None:
    python = ensure_venv()
    torch_probe = subprocess.run(
        [str(python), "-c", "import torch; print(torch.version.cuda or '')"],
        capture_output=True,
        text=True,
        check=False,
    )
    if torch_probe.returncode != 0 or not torch_probe.stdout.strip():
        run([
            str(python), "-m", "pip", "install", "--disable-pip-version-check",
            "--upgrade", "--force-reinstall",
            "--index-url", "https://download.pytorch.org/whl/cu121", "torch",
        ])
    run([
        str(python), "-m", "pip", "install", "--disable-pip-version-check",
        "numpy", "scipy", "trimesh", "tqdm", "pyhocon", "scikit-image",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create the local, minimal NPN repository and install its CUDA dependencies."
    )
    parser.add_argument("--clone-only", action="store_true", help="Only create the standalone NPN repository.")
    args = parser.parse_args()

    ensure_repo()
    if args.clone_only:
        print(f"[done] Standalone NPN repository created at {REPO_DIR}")
        return 0
    install_environment()
    print(f"[done] Shared Python environment: {python_executable().parent.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
