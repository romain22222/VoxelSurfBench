from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from command_runner import run_command
from python_env import ensure_venv, python_executable


REPO_URL = "https://github.com/czq142857/NDC.git"
REPO_DIR = THIS_DIR / "NDC"
WEIGHTS_DIR = REPO_DIR / "weights_examples"
PRETRAINED_DIR = REPO_DIR / "pretrained"
WEIGHTS_URL = "https://drive.google.com/file/d/1rLog9aX3gcKYF3FCHeL4BHDBHhXoPH3V/view?usp=sharing"
METHODS = ("ndc", "ndcx", "undc")


def ensure_repo() -> None:
    if REPO_DIR.exists():
        if not (REPO_DIR / ".git").exists() or not (REPO_DIR / "main.py").is_file():
            raise RuntimeError(f"{REPO_DIR} exists but is not an NDC clone.")
        return
    run_command("git", ["clone", "--depth", "1", REPO_URL, REPO_DIR])


def install_environment(torch_backend: str) -> None:
    python = ensure_venv()
    torch_probe = run_command(
        python,
        ["-c", "import torch; print(torch.version.cuda or '')"],
        check=False,
        echo=False,
        capture_output=True,
        text=True,
    )
    torch_installed = torch_probe.returncode == 0
    has_cuda_torch = torch_installed and bool(torch_probe.stdout.strip())
    if not torch_installed or (torch_backend == "cuda" and not has_cuda_torch):
        index = "https://download.pytorch.org/whl/cu121" if torch_backend == "cuda" else "https://download.pytorch.org/whl/cpu"
        arguments = ["-m", "pip", "install", "--disable-pip-version-check", "--index-url", index]
        if torch_installed:
            arguments.extend(["--upgrade", "--force-reinstall"])
        arguments.append("torch")
        run_command(python, arguments)
    run_command(python, [
        "-m", "pip", "install", "--disable-pip-version-check",
        "numpy", "h5py", "scipy", "scikit-learn", "trimesh", "cython", "py7zr", "gdown",
    ])
    run_command(python, ["setup.py", "build_ext", "--inplace"], cwd=REPO_DIR)


def download_and_extract_weights() -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    archive = WEIGHTS_DIR / "weights_examples.7z"
    if not archive.is_file():
        download_script = f"""
import gdown

ok = gdown.download(url={WEIGHTS_URL!r}, output={str(archive)!r}, quiet=False, fuzzy=True)
if not ok:
    raise RuntimeError("Google Drive download failed")
"""
        run_command(python_executable(), ["-c", download_script])
    extract_script = f"""
import py7zr

with py7zr.SevenZipFile({str(archive)!r}, mode="r") as weights:
    weights.extractall(path={str(WEIGHTS_DIR)!r})
"""
    run_command(python_executable(), ["-c", extract_script])


def find_checkpoint(method: str, kind: str) -> Path | None:
    names = (
        f"weights_{method}_voxel_{kind}.pth",
        f"weights_{method}_{kind}.pth",
    )
    files = sorted(path for path in WEIGHTS_DIR.rglob("*.pth") if path.is_file())
    for name in names:
        match = next((path for path in files if path.name == name), None)
        if match:
            return match
    return next(
        (
            path for path in files
            if method in str(path).lower() and kind in path.name.lower()
        ),
        None,
    )


def stage_weights() -> None:
    for method in METHODS:
        destination = PRETRAINED_DIR / method
        destination.mkdir(parents=True, exist_ok=True)
        for kind in ("bool", "float"):
            source = find_checkpoint(method, kind)
            if source is None:
                raise FileNotFoundError(f"Missing {method} {kind} checkpoint in {WEIGHTS_DIR}")
            shutil.copy2(source, destination / f"weights_{method}_voxel_{kind}.pth")


def has_staged_weights() -> bool:
    return all(
        (PRETRAINED_DIR / method / f"weights_{method}_voxel_{kind}.pth").is_file()
        for method in METHODS
        for kind in ("bool", "float")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Clone NDC, install dependencies, compile its extension and fetch weights.")
    parser.add_argument("--clone-only", action="store_true", help="Only clone the NDC repository.")
    parser.add_argument("--torch-backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--skip-weights", action="store_true", help="Do not download pretrained weights.")
    args = parser.parse_args()

    ensure_repo()
    if args.clone_only:
        print(f"[done] Repository cloned to {REPO_DIR}")
        return 0
    install_environment(args.torch_backend)
    if not args.skip_weights and not has_staged_weights():
        download_and_extract_weights()
        stage_weights()
    print(f"[done] Shared Python environment: {python_executable().parent.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
