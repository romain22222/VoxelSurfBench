from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_URL = "https://github.com/JacquesOlivierLachaud/Reconstruction.git"
REPO_DIR = THIS_DIR / "Reconstruction"
BUILD_DIR = REPO_DIR / "build"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def ensure_repo() -> None:
    if REPO_DIR.exists():
        if not (REPO_DIR / ".git").exists() or not (REPO_DIR / "CMakeLists.txt").is_file():
            raise RuntimeError(f"{REPO_DIR} exists but is not a Reconstruction clone.")
        return
    run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])


def wsl_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    if len(value) >= 2 and value[1] == ":":
        return f"/mnt/{value[0].lower()}{value[2:]}"
    return value


def build_in_wsl() -> None:
    if not shutil.which("wsl"):
        raise FileNotFoundError("WSL2 is required to build Reconstruction on Windows.")
    source = shlex.quote(wsl_path(REPO_DIR))
    build = shlex.quote(wsl_path(BUILD_DIR))
    script = (
        "set -e; "
        "if ! command -v cmake >/dev/null || ! command -v g++ >/dev/null || ! command -v make >/dev/null; then "
        "sudo apt-get update && sudo apt-get install -y cmake g++ make; fi; "
        f"cmake -S {source} -B {build} -G 'Unix Makefiles'; "
        f"cmake --build {build} --parallel"
    )
    run(["wsl", "bash", "-lc", script])


def build_native() -> None:
    if not shutil.which("cmake") or not shutil.which("g++") or not shutil.which("make"):
        raise FileNotFoundError("CMake, g++, and make are required. Install them before building Reconstruction.")
    run(["cmake", "-S", str(REPO_DIR), "-B", str(BUILD_DIR), "-G", "Unix Makefiles"])
    run(["cmake", "--build", str(BUILD_DIR), "--parallel"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Clone and build the Reconstruction tools.")
    parser.add_argument("--clone-only", action="store_true", help="Only clone the source repository.")
    args = parser.parse_args()

    ensure_repo()
    if args.clone_only:
        print(f"[done] Repository cloned to {REPO_DIR}")
        return 0

    if os.name == "nt":
        build_in_wsl()
    else:
        build_native()
    print(f"[done] Build completed in {BUILD_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
