from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from command_runner import run_command


METHODS = {
    "marchingcubes",
    "dualcontouring",
    "antialiasing",
    "fineantialiasing",
    "poisson",
    "shroud",
    "regularization",
}
EXTENSIONS = {".obj", ".off", ".ply"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Reconstruction method on a VOL input.")
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--vol", type=Path, required=True)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-mesh", type=Path, required=True)
    return parser.parse_args()


def wsl_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    if len(value) >= 2 and value[1] == ":":
        return f"/mnt/{value[0].lower()}{value[2:]}"
    return value


def executable_path(build_dir: Path, name: str) -> Path:
    for candidate in (build_dir / name, build_dir / "bin" / name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing {name}; run methods/Reproduction/build.py first.")


def run_tool(executable: Path, args: list[str], repo_dir: Path) -> None:
    if os.name == "nt":
        if not shutil.which("wsl"):
            raise FileNotFoundError("WSL2 is required to run Reconstruction on Windows.")
        command = [
            wsl_path(executable),
            *(wsl_path(Path(value)) if Path(value).is_absolute() else value for value in args),
        ]
        script = f"cd {shlex.quote(wsl_path(repo_dir))} && " + " ".join(shlex.quote(value) for value in command)
        run_command("wsl", ["bash", "-lc", script])
        return
    run_command(executable, args, cwd=repo_dir)


def find_mesh(work_dir: Path, extensions: set[str] = EXTENSIONS) -> Path | None:
    candidates = [
        path for path in work_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions and path.stat().st_size > 0
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def run_dual_contouring(vol: Path, prefix: Path, executable: Path, repo_dir: Path) -> None:
    last_error: subprocess.CalledProcessError | None = None
    for step in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0):
        for stale in prefix.parent.glob("mesh*.obj"):
            stale.unlink()
        try:
            run_tool(executable, ["-i", str(vol), "-o", str(prefix), "-g", str(step)], repo_dir)
        except subprocess.CalledProcessError as exc:
            last_error = exc
            continue
        mesh = find_mesh(prefix.parent, {".obj"})
        if mesh is not None:
            with mesh.open("r", encoding="utf-8", errors="replace") as stream:
                if any(line.startswith("f ") for line in stream):
                    return
    details = f" Last command error: {last_error}" if last_error else ""
    raise RuntimeError(f"Dual contouring did not produce a mesh for any supported grid step.{details}")


def main() -> int:
    args = parse_args()
    repo_dir = args.repo_dir.resolve()
    vol = args.vol.resolve()
    work_dir = args.work_dir.resolve()
    output_mesh = args.output_mesh.resolve()
    build_dir = repo_dir / "build"

    if not (repo_dir / "CMakeLists.txt").is_file():
        raise FileNotFoundError(f"Reconstruction repository not found: {repo_dir}")
    if not vol.is_file():
        raise FileNotFoundError(f"VOL input not found: {vol}")
    work_dir.mkdir(parents=True, exist_ok=True)
    prefix = work_dir / "mesh"

    mc = executable_path(build_dir, "volMarchingCubes")
    dc = executable_path(build_dir, "volIGLDualContouring")
    aa = executable_path(build_dir, "volAntialiasing")
    fine_aa = executable_path(build_dir, "volFineAntialiasing")
    poisson = executable_path(build_dir, "volCGALPoissonReconstruction")
    shroud = executable_path(build_dir, "volShroud")
    regularization = executable_path(build_dir, "volRegularization")

    if args.method == "marchingcubes":
        run_tool(mc, ["-i", str(vol), "-t", "0.5", "-o", str(prefix)], repo_dir)
    elif args.method == "dualcontouring":
        run_dual_contouring(vol, prefix, dc, repo_dir)
    elif args.method in {"antialiasing", "fineantialiasing"}:
        aa_executable = fine_aa if args.method == "fineantialiasing" else aa
        aa_vol = work_dir / "anti_aliased.vol"
        aa_args = ["-i", str(vol), "-e", "II", "-r", "4.0", "--lambda", "1.0", "-o", str(aa_vol)]
        if args.method == "fineantialiasing":
            aa_args.append("--constrain")
        try:
            run_tool(aa_executable, aa_args, repo_dir)
        except subprocess.CalledProcessError:
            if args.method != "fineantialiasing":
                raise
            aa_args.remove("--constrain")
            run_tool(aa_executable, aa_args, repo_dir)
        run_dual_contouring(aa_vol, prefix, dc, repo_dir)
    elif args.method == "poisson":
        poisson_output = work_dir / "mesh.off"
        run_tool(
            poisson,
            ["-i", str(vol), "-o", str(poisson_output), "-e", "II", "-r", "4.0", "-R", "5.0", "-s", "0.15"],
            repo_dir,
        )
    elif args.method == "shroud":
        run_tool(
            shroud,
            ["-i", str(vol), "-o", str(work_dir / "mesh.obj"), "--regularization", "squared-curvature", "--iterations", "100"],
            repo_dir,
        )
    else:
        run_tool(
            regularization,
            ["-i", str(vol), "-o", str(work_dir / "mesh.obj"), "--iterations", "200", "--time-step", "1.0", "--epsilon", "0.001"],
            repo_dir,
        )

    mesh = find_mesh(work_dir, {".off"} if args.method == "poisson" else {".obj", ".ply"})
    if mesh is None:
        raise FileNotFoundError(f"{args.method} did not produce a mesh in {work_dir}")
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    if mesh.resolve() != output_mesh:
        shutil.copy2(mesh, output_mesh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
