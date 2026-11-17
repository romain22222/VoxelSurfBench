from __future__ import annotations

import argparse
import os
import shlex
import shutil
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
REPO_URL = "https://github.com/romain22222/visibilityLattices.git"
REPO_DIR = THIS_DIR / "visibilityLattices"
BUILD_DIR = REPO_DIR / "build"
BUILD_JOBS = 2

sys.path.insert(0, str(ROOT))

from methods.command_runner import run_command


def wsl_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/")
    if len(value) >= 2 and value[1] == ":":
        return f"/mnt/{value[0].lower()}{value[2:]}"
    return value


def ensure_repo() -> None:
    if REPO_DIR.exists():
        if not (REPO_DIR / ".git").exists() or not (REPO_DIR / "CMakeLists.txt").is_file():
            raise RuntimeError(f"{REPO_DIR} exists but is not a visibilityLattices clone.")
    else:
        run_command("git", ["clone", "--depth", "1", REPO_URL, REPO_DIR])
    patch_headless_normals_export()


def patch_headless_normals_export() -> None:
    """Bridge the upstream checkout to the local saveNormals CLI until its PR is released."""
    source_path = REPO_DIR / "main.cpp"
    source = source_path.read_text(encoding="utf-8")
    if "savePointNormals(saveNormalsFilename" in source and 'app.add_option("--saveNormals"' in source:
        return

    global_anchor = "bool noInterface = false;"
    if source.count(global_anchor) != 1:
        raise RuntimeError("Cannot patch visibilityLattices: unexpected main.cpp global declarations.")
    source = source.replace(global_anchor, global_anchor + "\nstd::string saveNormalsFilename;", 1)

    headless_start = source.find("int runHeadlessNormalComparison() {")
    headless_end = source.find("\nstd::vector<double> operator-", headless_start)
    if headless_start < 0 or headless_end < 0:
        raise RuntimeError("Cannot patch visibilityLattices: headless normal function changed upstream.")
    headless = '''void savePointNormals(const std::string &filename,
                      const std::vector<RealPoint> &positions,
                      const std::vector<RealVector> &normals) {
        if (positions.size() != normals.size()) {
                throw std::runtime_error("Point/normal count mismatch while exporting visibility normals.");
        }
        std::ofstream output(filename);
        if (!output) {
                throw std::runtime_error("Cannot write visibility normals to " + filename);
        }
        output << std::setprecision(10);
        for (size_t i = 0; i < positions.size(); ++i) {
                output << positions[i][0] << " " << positions[i][1] << " " << positions[i][2]
                       << " " << normals[i][0] << " " << normals[i][1] << " " << normals[i][2] << "\\n";
        }
}

int runHeadlessNormalComparison() {
        trace.beginBlock("Headless visibility normals");
        computeVisibilityOmp(VisibilityRadius);
        computeVisibilityNormals();
        reorientVisibilityNormals();
        if (saveNormalsFilename.empty()) {
                std::cerr << "--saveNormals is required in headless mode." << std::endl;
                trace.endBlock();
                return 1;
        }
        try {
                savePointNormals(saveNormalsFilename, primal_positions_cache, visibility_normals);
        } catch (const std::exception &error) {
                std::cerr << error.what() << std::endl;
                trace.endBlock();
                return 1;
        }
        trace.endBlock();
        return 0;
}
'''
    source = source[:headless_start] + headless + source[headless_end:]

    include_anchor = "#include <fstream>"
    if include_anchor not in source:
        raise RuntimeError("Cannot patch visibilityLattices: expected <fstream> include was not found.")
    if "#include <iomanip>" not in source:
        source = source.replace(include_anchor, include_anchor + "\n#include <iomanip>", 1)
    if "#include <stdexcept>" not in source:
        source = source.replace(include_anchor, include_anchor + "\n#include <stdexcept>", 1)

    cli_anchor = 'app.add_option("--saveVisibility", _, "gpuRun only : a filename to save the computed visibility");'
    if source.count(cli_anchor) != 1:
        raise RuntimeError("Cannot patch visibilityLattices: expected CLI option declaration was not found.")
    source = source.replace(
        cli_anchor,
        cli_anchor + '\n        app.add_option("--saveNormals", saveNormalsFilename, "output file for visibility point-normal pairs");',
        1,
    )
    source_path.write_text(source, encoding="utf-8")


def build_in_wsl() -> None:
    if not shutil.which("wsl"):
        raise FileNotFoundError("WSL2 is required to build visibilityLattices on Windows.")
    source = shlex.quote(wsl_path(REPO_DIR))
    output = shlex.quote(wsl_path(BUILD_DIR))
    build = '"$HOME/.cache/voxelsurfbench/visibilitylattices"'
    script = (
        "set -e; "
        "if ! command -v cmake >/dev/null || ! command -v g++ >/dev/null || ! command -v make >/dev/null; then "
        "sudo apt-get update && sudo apt-get install -y cmake g++ make; fi; "
        f"cmake -S {source} -B {build} -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20; "
        f"cmake --build {build} --parallel {BUILD_JOBS} --target visibilityLattices; "
        f"mkdir -p {output}; "
        f"install -m 755 {build}/visibilityLattices {output}/visibilityLattices"
    )
    run_command("wsl", ["bash", "-lc", script])


def build_native() -> None:
    for tool in ("cmake", "g++", "make"):
        if not shutil.which(tool):
            raise FileNotFoundError(f"{tool} is required to build visibilityLattices.")
    run_command("cmake", ["-S", REPO_DIR, "-B", BUILD_DIR, "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_STANDARD=20"])
    run_command("cmake", ["--build", BUILD_DIR, "--parallel", str(BUILD_JOBS), "--target", "visibilityLattices"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Clone and build DGtal visibilityLattices.")
    parser.add_argument("--clone-only", action="store_true", help="Only clone and prepare the source checkout.")
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

