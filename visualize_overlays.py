"""Interactive Polyscope overlay of the GT, the .vol voxelization and every method result of a shape.

    python visualize_overlays.py --shape-dir experiments/shapes/barrel_stove [--raw] [--show-all] [--methods mc,dc]

The sampled point cloud is read from <shape>/gt/<shape>.ply. Camera, ground plane and 'gt' / 'vol' appearance come
from <shape>/gt/camera.json when it exists (see adjust_camera.py). Method meshes are shown aligned (aligned.<ext>),
or as output by the method with --raw.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import (
    ALIGNED_STEM,
    BASE_STEM,
    GT_DIRNAME,
    get_handler,
    load_grid_from_vol,
    load_gt_inputs,
    load_mesh,
    mesh_extensions,
)
from scene import apply_scene, apply_view, default_style, load_camera, register_voxels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape-dir", type=Path, required=True)
    parser.add_argument("--methods", type=str, default="all", help="Comma-separated method folders or 'all'.")
    parser.add_argument("--camera-json", type=Path, default=None, help="Default: <shape>/gt/camera.json")
    parser.add_argument("--alpha", type=float, default=0.38, help="Transparency of the method meshes.")
    parser.add_argument("--gt-alpha", type=float, default=0.8, help="GT opacity (higher = less transparent).")
    parser.add_argument("--show-all", action="store_true", help="Enable every mesh at startup (default: GT only).")
    parser.add_argument("--raw", action="store_true", help="Use base.<ext> (raw method output) instead of aligned.<ext>.")
    parser.add_argument("--no-vol", action="store_true", help="Do not add the .vol voxelization.")
    parser.add_argument("--no-points", action="store_true", help="Hide the sampled point cloud <shape>.ply.")
    return parser.parse_args()


def find_method_meshes(shape_dir: Path, wanted: list[str] | None, stem: str) -> list[tuple[str, Path]]:
    found = []
    for directory in sorted(p for p in shape_dir.iterdir() if p.is_dir() and p.name != GT_DIRNAME):
        if wanted is not None and directory.name not in wanted:
            continue
        for ext in mesh_extensions():
            candidate = directory / f"{stem}{ext}"
            if candidate.exists():
                found.append((directory.name, candidate))
                break
    return found


def main() -> int:
    args = parse_args()
    import polyscope as ps

    inputs = load_gt_inputs(args.shape_dir, require_sidecars=False)
    wanted = None if args.methods.strip().lower() == "all" else [m.strip() for m in args.methods.split(",") if m.strip()]
    camera = load_camera(args.camera_json or inputs.camera_json)
    saved_styles = (camera or {}).get("structures", {})

    def style_of(label: str, index: int = 0) -> dict:
        return {**default_style(label, index, alpha=args.alpha, gt_alpha=args.gt_alpha), **saved_styles.get(label, {})}

    ps.init()
    apply_scene(ps, (camera or {}).get("scene"))

    get_handler(inputs.gt_mesh).register_polyscope(ps, "gt", load_mesh(inputs.gt_mesh), {**style_of("gt"), "enabled": True})
    if not args.no_vol and inputs.vol.exists():
        register_voxels(ps, "vol", load_grid_from_vol(inputs.vol), {**style_of("vol"), "enabled": args.show_all})

    if not args.no_points and inputs.points.exists():
        points = get_handler(inputs.points).load(inputs.points)
        get_handler(inputs.points).register_polyscope(
            ps,
            "sampled_points",
            points,
            {"color": (0.10, 0.65, 0.28), "point_radius": 0.002, "enabled": True},
        )

    meshes = find_method_meshes(inputs.shape_dir, wanted, BASE_STEM if args.raw else ALIGNED_STEM)
    for index, (label, path) in enumerate(meshes):
        try:
            geometry = load_mesh(path)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] cannot load {path}: {exc}")
            continue
        get_handler(path).register_polyscope(ps, label, geometry, {**style_of(label, index), "enabled": args.show_all})
    print("[info] meshes:", ", ".join(["gt"] + [label for label, _ in meshes]))

    apply_view(ps, camera)
    ps.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
