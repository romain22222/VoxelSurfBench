"""Pick a camera angle for a shape: opens Polyscope with the GT and the .vol, and saves <shape>/gt/camera.json.

    python adjust_camera.py --shape-dir experiments/shapes/barrel_stove

Move the camera with the mouse, then use the "Camera" panel: ground plane, background, "Save camera".
The colors, transparency, edges and material of 'gt' and 'vol' are the ones set in the structure panels
(left side) at the time of saving. Opening the tool again restores the saved camera and look.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import get_handler, load_grid_from_vol, load_gt_inputs, load_mesh
from scene import (
    GROUND_PLANE_MODES,
    apply_scene,
    apply_view,
    default_style,
    load_camera,
    register_voxels,
    save_camera,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape-dir", type=Path, required=True)
    parser.add_argument("--camera-json", type=Path, default=None, help="Output file. Default: <shape>/gt/camera.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import polyscope as ps
    import polyscope.imgui as psim

    inputs = load_gt_inputs(args.shape_dir, require_sidecars=False)
    out_path = args.camera_json or inputs.camera_json
    camera = load_camera(out_path)
    saved_styles = (camera or {}).get("structures", {})

    ps.init()
    scene = apply_scene(ps, (camera or {}).get("scene"))

    structures = {
        "gt": get_handler(inputs.gt_mesh).register_polyscope(
            ps, "gt", load_mesh(inputs.gt_mesh), {**default_style("gt"), **saved_styles.get("gt", {})}
        ),
        "vol": register_voxels(ps, "vol", load_grid_from_vol(inputs.vol), {**default_style("vol"), **saved_styles.get("vol", {})}),
    }
    apply_view(ps, camera)

    state = {"status": f"Camera file: {out_path}"}

    def callback() -> None:
        psim.TextUnformatted("Camera")
        changed, idx = psim.Combo("Ground plane", GROUND_PLANE_MODES.index(scene["ground_plane_mode"]), GROUND_PLANE_MODES)
        if changed:
            scene["ground_plane_mode"] = GROUND_PLANE_MODES[idx]
            ps.set_ground_plane_mode(scene["ground_plane_mode"])
        changed, height = psim.SliderFloat("Ground height factor", float(scene["ground_plane_height_factor"]), -1.0, 1.0)
        if changed:
            scene["ground_plane_height_factor"] = height
            ps.set_ground_plane_height_factor(height, True)
        changed, color = psim.ColorEdit3("Background", [float(c) for c in ps.get_background_color()[:3]])
        if changed:
            ps.set_background_color((*color, 1.0))
        if psim.Button("Save camera"):
            save_camera(out_path, ps, scene, structures)
            state["status"] = f"Saved {out_path}"
            print(f"[done] {out_path}")
        psim.TextUnformatted(state["status"])

    ps.set_user_callback(callback)
    ps.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
