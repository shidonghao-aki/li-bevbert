#!/usr/bin/env python3
# Worker process for Habitat topdown rendering.  It is intentionally isolated
# because old Habitat-Sim builds may segfault while rasterizing topdown maps.

import argparse
import json
import os
import sys

import cv2

if __name__ == "__main__" and __package__ is None:
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from visualize_result.habitat_traj_renderer import HabitatTrajectoryRenderer
from visualize_result.viz_io import draw_habitat_topdown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()

    with open(args.request, "r", encoding="utf-8") as f:
        req = json.load(f)

    renderer = HabitatTrajectoryRenderer(
        width=64,
        height=64,
        hfov=90,
        gpu_id=int(req.get("gpu_id", 0)),
    )
    try:
        renderer.load_scene(req["scene_path"])
        topdown, pixels = draw_habitat_topdown(
            renderer.sim,
            req["traj"],
            episode_meta=req.get("episode_meta"),
            map_resolution=800,
        )
        cv2.imwrite(req["output_path"], topdown)
        with open(req["output_path"] + ".pixels.json", "w", encoding="utf-8") as f:
            json.dump([[int(x), int(y)] for x, y in pixels], f)
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
