#!/usr/bin/env python3
# Copyright: ETPNav 风格 — MatterSim 驱动 36 视角，Habitat 渲染
"""
Matterport3DSimulator 与 Habitat-Sim 对齐，为单个 scan/viewpoint 导出 36 张图 + 拼图。

需本地编译 Matterport3DSimulator 并设置 --matterport_build 为其 build 目录（内含 MatterSim 模块）。

示例:
  python -m visualize_result.render_matterport_viewpoint \\
    --matterport_build E:/Matterport3DSimulator/build \\
    --connectivity bevbert_ce/precompute_img_features/connectivity \\
    --scan_id gTV8FGcVJC9 --viewpoint_id <vp_id> \\
    --scene E:/data/mp3d/gTV8FGcVJC9/gTV8FGcVJC9.glb \\
    --out_dir vis_viewpoint
"""

import argparse
import math
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np
import quaternion  # noqa: F401  # 注册 np.quaternion
from scipy.spatial.transform import Rotation as R

if __name__ == "__main__" and __package__ is None:
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)

import habitat_sim
from habitat_sim import AgentState

# 与 ETPNav precompute save_img 一致
VIEWPOINT_SIZE = 36
DEFAULT_WIDTH = 256
DEFAULT_HEIGHT = 256
DEFAULT_VFOV = 60


def _build_matterport_sim(connectivity_dir: str, width: int, height: int, vfov_deg: float):
    if not os.path.isdir(connectivity_dir):
        raise FileNotFoundError(connectivity_dir)
    import MatterSim

    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setCameraResolution(width, height)
    sim.setCameraVFOV(math.radians(vfov_deg))
    sim.setDiscretizedViewingAngles(True)
    sim.setRenderingEnabled(False)
    sim.setDepthEnabled(False)
    sim.setPreloadingEnabled(False)
    sim.setBatchSize(1)
    sim.initialize()
    return sim


def _hfov_from_vfov(vfov_deg: float, width: int, height: int) -> float:
    v = math.radians(vfov_deg)
    hfov = 2.0 * math.atan(math.tan(v / 2.0) * (width / float(height)))
    return math.degrees(hfov)


def _build_habitat_for_scene(
    glb_path: str,
    width: int,
    height: int,
    hfov: float,
    gpu_id: int = 0,
) -> habitat_sim.Simulator:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = os.path.normpath(os.path.abspath(glb_path))
    sim_cfg.gpu_device_id = int(gpu_id)

    def cam(uuid: str, stype):
        sensor_spec_cls = getattr(habitat_sim, "CameraSensorSpec", habitat_sim.SensorSpec)
        s = sensor_spec_cls()
        s.uuid = uuid
        s.sensor_type = stype
        s.resolution = [height, width]
        s.position = [0.0, 0.0, 0.0]
        if hasattr(s, "hfov"):
            s.hfov = float(hfov)
        else:
            s.parameters["hfov"] = str(float(hfov))
        return s

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [
        cam("rgb", habitat_sim.SensorType.COLOR),
        cam("depth", habitat_sim.SensorType.DEPTH),
    ]
    return habitat_sim.Simulator(
        habitat_sim.Configuration(sim_cfg, [agent_cfg])
    )


def _matter_to_habitat_pose(
    x: float, y: float, z: float, heading: float, elev: float
) -> Tuple[np.ndarray, np.quaternion]:
    habitat_position = np.array([x, z - 1.25, -y], dtype=np.float32)
    mp3d_h = np.array([0.0, 2.0 * math.pi - heading, 0.0])
    mp3d_e = np.array([elev, 0.0, 0.0])
    rot_h = R.from_rotvec(mp3d_h)
    rot_e = R.from_rotvec(mp3d_e)
    qx, qy, qz, qw = (rot_h * rot_e).as_quat()  # scipy: x,y,z,w
    rot = np.quaternion(qw, qx, qy, qz)  # numpy-quaternion: w,x,y,z
    return habitat_position, rot


def run(
    matterport_build: str,
    connectivity_dir: str,
    scan_id: str,
    viewpoint_id: str,
    scene_glb: str,
    out_dir: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    vfov: float = DEFAULT_VFOV,
    depth: bool = False,
    gpu_id: int = 0,
) -> None:
    if matterport_build not in sys.path:
        sys.path.insert(0, matterport_build)

    mp3d = _build_matterport_sim(connectivity_dir, width, height, vfov)
    hfov = _hfov_from_vfov(vfov, width, height)
    hab = _build_habitat_for_scene(scene_glb, width, height, hfov, gpu_id=gpu_id)

    os.makedirs(out_dir, exist_ok=True)
    tiles: List[np.ndarray] = []

    try:
        for ix in range(VIEWPOINT_SIZE):
            if ix == 0:
                mp3d.newEpisode(
                    [scan_id], [viewpoint_id], [0.0], [math.radians(-30.0)]
                )
            elif ix % 12 == 0:
                mp3d.makeAction([0], [1.0], [1.0])
            else:
                mp3d.makeAction([0], [1.0], [0.0])

            state = mp3d.getState()[0]
            assert state.viewIndex == ix

            x, y, z = state.location.x, state.location.y, state.location.z
            h, e = state.heading, state.elevation
            pos, rot = _matter_to_habitat_pose(x, y, z, h, e)
            agent_state = AgentState()
            agent_state.position = pos
            agent_state.rotation = rot
            hab.get_agent(0).set_state(agent_state, reset_sensors=True)
            obs = hab.get_sensor_observations()
            if depth:
                d = np.asarray(obs["depth"], dtype=np.float32)
                d_vis = np.clip(d / 10.0, 0.0, 1.0)
                d_vis = (d_vis * 255).astype(np.uint8)
                cv2.imwrite(
                    os.path.join(out_dir, f"view_{ix:02d}_depth.png"), d_vis
                )

            # 与 ETPNav save_img 一致: habitat RGB -> 存盘用 BGR
            rgb = obs["rgb"]
            if rgb.shape[-1] == 4:
                rgb = rgb[:, :, :3]
            bgr = cv2.cvtColor(
                np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR
            )

            out_path = os.path.join(out_dir, f"view_{ix:02d}.png")
            cv2.imwrite(out_path, bgr)
            tiles.append(
                cv2.resize(bgr, (min(200, width), min(200, int(height * 200 / width))))
            )
    finally:
        hab.close()
    # 简易拼图 6x6
    if len(tiles) == 36:
        rows = []
        for r in range(6):
            row = np.concatenate(tiles[r * 6 : (r + 1) * 6], axis=1)
            rows.append(row)
        contact = np.concatenate(rows, axis=0)
        cv2.imwrite(os.path.join(out_dir, "contact_sheet_36.png"), contact)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MatterSim 36 视角 + Habitat 渲染（ETPNav 坐标转换）"
    )
    ap.add_argument(
        "--matterport_build", type=str, required=True, help="Matterport3DSimulator/build"
    )
    ap.add_argument(
        "--connectivity", type=str, required=True, help="含 *_connectivity.json 的目录"
    )
    ap.add_argument("--scan_id", type=str, required=True)
    ap.add_argument("--viewpoint_id", type=str, required=True)
    ap.add_argument("--scene", type=str, required=True, help="该 scan 的 .glb 绝对/相对路径")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    ap.add_argument("--vfov", type=float, default=DEFAULT_VFOV)
    ap.add_argument("--depth", action="store_true", help="额外保存 depth 可视化 png")
    ap.add_argument("--gpu_id", type=int, default=0)
    args = ap.parse_args()

    run(
        matterport_build=args.matterport_build,
        connectivity_dir=args.connectivity,
        scan_id=args.scan_id,
        viewpoint_id=args.viewpoint_id,
        scene_glb=args.scene,
        out_dir=args.out_dir,
        width=args.width,
        height=args.height,
        vfov=args.vfov,
        depth=args.depth,
        gpu_id=args.gpu_id,
    )
    print("done ->", args.out_dir)


if __name__ == "__main__":
    main()
