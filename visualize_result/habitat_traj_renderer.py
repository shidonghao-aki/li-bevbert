# Copyright: visualization helper for BEVBert / VLN-CE preds.json trajectories
"""Habitat-Sim 轨迹渲染器：在 MP3D .glb 中按 preds 的 position + heading 逐步渲染。"""

import os
from typing import List, Optional, Tuple

import numpy as np

import habitat_sim
from habitat_sim import AgentState


class HabitatTrajectoryRenderer:
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        hfov: float = 90.0,
        sensor_height: float = 0.0,
        gpu_id: int = 0,
    ) -> None:
        self.width = width
        self.height = height
        self.hfov = hfov
        self.sensor_height = sensor_height
        self.gpu_id = gpu_id
        self._sim: Optional[habitat_sim.Simulator] = None
        self._current_scene: Optional[str] = None

    @property
    def sim(self):
        return self._sim

    def close(self) -> None:
        if self._sim is not None:
            self._sim.close()
            self._sim = None
        self._current_scene = None

    def _make_camera_specs(self) -> List:
        sensor_spec_cls = getattr(habitat_sim, "CameraSensorSpec", habitat_sim.SensorSpec)

        rgb = sensor_spec_cls()
        rgb.uuid = "rgb"
        rgb.sensor_type = habitat_sim.SensorType.COLOR
        rgb.resolution = [self.height, self.width]
        rgb.position = [0.0, self.sensor_height, 0.0]
        self._set_hfov(rgb, self.hfov)

        depth = sensor_spec_cls()
        depth.uuid = "depth"
        depth.sensor_type = habitat_sim.SensorType.DEPTH
        depth.resolution = [self.height, self.width]
        depth.position = [0.0, self.sensor_height, 0.0]
        self._set_hfov(depth, self.hfov)

        return [rgb, depth]

    @staticmethod
    def _set_hfov(sensor_spec, hfov):
        if hasattr(sensor_spec, "hfov"):
            sensor_spec.hfov = float(hfov)
        else:
            sensor_spec.parameters["hfov"] = str(float(hfov))

    def load_scene(self, scene_path: str) -> None:
        scene_path = os.path.normpath(os.path.abspath(scene_path))
        if self._current_scene == scene_path and self._sim is not None:
            return
        self.close()

        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id = scene_path
        sim_cfg.gpu_device_id = int(self.gpu_id)
        if hasattr(sim_cfg, "allow_sliding"):
            sim_cfg.allow_sliding = True

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = self._make_camera_specs()

        self._sim = habitat_sim.Simulator(
            habitat_sim.Configuration(sim_cfg, [agent_cfg])
        )
        self._current_scene = scene_path

    @staticmethod
    def heading_to_rotation(heading: float):
        """与 Habitat 中绕世界 UP 的 yaw 一致；heading 为弧度。"""
        return habitat_sim.utils.quat_from_angle_axis(heading, habitat_sim.geo.UP)

    def render_step(
        self, position: List[float], heading: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._sim is None:
            raise RuntimeError("load_scene() first")

        pos = np.array(position, dtype=np.float32)
        rot = self.heading_to_rotation(float(heading))
        state = AgentState()
        state.position = pos
        state.rotation = rot
        self._sim.get_agent(0).set_state(state, reset_sensors=True)
        obs = self._sim.get_sensor_observations()
        rgb = obs["rgb"]
        if rgb.shape[-1] == 4:
            rgb = rgb[:, :, :3]
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        depth = np.asarray(obs["depth"], dtype=np.float32)
        return rgb, depth
