# Copyright: I/O and drawing helpers for visualize_result

import gzip
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def load_preds(preds_path: str) -> Dict[str, List[dict]]:
    """读取 infer preds 或 eval_vis；支持逗号分隔多个 rank 文件并合并。"""
    paths = [p.strip() for p in preds_path.split(",") if p.strip()]
    merged = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged.update(data)
    return merged


def load_episode_index(dataset_path: str) -> Dict[str, dict]:
    """从 VLN-CE 预处理 json / json.gz 读 episode 元信息，key 为 str(episode_id)。"""
    if dataset_path.lower().endswith(".gz"):
        with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    if isinstance(data, dict) and "episodes" in data:
        episodes = data["episodes"]
    elif isinstance(data, list):
        episodes = data
    else:
        raise ValueError("dataset 须为 {episodes: [...]} 或 episode 列表")

    out: Dict[str, dict] = {}
    for ep in episodes:
        eid = ep.get("episode_id", ep.get("id"))
        if eid is None:
            continue
        out[str(eid)] = ep
    return out


def load_gt_index(gt_path: str) -> Dict[str, dict]:
    """读取 eval 使用的 *_gt.json.gz，通常包含 locations 作为 GT path。"""
    if gt_path.lower().endswith(".gz"):
        with gzip.open(gt_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(gt_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    if isinstance(data, dict):
        return {str(k): v for k, v in data.items()}
    raise ValueError("gt 文件应为 {episode_id: {...}} 格式")


def infer_gt_path_from_dataset(dataset_path: Optional[str]) -> Optional[str]:
    """从 BERTidx dataset 路径推断 eval 代码使用的 GT_PATH。"""
    if not dataset_path:
        return None
    path = dataset_path
    path = path.replace("R2R_VLNCE_v1-2_preprocessed_BERTidx", "R2R_VLNCE_v1-2_preprocessed")
    path = path.replace("_bertidx.json.gz", "_gt.json.gz")
    path = path.replace("_bertidx.json", "_gt.json")
    return path


def merge_gt_into_episode(episode_meta: Optional[dict], gt_item: Optional[dict]) -> Optional[dict]:
    if episode_meta is None and gt_item is None:
        return None
    merged = dict(episode_meta or {})
    if gt_item is not None:
        merged["_gt_item"] = gt_item
        if "locations" in gt_item:
            merged["_gt_locations"] = gt_item["locations"]
        if "goal_position" in gt_item:
            merged["_gt_goal_position"] = gt_item["goal_position"]
    return merged


def parse_scan_id_from_scene(scene_id: str) -> str:
    base = os.path.basename(scene_id)
    return os.path.splitext(base)[0]


def resolve_scene_path(
    episode: dict,
    scene_root: Optional[str],
    force_scene_path: Optional[str] = None,
) -> str:
    if force_scene_path:
        return os.path.normpath(os.path.abspath(force_scene_path))

    scene_id = episode.get("scene_id", "")
    if not scene_id:
        raise ValueError("episode 无 scene_id，请使用 --force_scene")

    if os.path.isfile(scene_id):
        return os.path.normpath(os.path.abspath(scene_id))

    scan_id = parse_scan_id_from_scene(scene_id)
    root = scene_root or "."
    candidates = [
        os.path.join(root, "mp3d", scan_id, f"{scan_id}.glb"),
        os.path.join(root, "scene_datasets", "mp3d", scan_id, f"{scan_id}.glb"),
        os.path.join(root, "data", "scene_datasets", "mp3d", scan_id, f"{scan_id}.glb"),
        os.path.join(root, scan_id, f"{scan_id}.glb"),
        os.path.join(root, f"{scan_id}.glb"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return os.path.normpath(os.path.abspath(p))
    raise FileNotFoundError(
        f"找不到 scan={scan_id} 的 .glb，已尝试: {candidates[:3]}..."
    )


def draw_topdown(
    traj: List[dict], size: int = 800, margin: int = 40, line_bgr: Tuple[int, int, int] = (0, 0, 255)
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Habitat 俯视图: x–z 平面, y 为高度。"""
    pts = np.array([s["position"] for s in traj], dtype=np.float32)
    xs, zs = pts[:, 0], pts[:, 2]
    min_x, max_x = float(xs.min()), float(xs.max())
    min_z, max_z = float(zs.min()), float(zs.max())
    span_x = max(max_x - min_x, 1e-3)
    span_z = max(max_z - min_z, 1e-3)

    canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
    pixels: List[Tuple[int, int]] = []
    for x, z in zip(xs, zs):
        px = margin + int((x - min_x) / span_x * (size - 2 * margin))
        py = margin + int((z - min_z) / span_z * (size - 2 * margin))
        pixels.append((px, py))

    for a, b in zip(pixels[:-1], pixels[1:]):
        cv2.line(canvas, a, b, line_bgr, 3)
    if pixels:
        cv2.circle(canvas, pixels[0], 8, (0, 255, 0), -1)
        cv2.circle(canvas, pixels[-1], 8, (0, 0, 255), -1)
    return canvas, pixels


def _to_grid_xy(habitat_maps, sim, pos, map_shape):
    """Habitat position [x,y,z] -> OpenCV pixel (x=col, y=row)."""
    grid_x, grid_y = habitat_maps.to_grid(
        pos[2],
        pos[0],
        map_shape[0:2],
        sim,
    )
    return int(grid_y), int(grid_x)


def _colorize_habitat_map(habitat_maps, topdown):
    if hasattr(habitat_maps, "colorize_topdown_map"):
        colored = habitat_maps.colorize_topdown_map(topdown)
    else:
        palette = np.zeros((256, 3), dtype=np.uint8)
        palette[0] = [255, 255, 255]
        palette[1] = [170, 170, 170]
        palette[2] = [80, 80, 80]
        palette[3:] = [120, 120, 120]
        colored = palette[topdown.astype(np.uint8)]
    if colored.ndim == 2:
        colored = cv2.cvtColor(colored.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(colored[:, :, :3], dtype=np.uint8)


def extract_instruction_text(episode_meta: Optional[dict]) -> str:
    if episode_meta is None:
        return ""
    instr = episode_meta.get("instruction", "")
    if isinstance(instr, dict):
        return instr.get("instruction_text", instr.get("text", ""))
    if isinstance(instr, str):
        return instr
    return ""


def _is_xyz_position(item) -> bool:
    if not isinstance(item, (list, tuple)) or len(item) != 3:
        return False
    return all(isinstance(x, (int, float)) for x in item)


def _normalize_position_path(path) -> List[List[float]]:
    if not isinstance(path, list):
        return []
    out = []
    for item in path:
        if _is_xyz_position(item):
            out.append([float(x) for x in item])
        elif isinstance(item, dict) and _is_xyz_position(item.get("position")):
            out.append([float(x) for x in item["position"]])
    return out


def extract_gt_path(episode_meta: Optional[dict]) -> List[List[float]]:
    """读取 dataset 中可能存在的 GT/reference path；test split 可能没有。"""
    if episode_meta is None:
        return []

    for key in ["_gt_locations", "reference_path", "gt_path", "locations"]:
        path = _normalize_position_path(episode_meta.get(key))
        if len(path) >= 2:
            return path

    shortest_paths = episode_meta.get("shortest_paths")
    if isinstance(shortest_paths, list) and shortest_paths:
        for candidate in shortest_paths:
            path = _normalize_position_path(candidate)
            if len(path) >= 2:
                return path

    return []


def extract_goal_position(episode_meta: Optional[dict]) -> Optional[List[float]]:
    if episode_meta is None:
        return None
    if _is_xyz_position(episode_meta.get("_gt_goal_position")):
        return [float(x) for x in episode_meta["_gt_goal_position"]]
    if episode_meta.get("_gt_item"):
        gt_item = episode_meta["_gt_item"]
        if isinstance(gt_item, dict):
            for key in ["goal_position", "target_position"]:
                if _is_xyz_position(gt_item.get(key)):
                    return [float(x) for x in gt_item[key]]
            locs = _normalize_position_path(gt_item.get("locations"))
            if locs:
                return locs[-1]
    goals = episode_meta.get("goals")
    if isinstance(goals, list) and goals:
        goal = goals[0]
        if isinstance(goal, dict) and _is_xyz_position(goal.get("position")):
            return [float(x) for x in goal["position"]]
        if hasattr(goal, "position") and _is_xyz_position(goal.position):
            return [float(x) for x in goal.position]
    gt_path = extract_gt_path(episode_meta)
    if gt_path:
        return gt_path[-1]
    return None


def add_instruction_panel(
    image_bgr: np.ndarray,
    instruction: str,
    episode_id: str,
    scan_id: str,
    has_gt: bool,
    panel_width: int = 460,
) -> np.ndarray:
    """在 topdown 图右侧拼接 instruction 和颜色图例。"""
    h = image_bgr.shape[0]
    panel = np.ones((h, panel_width, 3), dtype=np.uint8) * 255

    y = 30
    cv2.putText(panel, "Episode: %s" % episode_id, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    y += 30
    cv2.putText(panel, "Scan: %s" % scan_id, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    y += 38

    legend = [
        ((0, 0, 255), "Prediction path"),
        ((255, 0, 0), "Ground truth path" if has_gt else "Ground truth: not available"),
        ((0, 255, 0), "Start"),
        ((0, 0, 255), "Predicted stop"),
        ((255, 0, 255), "Goal"),
    ]
    for color, text in legend:
        cv2.rectangle(panel, (20, y - 12), (42, y + 8), color, -1)
        cv2.putText(panel, text, (54, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1)
        y += 28

    y += 14
    cv2.putText(panel, "Instruction:", (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2)
    y += 28
    text = instruction or "(instruction not found in dataset)"
    words = text.replace("\n", " ").split(" ")
    line = ""
    max_chars = 48
    for word in words:
        maybe = (line + " " + word).strip()
        if len(maybe) > max_chars and line:
            cv2.putText(panel, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            y += 23
            line = word
            if y > h - 20:
                break
        else:
            line = maybe
    if line and y <= h - 20:
        cv2.putText(panel, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    return np.concatenate([image_bgr, panel], axis=1)


def draw_habitat_topdown(
    sim,
    traj: List[dict],
    episode_meta: Optional[dict] = None,
    map_resolution: int = 800,
    meters_per_pixel: Optional[float] = None,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """用 Habitat pathfinder 生成真实可通行区域 topdown，并叠加 preds 轨迹。"""
    from habitat.utils.visualizations import maps as habitat_maps

    if sim is None:
        raise RuntimeError("sim is None")
    if not traj:
        raise ValueError("empty trajectory")

    # 让地图高度贴近当前轨迹所在楼层。
    start = traj[0]["position"]
    agent = sim.get_agent(0)
    state = agent.get_state() if hasattr(agent, "get_state") else agent.state
    state.position = np.array(start, dtype=np.float32)
    agent.set_state(state, reset_sensors=True)
    base_height = float(start[1])

    if meters_per_pixel is None:
        try:
            meters_per_pixel = habitat_maps.calculate_meters_per_pixel(
                map_resolution, sim=sim
            )
        except TypeError:
            meters_per_pixel = habitat_maps.calculate_meters_per_pixel(
                map_resolution, sim
            )
        except Exception:
            meters_per_pixel = None

    try:
        topdown = habitat_maps.get_topdown_map(
            sim.pathfinder,
            base_height,
            map_resolution,
            False,
            meters_per_pixel,
        )
    except TypeError:
        try:
            topdown = habitat_maps.get_topdown_map(
                sim.pathfinder,
                base_height,
                map_resolution,
                False,
            )
        except TypeError:
            topdown = habitat_maps.get_topdown_map(
                sim,
                map_resolution=(map_resolution, map_resolution),
                num_samples=20000,
                draw_border=True,
            )

    canvas = _colorize_habitat_map(habitat_maps, topdown)

    gt_path = extract_gt_path(episode_meta)
    gt_pixels = []
    for pos in gt_path:
        try:
            gt_pixels.append(_to_grid_xy(habitat_maps, sim, pos, canvas.shape))
        except Exception:
            continue

    for a, b in zip(gt_pixels[:-1], gt_pixels[1:]):
        cv2.line(canvas, a, b, (255, 0, 0), 3)
    if gt_pixels:
        cv2.circle(canvas, gt_pixels[0], 7, (0, 180, 0), 2)
        cv2.circle(canvas, gt_pixels[-1], 9, (255, 0, 0), 2)

    pixels = []
    for step in traj:
        try:
            pixels.append(_to_grid_xy(habitat_maps, sim, step["position"], canvas.shape))
        except Exception:
            continue

    for a, b in zip(pixels[:-1], pixels[1:]):
        cv2.line(canvas, a, b, (0, 0, 255), 3)
    if pixels:
        cv2.circle(canvas, pixels[0], 8, (0, 255, 0), -1)
        cv2.circle(canvas, pixels[-1], 8, (0, 0, 255), -1)

    goal_pos = extract_goal_position(episode_meta)
    if goal_pos is not None:
        try:
            gxy = _to_grid_xy(habitat_maps, sim, goal_pos, canvas.shape)
            cv2.circle(canvas, gxy, 11, (255, 0, 255), 2)
        except Exception:
            pass

    cv2.putText(canvas, "Pred", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    cv2.putText(canvas, "GT" if gt_pixels else "GT N/A", (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

    return canvas, pixels


def save_video_bgr(path: str, frames_bgr: List[np.ndarray], fps: float) -> None:
    if not frames_bgr:
        return
    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, float(fps), (w, h))
    for fr in frames_bgr:
        writer.write(fr)
    writer.release()
