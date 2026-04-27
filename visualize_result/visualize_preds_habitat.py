#!/usr/bin/env python3
# Copyright: CLI — render preds.json trajectories in Habitat-Sim
"""
从 preds.json 读取 Habitat 连续轨迹，在对应 MP3D .glb 中渲染并导出视频/俯视图。

示例:
  python -m visualize_result.visualize_preds_habitat --preds e:/bevbert_data/preds.json \\
    --dataset data/datasets/.../val_seen_bertidx.json.gz --scene_root e:/bevbert_data \\
    --out_dir e:/bevbert_data/vis_out

  仅单场景调试用:
  python -m visualize_result.visualize_preds_habitat --preds e:/bevbert_data/preds.json \\
    --force_scene e:/bevbert_data/gTV8FGcVJC9.glb --out_dir e:/bevbert_data/vis_out
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# 包内与脚本直接运行均可
if __name__ == "__main__" and __package__ is None:
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)

from visualize_result.habitat_traj_renderer import HabitatTrajectoryRenderer
from visualize_result.viz_io import (
    add_instruction_panel,
    draw_habitat_topdown,
    draw_topdown,
    extract_goal_position,
    extract_gt_path,
    extract_instruction_text,
    infer_gt_path_from_dataset,
    load_episode_index,
    load_gt_index,
    load_preds,
    merge_gt_into_episode,
    parse_scan_id_from_scene,
    resolve_scene_path,
    save_video_bgr,
)


def log_msg(message: str) -> None:
    """Python 3.6 服务器 stdout 可能是 ASCII，避免中文日志导致崩溃。"""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", "ignore").decode("ascii"))


def render_one_episode(
    renderer: HabitatTrajectoryRenderer,
    episode_id: str,
    traj: List[dict],
    episode_meta: Optional[dict],
    scene_root: Optional[str],
    out_dir: str,
    force_scene: Optional[str],
    fps: float,
    save_depth: bool,
    save_frames: bool,
    save_render_video: bool,
    topdown_mode: str,
) -> str:
    if episode_meta is not None:
        scene_path = resolve_scene_path(
            episode_meta, scene_root=scene_root, force_scene_path=force_scene
        )
    else:
        if not force_scene:
            raise ValueError("episode metadata is missing; please provide --force_scene")
        scene_path = os.path.normpath(os.path.abspath(force_scene))

    scan_id = parse_scan_id_from_scene(scene_path)
    ep_out = os.path.join(out_dir, f"ep_{episode_id}")
    frame_dir = os.path.join(ep_out, "frames")
    os.makedirs(ep_out, exist_ok=True)
    if save_frames or save_depth:
        os.makedirs(frame_dir, exist_ok=True)

    renderer.load_scene(scene_path)
    instruction_text = extract_instruction_text(episode_meta)
    gt_path = extract_gt_path(episode_meta)
    has_gt = len(gt_path) >= 2
    goal_position = extract_goal_position(episode_meta)
    final_to_goal_euclidean = None
    if goal_position is not None and traj:
        final_to_goal_euclidean = float(
            np.linalg.norm(
                np.array(traj[-1]["position"], dtype=np.float32)
                - np.array(goal_position, dtype=np.float32)
            )
        )
    topdown_source = "plain"
    if topdown_mode in ["habitat", "auto"]:
        try:
            topdown_full, topdown_pixels = draw_habitat_topdown(
                renderer.sim,
                traj,
                episode_meta=episode_meta,
                map_resolution=800,
            )
            topdown_source = "habitat"
        except Exception as ex:
            if topdown_mode == "habitat":
                raise
            log_msg(f"[warn] {episode_id}: Habitat topdown failed; fallback to plain map: {ex}")
            topdown_full, topdown_pixels = draw_topdown(traj)
    else:
        topdown_full, topdown_pixels = draw_topdown(traj)

    frames_render_bgr: List[np.ndarray] = []
    frames_panel_bgr: List[np.ndarray] = []

    n = len(traj)
    for t, step in enumerate(traj):
        rgb, depth = renderer.render_step(
            position=step["position"],
            heading=step["heading"],
        )

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if save_frames:
            cv2.imwrite(os.path.join(frame_dir, f"{t:04d}_rgb.png"), bgr)
        if save_depth:
            np.save(os.path.join(frame_dir, f"{t:04d}_depth.npy"), depth)
            d_vis = np.clip(depth / 10.0, 0.0, 1.0)
            d_vis = (d_vis * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(frame_dir, f"{t:04d}_depth.png"), d_vis)

        if save_render_video:
            frames_render_bgr.append(bgr)

        topdown_now = topdown_full.copy()
        for p in topdown_pixels[: t + 1]:
            cv2.circle(topdown_now, p, 4, (0, 0, 0), -1)
        th, tw = rgb.shape[0], rgb.shape[1]
        topdown_resized = cv2.resize(topdown_now, (tw, th))

        panel = np.concatenate([bgr, topdown_resized], axis=1)
        label = f"ep={episode_id} scan={scan_id} step={t}/{max(n - 1, 0)}"
        cv2.putText(
            panel,
            label,
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        frames_panel_bgr.append(panel)

    topdown_path = os.path.join(ep_out, f"ep_{episode_id}_topdown.png")
    topdown_with_text = add_instruction_panel(
        topdown_full,
        instruction_text,
        episode_id=episode_id,
        scan_id=scan_id,
        has_gt=has_gt,
    )
    cv2.imwrite(topdown_path, topdown_with_text)
    cv2.imwrite(os.path.join(ep_out, f"ep_{episode_id}_topdown_map_only.png"), topdown_full)

    if save_render_video:
        save_video_bgr(
            os.path.join(ep_out, f"ep_{episode_id}_render.mp4"), frames_render_bgr, fps
        )
    save_video_bgr(
        os.path.join(ep_out, f"ep_{episode_id}_panel.mp4"), frames_panel_bgr, fps
    )

    meta: Dict[str, Any] = {
        "episode_id": episode_id,
        "scan_id": scan_id,
        "scene_path": scene_path,
        "num_steps": n,
        "start": traj[0] if traj else None,
        "end": traj[-1] if traj else None,
        "topdown_source": topdown_source,
        "has_ground_truth_path": has_gt,
        "gt_path_length": len(gt_path),
        "goal_position": goal_position,
        "final_to_goal_euclidean": final_to_goal_euclidean,
        "success_by_euclidean_3m": (
            final_to_goal_euclidean <= 3.0
            if final_to_goal_euclidean is not None
            else None
        ),
        "saved_frames": bool(save_frames),
        "saved_render_video": bool(save_render_video),
    }
    if episode_meta is not None and "_eval_metrics" in episode_meta:
        meta["eval_metrics"] = episode_meta["_eval_metrics"]
    if episode_meta is not None:
        if "instruction" in episode_meta:
            meta["instruction_text"] = instruction_text
        if goal_position is not None:
            meta["goal_position"] = goal_position

    with open(os.path.join(ep_out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return ep_out


def main() -> None:
    p = argparse.ArgumentParser(description="将 preds.json 在 Habitat 中可视化并导出")
    p.add_argument("--preds", type=str, required=True, help="preds.json 路径")
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="VLN-CE 预处理带 scene_id 的 json / json.gz；可与 --force_scene 联用",
    )
    p.add_argument(
        "--gt_path",
        type=str,
        default=None,
        help="可选：eval 使用的 *_gt.json.gz；不填时会尝试从 --dataset 自动推断",
    )
    p.add_argument(
        "--scene_root",
        type=str,
        default=None,
        help="在 scene_id 为相对路径时，用于拼 mp3d/scan/scan.glb",
    )
    p.add_argument("--out_dir", type=str, default="vis_out", help="输出根目录 (vis_out/...)")
    p.add_argument(
        "--force_scene",
        type=str,
        default=None,
        help="强制使用此 .glb；若提供，所有 episode 共用该场景",
    )
    p.add_argument("--episodes", type=str, default=None, help="仅渲染这些 id，逗号分隔，如 285,300")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--hfov", type=float, default=90.0)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--save_frames", action="store_true", help="保存每一步 rgb 到 frames/；默认不保存")
    p.add_argument("--save_render_video", action="store_true", help="额外保存纯第一视角 render.mp4；默认只保存 panel.mp4")
    p.add_argument("--save_depth", action="store_true", help="同时保存每步 depth .npy 与 .png；会自动创建 frames/")
    p.add_argument(
        "--topdown_mode",
        choices=["auto", "habitat", "plain"],
        default="auto",
        help="auto/habitat 使用 Habitat 可通行区域地图；plain 为原始白底 x-z 投影",
    )
    args = p.parse_args()
    if not args.dataset and not args.force_scene:
        p.error("必须提供 --dataset，或提供 --force_scene 以在无元数据时指定 .glb")

    preds = load_preds(args.preds)
    ep_index: Optional[Dict[str, dict]] = None
    if args.dataset:
        ep_index = load_episode_index(args.dataset)
    gt_index: Optional[Dict[str, dict]] = None
    gt_path = args.gt_path or infer_gt_path_from_dataset(args.dataset)
    if gt_path and os.path.isfile(gt_path):
        gt_index = load_gt_index(gt_path)
        log_msg(f"[info] loaded gt: {gt_path}")
    elif args.gt_path:
        log_msg(f"[warn] specified gt_path does not exist: {args.gt_path}")
    elif gt_path:
        log_msg(f"[warn] inferred gt_path does not exist: {gt_path}")

    episode_filter = None
    if args.episodes:
        episode_filter = {x.strip() for x in args.episodes.split(",") if x.strip()}

    os.makedirs(args.out_dir, exist_ok=True)

    renderer = HabitatTrajectoryRenderer(
        width=args.width,
        height=args.height,
        hfov=args.hfov,
        gpu_id=args.gpu_id,
    )

    try:
        for eid_raw, traj in preds.items():
            eid = str(eid_raw)
            if episode_filter is not None and eid not in episode_filter:
                continue
            eval_vis_item = None
            if isinstance(traj, dict) and "pred_path" in traj:
                eval_vis_item = traj
                traj = traj["pred_path"]
            meta = ep_index.get(eid) if ep_index is not None else None
            gt_item = gt_index.get(eid) if gt_index is not None else None
            meta = merge_gt_into_episode(meta, gt_item)
            if eval_vis_item is not None:
                meta = dict(meta or {})
                if "gt_path" in eval_vis_item:
                    meta["gt_path"] = eval_vis_item["gt_path"]
                if "goal_position" in eval_vis_item:
                    meta["_gt_goal_position"] = eval_vis_item["goal_position"]
                if "metrics" in eval_vis_item:
                    meta["_eval_metrics"] = eval_vis_item["metrics"]
            if ep_index is not None and meta is None and not args.force_scene:
                log_msg(f"[skip] {eid}: not found in dataset and --force_scene is not provided")
                continue
            if not isinstance(traj, list) or len(traj) == 0:
                log_msg(f"[skip] {eid}: empty trajectory")
                continue

            try:
                out = render_one_episode(
                    renderer=renderer,
                    episode_id=eid,
                    traj=traj,
                    episode_meta=meta,
                    scene_root=args.scene_root,
                    out_dir=args.out_dir,
                    force_scene=args.force_scene,
                    fps=args.fps,
                    save_depth=args.save_depth,
                    save_frames=args.save_frames,
                    save_render_video=args.save_render_video,
                    topdown_mode=args.topdown_mode,
                )
                log_msg(f"[ok] {eid} -> {out}")
            except Exception as ex:
                log_msg(f"[fail] {eid}: {ex}")
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
