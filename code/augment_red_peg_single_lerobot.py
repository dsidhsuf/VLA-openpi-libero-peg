#!/usr/bin/env python3
"""
Build an augmented single-episode LeRobot dataset from a raw LIBERO episode.

The augmentation is sample reweighting by repetition: key frames are repeated in
chronological order so training sees more rare but important phases without
breaking image/state/action alignment.

Default emphasis:
  - gripper negative frames
  - upward/lift frames (z > threshold)
  - large XY motion frames
  - a window around gripper sign transitions

Input raw episode layout:
  episode_seed.../
    trajectory.npz
    metadata.json
    videos/agentview.mp4
    videos/robot0_eye_in_hand.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_TASK = "Pick up the red rectangular peg, keep it vertical, and insert it into the rectangular slot."


def read_frames(video_path: Path, image_size: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for frame in iio.imiter(video_path):
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        if image_size > 0 and (arr.shape[0] != image_size or arr.shape[1] != image_size):
            arr = np.asarray(Image.fromarray(arr).resize((image_size, image_size), Image.BILINEAR))
        frames.append(arr.astype(np.uint8))
    return frames


def infer_task_text(meta: dict, default_task: str) -> str:
    for key in ("task", "task_description", "instruction", "language_instruction", "prompt", "language"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default_task


def align_indices(num_steps: int, frame_count: int, frame_stride: int) -> tuple[np.ndarray, int]:
    stride = max(1, int(frame_stride))
    stride_indices = np.arange(stride - 1, num_steps, stride, dtype=np.int64)

    if frame_count == len(stride_indices) + 1:
        return stride_indices, 1
    if frame_count == len(stride_indices):
        return stride_indices, 0
    if frame_count == num_steps + 1:
        return np.arange(num_steps, dtype=np.int64), 1
    if frame_count == num_steps:
        return np.arange(num_steps, dtype=np.int64), 0

    linear = np.linspace(0, num_steps - 1, num=frame_count, dtype=np.int64)
    return linear, 0


def build_state(traj) -> np.ndarray:
    eef_pos = np.asarray(traj["robot0_eef_pos"], dtype=np.float32)
    eef_quat = np.asarray(traj["robot0_eef_quat_wxyz"], dtype=np.float32)
    gripper_qpos = np.asarray(traj["robot0_gripper_qpos"], dtype=np.float32)
    gripper_mean = np.mean(np.abs(gripper_qpos), axis=1, keepdims=True).astype(np.float32)
    return np.concatenate([eef_pos, eef_quat, gripper_mean], axis=1).astype(np.float32)


def summarize_actions(actions: np.ndarray) -> dict:
    g = actions[:, 6]
    z = actions[:, 2]
    xy_norm = np.linalg.norm(actions[:, :2], axis=1)
    return {
        "samples": int(actions.shape[0]),
        "gripper_min": float(g.min()),
        "gripper_max": float(g.max()),
        "gripper_mean": float(g.mean()),
        "gripper_neg_ratio_lt_-0.5": float((g < -0.5).mean()),
        "gripper_pos_ratio_gt_+0.5": float((g > 0.5).mean()),
        "z_mean": float(z.mean()),
        "z_neg_ratio_lt_-0.01": float((z < -0.01).mean()),
        "z_pos_ratio_gt_+0.01": float((z > 0.01).mean()),
        "xy_big_ratio_gt_0.05": float((xy_norm > 0.05).mean()),
        "xyz_mean": [float(x) for x in actions[:, :3].mean(axis=0)],
    }


def compute_repeat_counts(actions: np.ndarray, args) -> tuple[np.ndarray, dict[str, int]]:
    n = actions.shape[0]
    counts = np.ones(n, dtype=np.int64)
    reasons = {
        "gripper_negative": 0,
        "z_positive": 0,
        "large_xy": 0,
        "gripper_transition_window": 0,
    }

    g = actions[:, 6]
    z = actions[:, 2]
    xy_norm = np.linalg.norm(actions[:, :2], axis=1)

    neg_mask = g < float(args.gripper_neg_threshold)
    z_pos_mask = z > float(args.z_pos_threshold)
    xy_mask = xy_norm > float(args.xy_threshold)

    counts[neg_mask] = np.maximum(counts[neg_mask], int(args.repeat_gripper_neg))
    counts[z_pos_mask] = np.maximum(counts[z_pos_mask], int(args.repeat_z_pos))
    counts[xy_mask] = np.maximum(counts[xy_mask], int(args.repeat_large_xy))

    reasons["gripper_negative"] = int(neg_mask.sum())
    reasons["z_positive"] = int(z_pos_mask.sum())
    reasons["large_xy"] = int(xy_mask.sum())

    signs = np.sign(g)
    transition_idx = np.where(np.abs(np.diff(signs, prepend=signs[0])) > 0)[0]
    window = max(0, int(args.transition_window))
    transition_mask = np.zeros(n, dtype=bool)
    for idx in transition_idx:
        lo = max(0, int(idx) - window)
        hi = min(n, int(idx) + window + 1)
        transition_mask[lo:hi] = True
    counts[transition_mask] = np.maximum(counts[transition_mask], int(args.repeat_transition))
    reasons["gripper_transition_window"] = int(transition_mask.sum())

    if args.max_repeat > 0:
        counts = np.minimum(counts, int(args.max_repeat))

    return counts, reasons


def build_features(height: int, width: int) -> dict:
    state_names = ["eef_x", "eef_y", "eef_z", "quat_w", "quat_x", "quat_y", "quat_z", "gripper"]
    action_names = [f"a{i}" for i in range(7)]
    return {
        "observation.images.image": {
            "dtype": "video",
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        },
        "observation.images.image2": {
            "dtype": "video",
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        },
        "observation.state": {"dtype": "float32", "shape": (8,), "names": state_names},
        "action": {"dtype": "float32", "shape": (7,), "names": action_names},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-episode", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--camera-primary", default="agentview")
    parser.add_argument("--camera-secondary", default="robot0_eye_in_hand")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--default-task", default=DEFAULT_TASK)
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--force-overwrite", action="store_true")

    parser.add_argument("--gripper-neg-threshold", type=float, default=-0.5)
    parser.add_argument("--z-pos-threshold", type=float, default=0.01)
    parser.add_argument("--xy-threshold", type=float, default=0.05)
    parser.add_argument("--transition-window", type=int, default=4)
    parser.add_argument("--repeat-gripper-neg", type=int, default=3)
    parser.add_argument("--repeat-z-pos", type=int, default=6)
    parser.add_argument("--repeat-large-xy", type=int, default=4)
    parser.add_argument("--repeat-transition", type=int, default=6)
    parser.add_argument("--max-repeat", type=int, default=8)
    args = parser.parse_args()

    src_episode = args.src_episode.resolve()
    output_root = args.output_root.resolve()

    if not (src_episode / "trajectory.npz").exists():
        raise FileNotFoundError(f"Missing trajectory.npz under {src_episode}")
    if not (src_episode / "metadata.json").exists():
        raise FileNotFoundError(f"Missing metadata.json under {src_episode}")

    if output_root.exists():
        if not args.force_overwrite:
            raise FileExistsError(f"{output_root} exists. Use --force-overwrite to recreate it.")
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((src_episode / "metadata.json").read_text(encoding="utf-8"))
    traj = np.load(src_episode / "trajectory.npz")
    action = np.asarray(traj["action"], dtype=np.float32)
    state = build_state(traj)
    if state.shape[0] != action.shape[0]:
        raise RuntimeError(f"state/action length mismatch: state={state.shape[0]} action={action.shape[0]}")

    video_dir = src_episode / "videos"
    primary_path = video_dir / f"{args.camera_primary}.mp4"
    secondary_path = video_dir / f"{args.camera_secondary}.mp4"
    if not primary_path.exists():
        raise FileNotFoundError(f"Missing primary video: {primary_path}")
    if not secondary_path.exists():
        raise FileNotFoundError(f"Missing secondary video: {secondary_path}")

    frames_primary = read_frames(primary_path, image_size=int(args.image_size))
    frames_secondary = read_frames(secondary_path, image_size=int(args.image_size))
    min_frames = min(len(frames_primary), len(frames_secondary))
    frame_stride = int(metadata.get("frame_stride", 1))
    control_hz = int(metadata.get("control_hz", 20))
    dataset_fps = max(1, int(round(control_hz / max(1, frame_stride))))
    step_indices, frame_offset = align_indices(action.shape[0], min_frames, frame_stride)
    usable = min(len(step_indices), max(0, min_frames - frame_offset))
    if usable <= 0:
        raise RuntimeError("No aligned samples after frame/action alignment.")

    step_indices = step_indices[:usable]
    frames_primary = frames_primary[frame_offset : frame_offset + usable]
    frames_secondary = frames_secondary[frame_offset : frame_offset + usable]
    base_action = action[step_indices]
    repeat_counts, reasons = compute_repeat_counts(base_action, args)
    expanded_local_indices = np.repeat(np.arange(usable, dtype=np.int64), repeat_counts)
    expanded_step_indices = step_indices[expanded_local_indices]
    expanded_actions = action[expanded_step_indices]

    task = infer_task_text(metadata, default_task=args.default_task)
    h = int(frames_primary[0].shape[0])
    w = int(frames_primary[0].shape[1])
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_root,
        fps=dataset_fps,
        features=build_features(h, w),
        robot_type="libero",
        use_videos=True,
        vcodec=args.vcodec,
    )

    for local_idx, step_idx in zip(expanded_local_indices, expanded_step_indices):
        frame = {
            "observation.images.image": frames_primary[int(local_idx)],
            "observation.images.image2": frames_secondary[int(local_idx)],
            "observation.state": state[int(step_idx)],
            "action": action[int(step_idx)],
            "task": task,
        }
        dataset.add_frame(frame)

    dataset.save_episode(parallel_encoding=True)

    before = summarize_actions(base_action)
    after = summarize_actions(expanded_actions)
    report = {
        "src_episode": str(src_episode),
        "output_root": str(output_root),
        "repo_id": args.repo_id,
        "task": task,
        "usable_original_samples": int(usable),
        "augmented_samples": int(expanded_actions.shape[0]),
        "repeat_total_mean": float(repeat_counts.mean()),
        "repeat_total_max": int(repeat_counts.max()),
        "reason_counts": reasons,
        "before": before,
        "after": after,
    }
    report_path = output_root / "augmentation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[augment] source:", src_episode)
    print("[augment] output:", output_root)
    print("[augment] repo_id:", args.repo_id)
    print("[augment] samples:", usable, "->", int(expanded_actions.shape[0]))
    print("[augment] reason_counts:", reasons)
    print("[before]", json.dumps(before, ensure_ascii=False))
    print("[after ]", json.dumps(after, ensure_ascii=False))
    print("[saved]", report_path)


if __name__ == "__main__":
    main()
