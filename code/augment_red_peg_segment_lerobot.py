#!/usr/bin/env python3
"""
Build a segment-augmented LeRobot dataset from one raw LIBERO episode.

Unlike row repetition, this script preserves temporal order inside every
training episode. This matters for PI0/PI0.5 because they train on future action
chunks; duplicating individual frames inside one episode corrupts those chunks.

Output dataset contains:
  - the full original episode once by default
  - repeated short episodes around rare / critical phases:
      * z-positive lift/move-up frames
      * large XY transfer frames
      * gripper sign transition windows
      * optional gripper-negative windows
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_TASK = "Pick up the red rectangular peg, keep it vertical, and insert it into the rectangular slot."


@dataclass(frozen=True)
class Segment:
    name: str
    start: int
    end: int
    repeat: int


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


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    runs = []
    start = None
    for i, keep in enumerate(mask):
        if keep and start is None:
            start = i
        elif not keep and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def expand_window(start: int, end: int, n: int, context: int, min_len: int) -> tuple[int, int]:
    start = max(0, int(start) - int(context))
    end = min(n, int(end) + int(context))
    cur_len = end - start
    if cur_len >= min_len:
        return start, end
    extra = int(min_len) - cur_len
    left = extra // 2
    right = extra - left
    start = max(0, start - left)
    end = min(n, end + right)
    # If clamped at one side, expand the other side.
    if end - start < min_len:
        if start == 0:
            end = min(n, start + min_len)
        if end == n:
            start = max(0, end - min_len)
    return start, end


def add_segment(
    segments: list[Segment],
    name: str,
    start: int,
    end: int,
    n: int,
    context: int,
    min_len: int,
    repeat: int,
):
    if repeat <= 0:
        return
    start, end = expand_window(start, end, n=n, context=context, min_len=min_len)
    if end - start <= 1:
        return
    segments.append(Segment(name=name, start=start, end=end, repeat=int(repeat)))


def build_segments(actions: np.ndarray, args) -> list[Segment]:
    n = actions.shape[0]
    z = actions[:, 2]
    g = actions[:, 6]
    xy_norm = np.linalg.norm(actions[:, :2], axis=1)
    segments: list[Segment] = []

    for i, (start, end) in enumerate(contiguous_runs(z > float(args.z_pos_threshold))):
        add_segment(
            segments,
            name=f"z_pos_{i}",
            start=start,
            end=end,
            n=n,
            context=args.segment_context,
            min_len=args.min_segment_len,
            repeat=args.repeat_z_pos_segments,
        )

    for i, (start, end) in enumerate(contiguous_runs(xy_norm > float(args.xy_threshold))):
        add_segment(
            segments,
            name=f"large_xy_{i}",
            start=start,
            end=end,
            n=n,
            context=args.segment_context,
            min_len=args.min_segment_len,
            repeat=args.repeat_large_xy_segments,
        )

    signs = np.sign(g)
    transition_idx = np.where(np.abs(np.diff(signs, prepend=signs[0])) > 0)[0]
    for i, idx in enumerate(transition_idx):
        add_segment(
            segments,
            name=f"gripper_transition_{i}",
            start=int(idx) - args.transition_context,
            end=int(idx) + args.transition_context + 1,
            n=n,
            context=0,
            min_len=args.min_segment_len,
            repeat=args.repeat_transition_segments,
        )

    if args.repeat_gripper_neg_segments > 0:
        for i, (start, end) in enumerate(contiguous_runs(g < float(args.gripper_neg_threshold))):
            add_segment(
                segments,
                name=f"gripper_neg_{i}",
                start=start,
                end=end,
                n=n,
                context=args.segment_context,
                min_len=args.min_segment_len,
                repeat=args.repeat_gripper_neg_segments,
            )

    # Deduplicate exact same windows by keeping the larger repeat count and a
    # combined name. Overlapping but non-identical windows are intentionally kept:
    # they are separate curriculum episodes centered on different rare events.
    merged: dict[tuple[int, int], Segment] = {}
    for seg in segments:
        key = (seg.start, seg.end)
        prev = merged.get(key)
        if prev is None or seg.repeat > prev.repeat:
            merged[key] = seg
    return list(merged.values())


def summarize_actions(actions: np.ndarray) -> dict:
    g = actions[:, 6]
    z = actions[:, 2]
    xy_norm = np.linalg.norm(actions[:, :2], axis=1)
    return {
        "samples": int(actions.shape[0]),
        "gripper_neg_ratio_lt_-0.5": float((g < -0.5).mean()),
        "gripper_pos_ratio_gt_+0.5": float((g > 0.5).mean()),
        "z_neg_ratio_lt_-0.01": float((z < -0.01).mean()),
        "z_pos_ratio_gt_+0.01": float((z > 0.01).mean()),
        "big_xy_ratio_gt_0.05": float((xy_norm > 0.05).mean()),
        "xyz_mean": [float(x) for x in actions[:, :3].mean(axis=0)],
    }


def add_episode_to_dataset(
    dataset: LeRobotDataset,
    frames_primary: list[np.ndarray],
    frames_secondary: list[np.ndarray],
    state: np.ndarray,
    action: np.ndarray,
    local_indices: np.ndarray,
    step_indices: np.ndarray,
    task: str,
):
    for local_idx in local_indices:
        step_idx = int(step_indices[int(local_idx)])
        dataset.add_frame(
            {
                "observation.images.image": frames_primary[int(local_idx)],
                "observation.images.image2": frames_secondary[int(local_idx)],
                "observation.state": state[step_idx],
                "action": action[step_idx],
                "task": task,
            }
        )
    dataset.save_episode(parallel_encoding=True)


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
    parser.add_argument("--no-full-episode", action="store_true")

    parser.add_argument("--z-pos-threshold", type=float, default=0.01)
    parser.add_argument("--xy-threshold", type=float, default=0.05)
    parser.add_argument("--gripper-neg-threshold", type=float, default=-0.5)
    parser.add_argument("--min-segment-len", type=int, default=80)
    parser.add_argument("--segment-context", type=int, default=18)
    parser.add_argument("--transition-context", type=int, default=24)
    parser.add_argument("--repeat-z-pos-segments", type=int, default=6)
    parser.add_argument("--repeat-large-xy-segments", type=int, default=4)
    parser.add_argument("--repeat-transition-segments", type=int, default=6)
    parser.add_argument("--repeat-gripper-neg-segments", type=int, default=1)
    args = parser.parse_args()

    src_episode = args.src_episode.resolve()
    output_root = args.output_root.resolve()
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
    frames_primary = read_frames(primary_path, image_size=args.image_size)
    frames_secondary = read_frames(secondary_path, image_size=args.image_size)
    min_frames = min(len(frames_primary), len(frames_secondary))
    frame_stride = int(metadata.get("frame_stride", 1))
    control_hz = int(metadata.get("control_hz", 20))
    fps = max(1, int(round(control_hz / max(1, frame_stride))))
    step_indices, frame_offset = align_indices(action.shape[0], min_frames, frame_stride)
    usable = min(len(step_indices), max(0, min_frames - frame_offset))
    if usable <= 0:
        raise RuntimeError("No aligned samples after frame/action alignment.")

    step_indices = step_indices[:usable]
    frames_primary = frames_primary[frame_offset : frame_offset + usable]
    frames_secondary = frames_secondary[frame_offset : frame_offset + usable]
    base_actions = action[step_indices]
    task = infer_task_text(metadata, default_task=args.default_task)

    h = int(frames_primary[0].shape[0])
    w = int(frames_primary[0].shape[1])
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_root,
        fps=fps,
        features=build_features(h, w),
        robot_type="libero",
        use_videos=True,
        vcodec=args.vcodec,
    )

    written_actions = []
    written_episodes = []
    if not args.no_full_episode:
        local_indices = np.arange(usable, dtype=np.int64)
        add_episode_to_dataset(dataset, frames_primary, frames_secondary, state, action, local_indices, step_indices, task)
        written_actions.append(base_actions)
        written_episodes.append({"name": "full_episode", "start": 0, "end": int(usable), "repeat_index": 0})

    segments = build_segments(base_actions, args)
    for seg in segments:
        local_indices = np.arange(seg.start, seg.end, dtype=np.int64)
        seg_actions = action[step_indices[local_indices]]
        for repeat_i in range(seg.repeat):
            add_episode_to_dataset(
                dataset,
                frames_primary,
                frames_secondary,
                state,
                action,
                local_indices,
                step_indices,
                task,
            )
            written_actions.append(seg_actions)
            written_episodes.append(
                {
                    "name": seg.name,
                    "start": int(seg.start),
                    "end": int(seg.end),
                    "repeat_index": int(repeat_i),
                    "length": int(seg.end - seg.start),
                }
            )

    all_actions = np.concatenate(written_actions, axis=0)
    report = {
        "src_episode": str(src_episode),
        "output_root": str(output_root),
        "repo_id": args.repo_id,
        "task": task,
        "usable_original_samples": int(usable),
        "num_written_episodes": len(written_episodes),
        "augmented_samples": int(all_actions.shape[0]),
        "segments": [seg.__dict__ for seg in segments],
        "written_episodes_preview": written_episodes[:50],
        "before": summarize_actions(base_actions),
        "after": summarize_actions(all_actions),
    }
    report_path = output_root / "segment_augmentation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[segment-augment] source:", src_episode)
    print("[segment-augment] output:", output_root)
    print("[segment-augment] repo_id:", args.repo_id)
    print("[segment-augment] episodes:", len(written_episodes))
    print("[segment-augment] samples:", usable, "->", int(all_actions.shape[0]))
    print("[before]", json.dumps(report["before"], ensure_ascii=False))
    print("[after ]", json.dumps(report["after"], ensure_ascii=False))
    print("[saved]", report_path)


if __name__ == "__main__":
    main()
