#!/usr/bin/env python3
"""
Build a one-episode LeRobot dataset from raw demo observations and teacher actions.

This is for distilling the successful waypoint/servo rollout:
  observation = raw episode images + raw robot state + task text
  action      = eval action_trace_csv from the successful teacher policy

It avoids training on the raw demo action labels, which were not the action
interface that made the corrected benchmark succeed.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from convert_libero_tree_to_single_lerobot_small import (
    align_indices,
    build_features,
    infer_task_text,
    read_frames,
)


DEFAULT_TASK = "Pick up the red rectangular peg, keep it vertical, and insert it into the rectangular slot."


def load_teacher_actions(csv_path: Path, episode_id: int, truncate_end_step: int) -> np.ndarray:
    by_step: dict[int, np.ndarray] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row.get("episode_id", 0)) != int(episode_id):
                continue
            step = int(row["step"])
            by_step[step] = np.asarray(
                [
                    float(row["a0"]),
                    float(row["a1"]),
                    float(row["a2"]),
                    float(row["a3"]),
                    float(row["a4"]),
                    float(row["a5"]),
                    float(row["gripper"]),
                ],
                dtype=np.float32,
            )

    if not by_step:
        raise RuntimeError(f"No teacher actions found for episode_id={episode_id} in {csv_path}")

    end = int(truncate_end_step)
    if end <= 0:
        end = max(by_step) + 1
    actions = np.zeros((end, 7), dtype=np.float32)
    missing = []
    for step in range(end):
        value = by_step.get(step)
        if value is None:
            missing.append(step)
        else:
            actions[step] = value
    if missing:
        preview = ", ".join(str(x) for x in missing[:20])
        raise RuntimeError(f"Missing teacher action rows for steps: {preview}")
    return actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-episode", required=True, type=Path)
    parser.add_argument("--teacher-action-csv", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--teacher-episode-id", type=int, default=0)
    parser.add_argument("--truncate-end-step", type=int, default=0, help="Exclusive end step. 0 means max teacher step + 1.")
    parser.add_argument("--camera-primary", default="agentview")
    parser.add_argument("--camera-secondary", default="robot0_eye_in_hand")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--default-task", default=DEFAULT_TASK)
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--force-overwrite", action="store_true")
    args = parser.parse_args()

    raw_episode = args.raw_episode.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.force_overwrite:
            raise FileExistsError(f"{output_root} exists. Use --force-overwrite to recreate it.")
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)

    meta_path = raw_episode / "metadata.json"
    traj_path = raw_episode / "trajectory.npz"
    video_dir = raw_episode / "videos"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    control_hz = int(meta.get("control_hz", 20))
    frame_stride = int(meta.get("frame_stride", 1))
    fps = max(1, int(round(control_hz / max(1, frame_stride))))
    task = infer_task_text(meta, args.default_task)

    traj = np.load(traj_path)
    raw_action = np.asarray(traj["action"], dtype=np.float32)
    eef_pos = np.asarray(traj["robot0_eef_pos"], dtype=np.float32)
    eef_quat = np.asarray(traj["robot0_eef_quat_wxyz"], dtype=np.float32)
    gripper_qpos = np.asarray(traj["robot0_gripper_qpos"], dtype=np.float32)
    gripper_mean = np.mean(np.abs(gripper_qpos), axis=1, keepdims=True).astype(np.float32)
    state = np.concatenate([eef_pos, eef_quat, gripper_mean], axis=1).astype(np.float32)

    teacher_action = load_teacher_actions(
        args.teacher_action_csv.resolve(),
        episode_id=args.teacher_episode_id,
        truncate_end_step=args.truncate_end_step,
    )
    end_step = min(len(raw_action), len(state), len(teacher_action))
    if end_step <= 0:
        raise RuntimeError("No usable samples after teacher/raw length alignment.")
    teacher_action = teacher_action[:end_step]

    frames_primary = read_frames(video_dir / f"{args.camera_primary}.mp4", args.image_size)
    frames_secondary = read_frames(video_dir / f"{args.camera_secondary}.mp4", args.image_size)
    min_frames = min(len(frames_primary), len(frames_secondary))
    step_indices, frame_offset = align_indices(
        num_steps=len(raw_action),
        frame_count=min_frames,
        frame_stride=frame_stride,
    )
    usable = min(len(step_indices), max(0, min_frames - frame_offset))
    step_indices = step_indices[:usable]
    frames_primary = frames_primary[frame_offset : frame_offset + usable]
    frames_secondary = frames_secondary[frame_offset : frame_offset + usable]

    keep = step_indices < end_step
    step_indices = step_indices[keep]
    frames_primary = [frame for frame, ok in zip(frames_primary, keep) if bool(ok)]
    frames_secondary = [frame for frame, ok in zip(frames_secondary, keep) if bool(ok)]
    if len(step_indices) == 0:
        raise RuntimeError("No samples remained after truncate/end-step filtering.")

    h = int(frames_primary[0].shape[0])
    w = int(frames_primary[0].shape[1])
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_root,
        fps=fps,
        features=build_features(h, w, include_aux_camera=False),
        robot_type="libero",
        use_videos=True,
        vcodec=args.vcodec,
    )

    for sample_idx, step_idx in enumerate(step_indices):
        dataset.add_frame(
            {
                "observation.images.image": frames_primary[sample_idx],
                "observation.images.image2": frames_secondary[sample_idx],
                "observation.state": state[int(step_idx)],
                "action": teacher_action[int(step_idx)],
                "task": task,
            }
        )
    dataset.save_episode(parallel_encoding=True)
    dataset.finalize()

    report = {
        "raw_episode": str(raw_episode),
        "teacher_action_csv": str(args.teacher_action_csv.resolve()),
        "output_root": str(output_root),
        "repo_id": args.repo_id,
        "num_samples": int(len(step_indices)),
        "first_step": int(step_indices[0]),
        "last_step": int(step_indices[-1]),
        "task": task,
    }
    (output_root / "teacher_conversion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[done]", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
