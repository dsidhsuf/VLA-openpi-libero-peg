#!/usr/bin/env python3
"""
Convert one generated episode folder into a LeRobot v3 dataset
that is directly compatible with pi0_libero_base fine-tuning.

Input episode folder is expected to contain:
  - trajectory.npz
  - metadata.json
  - videos/agentview.mp4
  - videos/frontview.mp4
  - videos/robot0_eye_in_hand.mp4
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


def align_indices(num_steps: int, frame_count: int, frame_stride: int) -> tuple[np.ndarray, int]:
    """
    Returns:
      step_indices: selected indices in trajectory arrays
      frame_offset: drop this many leading frames from decoded videos
    """
    stride = max(1, int(frame_stride))
    stride_indices = np.arange(stride - 1, num_steps, stride, dtype=np.int64)

    # Recorder usually stores an initial frame before first action.
    if frame_count == len(stride_indices) + 1:
        return stride_indices, 1
    if frame_count == len(stride_indices):
        return stride_indices, 0

    # Fallback: linearly align if lengths don't match expected stride pattern.
    linear = np.linspace(0, num_steps - 1, num=frame_count, dtype=np.int64)
    return linear, 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode-dir",
        type=Path,
        default=Path("/root/autodl-tmp/openpi_earbud_proto/libero_lerobot_dataset/easy/episode_seed0_20260423_212633"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/autodl-tmp/openpi_earbud_proto/lerobot_ready/earbud_insert_single_v3"),
    )
    parser.add_argument("--repo-id", type=str, default="local/earbud_insert_single_v3")
    parser.add_argument("--task", type=str, default="Insert the earbud into the charging slot.")
    parser.add_argument("--camera-primary", type=str, default="agentview")
    parser.add_argument("--camera-secondary", type=str, default="robot0_eye_in_hand")
    parser.add_argument("--camera-aux", type=str, default="frontview")
    parser.add_argument("--include-aux-camera", action="store_true")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--force-overwrite", action="store_true")
    args = parser.parse_args()

    episode_dir = args.episode_dir.resolve()
    traj_path = episode_dir / "trajectory.npz"
    meta_path = episode_dir / "metadata.json"
    video_dir = episode_dir / "videos"

    if not traj_path.exists():
        raise FileNotFoundError(f"Missing file: {traj_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing file: {meta_path}")
    if not video_dir.exists():
        raise FileNotFoundError(f"Missing directory: {video_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        ep_meta = json.load(f)

    control_hz = int(ep_meta.get("control_hz", 20))
    frame_stride = int(ep_meta.get("frame_stride", 1))
    dataset_fps = max(1, int(round(control_hz / max(1, frame_stride))))

    traj = np.load(traj_path)
    action = np.asarray(traj["action"], dtype=np.float32)  # (T, 7)
    eef_pos = np.asarray(traj["robot0_eef_pos"], dtype=np.float32)  # (T, 3)
    eef_quat = np.asarray(traj["robot0_eef_quat_wxyz"], dtype=np.float32)  # (T, 4)
    gripper_qpos = np.asarray(traj["robot0_gripper_qpos"], dtype=np.float32)  # (T, 2)
    gripper_mean = np.mean(np.abs(gripper_qpos), axis=1, keepdims=True).astype(np.float32)  # (T, 1)
    state = np.concatenate([eef_pos, eef_quat, gripper_mean], axis=1).astype(np.float32)  # (T, 8)

    num_steps = action.shape[0]
    if state.shape[0] != num_steps:
        raise ValueError(f"State/action length mismatch: state={state.shape[0]} action={num_steps}")

    cam_primary_path = video_dir / f"{args.camera_primary}.mp4"
    cam_secondary_path = video_dir / f"{args.camera_secondary}.mp4"
    cam_aux_path = video_dir / f"{args.camera_aux}.mp4"

    if not cam_primary_path.exists():
        raise FileNotFoundError(f"Missing primary camera video: {cam_primary_path}")
    if not cam_secondary_path.exists():
        raise FileNotFoundError(f"Missing secondary camera video: {cam_secondary_path}")

    frames_primary = read_frames(cam_primary_path, args.image_size)
    frames_secondary = read_frames(cam_secondary_path, args.image_size)
    frames_aux = read_frames(cam_aux_path, args.image_size) if (args.include_aux_camera and cam_aux_path.exists()) else None

    min_frames = min(len(frames_primary), len(frames_secondary))
    if frames_aux is not None:
        min_frames = min(min_frames, len(frames_aux))
    if min_frames <= 0:
        raise RuntimeError("No decoded frames found in camera videos.")

    step_indices, frame_offset = align_indices(num_steps=num_steps, frame_count=min_frames, frame_stride=frame_stride)
    usable = min(len(step_indices), max(0, min_frames - frame_offset))
    if usable <= 0:
        raise RuntimeError("No aligned samples after frame/action alignment.")

    step_indices = step_indices[:usable]
    frames_primary = frames_primary[frame_offset : frame_offset + usable]
    frames_secondary = frames_secondary[frame_offset : frame_offset + usable]
    if frames_aux is not None:
        frames_aux = frames_aux[frame_offset : frame_offset + usable]

    if args.output_root.exists():
        if args.force_overwrite:
            shutil.rmtree(args.output_root)
        else:
            raise FileExistsError(
                f"{args.output_root} already exists. Use --force-overwrite to recreate it."
            )

    h = int(frames_primary[0].shape[0])
    w = int(frames_primary[0].shape[1])
    state_names = ["eef_x", "eef_y", "eef_z", "quat_w", "quat_x", "quat_y", "quat_z", "gripper"]
    action_names = [f"a{i}" for i in range(7)]

    features = {
        # Match pi0_libero_base expected visual keys exactly.
        "observation.images.image": {"dtype": "video", "shape": (3, h, w), "names": ["channels", "height", "width"]},
        "observation.images.image2": {"dtype": "video", "shape": (3, h, w), "names": ["channels", "height", "width"]},
        "observation.state": {"dtype": "float32", "shape": (8,), "names": state_names},
        "action": {"dtype": "float32", "shape": (7,), "names": action_names},
    }
    if frames_aux is not None:
        # Extra camera kept for analysis/future models; pi0_libero_base training uses image + image2.
        features["observation.images.frontview_aux"] = {
            "dtype": "video",
            "shape": (3, h, w),
            "names": ["channels", "height", "width"],
        }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output_root,
        fps=dataset_fps,
        features=features,
        robot_type="libero",
        use_videos=True,
        vcodec="h264",
    )

    for i, step_idx in enumerate(step_indices):
        frame = {
            "observation.images.image": frames_primary[i],
            "observation.images.image2": frames_secondary[i],
            "observation.state": state[step_idx],
            "action": action[step_idx],
            "task": args.task,
        }
        if frames_aux is not None:
            frame["observation.images.frontview_aux"] = frames_aux[i]
        dataset.add_frame(frame)

    dataset.save_episode(parallel_encoding=True)
    dataset.finalize()

    # Read-back check to ensure training compatibility.
    check_ds = LeRobotDataset(repo_id=args.repo_id, root=args.output_root)
    sample = check_ds[0]
    required = ["observation.images.image", "observation.images.image2", "observation.state", "action"]
    missing = [k for k in required if k not in sample]
    if missing:
        raise RuntimeError(f"Converted dataset missing required keys: {missing}")

    report = {
        "episode_dir": str(episode_dir),
        "output_root": str(args.output_root),
        "repo_id": args.repo_id,
        "dataset_fps": dataset_fps,
        "num_steps_raw": int(num_steps),
        "num_samples_converted": int(usable),
        "frame_stride_meta": frame_stride,
        "frame_offset_used": int(frame_offset),
        "step_index_head": step_indices[:10].tolist(),
        "sample_keys": sorted(list(sample.keys())),
        "camera_primary": args.camera_primary,
        "camera_secondary": args.camera_secondary,
        "camera_aux_included": bool(frames_aux is not None),
    }
    with open(args.output_root / "conversion_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Conversion done.")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
