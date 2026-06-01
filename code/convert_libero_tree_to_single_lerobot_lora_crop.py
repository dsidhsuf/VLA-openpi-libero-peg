#!/usr/bin/env python3
"""
Convert a directory tree of cropped/keyframe/raw LIBERO episodes into ONE LeRobot dataset root.

Designed for LoRA fine-tuning after episode cropping, e.g.:
  dense episode -> crop_episode_for_lora.py --start_ratio 0.5 -> cropped episode tree
  cropped episode tree -> this converter -> LeRobot dataset -> lerobot-train

Input layout:
  <src-root>/
    easy/episode_seed.../
    medium/episode_seed.../
    hard/episode_seed.../

Each episode directory:
  - trajectory.npz
  - metadata.json
  - videos/<camera>.mp4

Main improvements over the original converter:
  1. Cropped/keyframe episodes can be aligned one-to-one without re-applying frame_stride.
  2. observation_state can be used directly when it exists.
  3. PI0.5/OpenPI-style camera feature names can be emitted.
  4. Strict alignment is the default, so silent linspace misalignment is avoided.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset


CROPPED_RECORD_MODES = {
    "cropped",
    "crop",
    "last50",
    "last_50",
    "keyframe",
    "keyframes",
    "phase_keyframe",
    "phase-aware-keyframe",
    "phase_aware_keyframe",
}


def is_raw_episode_dir(path: Path) -> bool:
    return (
        (path / "trajectory.npz").exists()
        and (path / "metadata.json").exists()
        and (path / "videos").exists()
    )


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


def align_indices_legacy(num_steps: int, frame_count: int, frame_stride: int, allow_linear_fallback: bool) -> tuple[np.ndarray, int, str]:
    """
    Original recorder alignment:
      - frame_count == sampled_steps      -> offset 0
      - frame_count == sampled_steps + 1  -> skip initial frame by offset 1
    """
    stride = max(1, int(frame_stride))
    stride_indices = np.arange(stride - 1, num_steps, stride, dtype=np.int64)

    if frame_count == len(stride_indices) + 1:
        return stride_indices, 1, "legacy_offset1"
    if frame_count == len(stride_indices):
        return stride_indices, 0, "legacy_offset0"

    if allow_linear_fallback:
        linear = np.linspace(0, num_steps - 1, num=frame_count, dtype=np.int64)
        return linear, 0, "legacy_linear_fallback"

    raise ValueError(
        "Frame/action length mismatch under legacy alignment: "
        f"num_steps={num_steps}, frame_count={frame_count}, frame_stride={frame_stride}, "
        f"expected_frames={len(stride_indices)} or {len(stride_indices) + 1}. "
        "If this is a cropped/keyframe dataset, use --alignment-mode one_to_one or auto. "
        "If you intentionally want approximate alignment, add --allow-linear-fallback."
    )


def align_indices_one_to_one(num_steps: int, frame_count: int, allow_linear_fallback: bool) -> tuple[np.ndarray, int, str]:
    """
    Cropped/keyframe alignment:
      - trajectory.npz and videos were already cropped by the same indices.
      - therefore sample i should use state/action i and frame i.
      - if the video still has one initial frame, offset 1 is allowed.
    """
    step_indices = np.arange(num_steps, dtype=np.int64)

    if frame_count == num_steps:
        return step_indices, 0, "one_to_one_offset0"
    if frame_count == num_steps + 1:
        return step_indices, 1, "one_to_one_offset1"

    usable = min(num_steps, frame_count)
    if usable > 0 and allow_linear_fallback:
        return np.arange(usable, dtype=np.int64), 0, "one_to_one_truncated_fallback"

    raise ValueError(
        "Frame/action length mismatch under one-to-one alignment: "
        f"num_steps={num_steps}, frame_count={frame_count}. "
        "Your crop/keyframe script should write the same number of video frames and trajectory rows, "
        "or exactly one extra initial video frame. "
        "Check video_offset in the crop/keyframe script."
    )


def choose_alignment_mode(ep_meta: dict, traj: np.lib.npyio.NpzFile, requested: str) -> str:
    if requested != "auto":
        return requested

    record_mode = str(ep_meta.get("record_mode", "")).strip().lower()
    has_source_step_index = "source_step_index" in traj.files
    has_crop_marker = any(k in ep_meta for k in ("source_start_index", "source_end_index", "crop_start_index", "crop_end_index"))

    if record_mode in CROPPED_RECORD_MODES or has_source_step_index or has_crop_marker:
        return "one_to_one"

    return "legacy"


def collect_episode_dirs(src_root: Path, categories: list[str]) -> list[Path]:
    episodes: list[Path] = []
    for cat in categories:
        cat_dir = src_root / cat
        if not cat_dir.exists() or not cat_dir.is_dir():
            continue
        for child in sorted(cat_dir.iterdir()):
            if child.is_dir() and is_raw_episode_dir(child):
                episodes.append(child.resolve())
    return episodes


def infer_task_text(meta: dict, default_task: str) -> str:
    for key in ("task", "task_description", "instruction", "language_instruction", "prompt"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default_task


def make_state_from_eef8(traj: np.lib.npyio.NpzFile) -> np.ndarray:
    eef_pos = np.asarray(traj["robot0_eef_pos"], dtype=np.float32)  # (T, 3)
    eef_quat = np.asarray(traj["robot0_eef_quat_wxyz"], dtype=np.float32)  # (T, 4)
    gripper_qpos = np.asarray(traj["robot0_gripper_qpos"], dtype=np.float32)  # (T, 2)
    gripper_mean = np.mean(np.abs(gripper_qpos), axis=1, keepdims=True).astype(np.float32)  # (T, 1)
    return np.concatenate([eef_pos, eef_quat, gripper_mean], axis=1).astype(np.float32)  # (T, 8)


def load_state(traj: np.lib.npyio.NpzFile, state_source: str) -> tuple[np.ndarray, str, list[str]]:
    """
    Returns state, actually_used_source, state_names.
    """
    if state_source == "observation_state":
        if "observation_state" not in traj.files:
            raise KeyError("state_source=observation_state but trajectory.npz has no 'observation_state'.")
        state = np.asarray(traj["observation_state"], dtype=np.float32)
        names = [f"s{i}" for i in range(state.shape[1])]
        return state, "observation_state", names

    if state_source == "eef8":
        state = make_state_from_eef8(traj)
        names = ["eef_x", "eef_y", "eef_z", "quat_w", "quat_x", "quat_y", "quat_z", "gripper"]
        return state, "eef8", names

    if state_source == "auto":
        if "observation_state" in traj.files:
            state = np.asarray(traj["observation_state"], dtype=np.float32)
            names = [f"s{i}" for i in range(state.shape[1])]
            return state, "observation_state", names
        state = make_state_from_eef8(traj)
        names = ["eef_x", "eef_y", "eef_z", "quat_w", "quat_x", "quat_y", "quat_z", "gripper"]
        return state, "eef8", names

    raise ValueError(f"Unknown state_source: {state_source}")


def get_feature_keys(feature_layout: str, include_aux_camera: bool) -> dict:
    if feature_layout == "generic":
        keys = {
            "primary": "observation.images.image",
            "secondary": "observation.images.image2",
            "aux": "observation.images.frontview_aux",
        }
    elif feature_layout == "pi05":
        keys = {
            "primary": "observation.images.base_0_rgb",
            "secondary": "observation.images.left_wrist_0_rgb",
            "aux": "observation.images.right_wrist_0_rgb",
        }
    else:
        raise ValueError(f"Unknown feature_layout: {feature_layout}")

    if not include_aux_camera:
        keys["aux"] = None
    return keys


def iter_episode_payloads(
    episodes: list[Path],
    workers: int,
    prefetch: int,
    camera_primary: str,
    camera_secondary: str,
    camera_aux: str,
    include_aux_camera: bool,
    image_size: int,
    default_task: str,
    state_source: str,
    alignment_mode: str,
    allow_linear_fallback: bool,
    fps_override: int,
):
    total = len(episodes)
    if workers <= 1:
        for idx, episode_dir in enumerate(episodes, start=1):
            try:
                payload = load_episode_payload(
                    episode_dir=episode_dir,
                    camera_primary=camera_primary,
                    camera_secondary=camera_secondary,
                    camera_aux=camera_aux,
                    include_aux_camera=include_aux_camera,
                    image_size=image_size,
                    default_task=default_task,
                    state_source=state_source,
                    alignment_mode=alignment_mode,
                    allow_linear_fallback=allow_linear_fallback,
                    fps_override=fps_override,
                )
                yield idx, total, episode_dir, payload, None
            except Exception as e:
                yield idx, total, episode_dir, None, e
        return

    max_prefetch = max(workers, prefetch)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        in_flight = {}
        next_submit_idx = 0

        def submit_one(i: int):
            ep = episodes[i]
            fut = pool.submit(
                load_episode_payload,
                episode_dir=ep,
                camera_primary=camera_primary,
                camera_secondary=camera_secondary,
                camera_aux=camera_aux,
                include_aux_camera=include_aux_camera,
                image_size=image_size,
                default_task=default_task,
                state_source=state_source,
                alignment_mode=alignment_mode,
                allow_linear_fallback=allow_linear_fallback,
                fps_override=fps_override,
            )
            in_flight[fut] = (i + 1, ep)

        while next_submit_idx < total and len(in_flight) < max_prefetch:
            submit_one(next_submit_idx)
            next_submit_idx += 1

        while in_flight:
            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                idx, episode_dir = in_flight.pop(fut)
                try:
                    payload = fut.result()
                    yield idx, total, episode_dir, payload, None
                except Exception as e:
                    yield idx, total, episode_dir, None, e

                if next_submit_idx < total:
                    submit_one(next_submit_idx)
                    next_submit_idx += 1


def load_episode_payload(
    episode_dir: Path,
    camera_primary: str,
    camera_secondary: str,
    camera_aux: str,
    include_aux_camera: bool,
    image_size: int,
    default_task: str,
    state_source: str,
    alignment_mode: str,
    allow_linear_fallback: bool,
    fps_override: int,
) -> dict:
    traj_path = episode_dir / "trajectory.npz"
    meta_path = episode_dir / "metadata.json"
    video_dir = episode_dir / "videos"

    with meta_path.open("r", encoding="utf-8") as f:
        ep_meta = json.load(f)

    control_hz = float(ep_meta.get("control_hz", ep_meta.get("capture_hz", 20.0)))
    frame_stride = int(ep_meta.get("frame_stride", 1))
    if fps_override > 0:
        dataset_fps = int(fps_override)
    else:
        dataset_fps = max(1, int(round(control_hz / max(1, frame_stride))))

    traj = np.load(traj_path, allow_pickle=True)
    action = np.asarray(traj["action"], dtype=np.float32)  # (T, A)
    state, used_state_source, state_names = load_state(traj, state_source=state_source)

    num_steps = action.shape[0]
    if state.shape[0] != num_steps:
        raise ValueError(f"State/action length mismatch: state={state.shape[0]} action={num_steps}")

    cam_primary_path = video_dir / f"{camera_primary}.mp4"
    cam_secondary_path = video_dir / f"{camera_secondary}.mp4"
    cam_aux_path = video_dir / f"{camera_aux}.mp4"

    if not cam_primary_path.exists():
        raise FileNotFoundError(f"Missing primary camera video: {cam_primary_path}")
    if not cam_secondary_path.exists():
        raise FileNotFoundError(f"Missing secondary camera video: {cam_secondary_path}")

    frames_primary = read_frames(cam_primary_path, image_size)
    frames_secondary = read_frames(cam_secondary_path, image_size)

    frames_aux = None
    if include_aux_camera:
        if not cam_aux_path.exists():
            raise FileNotFoundError(f"Missing aux camera video: {cam_aux_path}")
        frames_aux = read_frames(cam_aux_path, image_size)

    min_frames = min(len(frames_primary), len(frames_secondary))
    if frames_aux is not None:
        min_frames = min(min_frames, len(frames_aux))
    if min_frames <= 0:
        raise RuntimeError("No decoded frames found in camera videos.")

    actual_alignment_mode = choose_alignment_mode(ep_meta, traj, requested=alignment_mode)
    if actual_alignment_mode == "one_to_one":
        step_indices, frame_offset, alignment_used = align_indices_one_to_one(
            num_steps=num_steps,
            frame_count=min_frames,
            allow_linear_fallback=allow_linear_fallback,
        )
    elif actual_alignment_mode == "legacy":
        step_indices, frame_offset, alignment_used = align_indices_legacy(
            num_steps=num_steps,
            frame_count=min_frames,
            frame_stride=frame_stride,
            allow_linear_fallback=allow_linear_fallback,
        )
    else:
        raise ValueError(f"Unknown alignment mode after auto resolution: {actual_alignment_mode}")

    usable = min(len(step_indices), max(0, min_frames - frame_offset))
    if usable <= 0:
        raise RuntimeError("No aligned samples after frame/action alignment.")

    step_indices = step_indices[:usable]
    frames_primary = frames_primary[frame_offset : frame_offset + usable]
    frames_secondary = frames_secondary[frame_offset : frame_offset + usable]
    if frames_aux is not None:
        frames_aux = frames_aux[frame_offset : frame_offset + usable]

    phase_first = None
    phase_last = None
    phase_count = None
    phase_path = episode_dir / "phases.json"
    if phase_path.exists():
        try:
            phases = json.loads(phase_path.read_text(encoding="utf-8"))
            phase_count = len(phases)
            if len(phases) == num_steps and len(step_indices) > 0:
                phase_first = phases[int(step_indices[0])]
                phase_last = phases[int(step_indices[-1])]
        except Exception:
            phase_first = None
            phase_last = None
            phase_count = None

    source_step_index = None
    if "source_step_index" in traj.files:
        source_step_index = np.asarray(traj["source_step_index"])

    task = infer_task_text(ep_meta, default_task=default_task)
    h = int(frames_primary[0].shape[0])
    w = int(frames_primary[0].shape[1])

    return {
        "episode_dir": episode_dir,
        "dataset_fps": int(dataset_fps),
        "action": action,
        "state": state,
        "state_names": state_names,
        "used_state_source": used_state_source,
        "step_indices": step_indices,
        "source_step_index": source_step_index,
        "frames_primary": frames_primary,
        "frames_secondary": frames_secondary,
        "frames_aux": frames_aux,
        "usable_samples": int(usable),
        "frame_offset": int(frame_offset),
        "alignment_used": alignment_used,
        "actual_alignment_mode": actual_alignment_mode,
        "task": task,
        "height": h,
        "width": w,
        "phase_count": phase_count,
        "phase_first": phase_first,
        "phase_last": phase_last,
    }


def build_features(
    height: int,
    width: int,
    include_aux_camera: bool,
    feature_layout: str,
    state_dim: int,
    action_dim: int,
    state_names: list[str],
) -> dict:
    keys = get_feature_keys(feature_layout, include_aux_camera=include_aux_camera)
    action_names = [f"a{i}" for i in range(action_dim)]

    features = {
        keys["primary"]: {
            "dtype": "video",
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        },
        keys["secondary"]: {
            "dtype": "video",
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": action_names,
        },
    }

    if keys["aux"] is not None:
        features[keys["aux"]] = {
            "dtype": "video",
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        }

    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path("/root/autodl-tmp/openpi_earbud_proto/libero_lerobot_dataset"),
        help="Raw/cropped episode tree root (contains easy/medium/hard).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output path of ONE merged LeRobot dataset root.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="local/earbud_insert_lora",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["easy", "medium", "hard"],
        help="Category folders under src-root to scan.",
    )
    parser.add_argument("--camera-primary", type=str, default="agentview")
    parser.add_argument("--camera-secondary", type=str, default="robot0_eye_in_hand")
    parser.add_argument("--camera-aux", type=str, default="frontview")
    parser.add_argument("--include-aux-camera", action="store_true")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--default-task", type=str, default="Insert the earbud into the charging slot.")
    parser.add_argument("--limit-episodes", type=int, default=0, help="Debug option. 0 means all episodes.")
    parser.add_argument("--skip-invalid", action="store_true", help="Skip broken episodes instead of failing immediately.")
    parser.add_argument("--vcodec", type=str, default="h264")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 4))),
        help="Worker threads for parallel episode decoding/preprocessing.",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=2,
        help="Max in-flight episodes to preload when workers > 1. Higher uses more RAM.",
    )
    parser.add_argument("--force-overwrite", action="store_true")

    parser.add_argument(
        "--state-source",
        choices=["auto", "observation_state", "eef8"],
        default="auto",
        help=(
            "auto uses trajectory.npz:observation_state if it exists; otherwise constructs an 8D state from "
            "eef pos + quat + gripper. Use eef8 to keep the old converter behavior."
        ),
    )
    parser.add_argument(
        "--alignment-mode",
        choices=["auto", "one_to_one", "legacy"],
        default="auto",
        help=(
            "auto uses one_to_one for cropped/keyframe episodes and legacy for raw episodes. "
            "one_to_one is recommended after crop_episode_for_lora.py."
        ),
    )
    parser.add_argument(
        "--allow-linear-fallback",
        action="store_true",
        help="Allow approximate/truncated fallback when frame and action lengths mismatch. Not recommended for training.",
    )
    parser.add_argument(
        "--fps-override",
        type=int,
        default=0,
        help="Override dataset fps. 0 means use control_hz / frame_stride from metadata.",
    )
    parser.add_argument(
        "--feature-layout",
        choices=["generic", "pi05"],
        default="generic",
        help=(
            "generic keeps observation.images.image/image2 like the old script. "
            "pi05 emits observation.images.base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb."
        ),
    )

    args = parser.parse_args()

    src_root = args.src_root.resolve()
    output_root = args.output_root.resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"Source root not found: {src_root}")

    episodes = collect_episode_dirs(src_root, categories=args.categories)
    if not episodes:
        raise RuntimeError(f"No episode directories found under {src_root} for categories {args.categories}")
    if args.limit_episodes > 0:
        episodes = episodes[: args.limit_episodes]

    if output_root.exists():
        if args.force_overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(f"{output_root} already exists. Use --force-overwrite to recreate it.")

    output_root.parent.mkdir(parents=True, exist_ok=True)

    dataset = None
    expected = None
    conversion_items = []
    failures = []
    converted = 0
    total_samples = 0
    feature_keys = get_feature_keys(args.feature_layout, include_aux_camera=args.include_aux_camera)

    try:
        print(
            f"[convert] workers={max(1, args.workers)} "
            f"prefetch={max(1, args.prefetch)} episodes={len(episodes)} "
            f"state_source={args.state_source} alignment_mode={args.alignment_mode} "
            f"feature_layout={args.feature_layout}"
        )
        payload_iter = iter_episode_payloads(
            episodes=episodes,
            workers=max(1, args.workers),
            prefetch=max(1, args.prefetch),
            camera_primary=args.camera_primary,
            camera_secondary=args.camera_secondary,
            camera_aux=args.camera_aux,
            include_aux_camera=args.include_aux_camera,
            image_size=args.image_size,
            default_task=args.default_task,
            state_source=args.state_source,
            alignment_mode=args.alignment_mode,
            allow_linear_fallback=args.allow_linear_fallback,
            fps_override=args.fps_override,
        )

        for idx, total, episode_dir, payload, load_error in payload_iter:
            category = episode_dir.parent.name
            short_name = f"{category}/{episode_dir.name}"
            print(f"[{idx:04d}/{total:04d}] converting {short_name}")

            if load_error is not None:
                err = f"{type(load_error).__name__}: {load_error}"
                if args.skip_invalid:
                    print(f"  [skip] {short_name} -> {err}")
                    failures.append({"episode_dir": str(episode_dir), "error": err})
                    continue
                raise load_error
            assert payload is not None

            if dataset is None:
                features = build_features(
                    height=payload["height"],
                    width=payload["width"],
                    include_aux_camera=args.include_aux_camera,
                    feature_layout=args.feature_layout,
                    state_dim=int(payload["state"].shape[1]),
                    action_dim=int(payload["action"].shape[1]),
                    state_names=payload["state_names"],
                )
                dataset = LeRobotDataset.create(
                    repo_id=args.repo_id,
                    root=output_root,
                    fps=payload["dataset_fps"],
                    features=features,
                    robot_type="libero",
                    use_videos=True,
                    vcodec=args.vcodec,
                )
                expected = {
                    "fps": int(payload["dataset_fps"]),
                    "height": int(payload["height"]),
                    "width": int(payload["width"]),
                    "state_dim": int(payload["state"].shape[1]),
                    "action_dim": int(payload["action"].shape[1]),
                    "state_names": payload["state_names"],
                    "used_state_source": payload["used_state_source"],
                }
                print(f"  [dataset] fps={expected['fps']} image={expected['height']}x{expected['width']} "
                      f"state_dim={expected['state_dim']} action_dim={expected['action_dim']} "
                      f"state_source={expected['used_state_source']}")
                print(f"  [features] primary={feature_keys['primary']} secondary={feature_keys['secondary']} aux={feature_keys['aux']}")
            else:
                assert expected is not None
                if int(payload["dataset_fps"]) != expected["fps"]:
                    raise ValueError(f"FPS mismatch in {short_name}: {payload['dataset_fps']} vs {expected['fps']}")
                if int(payload["height"]) != expected["height"] or int(payload["width"]) != expected["width"]:
                    raise ValueError(
                        f"Image size mismatch in {short_name}: {payload['height']}x{payload['width']} vs "
                        f"{expected['height']}x{expected['width']}"
                    )
                if int(payload["state"].shape[1]) != expected["state_dim"]:
                    raise ValueError(f"State dim mismatch in {short_name}: {payload['state'].shape[1]} vs {expected['state_dim']}")
                if int(payload["action"].shape[1]) != expected["action_dim"]:
                    raise ValueError(f"Action dim mismatch in {short_name}: {payload['action'].shape[1]} vs {expected['action_dim']}")

            assert dataset is not None
            step_indices = payload["step_indices"]

            for sample_idx, step_idx in enumerate(step_indices):
                frame = {
                    feature_keys["primary"]: payload["frames_primary"][sample_idx],
                    feature_keys["secondary"]: payload["frames_secondary"][sample_idx],
                    "observation.state": payload["state"][step_idx],
                    "action": payload["action"][step_idx],
                    "task": payload["task"],
                }
                if args.include_aux_camera and feature_keys["aux"] is not None:
                    frame[feature_keys["aux"]] = payload["frames_aux"][sample_idx]
                dataset.add_frame(frame)

            dataset.save_episode(parallel_encoding=True)
            converted += 1
            total_samples += int(payload["usable_samples"])

            src_first = None
            src_last = None
            if payload["source_step_index"] is not None and len(payload["source_step_index"]) > 0:
                src_first = int(payload["source_step_index"][int(step_indices[0])])
                src_last = int(payload["source_step_index"][int(step_indices[-1])])

            conversion_items.append(
                {
                    "index": idx,
                    "category": category,
                    "episode_dir": str(episode_dir),
                    "usable_samples": int(payload["usable_samples"]),
                    "frame_offset": int(payload["frame_offset"]),
                    "alignment_used": payload["alignment_used"],
                    "actual_alignment_mode": payload["actual_alignment_mode"],
                    "used_state_source": payload["used_state_source"],
                    "state_dim": int(payload["state"].shape[1]),
                    "action_dim": int(payload["action"].shape[1]),
                    "source_step_first": src_first,
                    "source_step_last": src_last,
                    "phase_count": payload["phase_count"],
                    "phase_first": payload["phase_first"],
                    "phase_last": payload["phase_last"],
                    "task": payload["task"],
                }
            )
            print(
                f"  [ok] samples={payload['usable_samples']} align={payload['alignment_used']} "
                f"state={payload['used_state_source']} phase={payload['phase_first']} -> {payload['phase_last']}"
            )

    except Exception:
        print("\nConversion failed. Stack trace:")
        print(traceback.format_exc())
        raise
    finally:
        if dataset is not None:
            dataset.finalize()

    if converted == 0:
        raise RuntimeError("No episodes were converted.")

    check_ds = LeRobotDataset(repo_id=args.repo_id, root=output_root)
    sample = check_ds[0]
    required = [feature_keys["primary"], feature_keys["secondary"], "observation.state", "action"]
    missing = [k for k in required if k not in sample]
    if missing:
        raise RuntimeError(f"Converted dataset missing required keys: {missing}")

    report = {
        "src_root": str(src_root),
        "output_root": str(output_root),
        "repo_id": args.repo_id,
        "categories": args.categories,
        "episodes_discovered": len(episodes),
        "episodes_converted": converted,
        "episodes_failed": len(failures),
        "total_samples": total_samples,
        "camera_primary": args.camera_primary,
        "camera_secondary": args.camera_secondary,
        "camera_aux": args.camera_aux,
        "include_aux_camera": bool(args.include_aux_camera),
        "image_size": int(args.image_size),
        "vcodec": args.vcodec,
        "state_source_arg": args.state_source,
        "alignment_mode_arg": args.alignment_mode,
        "allow_linear_fallback": bool(args.allow_linear_fallback),
        "fps_override": int(args.fps_override),
        "feature_layout": args.feature_layout,
        "feature_keys": feature_keys,
        "items": conversion_items,
        "failures": failures,
    }
    report_path = output_root / "conversion_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nConversion done.")
    print(f"episodes_converted: {converted}")
    print(f"total_samples: {total_samples}")
    print(f"output_root: {output_root}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
