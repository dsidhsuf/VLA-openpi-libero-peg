#!/usr/bin/env python3
"""
Convert a directory tree of raw LIBERO episodes into ONE LeRobot dataset root.

Input layout example:
  <src-root>/
    easy/episode_seed.../
    medium/episode_seed.../
    hard/episode_seed.../

Each episode directory is expected to contain:
  - trajectory.npz
  - metadata.json
  - videos/<camera>.mp4

Output:
  One LeRobot dataset root that contains meta/info.json and data/,
  directly usable by one-shot global-step lerobot-train.
"""

from __future__ import annotations

import argparse
import os
import json
import shutil
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset


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


def align_indices(num_steps: int, frame_count: int, frame_stride: int) -> tuple[np.ndarray, int]:
    stride = max(1, int(frame_stride))
    stride_indices = np.arange(stride - 1, num_steps, stride, dtype=np.int64)

    # Recorder may store an initial frame before first action.
    if frame_count == len(stride_indices) + 1:
        return stride_indices, 1
    if frame_count == len(stride_indices):
        return stride_indices, 0

    # Fallback for length mismatch.
    linear = np.linspace(0, num_steps - 1, num=frame_count, dtype=np.int64)
    return linear, 0


def collect_episode_dirs(
    src_root: Path,
    categories: list[str],
    category_limits: dict[str, int | None] | None = None,
) -> list[Path]:
    episodes: list[Path] = []
    for cat in categories:
        cat_dir = src_root / cat
        if not cat_dir.exists() or not cat_dir.is_dir():
            continue
        cat_episodes: list[Path] = []
        for child in sorted(cat_dir.iterdir()):
            if child.is_dir() and is_raw_episode_dir(child):
                cat_episodes.append(child.resolve())
        limit = category_limits.get(cat) if category_limits else None
        if limit is not None:
            cat_episodes = cat_episodes[:limit]
        episodes.extend(cat_episodes)
    return episodes


def build_category_limits(args) -> dict[str, int | None]:
    default_limit = int(args.limit_per_category)
    specific_limits = {
        "easy": args.limit_easy,
        "medium": args.limit_medium,
        "hard": args.limit_hard,
    }
    limits: dict[str, int | None] = {}
    for cat in args.categories:
        specific = specific_limits.get(cat)
        if specific is not None:
            if specific < 0:
                raise ValueError(f"--limit-{cat} must be >= 0, got {specific}")
            limits[cat] = int(specific)
        elif default_limit > 0:
            limits[cat] = default_limit
        else:
            limits[cat] = None
    return limits


def infer_task_text(meta: dict, default_task: str) -> str:
    for key in ("task", "task_description", "instruction", "language_instruction", "prompt"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default_task


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
) -> dict:
    traj_path = episode_dir / "trajectory.npz"
    meta_path = episode_dir / "metadata.json"
    video_dir = episode_dir / "videos"

    with meta_path.open("r", encoding="utf-8") as f:
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

    step_indices, frame_offset = align_indices(
        num_steps=num_steps,
        frame_count=min_frames,
        frame_stride=frame_stride,
    )
    usable = min(len(step_indices), max(0, min_frames - frame_offset))
    if usable <= 0:
        raise RuntimeError("No aligned samples after frame/action alignment.")

    step_indices = step_indices[:usable]
    frames_primary = frames_primary[frame_offset : frame_offset + usable]
    frames_secondary = frames_secondary[frame_offset : frame_offset + usable]
    if frames_aux is not None:
        frames_aux = frames_aux[frame_offset : frame_offset + usable]

    task = infer_task_text(ep_meta, default_task=default_task)
    h = int(frames_primary[0].shape[0])
    w = int(frames_primary[0].shape[1])

    return {
        "episode_dir": episode_dir,
        "dataset_fps": dataset_fps,
        "action": action,
        "state": state,
        "step_indices": step_indices,
        "frames_primary": frames_primary,
        "frames_secondary": frames_secondary,
        "frames_aux": frames_aux,
        "usable_samples": int(usable),
        "frame_offset": int(frame_offset),
        "task": task,
        "height": h,
        "width": w,
    }


def build_features(height: int, width: int, include_aux_camera: bool) -> dict:
    state_names = ["eef_x", "eef_y", "eef_z", "quat_w", "quat_x", "quat_y", "quat_z", "gripper"]
    action_names = [f"a{i}" for i in range(7)]
    features = {
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
    if include_aux_camera:
        features["observation.images.frontview_aux"] = {
            "dtype": "video",
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        }
    return features


def load_existing_report(report_path: Path) -> dict:
    if not report_path.exists():
        return {}
    with report_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return data


def write_report(
    report_path: Path,
    *,
    args,
    src_root: Path,
    output_root: Path,
    episodes_discovered: int,
    converted: int,
    failures: list[dict],
    total_samples: int,
    conversion_items: list[dict],
):
    report = {
        "src_root": str(src_root),
        "output_root": str(output_root),
        "repo_id": args.repo_id,
        "categories": args.categories,
        "episodes_discovered": episodes_discovered,
        "episodes_converted": converted,
        "episodes_failed": len(failures),
        "total_samples": total_samples,
        "camera_primary": args.camera_primary,
        "camera_secondary": args.camera_secondary,
        "camera_aux": args.camera_aux,
        "include_aux_camera": bool(args.include_aux_camera),
        "image_size": int(args.image_size),
        "vcodec": args.vcodec,
        "resume": bool(args.resume),
        "workers": int(args.workers),
        "prefetch": int(args.prefetch),
        "items": conversion_items,
        "failures": failures,
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path("/root/autodl-tmp/openpi_earbud_proto/libero_lerobot_dataset"),
        help="Raw episode tree root (contains easy/medium/hard).",
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
        default="local/earbud_insert_batch_v3_global",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["easy", "medium", "hard"],
        help="Category folders under src-root to scan.",
    )
    parser.add_argument(
        "--camera-primary",
        type=str,
        default="agentview",
    )
    parser.add_argument(
        "--camera-secondary",
        type=str,
        default="robot0_eye_in_hand",
    )
    parser.add_argument(
        "--camera-aux",
        type=str,
        default="frontview",
    )
    parser.add_argument("--include-aux-camera", action="store_true")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--default-task",
        type=str,
        default="Insert the earbud into the charging slot.",
    )
    parser.add_argument(
        "--limit-episodes",
        type=int,
        default=0,
        help="Debug option. 0 means all episodes.",
    )
    parser.add_argument(
        "--limit-per-category",
        type=int,
        default=0,
        help="Take at most this many sorted episodes from each category. 0 means all episodes.",
    )
    parser.add_argument(
        "--limit-easy",
        type=int,
        default=None,
        help="Take this many sorted easy episodes. Overrides --limit-per-category for easy.",
    )
    parser.add_argument(
        "--limit-medium",
        type=int,
        default=None,
        help="Take this many sorted medium episodes. Overrides --limit-per-category for medium.",
    )
    parser.add_argument(
        "--limit-hard",
        type=int,
        default=None,
        help="Take this many sorted hard episodes. Overrides --limit-per-category for hard.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip broken episodes instead of failing immediately.",
    )
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
        help="Max in-flight episodes to preload when workers > 1 (higher uses more RAM).",
    )
    parser.add_argument("--force-overwrite", action="store_true")
    args = parser.parse_args()

    src_root = args.src_root.resolve()
    output_root = args.output_root.resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"Source root not found: {src_root}")

    category_limits = build_category_limits(args)
    episodes = collect_episode_dirs(
        src_root,
        categories=args.categories,
        category_limits=category_limits,
    )
    if not episodes:
        raise RuntimeError(
            f"No raw episode directories found under {src_root} for categories {args.categories}"
        )
    if args.limit_episodes > 0:
        episodes = episodes[: args.limit_episodes]

    if output_root.exists():
        if args.force_overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(
                f"{output_root} already exists. Use --force-overwrite to recreate it."
            )

    output_root.parent.mkdir(parents=True, exist_ok=True)

    dataset = None
    expected = None
    conversion_items = []
    failures = []
    converted = 0
    total_samples = 0

    try:
        print(
            f"[convert] workers={max(1, args.workers)} "
            f"prefetch={max(1, args.prefetch)} episodes={len(episodes)} "
            f"category_limits={category_limits}"
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
                }
            else:
                assert expected is not None
                if int(payload["dataset_fps"]) != expected["fps"]:
                    raise ValueError(
                        f"FPS mismatch in {short_name}: {payload['dataset_fps']} vs {expected['fps']}"
                    )
                if int(payload["height"]) != expected["height"] or int(payload["width"]) != expected["width"]:
                    raise ValueError(
                        f"Image size mismatch in {short_name}: "
                        f"{payload['height']}x{payload['width']} vs "
                        f"{expected['height']}x{expected['width']}"
                    )
                if int(payload["state"].shape[1]) != expected["state_dim"]:
                    raise ValueError(
                        f"State dim mismatch in {short_name}: {payload['state'].shape[1]} vs {expected['state_dim']}"
                    )
                if int(payload["action"].shape[1]) != expected["action_dim"]:
                    raise ValueError(
                        f"Action dim mismatch in {short_name}: {payload['action'].shape[1]} vs {expected['action_dim']}"
                    )

            assert dataset is not None
            step_indices = payload["step_indices"]
            for sample_idx, step_idx in enumerate(step_indices):
                frame = {
                    "observation.images.image": payload["frames_primary"][sample_idx],
                    "observation.images.image2": payload["frames_secondary"][sample_idx],
                    "observation.state": payload["state"][step_idx],
                    "action": payload["action"][step_idx],
                    "task": payload["task"],
                }
                if args.include_aux_camera:
                    frame["observation.images.frontview_aux"] = payload["frames_aux"][sample_idx]
                dataset.add_frame(frame)

            dataset.save_episode(parallel_encoding=True)
            converted += 1
            total_samples += int(payload["usable_samples"])
            conversion_items.append(
                {
                    "index": idx,
                    "category": category,
                    "episode_dir": str(episode_dir),
                    "usable_samples": int(payload["usable_samples"]),
                    "frame_offset": int(payload["frame_offset"]),
                    "task": payload["task"],
                }
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
    required = ["observation.images.image", "observation.images.image2", "observation.state", "action"]
    missing = [k for k in required if k not in sample]
    if missing:
        raise RuntimeError(f"Converted dataset missing required keys: {missing}")

    report = {
        "src_root": str(src_root),
        "output_root": str(output_root),
        "repo_id": args.repo_id,
        "categories": args.categories,
        "limit_per_category": int(args.limit_per_category),
        "limit_easy": args.limit_easy,
        "limit_medium": args.limit_medium,
        "limit_hard": args.limit_hard,
        "category_limits": category_limits,
        "limit_episodes": int(args.limit_episodes),
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
