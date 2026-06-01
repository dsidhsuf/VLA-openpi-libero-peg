#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把一个 dense episode 裁剪成 LoRA 微调只学习的片段，例如只保留后半段。
会同步裁剪：trajectory.npz / phases.json / steps.jsonl / videos/*.mp4
避免 image、state、action、phase 标签错位。

用法示例：
python crop_episode_for_lora.py \
  --src_episode /path/to/episode_seed0_xxx \
  --dst_episode /path/to/episode_seed0_xxx_last50 \
  --start_ratio 0.5

也可以从某个 phase 开始：
python crop_episode_for_lora.py \
  --src_episode /path/to/episode_seed0_xxx \
  --dst_episode /path/to/episode_seed0_xxx_from_squeeze \
  --start_phase squeeze
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import cv2
except Exception as e:  # pragma: no cover
    cv2 = None


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def find_phase_start(phases: List[str], phase_name: str) -> int:
    for i, p in enumerate(phases):
        if p == phase_name:
            return i
    available = []
    for p in phases:
        if p not in available:
            available.append(p)
    raise ValueError(
        f"phase {phase_name!r} not found. Available phases: {available}"
    )


def decide_start_index(n: int, phases: Optional[List[str]], args: argparse.Namespace) -> int:
    modes = [
        args.start_index is not None,
        args.start_ratio is not None,
        args.start_phase is not None,
    ]
    if sum(modes) != 1:
        raise ValueError("必须且只能指定一个：--start_index / --start_ratio / --start_phase")

    if args.start_index is not None:
        start = int(args.start_index)
    elif args.start_ratio is not None:
        if not (0.0 <= args.start_ratio < 1.0):
            raise ValueError("--start_ratio 必须在 [0, 1) 内，例如 0.5 表示只保留后半段")
        start = int(np.floor(n * float(args.start_ratio)))
    else:
        if phases is None:
            raise ValueError("使用 --start_phase 需要 src_episode/phases.json 存在")
        start = find_phase_start(phases, args.start_phase)

    start = max(0, min(start, n - 1))
    return start


def slice_npz(src_npz: Path, dst_npz: Path, selected: np.ndarray) -> None:
    src = np.load(src_npz, allow_pickle=True)
    n_src = int(src["action"].shape[0])
    out: Dict[str, Any] = {}

    for k in src.files:
        arr = src[k]
        if hasattr(arr, "shape") and len(arr.shape) > 0 and arr.shape[0] == n_src:
            out[k] = arr[selected]
        else:
            out[k] = arr

    # 保留原始 step，便于调试标签是否错位
    out["source_step_index"] = selected.astype(np.int32)

    # 重新编号，LeRobot 里每个裁剪后的 episode 从 0 开始更干净
    if "step" in out:
        out["step"] = np.arange(len(selected), dtype=np.int32)

    # 时间戳也从 0 开始，避免裁剪后 episode 的 timestamp 从几十秒开始
    if "timestamp_s" in out:
        ts = out["timestamp_s"].astype(np.float32)
        if len(ts) > 0:
            ts = ts - ts[0]
        out["timestamp_s"] = ts

    # 裁剪后的最后一帧作为 episode 结束
    if "done" in out:
        done = np.zeros(len(selected), dtype=bool)
        done[-1] = True
        out["done"] = done

    dst_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst_npz, **out)


def get_video_frame_count(video_path: Path) -> int:
    if cv2 is None:
        raise RuntimeError("需要 opencv-python 或 opencv-python-headless 才能裁剪视频")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def resolve_video_offset(frame_count: int, n_traj: int, requested: str) -> int:
    if requested != "auto":
        return int(requested)

    if frame_count == n_traj:
        return 0
    if frame_count == n_traj + 1:
        # 常见情况：视频多一个初始帧，trajectory 第 i 行对应视频第 i+1 帧
        return 1

    # 保守默认：不偏移，但后面会做越界检查
    return 0


def slice_video(src_video: Path, dst_video: Path, selected: np.ndarray, n_traj: int, video_offset: str) -> None:
    if cv2 is None:
        raise RuntimeError("需要 opencv-python 或 opencv-python-headless 才能裁剪视频")

    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src_video}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    offset = resolve_video_offset(frame_count, n_traj, video_offset)
    wanted = (selected + offset).astype(np.int64)

    if len(wanted) == 0:
        raise RuntimeError("selected 为空，无法裁剪视频")
    if wanted.max() >= frame_count or wanted.min() < 0:
        raise RuntimeError(
            f"视频索引越界: min={wanted.min()}, max={wanted.max()}, "
            f"frame_count={frame_count}, offset={offset}. "
            f"可以尝试 --video_offset 0 或 --video_offset 1"
        )

    dst_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst_video), fourcc, fps, (width, height))

    wanted_set = set(int(x) for x in wanted.tolist())
    idx = 0
    written = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted_set:
            writer.write(frame)
            written += 1
        idx += 1

    cap.release()
    writer.release()

    if written != len(selected):
        raise RuntimeError(
            f"视频裁剪帧数不一致: written={written}, expected={len(selected)}, "
            f"src={src_video}, offset={offset}"
        )


def maybe_copy_file(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_episode", required=True, help="原始 episode 目录，里面有 trajectory.npz")
    parser.add_argument("--dst_episode", required=True, help="裁剪后的 episode 输出目录")

    group = parser.add_argument_group("裁剪起点，三选一")
    group.add_argument("--start_ratio", type=float, default=None, help="例如 0.5 表示只保留后半段")
    group.add_argument("--start_index", type=int, default=None, help="从指定 step index 开始保留")
    group.add_argument("--start_phase", type=str, default=None, help="从指定 phase 第一次出现处开始保留，例如 squeeze")

    parser.add_argument(
        "--video_offset",
        default="auto",
        help="auto/0/1。若视频比 trajectory 多 1 帧，auto 会使用 offset=1",
    )
    parser.add_argument("--no_video", action="store_true", help="只裁剪 npz/json，不裁剪视频")
    args = parser.parse_args()

    src = Path(args.src_episode)
    dst = Path(args.dst_episode)
    traj_path = src / "trajectory.npz"
    phase_path = src / "phases.json"
    steps_path = src / "steps.jsonl"
    meta_path = src / "metadata.json"

    if not traj_path.exists():
        raise FileNotFoundError(traj_path)

    traj = np.load(traj_path, allow_pickle=True)
    n = int(traj["action"].shape[0])

    phases = read_json(phase_path) if phase_path.exists() else None
    if phases is not None and len(phases) != n:
        raise RuntimeError(f"phases length mismatch: {len(phases)} vs trajectory {n}")

    start = decide_start_index(n, phases, args)
    selected = np.arange(start, n, dtype=np.int64)

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    slice_npz(traj_path, dst / "trajectory.npz", selected)

    if phases is not None:
        write_json(dst / "phases.json", [phases[int(i)] for i in selected])

    if steps_path.exists():
        rows = read_jsonl(steps_path)
        if len(rows) != n:
            raise RuntimeError(f"steps length mismatch: {len(rows)} vs trajectory {n}")
        out_rows = []
        t0 = float(traj["timestamp_s"][start]) if "timestamp_s" in traj.files else 0.0
        for new_i, old_i in enumerate(selected.tolist()):
            r = dict(rows[old_i])
            r["source_step_index"] = int(old_i)
            r["step"] = int(new_i)
            r["frame_index"] = int(new_i)
            if "timestamp_s" in r:
                r["timestamp_s"] = float(r["timestamp_s"] - t0)
            out_rows.append(r)
        write_jsonl(dst / "steps.jsonl", out_rows)

    meta = read_json(meta_path) if meta_path.exists() else {}
    meta.update({
        "record_mode": "cropped_for_lora",
        "crop_mode": "last_segment",
        "source_num_steps": int(n),
        "num_steps": int(len(selected)),
        "crop_start_index": int(start),
        "crop_end_index": int(n - 1),
        "crop_keep_ratio": float(len(selected) / n),
        "video_offset": args.video_offset,
        "note": "trajectory/action/state/phase/video are synchronously cropped for LoRA fine-tuning.",
    })
    write_json(dst / "metadata.json", meta)

    # 复制初始图，仅用于查看，不参与训练
    maybe_copy_file(src / "init_agentview.png", dst / "init_agentview.png")

    if not args.no_video:
        src_video_dir = src / "videos"
        if src_video_dir.exists():
            for video in sorted(src_video_dir.glob("*.mp4")):
                print(f"[INFO] slicing video: {video.name}")
                slice_video(video, dst / "videos" / video.name, selected, n, args.video_offset)
        else:
            # 有些 episode 把视频直接放在 episode 根目录
            for video in sorted(src.glob("*.mp4")):
                print(f"[INFO] slicing video: {video.name}")
                slice_video(video, dst / "videos" / video.name, selected, n, args.video_offset)

    print("[DONE] cropped episode saved to:", dst)
    print(f"[INFO] source steps: {n}")
    print(f"[INFO] kept steps  : {len(selected)}")
    print(f"[INFO] start index : {start}")
    if phases is not None:
        print(f"[INFO] start phase : {phases[start]}")


if __name__ == "__main__":
    main()
