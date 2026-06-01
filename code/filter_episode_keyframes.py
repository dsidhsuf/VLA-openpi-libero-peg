import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


CRITICAL_PHASES = {
    "preclose_gripper",
    "descend_to_cage",
    "squeeze",
    "lift",
    "object_axis_align",
    "slot_hover",
    "slot_fine_rotate",
    "descend_pre_insert",
    "descend_final_insert",
    "open_gripper",
}

SPARSE_PHASES = {
    "rise_safe",
    "move_above_object",
    "align_wrist_yaw",
    "pregrasp",
    "preclose_height",
    "move_safe_above_slot",
    "retreat",
}


def phase_stride(phase: str, mode: str) -> int:
    if phase in CRITICAL_PHASES:
        return 1
    if mode == "keyframe_critical":
        if phase in SPARSE_PHASES:
            return 5
        if "settle" in phase or "hold" in phase or "wait" in phase or phase in {"start", "pre_release_hold"}:
            return 12
        return 4
    if phase in SPARSE_PHASES:
        return 3
    if "settle" in phase or "hold" in phase or "wait" in phase or phase in {"start", "pre_release_hold"}:
        return 8
    return 2


def select_keyframe_indices(actions: np.ndarray, phases: list[str], mode: str) -> np.ndarray:
    mode = str(mode or "keyframe").lower()
    if mode not in ("keyframe", "keyframe_critical"):
        raise ValueError(f"Unsupported mode={mode!r}; use keyframe or keyframe_critical.")
    selected = []
    phase_counts = defaultdict(int)
    for idx, (action, phase) in enumerate(zip(actions, phases)):
        phase_counts[phase] += 1
        phase_i = phase_counts[phase]
        action = np.asarray(action, dtype=np.float32)

        keep = False
        if idx == 0 or idx == len(actions) - 1:
            keep = True
        elif phase_i <= 2:
            keep = True
        elif phase in CRITICAL_PHASES:
            keep = True
        elif mode == "keyframe" and (np.linalg.norm(action[:3]) >= 0.06 or abs(float(action[5])) >= 0.06):
            keep = True
        elif (
            mode == "keyframe_critical"
            and phase not in SPARSE_PHASES
            and not ("settle" in phase or "hold" in phase or "wait" in phase)
            and (np.linalg.norm(action[:3]) >= 0.078 or abs(float(action[5])) >= 0.078)
        ):
            keep = True
        elif phase_i % phase_stride(phase, mode) == 0:
            keep = True

        if keep:
            selected.append(idx)
    return np.asarray(sorted(set(selected)), dtype=np.int64)


def read_video(video_path: Path) -> list[np.ndarray]:
    return [np.asarray(frame, dtype=np.uint8) for frame in imageio.get_reader(str(video_path))]


def infer_frame_offset(num_steps: int, frame_count: int, frame_stride: int) -> int:
    stride = max(1, int(frame_stride))
    stride_indices = np.arange(stride - 1, num_steps, stride, dtype=np.int64)
    if frame_count == len(stride_indices) + 1:
        return 1
    if frame_count == len(stride_indices):
        return 0
    # Common generated format: initial frame + one frame per action.
    if frame_count == num_steps + 1:
        return 1
    return 0


def filter_episode(src_episode: Path, dst_episode: Path, fps=None, force: bool = False, mode: str = "keyframe") -> None:
    src_episode = src_episode.resolve()
    dst_episode = dst_episode.resolve()
    if dst_episode.exists():
        if not force:
            raise FileExistsError(f"Output already exists: {dst_episode}")
        shutil.rmtree(dst_episode)
    dst_episode.mkdir(parents=True, exist_ok=True)
    (dst_episode / "videos").mkdir(parents=True, exist_ok=True)

    traj = np.load(src_episode / "trajectory.npz")
    metadata = json.loads((src_episode / "metadata.json").read_text(encoding="utf-8"))
    phases = json.loads((src_episode / "phases.json").read_text(encoding="utf-8"))
    actions = np.asarray(traj["action"], dtype=np.float32)

    if len(phases) != actions.shape[0]:
        raise RuntimeError(f"phase/action length mismatch: phases={len(phases)} actions={actions.shape[0]}")

    selected = select_keyframe_indices(actions, phases, mode=mode)
    if selected.size <= 0:
        raise RuntimeError("No keyframes selected.")

    arrays = {}
    for key in traj.files:
        arr = traj[key]
        if arr.shape[0] == actions.shape[0]:
            arrays[key] = arr[selected]
        else:
            arrays[key] = arr
    np.savez_compressed(dst_episode / "trajectory.npz", **arrays)

    selected_phases = [phases[int(i)] for i in selected]
    (dst_episode / "phases.json").write_text(json.dumps(selected_phases, ensure_ascii=False), encoding="utf-8")

    step_rows = []
    src_steps = src_episode / "steps.jsonl"
    if src_steps.exists():
        rows = [json.loads(line) for line in src_steps.read_text(encoding="utf-8").splitlines() if line.strip()]
        step_rows = [rows[int(i)] for i in selected]
    else:
        step_rows = [
            {"step": int(arrays["step"][j]), "timestamp_s": float(arrays["timestamp_s"][j]), "phase": selected_phases[j]}
            for j in range(len(selected))
        ]
    with (dst_episode / "steps.jsonl").open("w", encoding="utf-8") as f:
        for row in step_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    frame_stride = int(metadata.get("frame_stride", 1))
    video_fps = int(fps or metadata.get("video_fps_resolved") or metadata.get("video_fps_base") or 20)
    video_paths = {}
    for video_path in sorted((src_episode / "videos").glob("*.mp4")):
        frames = read_video(video_path)
        offset = infer_frame_offset(actions.shape[0], len(frames), frame_stride)
        keep_frame_indices = [0] + [int(offset + i) for i in selected if int(offset + i) < len(frames)]
        keep_frame_indices = sorted(set(keep_frame_indices))
        filtered_frames = [frames[i] for i in keep_frame_indices]
        out_video = dst_episode / "videos" / video_path.name
        imageio.mimwrite(str(out_video), filtered_frames, fps=video_fps)
        video_paths[video_path.stem] = str(out_video)

        if video_path.stem == "agentview":
            imageio.imwrite(str(dst_episode / "init_agentview.png"), filtered_frames[0])
            imageio.mimwrite(str(dst_episode / "preview.mp4"), filtered_frames, fps=video_fps)

    metadata.update(
        {
            "source_episode_dir": str(src_episode),
            "record_mode": f"post_{mode}",
            "original_num_steps": int(actions.shape[0]),
            "num_steps": int(selected.size),
            "keyframe_keep_ratio": float(selected.size / max(1, actions.shape[0])),
            "frame_stride": 1,
            "requested_frame_stride": frame_stride,
            "max_recorded_frames": int(selected.size + 1),
            "video_fps_resolved": video_fps,
        }
    )
    metadata.setdefault("paths", {})
    metadata["paths"].update(
        {
            "trajectory_npz": str(dst_episode / "trajectory.npz"),
            "phases_json": str(dst_episode / "phases.json"),
            "steps_jsonl": str(dst_episode / "steps.jsonl"),
            "videos": video_paths,
            "preview_video": str(dst_episode / "preview.mp4"),
            "metadata_json": str(dst_episode / "metadata.json"),
        }
    )
    (dst_episode / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"source: {src_episode}")
    print(f"output: {dst_episode}")
    print(f"steps: {actions.shape[0]} -> {selected.size} ({selected.size / max(1, actions.shape[0]):.2%})")
    print(f"preview: {dst_episode / 'preview.mp4'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-episode", required=True)
    parser.add_argument("--dst-episode", required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--mode", choices=["keyframe", "keyframe_critical"], default="keyframe")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    filter_episode(Path(args.src_episode), Path(args.dst_episode), fps=args.fps, force=args.force, mode=args.mode)


if __name__ == "__main__":
    main()
