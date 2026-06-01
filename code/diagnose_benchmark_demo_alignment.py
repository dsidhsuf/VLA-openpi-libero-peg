#!/usr/bin/env python3
"""
Diagnose whether a recorded LIBERO demo can be replayed in a benchmark task.

Use this before more training when direct action replay fails. If the benchmark
initial state is not the same as the raw demo's initial state, even a perfect
action sequence can close the gripper at the wrong place.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TaskSpec:
    name: str
    level: str
    bddl_file: str
    language: str
    max_steps: int
    state_file: Path


def fmt_vec(x: np.ndarray | list[float], n: int | None = None) -> str:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if n is not None:
        arr = arr[:n]
    return "[" + ", ".join(f"{v:+.5f}" for v in arr) + "]"


def resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_task(assets_dir: Path, level: str) -> TaskSpec:
    tasks_json = assets_dir / "tasks.json"
    with tasks_json.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    levels = raw.get("levels", {})
    if level not in levels:
        raise KeyError(f"Level {level!r} not found in {tasks_json}. Available: {sorted(levels)}")
    t = levels[level]
    return TaskSpec(
        name=str(t["name"]),
        level=level,
        bddl_file=str(t["bddl"]),
        language=str(t.get("language", "")),
        max_steps=int(t.get("max_steps", 0)),
        state_file=resolve_path(assets_dir, t["state_file"]),
    )


def load_benchmark_state(task: TaskSpec, init_index: int) -> dict[str, np.ndarray]:
    pack = np.load(task.state_file, allow_pickle=True)
    qpos = np.asarray(pack["qpos"], dtype=np.float64)
    qvel = np.asarray(pack["qvel"], dtype=np.float64)
    idx = int(np.clip(init_index, 0, len(qpos) - 1))
    return {
        "qpos": qpos[idx].copy(),
        "qvel": qvel[idx].copy(),
        "num_init_states": np.asarray([len(qpos)], dtype=np.int64),
        "init_index": np.asarray([idx], dtype=np.int64),
    }


def summarize_npz(path: Path) -> dict[str, tuple[int, ...]]:
    pack = np.load(path, allow_pickle=True)
    return {key: tuple(np.asarray(pack[key]).shape) for key in pack.files}


def first_available(pack: Any, names: list[str]) -> np.ndarray | None:
    for name in names:
        if name in pack.files:
            return np.asarray(pack[name])
    return None


def get_raw_positions(traj: Any, idx: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key in (
        "robot0_eef_pos",
        "robot0_gripper_qpos",
        "earbud_1_pos",
        "charging_slot_1_pos",
        "earbud_1_quat_wxyz",
        "charging_slot_1_quat_wxyz",
    ):
        if key in traj.files:
            arr = np.asarray(traj[key])
            if arr.ndim > 0 and arr.shape[0] > 0:
                out[key] = np.asarray(arr[int(np.clip(idx, 0, arr.shape[0] - 1))], dtype=np.float64)
    return out


def print_pose_block(title: str, data: dict[str, np.ndarray]) -> None:
    print(f"\n[{title}]")
    for key in ("robot0_eef_pos", "earbud_1_pos", "charging_slot_1_pos", "robot0_gripper_qpos"):
        if key in data:
            print(f"{key}: {fmt_vec(data[key])}")
    if "robot0_eef_pos" in data and "earbud_1_pos" in data:
        print(f"eef_obj_dist: {np.linalg.norm(data['robot0_eef_pos'] - data['earbud_1_pos']):.6f}")
        print(f"obj_minus_eef: {fmt_vec(data['earbud_1_pos'] - data['robot0_eef_pos'])}")
    if "earbud_1_pos" in data and "charging_slot_1_pos" in data:
        obj_slot = data["earbud_1_pos"] - data["charging_slot_1_pos"]
        print(f"obj_slot_xy: {np.linalg.norm(obj_slot[:2]):.6f}")
        print(f"obj_slot_z: {obj_slot[2]:+.6f}")


def print_pose_diff(a_name: str, a: dict[str, np.ndarray], b_name: str, b: dict[str, np.ndarray]) -> None:
    print(f"\n[pose diff] {a_name} - {b_name}")
    for key in ("robot0_eef_pos", "earbud_1_pos", "charging_slot_1_pos"):
        if key in a and key in b:
            diff = np.asarray(a[key]) - np.asarray(b[key])
            print(f"{key}: norm={np.linalg.norm(diff):.6f} diff={fmt_vec(diff)}")
    if "robot0_eef_pos" in a and "earbud_1_pos" in a and "robot0_eef_pos" in b and "earbud_1_pos" in b:
        rel_a = a["earbud_1_pos"] - a["robot0_eef_pos"]
        rel_b = b["earbud_1_pos"] - b["robot0_eef_pos"]
        diff = rel_a - rel_b
        print(f"obj_minus_eef: norm={np.linalg.norm(diff):.6f} diff={fmt_vec(diff)}")


def build_env_and_get_obs(task: TaskSpec, init_state: dict[str, np.ndarray], camera_size: int) -> dict[str, np.ndarray]:
    from libero.libero.envs import OffScreenRenderEnv

    kwargs = dict(
        bddl_file_name=task.bddl_file,
        camera_heights=int(camera_size),
        camera_widths=int(camera_size),
        ignore_done=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
    )
    try:
        env = OffScreenRenderEnv(**kwargs)
    except TypeError:
        kwargs.pop("camera_names", None)
        env = OffScreenRenderEnv(**kwargs)

    try:
        base = env.env if hasattr(env, "env") else env
        base.sim.data.qpos[:] = np.asarray(init_state["qpos"], dtype=np.float64)
        base.sim.data.qvel[:] = np.asarray(init_state["qvel"], dtype=np.float64)
        base.sim.forward()
        if hasattr(base, "_get_observations"):
            try:
                obs = base._get_observations(force_update=True)
            except TypeError:
                obs = base._get_observations()
        else:
            raise RuntimeError("Environment does not expose _get_observations().")
        return {
            key: np.asarray(obs[key], dtype=np.float64)
            for key in ("robot0_eef_pos", "robot0_gripper_qpos", "earbud_1_pos", "charging_slot_1_pos")
            if key in obs
        }
    finally:
        try:
            env.close()
        except Exception:
            pass


def transition_report(actions: np.ndarray, max_rows: int = 12) -> None:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < 7:
        print("[actions] unexpected shape:", actions.shape)
        return
    g = actions[:, 6]
    z = actions[:, 2]
    xy = np.linalg.norm(actions[:, :2], axis=1)
    signs = np.sign(g)
    transitions = np.where(np.abs(np.diff(signs, prepend=signs[0])) > 0)[0]
    print("\n[action landmarks]")
    print(f"num_actions: {len(actions)}")
    print(f"gripper unique approx: min={g.min():+.4f} max={g.max():+.4f} mean={g.mean():+.4f}")
    print(f"first gripper transitions: {transitions[:max_rows].tolist()}")
    print(f"first z_pos idx: {np.where(z > 0.01)[0][:max_rows].tolist()}")
    print(f"first z_neg idx: {np.where(z < -0.01)[0][:max_rows].tolist()}")
    print(f"first large_xy idx: {np.where(xy > 0.05)[0][:max_rows].tolist()}")
    for idx in transitions[:max_rows]:
        lo = max(0, int(idx) - 2)
        hi = min(len(actions), int(idx) + 3)
        print(f"  transition@{int(idx)}")
        for j in range(lo, hi):
            print(f"    {j:04d}: xyz={fmt_vec(actions[j, :3])} g={actions[j, 6]:+.3f}")


def compare_lerobot_actions(dataset_root: Path, repo_id: str, episode_index: int, raw_actions: np.ndarray, action_key: str) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    try:
        ds = LeRobotDataset(repo_id=repo_id, root=str(dataset_root), return_uint8=True)
    except TypeError:
        ds = LeRobotDataset(repo_id=repo_id, root=str(dataset_root))

    converted = []
    for i in range(len(ds)):
        sample = ds[i]
        ep = sample.get("episode_index", None)
        if ep is not None and int(np.asarray(ep).reshape(-1)[0]) != int(episode_index):
            continue
        converted.append(np.asarray(sample[action_key], dtype=np.float64).reshape(-1)[:7])
    if not converted:
        print("\n[lerobot action compare] no converted actions found")
        return
    converted_actions = np.asarray(converted, dtype=np.float64)

    print("\n[lerobot action compare]")
    print(f"raw shape: {raw_actions.shape}")
    print(f"converted shape: {converted_actions.shape}")
    print(f"raw first:       {fmt_vec(raw_actions[0])}")
    print(f"converted first: {fmt_vec(converted_actions[0])}")
    best: tuple[float, int] | None = None
    for offset in range(-20, 21):
        if offset >= 0:
            a = raw_actions[offset : offset + min(len(raw_actions) - offset, len(converted_actions))]
            b = converted_actions[: len(a)]
        else:
            b = converted_actions[-offset : -offset + min(len(converted_actions) + offset, len(raw_actions))]
            a = raw_actions[: len(b)]
        if len(a) < 10:
            continue
        mae = float(np.mean(np.abs(a[:, :7] - b[:, :7])))
        if best is None or mae < best[0]:
            best = (mae, offset)
    if best is not None:
        print(f"best raw_offset_vs_converted in [-20,20]: offset={best[1]} mae={best[0]:.8f}")
        if best[1] != 0:
            print("WARNING: converted actions appear shifted relative to raw actions.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-assets", required=True, type=Path)
    parser.add_argument("--level", default="easy")
    parser.add_argument("--init-index", type=int, default=0)
    parser.add_argument("--raw-episode", required=True, type=Path)
    parser.add_argument("--raw-index", type=int, default=0)
    parser.add_argument("--camera-size", type=int, default=224)
    parser.add_argument("--skip-env", action="store_true", help="Only print npz/action summaries; do not instantiate LIBERO.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Optional converted LeRobot dataset to compare.")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--action-key", default="action")
    args = parser.parse_args()

    assets = args.benchmark_assets.resolve()
    raw_episode = args.raw_episode.resolve()
    raw_traj_path = raw_episode / "trajectory.npz"
    raw_meta_path = raw_episode / "metadata.json"
    if not raw_traj_path.exists():
        raise FileNotFoundError(f"Missing raw trajectory: {raw_traj_path}")

    task = load_task(assets, args.level)
    init_state = load_benchmark_state(task, args.init_index)
    raw_traj = np.load(raw_traj_path, allow_pickle=True)
    raw_actions = np.asarray(raw_traj["action"], dtype=np.float64)

    print("[task]")
    print(f"name: {task.name}")
    print(f"level: {task.level}")
    print(f"language: {task.language}")
    print(f"bddl: {task.bddl_file}")
    print(f"state_file: {task.state_file}")
    print(f"num_init_states: {int(init_state['num_init_states'][0])}")
    print(f"init_index_used: {int(init_state['init_index'][0])}")

    print("\n[raw episode]")
    print(f"path: {raw_episode}")
    if raw_meta_path.exists():
        with raw_meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        keep = {k: meta.get(k) for k in ("task", "task_description", "language", "seed", "control_hz", "frame_stride") if k in meta}
        print("metadata_subset:", json.dumps(keep, ensure_ascii=False))
    print("trajectory keys/shapes:", summarize_npz(raw_traj_path))
    transition_report(raw_actions)

    raw_pose = get_raw_positions(raw_traj, args.raw_index)
    print_pose_block(f"raw trajectory index {args.raw_index}", raw_pose)

    bench_pose: dict[str, np.ndarray] = {}
    if not args.skip_env:
        try:
            bench_pose = build_env_and_get_obs(task, init_state, args.camera_size)
            print_pose_block(f"benchmark reset init {int(init_state['init_index'][0])}", bench_pose)
            print_pose_diff("benchmark_init", bench_pose, f"raw_idx_{args.raw_index}", raw_pose)
        except Exception as exc:
            print("\n[benchmark env]")
            print(f"FAILED to instantiate/read env: {type(exc).__name__}: {exc}")
            print("Tip: rerun with --skip-env if you only need action/conversion checks.")

    qpos_raw = first_available(raw_traj, ["qpos", "sim_qpos", "initial_qpos"])
    qvel_raw = first_available(raw_traj, ["qvel", "sim_qvel", "initial_qvel"])
    print("\n[qpos/qvel availability]")
    print(f"raw qpos present: {qpos_raw is not None}")
    print(f"raw qvel present: {qvel_raw is not None}")
    if qpos_raw is not None:
        raw_qpos0 = np.asarray(qpos_raw[0] if np.asarray(qpos_raw).ndim > 1 else qpos_raw, dtype=np.float64)
        diff = np.asarray(init_state["qpos"], dtype=np.float64) - raw_qpos0[: len(init_state["qpos"])]
        print(f"qpos diff norm benchmark-raw0: {np.linalg.norm(diff):.6f}")
    if qvel_raw is not None:
        raw_qvel0 = np.asarray(qvel_raw[0] if np.asarray(qvel_raw).ndim > 1 else qvel_raw, dtype=np.float64)
        diff = np.asarray(init_state["qvel"], dtype=np.float64) - raw_qvel0[: len(init_state["qvel"])]
        print(f"qvel diff norm benchmark-raw0: {np.linalg.norm(diff):.6f}")

    if args.dataset_root is not None and args.repo_id:
        compare_lerobot_actions(args.dataset_root.resolve(), args.repo_id, args.episode_index, raw_actions, args.action_key)

    print("\n[interpretation]")
    print("If benchmark_init vs raw_idx_0 object/eef pose differs by more than a few millimeters, direct replay is not a valid model test.")
    print("If converted actions are shifted relative to raw actions, fix conversion/replay offset before training.")


if __name__ == "__main__":
    main()
