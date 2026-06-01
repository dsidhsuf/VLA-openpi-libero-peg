#!/usr/bin/env python3
"""
Create a benchmark copy whose object initial pose matches a raw demo.

This fixes a very specific failure mode:
  direct replay of a successful demo fails because the benchmark object's
  initial z/pose is different from the recorded demo's initial z/pose.

The script does not overwrite the source benchmark. It copies the benchmark
directory, patches the selected level's state npz in the copy, and updates
tasks.json to point at the copied state file.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def make_relative(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def load_tasks(assets_dir: Path) -> dict[str, Any]:
    tasks_json = assets_dir / "tasks.json"
    with tasks_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_tasks(assets_dir: Path, tasks: dict[str, Any]) -> None:
    tasks_json = assets_dir / "tasks.json"
    with tasks_json.open("w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)


def get_joint_name(env: Any, object_name: str) -> str:
    base = env.env if hasattr(env, "env") else env
    obj = base.objects_dict[object_name]
    joints = getattr(obj, "joints", None)
    if joints and len(joints) > 0:
        return str(joints[0])
    raise RuntimeError(f"Could not find free joint for {object_name!r}")


def get_obs(env: Any) -> dict[str, Any]:
    base = env.env if hasattr(env, "env") else env
    if hasattr(base, "_get_observations"):
        try:
            return base._get_observations(force_update=True)
        except TypeError:
            return base._get_observations()
    raise RuntimeError("Environment does not expose _get_observations().")


def build_env(bddl_file: str, camera_size: int):
    from libero.libero.envs import OffScreenRenderEnv

    kwargs = dict(
        bddl_file_name=bddl_file,
        camera_heights=int(camera_size),
        camera_widths=int(camera_size),
        ignore_done=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
    )
    try:
        return OffScreenRenderEnv(**kwargs)
    except TypeError:
        kwargs.pop("camera_names", None)
        return OffScreenRenderEnv(**kwargs)


def set_state(env: Any, qpos: np.ndarray, qvel: np.ndarray) -> None:
    base = env.env if hasattr(env, "env") else env
    base.sim.data.qpos[:] = np.asarray(qpos, dtype=np.float64)
    base.sim.data.qvel[:] = np.asarray(qvel, dtype=np.float64)
    base.sim.forward()


def zero_joint_qvel(sim: Any, joint_name: str) -> None:
    try:
        qvel = np.asarray(sim.data.get_joint_qvel(joint_name), dtype=np.float64)
        sim.data.set_joint_qvel(joint_name, np.zeros_like(qvel))
    except Exception:
        pass


def patch_one_state(
    env: Any,
    qpos: np.ndarray,
    qvel: np.ndarray,
    object_name: str,
    raw_object_pos: np.ndarray,
    raw_object_quat_wxyz: np.ndarray,
    slot_name: str | None,
    raw_slot_pos: np.ndarray | None,
    raw_slot_quat_wxyz: np.ndarray | None,
    patch_slot: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    set_state(env, qpos, qvel)
    base = env.env if hasattr(env, "env") else env
    sim = base.sim

    before_obs = get_obs(env)
    object_joint = get_joint_name(env, object_name)
    q_obj = np.asarray(sim.data.get_joint_qpos(object_joint), dtype=np.float64).copy()
    q_obj[:3] = np.asarray(raw_object_pos, dtype=np.float64)
    q_obj[3:7] = np.asarray(raw_object_quat_wxyz, dtype=np.float64)
    sim.data.set_joint_qpos(object_joint, q_obj)
    zero_joint_qvel(sim, object_joint)

    slot_joint = None
    if patch_slot and slot_name and raw_slot_pos is not None and raw_slot_quat_wxyz is not None:
        slot_joint = get_joint_name(env, slot_name)
        q_slot = np.asarray(sim.data.get_joint_qpos(slot_joint), dtype=np.float64).copy()
        q_slot[:3] = np.asarray(raw_slot_pos, dtype=np.float64)
        q_slot[3:7] = np.asarray(raw_slot_quat_wxyz, dtype=np.float64)
        sim.data.set_joint_qpos(slot_joint, q_slot)
        zero_joint_qvel(sim, slot_joint)

    sim.forward()
    after_obs = get_obs(env)
    return (
        np.asarray(sim.data.qpos, dtype=np.float32).copy(),
        np.asarray(sim.data.qvel, dtype=np.float32).copy(),
        {
            "object_joint": object_joint,
            "slot_joint": slot_joint,
            "before": {
                f"{object_name}_pos": np.asarray(before_obs.get(f"{object_name}_pos", []), dtype=float).tolist(),
                f"{slot_name}_pos": np.asarray(before_obs.get(f"{slot_name}_pos", []), dtype=float).tolist() if slot_name else [],
            },
            "after": {
                f"{object_name}_pos": np.asarray(after_obs.get(f"{object_name}_pos", []), dtype=float).tolist(),
                f"{slot_name}_pos": np.asarray(after_obs.get(f"{slot_name}_pos", []), dtype=float).tolist() if slot_name else [],
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-assets", required=True, type=Path)
    parser.add_argument("--output-assets", required=True, type=Path)
    parser.add_argument("--level", default="easy")
    parser.add_argument("--raw-episode", required=True, type=Path)
    parser.add_argument("--raw-index", type=int, default=0)
    parser.add_argument("--object-name", default="earbud_1")
    parser.add_argument("--slot-name", default="charging_slot_1")
    parser.add_argument("--patch-slot", action="store_true")
    parser.add_argument("--camera-size", type=int, default=224)
    parser.add_argument("--force-overwrite", action="store_true")
    args = parser.parse_args()

    src_assets = args.benchmark_assets.resolve()
    out_assets = args.output_assets.resolve()
    raw_episode = args.raw_episode.resolve()
    raw_traj_path = raw_episode / "trajectory.npz"
    if not raw_traj_path.exists():
        raise FileNotFoundError(f"Missing raw trajectory: {raw_traj_path}")

    if out_assets.exists():
        if args.force_overwrite:
            shutil.rmtree(out_assets)
        else:
            raise FileExistsError(f"{out_assets} already exists. Use --force-overwrite to recreate it.")
    shutil.copytree(src_assets, out_assets)

    tasks = load_tasks(out_assets)
    levels = tasks.get("levels", {})
    if args.level not in levels:
        raise KeyError(f"Level {args.level!r} not in tasks.json. Available: {sorted(levels)}")
    task = levels[args.level]
    state_file = resolve_path(out_assets, task["state_file"])

    raw = np.load(raw_traj_path, allow_pickle=True)
    object_pos_key = f"{args.object_name}_pos"
    object_quat_key = f"{args.object_name}_quat_wxyz"
    slot_pos_key = f"{args.slot_name}_pos"
    slot_quat_key = f"{args.slot_name}_quat_wxyz"
    for key in (object_pos_key, object_quat_key):
        if key not in raw.files:
            raise KeyError(f"Raw trajectory missing {key!r}. Available keys: {raw.files}")

    raw_idx = int(np.clip(args.raw_index, 0, len(raw[object_pos_key]) - 1))
    raw_object_pos = np.asarray(raw[object_pos_key][raw_idx], dtype=np.float64)
    raw_object_quat = np.asarray(raw[object_quat_key][raw_idx], dtype=np.float64)
    raw_slot_pos = np.asarray(raw[slot_pos_key][raw_idx], dtype=np.float64) if slot_pos_key in raw.files else None
    raw_slot_quat = np.asarray(raw[slot_quat_key][raw_idx], dtype=np.float64) if slot_quat_key in raw.files else None

    pack = np.load(state_file, allow_pickle=True)
    qpos = np.asarray(pack["qpos"], dtype=np.float64)
    qvel = np.asarray(pack["qvel"], dtype=np.float64)
    other_arrays = {k: pack[k] for k in pack.files if k not in ("qpos", "qvel")}

    env = build_env(str(task["bddl"]), args.camera_size)
    reports = []
    patched_qpos = []
    patched_qvel = []
    try:
        for i in range(len(qpos)):
            qpos_i, qvel_i, report = patch_one_state(
                env=env,
                qpos=qpos[i],
                qvel=qvel[i],
                object_name=args.object_name,
                raw_object_pos=raw_object_pos,
                raw_object_quat_wxyz=raw_object_quat,
                slot_name=args.slot_name,
                raw_slot_pos=raw_slot_pos,
                raw_slot_quat_wxyz=raw_slot_quat,
                patch_slot=bool(args.patch_slot),
            )
            report["state_index"] = int(i)
            patched_qpos.append(qpos_i)
            patched_qvel.append(qvel_i)
            reports.append(report)
    finally:
        try:
            env.close()
        except Exception:
            pass

    np.savez_compressed(
        state_file,
        qpos=np.asarray(patched_qpos, dtype=np.float32),
        qvel=np.asarray(patched_qvel, dtype=np.float32),
        **other_arrays,
    )

    task["state_file"] = make_relative(state_file, out_assets)
    save_tasks(out_assets, tasks)

    summary = {
        "source_benchmark_assets": str(src_assets),
        "output_benchmark_assets": str(out_assets),
        "level": args.level,
        "state_file": str(state_file),
        "raw_episode": str(raw_episode),
        "raw_index": raw_idx,
        "object_name": args.object_name,
        "slot_name": args.slot_name,
        "patch_slot": bool(args.patch_slot),
        "raw_object_pos": raw_object_pos.tolist(),
        "raw_object_quat_wxyz": raw_object_quat.tolist(),
        "raw_slot_pos": raw_slot_pos.tolist() if raw_slot_pos is not None else None,
        "reports": reports,
    }
    report_path = out_assets / "patched_from_raw_demo_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[saved] {state_file}")
    print(f"[saved] {report_path}")


if __name__ == "__main__":
    main()
