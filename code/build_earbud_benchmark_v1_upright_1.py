#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from libero.libero.envs import OffScreenRenderEnv

BASE_BDDL_DIR = "/root/autodl-tmp/openpi_earbud_proto/third_party/libero/libero/libero/bddl_files/libero_90"
LEVEL_TO_BDDL = {
    "easy": "LIVING_ROOM_SCENE1_insert_the_earbud_into_the_charging_slot_easy.bddl",
    "medium": "LIVING_ROOM_SCENE1_insert_the_earbud_into_the_charging_slot_medium.bddl",
    "hard": "LIVING_ROOM_SCENE1_insert_the_earbud_into_the_charging_slot_hard.bddl",
}

SLOT_Y_DEG = -12.0
EARBUD_EDGE_REST_Z = 0.4435
EARBUD_FLAT_REST_Z = 0.4435
FLAT_REST_ROLL_DEG = 90.0
Q_VERTICAL = np.array([0.70710678, 0.0, -0.70710678, 0.0], dtype=np.float64)


def quat_wxyz_from_axis_angle(axis, deg):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    th = np.deg2rad(deg)
    w = np.cos(th / 2.0)
    xyz = axis * np.sin(th / 2.0)
    return np.array([w, xyz[0], xyz[1], xyz[2]], dtype=np.float64)


def quat_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def make_offscreen_env(bddl_file: str):
    kwargs = dict(
        bddl_file_name=bddl_file,
        camera_heights=128,
        camera_widths=128,
        ignore_done=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
    )
    try:
        return OffScreenRenderEnv(**kwargs)
    except TypeError:
        kwargs.pop("camera_names", None)
        return OffScreenRenderEnv(**kwargs)


def get_sim(env):
    return env.env.sim if hasattr(env, "env") else env.sim


def get_joint_name(env, object_name: str):
    base = env.env if hasattr(env, "env") else env
    obj = base.objects_dict[object_name]
    joints = getattr(obj, "joints", None)
    if not joints:
        raise RuntimeError(f"Could not find free joint for {object_name}")
    return joints[0]


def get_joint_qpos(sim, joint_name: str):
    return np.array(sim.data.get_joint_qpos(joint_name), dtype=np.float64)


def set_joint_qpos(sim, joint_name: str, qpos):
    sim.data.set_joint_qpos(joint_name, np.asarray(qpos, dtype=np.float64))


def set_joint_qvel_zero(sim, joint_name: str):
    try:
        qvel = np.array(sim.data.get_joint_qvel(joint_name), dtype=np.float64)
        sim.data.set_joint_qvel(joint_name, np.zeros_like(qvel))
    except Exception:
        pass


def parse_seed_from_name(name: str) -> int:
    m = re.search(r"seed(\d+)", name)
    return int(m.group(1)) if m else 0


def is_upright(meta: dict) -> bool:
    return str(meta.get("rest_pose_mode", "edge")).lower() != "flat"


def is_success(meta: dict) -> bool:
    for k in ("release_drop_success", "success", "is_success", "final_success"):
        if k in meta:
            return bool(meta[k])
    if "obj_slot_xy" in meta and "obj_slot_z" in meta:
        return (float(meta["obj_slot_xy"]) < 0.02) and (float(meta["obj_slot_z"]) < 0.03)
    return False


def reconstruct_init_state(env, meta: dict, earbud_joint: str, slot_joint: str):
    seed = int(meta.get("seed", 0))
    random_yaw_deg = float(meta.get("random_yaw_deg", 0.0))
    rest_pose_mode = str(meta.get("rest_pose_mode", "edge")).lower()

    env.seed(seed)
    obs = env.reset()
    sim = get_sim(env)

    earbud_pos = np.asarray(obs["earbud_1_pos"], dtype=np.float64).copy()
    if rest_pose_mode == "flat":
        earbud_pos[2] = EARBUD_FLAT_REST_Z
    else:
        earbud_pos[2] = EARBUD_EDGE_REST_Z

    q_random_yaw = quat_wxyz_from_axis_angle([0, 0, 1], random_yaw_deg)
    if rest_pose_mode == "flat":
        q_flat_roll_local = quat_wxyz_from_axis_angle([0, 0, 1], FLAT_REST_ROLL_DEG)
        q_rest_base = quat_mul_wxyz(Q_VERTICAL, q_flat_roll_local)
    else:
        q_rest_base = Q_VERTICAL
    earbud_quat = quat_mul_wxyz(q_random_yaw, q_rest_base)

    slot_pos = np.asarray(obs["charging_slot_1_pos"], dtype=np.float64).copy()
    slot_pos[2] = max(float(slot_pos[2]), 0.4680)
    slot_quat = quat_wxyz_from_axis_angle([0, 1, 0], SLOT_Y_DEG)

    q_ear = get_joint_qpos(sim, earbud_joint)
    q_ear[:3] = earbud_pos
    q_ear[3:7] = earbud_quat

    q_slot = get_joint_qpos(sim, slot_joint)
    q_slot[:3] = slot_pos
    q_slot[3:7] = slot_quat

    set_joint_qpos(sim, earbud_joint, q_ear)
    set_joint_qpos(sim, slot_joint, q_slot)
    set_joint_qvel_zero(sim, earbud_joint)
    set_joint_qvel_zero(sim, slot_joint)

    sim.forward()  # 关键：不调用 env.step()

    qpos = np.asarray(sim.data.qpos, dtype=np.float32).copy()
    qvel = np.asarray(sim.data.qvel, dtype=np.float32).copy()
    return qpos, qvel


def write_runtime_module(module_path: Path, assets_dir: Path):
    template = r'''from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from libero.libero.envs import OffScreenRenderEnv

_ASSETS_DIR = Path(r"__ASSETS_DIR__")
_TASKS_JSON = _ASSETS_DIR / "tasks.json"


@dataclass
class TaskSpec:
    name: str
    level: str
    bddl_file: str
    language: str
    max_steps: int
    state_file: str


def _read_tasks_json():
    with _TASKS_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_task_specs():
    raw = _read_tasks_json()
    out = []
    for level in ("easy", "medium", "hard"):
        if level not in raw["levels"]:
            continue
        t = raw["levels"][level]
        out.append(
            TaskSpec(
                name=t["name"],
                level=level,
                bddl_file=t["bddl"],
                language=t["language"],
                max_steps=int(t["max_steps"]),
                state_file=t["state_file"],
            )
        )
    return out


def build_env(task: TaskSpec, camera_size: int = 512):
    kwargs = dict(
        bddl_file_name=task.bddl_file,
        camera_heights=camera_size,
        camera_widths=camera_size,
        ignore_done=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
    )
    try:
        return OffScreenRenderEnv(**kwargs)
    except TypeError:
        kwargs.pop("camera_names", None)
        return OffScreenRenderEnv(**kwargs)


def load_init_states(task: TaskSpec):
    pack = np.load(task.state_file, allow_pickle=True)
    qpos = np.asarray(pack["qpos"], dtype=np.float32)
    qvel = np.asarray(pack["qvel"], dtype=np.float32)
    if "episode_dir" in pack:
        epi = pack["episode_dir"]
    else:
        epi = np.array([""] * len(qpos), dtype=object)

    states = []
    for i in range(len(qpos)):
        states.append({"qpos": qpos[i], "qvel": qvel[i], "episode_dir": str(epi[i])})
    return states


def set_state_qpos_qvel(env, init_state):
    if isinstance(init_state, dict):
        qpos = np.asarray(init_state["qpos"], dtype=np.float64)
        qvel = np.asarray(init_state["qvel"], dtype=np.float64)
    else:
        qpos, qvel = init_state
        qpos = np.asarray(qpos, dtype=np.float64)
        qvel = np.asarray(qvel, dtype=np.float64)

    base = env.env if hasattr(env, "env") else env
    sim = base.sim
    sim.data.qpos[:] = qpos
    sim.data.qvel[:] = qvel
    sim.forward()


def compute_success(obs):
    obj = np.asarray(obs["earbud_1_pos"], dtype=np.float32)
    slot = np.asarray(obs["charging_slot_1_pos"], dtype=np.float32)
    obj_slot_xy = float(np.linalg.norm(obj[:2] - slot[:2]))
    obj_slot_z = float(obj[2] - slot[2])
    success = float((obj_slot_xy < 0.02) and (obj_slot_z < 0.03))
    return {"success": success, "obj_slot_xy": obj_slot_xy, "obj_slot_z": obj_slot_z}
'''
    code = template.replace("__ASSETS_DIR__", str(assets_dir.resolve()))
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(code, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, default=Path("./benchmark/earbud_benchmark_v1_upright_assets"))
    parser.add_argument("--module-path", type=Path, default=Path("./earbud_benchmark_v1.py"))
    parser.add_argument("--include-fail", action="store_true")
    parser.add_argument("--max-per-level", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--language", type=str, default="Insert the earbud into the charging slot.")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    assets_dir = args.assets_dir.resolve()
    states_dir = assets_dir / "states"
    states_dir.mkdir(parents=True, exist_ok=True)

    tasks = {"created_at": datetime.now().isoformat(), "dataset_root": str(dataset_root), "levels": {}}

    for level in ("easy", "medium", "hard"):
        level_dir = dataset_root / level
        if not level_dir.exists():
            raise FileNotFoundError(f"Missing level dir: {level_dir}")

        candidates = []
        for ep in sorted(level_dir.glob("episode_*")):
            if not ep.is_dir():
                continue
            meta_path = ep / "metadata.json"
            if not meta_path.exists():
                continue
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)

            if not is_upright(meta):
                continue
            ok = is_success(meta)
            if (not args.include_fail) and (not ok):
                continue

            seed = int(meta.get("seed", parse_seed_from_name(ep.name)))
            candidates.append((seed, ep.name, ep, meta, ok))

        candidates.sort(key=lambda x: (x[0], x[1]))

        if len(candidates) < args.max_per_level:
            raise RuntimeError(
                f"{level}: upright+success only has {len(candidates)} episodes, "
                f"but --max-per-level={args.max_per_level}"
            )

        selected = candidates[: args.max_per_level]
        bddl_path = str((Path(BASE_BDDL_DIR) / LEVEL_TO_BDDL[level]).resolve())
        env = make_offscreen_env(bddl_path)
        sim = get_sim(env)
        _ = sim  # noqa
        earbud_joint = get_joint_name(env, "earbud_1")
        slot_joint = get_joint_name(env, "charging_slot_1")

        qpos_list, qvel_list, ep_list, seed_list = [], [], [], []
        for i, (_, ep_name, ep_dir, meta, ok) in enumerate(selected, start=1):
            qpos, qvel = reconstruct_init_state(env, meta, earbud_joint, slot_joint)
            qpos_list.append(qpos)
            qvel_list.append(qvel)
            ep_list.append(str(ep_dir))
            seed_list.append(int(meta.get("seed", parse_seed_from_name(ep_name))))
            print(f"[{level} {i}/{len(selected)}] {ep_dir.name} success={ok}")

        env.close()

        state_file = states_dir / f"{level}_init_states.npz"
        np.savez_compressed(
            state_file,
            qpos=np.asarray(qpos_list, dtype=np.float32),
            qvel=np.asarray(qvel_list, dtype=np.float32),
            episode_dir=np.asarray(ep_list, dtype=object),
            seed=np.asarray(seed_list, dtype=np.int32),
        )

        tasks["levels"][level] = {
            "name": f"earbud_insert_{level}_upright",
            "bddl": bddl_path,
            "language": args.language,
            "max_steps": int(args.max_steps),
            "state_file": str(state_file),
            "num_episodes": int(len(qpos_list)),
        }

    tasks_json = assets_dir / "tasks.json"
    with tasks_json.open("w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    write_runtime_module(args.module_path.resolve(), assets_dir)
    print(f"[saved] tasks: {tasks_json}")
    print(f"[saved] module: {args.module_path.resolve()}")
    for lvl in ("easy", "medium", "hard"):
        print(f"[count] {lvl} = {tasks['levels'][lvl]['num_episodes']}")


if __name__ == "__main__":
    main()
