from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from libero.libero.envs import OffScreenRenderEnv

_ASSETS_DIR = Path(r"/root/autodl-tmp/openpi_earbud_proto/benchmark/earbud_benchmark_red_peg_single1")
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
