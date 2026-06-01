from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np
import torch


def quat_wxyz_from_axis_angle(axis, deg):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    theta = np.deg2rad(deg)
    w = np.cos(theta / 2.0)
    xyz = axis * np.sin(theta / 2.0)
    return np.array([w, xyz[0], xyz[1], xyz[2]], dtype=float)


def quat_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def get_sim(env):
    return env.env.sim if hasattr(env, "env") else env.sim


def get_joint_name(env, object_name: str) -> str:
    base = env.env if hasattr(env, "env") else env
    obj = base.objects_dict[object_name]
    joints = getattr(obj, "joints", None)
    if joints and len(joints) > 0:
        return joints[0]
    raise RuntimeError(f"Could not find free joint for object '{object_name}'")


def _sim_data(sim):
    if hasattr(sim, "data"):
        return sim.data
    if hasattr(sim, "_data"):
        return sim._data
    return None


def get_joint_qpos(sim, joint_name: str) -> np.ndarray:
    data = _sim_data(sim)
    if data is not None and hasattr(data, "get_joint_qpos"):
        return np.asarray(data.get_joint_qpos(joint_name), dtype=float)
    if hasattr(sim, "get_joint_qpos"):
        return np.asarray(sim.get_joint_qpos(joint_name), dtype=float)
    raise AttributeError(
        "Could not read joint qpos from sim. Expected sim.data/sim._data.get_joint_qpos or sim.get_joint_qpos."
    )


def set_joint_qpos(sim, joint_name: str, qpos: np.ndarray) -> None:
    data = _sim_data(sim)
    if data is not None and hasattr(data, "set_joint_qpos"):
        data.set_joint_qpos(joint_name, np.asarray(qpos, dtype=float))
    elif hasattr(sim, "set_joint_qpos"):
        sim.set_joint_qpos(joint_name, np.asarray(qpos, dtype=float))
    else:
        raise AttributeError(
            "Could not set joint qpos on sim. Expected sim.data/sim._data.set_joint_qpos or sim.set_joint_qpos."
        )
    sim.forward()


def set_joint_qvel_zero(sim, joint_name: str) -> None:
    try:
        data = _sim_data(sim)
        if data is not None and hasattr(data, "get_joint_qvel") and hasattr(data, "set_joint_qvel"):
            qvel = np.asarray(data.get_joint_qvel(joint_name), dtype=float)
            data.set_joint_qvel(joint_name, np.zeros_like(qvel))
        elif hasattr(sim, "get_joint_qvel") and hasattr(sim, "set_joint_qvel"):
            qvel = np.asarray(sim.get_joint_qvel(joint_name), dtype=float)
            sim.set_joint_qvel(joint_name, np.zeros_like(qvel))
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build fixed init states (.pt) for a custom LIBERO insert task. "
            "This reuses the same object/slot reset strategy from the successful scripted rollout."
        )
    )
    parser.add_argument("--bddl-file", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num-states", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--settle-steps", type=int, default=25)

    parser.add_argument("--earbud-object-name", type=str, default="earbud_1")
    parser.add_argument("--slot-object-name", type=str, default="charging_slot_1")
    parser.add_argument("--earbud-pos-key", type=str, default="earbud_1_pos")
    parser.add_argument("--slot-pos-key", type=str, default="charging_slot_1_pos")

    parser.add_argument("--random-yaw-min-deg", type=float, default=-90.0)
    parser.add_argument("--random-yaw-max-deg", type=float, default=90.0)
    parser.add_argument("--flat-rest-prob", type=float, default=0.5)
    parser.add_argument("--flat-rest-roll-deg", type=float, default=90.0)

    parser.add_argument("--earbud-edge-rest-z", type=float, default=0.4435)
    parser.add_argument("--earbud-flat-rest-z", type=float, default=0.4435)

    parser.add_argument("--slot-y-deg", type=float, default=-12.0)
    parser.add_argument("--slot-min-z", type=float, default=0.4680)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from libero.libero.envs import OffScreenRenderEnv

    out_path = pathlib.Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = OffScreenRenderEnv(
        bddl_file_name=args.bddl_file,
        camera_heights=args.camera_size,
        camera_widths=args.camera_size,
        ignore_done=True,
    )
    env.seed(args.seed)
    rng = np.random.RandomState(args.seed)

    sim = get_sim(env)
    earbud_joint_name = get_joint_name(env, args.earbud_object_name)
    slot_joint_name = get_joint_name(env, args.slot_object_name)

    q_vertical = np.array([0.70710678, 0.0, -0.70710678, 0.0], dtype=float)
    q_flat_roll_local = quat_wxyz_from_axis_angle([0, 0, 1], args.flat_rest_roll_deg)
    slot_stable_quat = quat_wxyz_from_axis_angle([0, 1, 0], args.slot_y_deg)

    init_states: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []

    dummy_action = np.zeros(7, dtype=np.float32)
    dummy_action[-1] = -1.0

    for idx in range(args.num_states):
        obs = env.reset()

        earbud_stable_pos = np.asarray(obs[args.earbud_pos_key], dtype=float).copy()
        slot_stable_pos = np.asarray(obs[args.slot_pos_key], dtype=float).copy()

        random_yaw_deg = float(
            rng.uniform(args.random_yaw_min_deg, args.random_yaw_max_deg)
        )
        q_random_yaw = quat_wxyz_from_axis_angle([0, 0, 1], random_yaw_deg)

        rest_pose_mode = "flat" if rng.rand() < args.flat_rest_prob else "edge"
        if rest_pose_mode == "flat":
            earbud_stable_pos[2] = args.earbud_flat_rest_z
            q_rest_base = quat_mul_wxyz(q_vertical, q_flat_roll_local)
        else:
            earbud_stable_pos[2] = args.earbud_edge_rest_z
            q_rest_base = q_vertical
        earbud_stable_quat = quat_mul_wxyz(q_random_yaw, q_rest_base)

        slot_stable_pos[2] = max(slot_stable_pos[2], args.slot_min_z)

        for _ in range(args.settle_steps):
            q_ear = get_joint_qpos(sim, earbud_joint_name)
            q_ear[:3] = earbud_stable_pos
            q_ear[3:7] = earbud_stable_quat
            set_joint_qpos(sim, earbud_joint_name, q_ear)
            set_joint_qvel_zero(sim, earbud_joint_name)

            q_slot = get_joint_qpos(sim, slot_joint_name)
            q_slot[:3] = slot_stable_pos
            q_slot[3:7] = slot_stable_quat
            set_joint_qpos(sim, slot_joint_name, q_slot)
            set_joint_qvel_zero(sim, slot_joint_name)

            env.step(dummy_action.tolist())

        if hasattr(env, "get_sim_state"):
            state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        else:
            state = np.asarray(sim.get_state().flatten(), dtype=np.float64).copy()

        init_states.append(state)
        metadata.append(
            {
                "index": idx,
                "seed": int(args.seed),
                "random_yaw_deg": random_yaw_deg,
                "rest_pose_mode": rest_pose_mode,
            }
        )
        print(
            f"[{idx + 1}/{args.num_states}] rest_pose={rest_pose_mode} random_yaw_deg={random_yaw_deg:.2f}"
        )

    env.close()

    states_np = np.stack(init_states, axis=0)
    payload = {
        "init_states": torch.from_numpy(states_np),
        "metadata": metadata,
        "bddl_file": str(pathlib.Path(args.bddl_file).expanduser().resolve()),
        "earbud_object_name": args.earbud_object_name,
        "slot_object_name": args.slot_object_name,
    }
    torch.save(payload, out_path)

    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "output": str(out_path),
                "num_states": int(states_np.shape[0]),
                "state_dim": int(states_np.shape[1]),
                "seed": int(args.seed),
                "bddl_file": str(pathlib.Path(args.bddl_file).expanduser().resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved init states: {out_path}")
    print(f"saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
