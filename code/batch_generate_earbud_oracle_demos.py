import os
import csv
import json
import argparse
from datetime import datetime

import numpy as np
import imageio

from libero.libero.envs import OffScreenRenderEnv

BASE_DIR = "/root/autodl-tmp/openpi_earbud_proto/third_party/libero/libero/libero/bddl_files/libero_90"

LEVEL_CFG = {
    "easy": {
        "bddl": os.path.join(BASE_DIR, "LIVING_ROOM_SCENE1_insert_the_earbud_into_the_charging_slot_easy.bddl"),
        "xy_thresh": 0.03,
        "z_thresh": 0.025,
        "angle_thresh_deg": 25.0,
    },
    "medium": {
        "bddl": os.path.join(BASE_DIR, "LIVING_ROOM_SCENE1_insert_the_earbud_into_the_charging_slot_medium.bddl"),
        "xy_thresh": 0.02,
        "z_thresh": 0.02,
        "angle_thresh_deg": 20.0,
    },
    "hard": {
        "bddl": os.path.join(BASE_DIR, "LIVING_ROOM_SCENE1_insert_the_earbud_into_the_charging_slot_hard.bddl"),
        "xy_thresh": 0.01,
        "z_thresh": 0.015,
        "angle_thresh_deg": 10.0,
    },
}

def quat_wxyz_to_rotmat(q):
    q = np.asarray(q, dtype=float)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=float)

def axis_world_from_quat(quat, local_axis="z"):
    R = quat_wxyz_to_rotmat(quat)
    axis_map = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }
    return R @ axis_map[local_axis]

def angle_deg_between(v1, v2):
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    v1 = v1 / (np.linalg.norm(v1) + 1e-12)
    v2 = v2 / (np.linalg.norm(v2) + 1e-12)
    cosv = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))

def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    return q / (np.linalg.norm(q) + 1e-12)

def quat_slerp(q0, q1, t):
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)

    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return quat_normalize(q)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1

def custom_success(obs, xy_thresh, z_thresh, angle_thresh_deg):
    ep = obs["earbud_1_pos"]
    sp = obs["charging_slot_1_pos"]

    eq = obs["earbud_1_quat"]
    sq = obs["charging_slot_1_quat"]

    xy_dist = np.linalg.norm(ep[:2] - sp[:2])
    z_dist = abs(ep[2] - sp[2])

    earbud_axis = axis_world_from_quat(eq, local_axis="z")
    slot_axis = axis_world_from_quat(sq, local_axis="z")
    angle_deg = angle_deg_between(earbud_axis, slot_axis)

    success = (xy_dist < xy_thresh) and (z_dist < z_thresh) and (angle_deg < angle_thresh_deg)
    return success, xy_dist, z_dist, angle_deg

def get_sim(env):
    return env.env.sim if hasattr(env, "env") else env.sim

def get_earbud_joint_name(env):
    obj = env.env.objects_dict["earbud_1"] if hasattr(env, "env") else env.objects_dict["earbud_1"]
    joints = getattr(obj, "joints", None)
    if joints and len(joints) > 0:
        return joints[0]
    raise RuntimeError("Could not find free joint for earbud_1")

def get_joint_qpos(sim, joint_name):
    return np.array(sim.data.get_joint_qpos(joint_name), dtype=float)

def set_joint_qpos(sim, joint_name, qpos):
    sim.data.set_joint_qpos(joint_name, qpos)
    sim.forward()

def run_episode(level, cfg, seed, out_dir, camera_size):
    env_args = {
        "bddl_file_name": cfg["bddl"],
        "camera_heights": camera_size,
        "camera_widths": camera_size,
    }

    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)

    if hasattr(env, "env") and hasattr(env.env, "_check_success"):
        env.env._check_success = lambda: False

    obs = env.reset()
    sim = get_sim(env)
    joint_name = get_earbud_joint_name(env)

    slot_pos = obs["charging_slot_1_pos"].copy()
    earbud_pos = obs["earbud_1_pos"].copy()
    slot_quat = obs["charging_slot_1_quat"].copy()

    hover_pos = slot_pos.copy()
    hover_pos[2] = max(earbud_pos[2], slot_pos[2] + 0.05)
    insert_pos = slot_pos.copy()
    insert_pos[2] = slot_pos[2] + 0.01

    frames = []
    traj = {
        "earbud_pos": [],
        "earbud_quat": [],
        "slot_pos": [],
        "slot_quat": [],
        "xy_dist": [],
        "z_dist": [],
        "angle_deg": [],
        "success": [],
    }

    def record_obs(obs):
        if "agentview_image" in obs:
            frames.append(obs["agentview_image"][::-1])

        s, xy, zz, ang = custom_success(
            obs,
            cfg["xy_thresh"],
            cfg["z_thresh"],
            cfg["angle_thresh_deg"],
        )
        traj["earbud_pos"].append(obs["earbud_1_pos"].copy())
        traj["earbud_quat"].append(obs["earbud_1_quat"].copy())
        traj["slot_pos"].append(obs["charging_slot_1_pos"].copy())
        traj["slot_quat"].append(obs["charging_slot_1_quat"].copy())
        traj["xy_dist"].append(xy)
        traj["z_dist"].append(zz)
        traj["angle_deg"].append(ang)
        traj["success"].append(bool(s))
        return s, xy, zz, ang

    def move_only(target_pos, n_steps):
        qpos = get_joint_qpos(sim, joint_name)
        cur = qpos[:3].copy()
        last_obs = None
        for i in range(n_steps):
            alpha = (i + 1) / n_steps
            new_pos = (1 - alpha) * cur + alpha * target_pos
            qpos2 = get_joint_qpos(sim, joint_name)
            qpos2[:3] = new_pos
            set_joint_qpos(sim, joint_name, qpos2)
            obs2, reward, done, info = env.step(np.zeros(7, dtype=np.float32))
            record_obs(obs2)
            last_obs = obs2
        return last_obs

    def rotate_only(target_quat, n_steps):
        qpos = get_joint_qpos(sim, joint_name)
        cur_quat = quat_normalize(qpos[3:7].copy())
        tgt_quat = quat_normalize(target_quat.copy())
        last_obs = None
        for i in range(n_steps):
            alpha = (i + 1) / n_steps
            new_quat = quat_slerp(cur_quat, tgt_quat, alpha)
            qpos2 = get_joint_qpos(sim, joint_name)
            qpos2[3:7] = new_quat
            set_joint_qpos(sim, joint_name, qpos2)
            obs2, reward, done, info = env.step(np.zeros(7, dtype=np.float32))
            record_obs(obs2)
            last_obs = obs2
        return last_obs

    def move_with_rotation(target_pos, target_quat, n_steps):
        qpos = get_joint_qpos(sim, joint_name)
        cur_pos = qpos[:3].copy()
        cur_quat = quat_normalize(qpos[3:7].copy())
        tgt_quat = quat_normalize(target_quat.copy())
        last_obs = None
        for i in range(n_steps):
            alpha = (i + 1) / n_steps
            new_pos = (1 - alpha) * cur_pos + alpha * target_pos
            new_quat = quat_slerp(cur_quat, tgt_quat, alpha)
            qpos2 = get_joint_qpos(sim, joint_name)
            qpos2[:3] = new_pos
            qpos2[3:7] = new_quat
            set_joint_qpos(sim, joint_name, qpos2)
            obs2, reward, done, info = env.step(np.zeros(7, dtype=np.float32))
            record_obs(obs2)
            last_obs = obs2
        return last_obs

    def hold_with_rotation(target_quat, n_steps):
        tgt_quat = quat_normalize(target_quat.copy())
        last_obs = None
        for _ in range(n_steps):
            qpos2 = get_joint_qpos(sim, joint_name)
            qpos2[3:7] = tgt_quat
            set_joint_qpos(sim, joint_name, qpos2)
            obs2, reward, done, info = env.step(np.zeros(7, dtype=np.float32))
            record_obs(obs2)
            last_obs = obs2
        return last_obs

    # init frame
    if "agentview_image" in obs:
        frames.extend([obs["agentview_image"][::-1]] * 4)
    record_obs(obs)

    # scripted oracle rollout
    for _ in range(4):
        obs, reward, done, info = env.step(np.zeros(7, dtype=np.float32))
        record_obs(obs)

    obs = move_only(hover_pos, n_steps=18)
    obs = rotate_only(slot_quat, n_steps=14)
    obs = move_with_rotation(insert_pos, slot_quat, n_steps=16)
    obs = hold_with_rotation(slot_quat, n_steps=8)

    final_success, final_xy, final_z, final_ang = custom_success(
        obs,
        cfg["xy_thresh"],
        cfg["z_thresh"],
        cfg["angle_thresh_deg"],
    )

    # save outputs
    ep_id = f"seed{seed}"
    mp4_path = os.path.join(out_dir, f"{ep_id}.mp4")
    npz_path = os.path.join(out_dir, f"{ep_id}.npz")
    json_path = os.path.join(out_dir, f"{ep_id}.json")

    imageio.mimwrite(mp4_path, frames, fps=10)

    np.savez_compressed(
        npz_path,
        earbud_pos=np.array(traj["earbud_pos"]),
        earbud_quat=np.array(traj["earbud_quat"]),
        slot_pos=np.array(traj["slot_pos"]),
        slot_quat=np.array(traj["slot_quat"]),
        xy_dist=np.array(traj["xy_dist"]),
        z_dist=np.array(traj["z_dist"]),
        angle_deg=np.array(traj["angle_deg"]),
        success=np.array(traj["success"], dtype=bool),
    )

    meta = {
        "level": level,
        "seed": seed,
        "bddl": cfg["bddl"],
        "final_success": bool(final_success),
        "final_xy_dist": float(final_xy),
        "final_z_dist": float(final_z),
        "final_angle_deg": float(final_ang),
        "num_frames": len(frames),
        "video_path": mp4_path,
        "trajectory_path": npz_path,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    env.close()
    return meta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", nargs="+", default=["easy", "medium", "hard"])
    parser.add_argument("--episodes-per-level", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--camera-size", type=int, default=512)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = f"/root/autodl-tmp/openpi_earbud_proto/oracle_demos_{run_id}"
    os.makedirs(root_dir, exist_ok=True)

    summary_csv = os.path.join(root_dir, "summary.csv")
    rows = []

    print("saving to:", root_dir)

    for level in args.levels:
        cfg = LEVEL_CFG[level]
        level_dir = os.path.join(root_dir, level)
        os.makedirs(level_dir, exist_ok=True)

        print(f"\n=== level: {level} ===")
        for i in range(args.episodes_per_level):
            seed = args.seed_base + i
            try:
                meta = run_episode(level, cfg, seed, level_dir, args.camera_size)
                rows.append(meta)
                print(
                    f"[ok] {level} seed={seed} "
                    f"success={meta['final_success']} "
                    f"xy={meta['final_xy_dist']:.4f} "
                    f"z={meta['final_z_dist']:.4f} "
                    f"ang={meta['final_angle_deg']:.2f}"
                )
            except Exception as e:
                err = {
                    "level": level,
                    "seed": seed,
                    "bddl": cfg["bddl"],
                    "final_success": False,
                    "final_xy_dist": None,
                    "final_z_dist": None,
                    "final_angle_deg": None,
                    "num_frames": 0,
                    "video_path": None,
                    "trajectory_path": None,
                    "error": repr(e),
                }
                rows.append(err)
                print(f"[fail] {level} seed={seed} error={e}")

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "level",
                "seed",
                "bddl",
                "final_success",
                "final_xy_dist",
                "final_z_dist",
                "final_angle_deg",
                "num_frames",
                "video_path",
                "trajectory_path",
                "error",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("\nsaved summary:", summary_csv)
    print("done")

if __name__ == "__main__":
    main()
