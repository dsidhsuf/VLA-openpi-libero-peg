import os
import argparse
import numpy as np
import imageio
from datetime import datetime

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
    q = q / np.linalg.norm(q)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["easy", "medium", "hard"], required=True)
    args = parser.parse_args()

    cfg = LEVEL_CFG[args.level]
    out_dir = f"/root/autodl-tmp/openpi_earbud_proto/levels_eval_rot_v2/{args.level}"
    os.makedirs(out_dir, exist_ok=True)

    env_args = {
        "bddl_file_name": cfg["bddl"],
        "camera_heights": 512,
        "camera_widths": 512,
    }

    env = OffScreenRenderEnv(**env_args)
    env.seed(0)

    if hasattr(env, "env") and hasattr(env.env, "_check_success"):
        env.env._check_success = lambda: False

    obs = env.reset()
    print("level:", args.level)
    print("reset ok")

    sim = get_sim(env)
    joint_name = get_earbud_joint_name(env)
    print("earbud joint:", joint_name)

    slot_pos = obs["charging_slot_1_pos"].copy()
    earbud_pos = obs["earbud_1_pos"].copy()
    slot_quat = obs["charging_slot_1_quat"].copy()

    print("start earbud_pos:", earbud_pos)
    print("slot_pos:", slot_pos)
    print("slot_quat:", slot_quat)

    hover_pos = slot_pos.copy()
    hover_pos[2] = max(earbud_pos[2], slot_pos[2] + 0.05)

    insert_pos = slot_pos.copy()
    insert_pos[2] = slot_pos[2] + 0.01

    frames = []
    img = obs["agentview_image"][::-1]
    frames.extend([img] * 4)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    init_path = os.path.join(out_dir, f"init_{ts}.png")
    imageio.imwrite(init_path, img)
    print("saved:", init_path)

    dummy_action = np.zeros(7, dtype=np.float32)

    def record_current_frame(obs):
        if "agentview_image" in obs:
            frames.append(obs["agentview_image"][::-1])

    def log_obs(tag, obs2):
        s, xy, zz, ang = custom_success(
            obs2,
            cfg["xy_thresh"],
            cfg["z_thresh"],
            cfg["angle_thresh_deg"],
        )
        print(f"{tag} success={s} xy_dist={xy:.4f} z_dist={zz:.4f} angle_deg={ang:.2f}")
        return obs2

    def move_only(target_pos, n_steps, tag="move step"):
        qpos = get_joint_qpos(sim, joint_name)
        cur = qpos[:3].copy()
        last_obs = None
        for i in range(n_steps):
            alpha = (i + 1) / n_steps
            new_pos = (1 - alpha) * cur + alpha * target_pos
            qpos2 = get_joint_qpos(sim, joint_name)
            qpos2[:3] = new_pos
            set_joint_qpos(sim, joint_name, qpos2)

            obs2, reward, done, info = env.step(dummy_action)
            log_obs(tag, obs2)
            record_current_frame(obs2)
            last_obs = obs2
        return last_obs

    def rotate_only(target_quat, n_steps, tag="rot  step"):
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

            obs2, reward, done, info = env.step(dummy_action)
            log_obs(tag, obs2)
            record_current_frame(obs2)
            last_obs = obs2
        return last_obs

    def move_with_rotation(target_pos, target_quat, n_steps, tag="desc step"):
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

            obs2, reward, done, info = env.step(dummy_action)
            log_obs(tag, obs2)
            record_current_frame(obs2)
            last_obs = obs2
        return last_obs

    def hold_with_rotation(target_quat, n_steps, tag="hold step"):
        last_obs = None
        tgt_quat = quat_normalize(target_quat.copy())
        for _ in range(n_steps):
            qpos2 = get_joint_qpos(sim, joint_name)
            qpos2[3:7] = tgt_quat
            set_joint_qpos(sim, joint_name, qpos2)

            obs2, reward, done, info = env.step(dummy_action)
            log_obs(tag, obs2)
            record_current_frame(obs2)
            last_obs = obs2
        return last_obs

    # hold
    for _ in range(4):
        obs, reward, done, info = env.step(dummy_action)
        record_current_frame(obs)

    # phase 1: move above slot
    obs = move_only(hover_pos, n_steps=18, tag="move step")

    # phase 2: rotate to align with slot
    obs = rotate_only(slot_quat, n_steps=14, tag="rot  step")

    # phase 3: descend while continuing rotation correction
    obs = move_with_rotation(insert_pos, slot_quat, n_steps=16, tag="desc step")

    # phase 4: hold while pinning orientation
    obs = hold_with_rotation(slot_quat, n_steps=8, tag="hold step")

    success, xy_dist, z_dist, angle_deg = custom_success(
        obs,
        cfg["xy_thresh"],
        cfg["z_thresh"],
        cfg["angle_thresh_deg"],
    )
    print(f"final success={success} xy_dist={xy_dist:.4f} z_dist={z_dist:.4f} angle_deg={angle_deg:.2f}")

    mp4_path = os.path.join(out_dir, f"scripted_rollout_rot_v2_{ts}.mp4")
    imageio.mimwrite(mp4_path, frames, fps=10)
    print("saved:", mp4_path)

    env.close()
    print("done")

if __name__ == "__main__":
    main()
