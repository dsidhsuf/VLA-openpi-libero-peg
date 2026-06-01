import os
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

# ----------- 需要手调的参数 -----------
CAMERA_SIZE = 512

KP_POS = 10.0
KP_ROT = 3.0

PREGRASP_Z_OFFSET = 0.08
GRASP_Z_OFFSET = 0.005
LIFT_Z_OFFSET = 0.10
PRESLOT_Z_OFFSET = 0.08
INSERT_Z_OFFSET = 0.01

MAX_SERVO_STEPS = 60
HOLD_STEPS = 10
CLOSE_GRIPPER_STEPS = 24

# 如果日志显示“闭合阶段夹爪反而张开”，就把这两个值对调
GRIP_OPEN = 1.0
GRIP_CLOSE = -1.0
# ------------------------------------


def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    return q / (np.linalg.norm(q) + 1e-12)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=float)


def quat_inv(q):
    q = quat_normalize(q)
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)


def quat_wxyz_to_rotmat(q):
    q = quat_normalize(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=float)


def rotate_vec(q, v):
    return quat_wxyz_to_rotmat(q) @ np.asarray(v, dtype=float)


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


def quat_to_axis_angle(q):
    q = quat_normalize(q)
    w, x, y, z = q
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
    s = np.sqrt(max(1e-12, 1.0 - w*w))
    axis = np.array([x, y, z]) / s
    return axis * angle


def pose_rot_error(cur_quat, tgt_quat):
    q_err = quat_mul(tgt_quat, quat_inv(cur_quat))
    return quat_to_axis_angle(q_err)


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


def make_action(obs, target_pos, target_quat, gripper_cmd):
    cur_pos = obs["robot0_eef_pos"]
    cur_quat = quat_normalize(obs["robot0_eef_quat"])

    pos_err = target_pos - cur_pos
    rot_err = pose_rot_error(cur_quat, target_quat)

    action = np.zeros(7, dtype=np.float32)
    action[:3] = np.clip(KP_POS * pos_err, -1.0, 1.0)
    action[3:6] = np.clip(KP_ROT * rot_err, -1.0, 1.0)
    action[6] = float(np.clip(gripper_cmd, -1.0, 1.0))
    return action


def rollout(level):
    cfg = LEVEL_CFG[level]

    env = OffScreenRenderEnv(
        bddl_file_name=cfg["bddl"],
        camera_heights=CAMERA_SIZE,
        camera_widths=CAMERA_SIZE,
    )
    env.seed(0)

    # 先关掉 LIBERO 里原来的 contain_region success
    if hasattr(env, "env") and hasattr(env.env, "_check_success"):
        env.env._check_success = lambda: False

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"/root/autodl-tmp/openpi_earbud_proto/robot_pick_insert_v1/{level}"
    os.makedirs(out_dir, exist_ok=True)

    obs = env.reset()
    print("reset ok")

    frames = []
    if "agentview_image" in obs:
        frames.append(obs["agentview_image"][::-1])

    def step_and_record(action):
        nonlocal obs
        obs, reward, done, info = env.step(action)
        if "agentview_image" in obs:
            frames.append(obs["agentview_image"][::-1])
        s, xy, zz, ang = custom_success(
            obs,
            cfg["xy_thresh"],
            cfg["z_thresh"],
            cfg["angle_thresh_deg"],
        )
        return s, xy, zz, ang

    def debug_state(tag, obs):
        eef_pos = obs["robot0_eef_pos"]
        earbud_pos = obs["earbud_1_pos"]
        slot_pos = obs["charging_slot_1_pos"]
        grip = obs["robot0_gripper_qpos"]
        dist = np.linalg.norm(eef_pos - earbud_pos)

        s, xy, zz, ang = custom_success(
            obs,
            cfg["xy_thresh"],
            cfg["z_thresh"],
            cfg["angle_thresh_deg"],
        )

        print(
            f"[{tag}] "
            f"eef={np.round(eef_pos,4)} "
            f"earbud={np.round(earbud_pos,4)} "
            f"slot={np.round(slot_pos,4)} "
            f"grip={np.round(grip,4)} "
            f"eef_obj_dist={dist:.4f} "
            f"xy={xy:.4f} z={zz:.4f} ang={ang:.2f} succ={s}"
        )

    def servo_to_pose(target_pos, target_quat, grip_cmd, steps=MAX_SERVO_STEPS, pos_tol=0.01, rot_tol_deg=10.0):
        last = None
        for _ in range(steps):
            action = make_action(obs, target_pos, target_quat, grip_cmd)
            s, xy, zz, ang = step_and_record(action)

            pos_err = np.linalg.norm(target_pos - obs["robot0_eef_pos"])
            rot_err_deg = angle_deg_between(
                axis_world_from_quat(obs["robot0_eef_quat"], "z"),
                axis_world_from_quat(target_quat, "z"),
            )
            last = (s, xy, zz, ang)

            if pos_err < pos_tol and rot_err_deg < rot_tol_deg:
                break
        return last

    # -------- phase 0: 读取目标 --------
    earbud_pos = obs["earbud_1_pos"].copy()
    earbud_quat = quat_normalize(obs["earbud_1_quat"].copy())

    slot_pos = obs["charging_slot_1_pos"].copy()
    slot_quat = quat_normalize(obs["charging_slot_1_quat"].copy())

    eef_pos0 = obs["robot0_eef_pos"].copy()
    eef_quat0 = quat_normalize(obs["robot0_eef_quat"].copy())

    print("earbud_pos:", earbud_pos)
    print("slot_pos:", slot_pos)
    print("eef_pos0:", eef_pos0)
    print("eef_quat0:", eef_quat0)

    # v1：先用初始末端姿态去抓，先看能否抓住
    grasp_ee_quat = eef_quat0.copy()

    pregrasp_pos = earbud_pos + np.array([0.0, 0.0, PREGRASP_Z_OFFSET])
    grasp_pos = earbud_pos + np.array([0.0, 0.0, GRASP_Z_OFFSET])

    # -------- phase 1: 接近耳机上方 --------
    print("\n[phase 1] pregrasp")
    servo_to_pose(pregrasp_pos, grasp_ee_quat, GRIP_OPEN)
    debug_state("after_pregrasp", obs)

    # -------- phase 2: 下探 --------
    print("[phase 2] descend to grasp")
    servo_to_pose(grasp_pos, grasp_ee_quat, GRIP_OPEN, steps=80, pos_tol=0.006, rot_tol_deg=12.0)
    debug_state("after_descend", obs)

    # -------- phase 3: 闭合夹爪 --------
    print("[phase 3] close gripper")
    for _ in range(CLOSE_GRIPPER_STEPS):
        action = make_action(obs, grasp_pos, grasp_ee_quat, GRIP_CLOSE)
        step_and_record(action)
    debug_state("after_close", obs)

    # 抓取后记录耳机相对末端位姿
    ee_pos = obs["robot0_eef_pos"].copy()
    ee_quat = quat_normalize(obs["robot0_eef_quat"].copy())
    obj_pos = obs["earbud_1_pos"].copy()
    obj_quat = quat_normalize(obs["earbud_1_quat"].copy())

    p_obj_in_ee = rotate_vec(quat_inv(ee_quat), obj_pos - ee_pos)
    q_obj_in_ee = quat_mul(quat_inv(ee_quat), obj_quat)

    print("p_obj_in_ee:", np.round(p_obj_in_ee, 4))
    print("q_obj_in_ee:", np.round(q_obj_in_ee, 4))

    # -------- phase 4: 抬起 --------
    print("[phase 4] lift")
    lift_pos = obs["robot0_eef_pos"].copy() + np.array([0.0, 0.0, LIFT_Z_OFFSET])
    servo_to_pose(lift_pos, ee_quat, GRIP_CLOSE, steps=80, pos_tol=0.008, rot_tol_deg=12.0)
    debug_state("after_lift", obs)

    # -------- phase 5: 移动到槽口上方 --------
    print("[phase 5] move above slot")
    desired_obj_quat = slot_quat.copy()
    desired_ee_quat = quat_mul(desired_obj_quat, quat_inv(q_obj_in_ee))

    preslot_obj_pos = slot_pos + np.array([0.0, 0.0, PRESLOT_Z_OFFSET])
    preslot_ee_pos = preslot_obj_pos - rotate_vec(desired_ee_quat, p_obj_in_ee)

    servo_to_pose(preslot_ee_pos, desired_ee_quat, GRIP_CLOSE, steps=100, pos_tol=0.010, rot_tol_deg=12.0)
    debug_state("after_move_above_slot", obs)

    # -------- phase 6: 下插 --------
    print("[phase 6] insert")
    insert_obj_pos = slot_pos + np.array([0.0, 0.0, INSERT_Z_OFFSET])
    insert_ee_pos = insert_obj_pos - rotate_vec(desired_ee_quat, p_obj_in_ee)

    servo_to_pose(insert_ee_pos, desired_ee_quat, GRIP_CLOSE, steps=100, pos_tol=0.006, rot_tol_deg=8.0)
    debug_state("after_insert", obs)

    # -------- phase 7: 保持 --------
    print("[phase 7] hold")
    last = None
    for _ in range(HOLD_STEPS):
        action = make_action(obs, insert_ee_pos, desired_ee_quat, GRIP_CLOSE)
        last = step_and_record(action)
    debug_state("after_hold", obs)

    s, xy, zz, ang = custom_success(
        obs,
        cfg["xy_thresh"],
        cfg["z_thresh"],
        cfg["angle_thresh_deg"],
    )
    print(f"final success={s} xy_dist={xy:.4f} z_dist={zz:.4f} angle_deg={ang:.2f}")

    init_path = os.path.join(out_dir, f"init_{ts}.png")
    video_path = os.path.join(out_dir, f"robot_pick_insert_v1_{ts}.mp4")

    imageio.imwrite(init_path, frames[0])
    imageio.mimwrite(video_path, frames, fps=10)

    print("saved:", init_path)
    print("saved:", video_path)

    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["easy", "medium", "hard"], default="easy")
    args = parser.parse_args()
    rollout(args.level)


if __name__ == "__main__":
    main()