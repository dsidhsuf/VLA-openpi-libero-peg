import os
from datetime import datetime
import argparse
import math

import numpy as np
import imageio

from libero.libero.envs import OffScreenRenderEnv

BASE_DIR = "/root/autodl-tmp/openpi_earbud_proto/third_party/libero/libero/libero/bddl_files/libero_90"

LEVEL_CFG = {
    "easy": {
        "bddl": os.path.join(BASE_DIR, "LIVING_ROOM_SCENE1_insert_the_earbud_into_the_charging_slot_easy.bddl"),
    },
    "medium": {
        "bddl": os.path.join(BASE_DIR, "LIVING_ROOM_SCENE1_insert_the_earbud_into_the_charging_slot_medium.bddl"),
    },
    "hard": {
        "bddl": os.path.join(BASE_DIR, "LIVING_ROOM_SCENE1_insert_the_earbud_into_the_charging_slot_hard.bddl"),
    },
}

CAMERA_SIZE = 768

# ===== 基本控制参数 =====
KP_POS = 2.0
POS_CLIP = 0.08

KP_ROT = 1.6
ROT_CLIP = 0.20

POS_CLIP_RELEASE = 0.02
ROT_CLIP_RELEASE = 0.08

SAFE_TRAVEL_Z = 0.62
PREGRASP_Z_OFFSET = 0.08
PRECLOSE_Z_OFFSET = 0.028
CAGE_Z_OFFSET = 0.014
LIFT_Z_OFFSET = 0.10
SLOT_HOVER_Z_OFFSET = 0.08

GRASP_X_OFFSET = -0.010
GRASP_Y_OFFSET = 0.000

MAX_SERVO_STEPS = 180
PRECLOSE_STEPS = 28
CLOSE_GRIPPER_STEPS = 40
POST_MOVE_HOLD_STEPS = 16

GRIP_OPEN = -1.0
GRIP_CLOSE = 1.0

# ===== release 段 =====
RELEASE_DESCEND_Z = 0.030
PRE_RELEASE_HOLD_STEPS = 16
RELEASE_HOLD_STEPS = 24
RETREAT_Z = 0.06
TARGET_OPEN_ABS = 0.039

# ===== slot 校准 =====
SLOT_Y_DEG = -12.0

# ===== 关键：孔的长边目标 yaw =====
# 如果你发现长边/宽边对反了，就把 90.0 改成 -90.0
SLOT_ALIGN_YAW_DEG = 90.0

# ===== 随机初始 yaw 范围 =====
DEFAULT_YAW_MIN = -60.0
DEFAULT_YAW_MAX = 60.0


def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def quat_mul(q1, q2):
    # wxyz
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


def quat_from_axis_angle(axis, deg):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    theta = np.deg2rad(deg)
    w = np.cos(theta / 2.0)
    xyz = axis * np.sin(theta / 2.0)
    return np.array([w, xyz[0], xyz[1], xyz[2]], dtype=float)


def quat_to_rotvec(q):
    q = quat_normalize(q)
    if q[0] < 0:
        q = -q
    w = np.clip(q[0], -1.0, 1.0)
    xyz = q[1:]
    s = np.linalg.norm(xyz)
    if s < 1e-8:
        return np.zeros(3, dtype=float)
    angle = 2.0 * np.arctan2(s, w)
    axis = xyz / s
    return axis * angle


def quat_angle_deg(q):
    rv = quat_to_rotvec(q)
    return np.linalg.norm(rv) * 180.0 / np.pi


def get_sim(env):
    return env.env.sim if hasattr(env, "env") else env.sim


def get_joint_name(env, object_name):
    obj = env.env.objects_dict[object_name] if hasattr(env, "env") else env.objects_dict[object_name]
    joints = getattr(obj, "joints", None)
    if joints and len(joints) > 0:
        return joints[0]
    raise RuntimeError(f"Could not find free joint for {object_name}")


def get_joint_qpos(sim, joint_name):
    return np.array(sim.data.get_joint_qpos(joint_name), dtype=float)


def set_joint_qpos(sim, joint_name, qpos):
    sim.data.set_joint_qpos(joint_name, qpos)
    sim.forward()


def set_joint_qvel_zero(sim, joint_name):
    try:
        qvel = np.array(sim.data.get_joint_qvel(joint_name), dtype=float)
        sim.data.set_joint_qvel(joint_name, np.zeros_like(qvel))
    except Exception:
        pass


def rollout(level: str, seed: int, yaw_min: float, yaw_max: float, xy_jitter: float):
    rng = np.random.RandomState(seed)

    cfg = LEVEL_CFG[level]

    env = OffScreenRenderEnv(
        bddl_file_name=cfg["bddl"],
        camera_heights=CAMERA_SIZE,
        camera_widths=CAMERA_SIZE,
        ignore_done=True,
    )
    env.seed(seed)

    if hasattr(env, "env") and hasattr(env.env, "_check_success"):
        env.env._check_success = lambda: False

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"/root/autodl-tmp/openpi_earbud_proto/full_chain_release_v3/{level}"
    os.makedirs(out_dir, exist_ok=True)

    obs = env.reset()
    print("reset ok")

    sim = get_sim(env)
    earbud_joint_name = get_joint_name(env, "earbud_1")
    slot_joint_name = get_joint_name(env, "charging_slot_1")

    frames = []

    # ===== 初始随机 yaw + 可选小范围 xy 抖动 =====
    earbud_stable_pos = obs["earbud_1_pos"].copy()
    earbud_stable_pos[0] += rng.uniform(-xy_jitter, xy_jitter)
    earbud_stable_pos[1] += rng.uniform(-xy_jitter, xy_jitter)
    earbud_stable_pos[2] = 0.4435

    rand_yaw_deg = rng.uniform(yaw_min, yaw_max)

    # 先竖起来，再加随机 yaw
    q_vertical = np.array([0.70710678, 0.0, -0.70710678, 0.0], dtype=float)
    q_rand_yaw = quat_from_axis_angle([0, 0, 1], rand_yaw_deg)
    earbud_stable_quat = quat_normalize(quat_mul(q_rand_yaw, q_vertical))

    slot_stable_pos = obs["charging_slot_1_pos"].copy()
    slot_stable_pos[2] = max(slot_stable_pos[2], 0.4680)
    slot_stable_quat = quat_from_axis_angle([0, 1, 0], SLOT_Y_DEG)

    def refresh_obs():
        nonlocal obs
        base = env.env if hasattr(env, "env") else env
        if hasattr(base, "_get_observations"):
            try:
                obs = base._get_observations(force_update=True)
            except TypeError:
                obs = base._get_observations()

    def enforce_slot():
        q_slot = get_joint_qpos(sim, slot_joint_name)
        q_slot[:3] = slot_stable_pos
        q_slot[3:7] = slot_stable_quat
        set_joint_qpos(sim, slot_joint_name, q_slot)
        set_joint_qvel_zero(sim, slot_joint_name)

    def step_and_record(action):
        nonlocal obs
        enforce_slot()
        obs, reward, done, info = env.step(action)
        enforce_slot()
        refresh_obs()
        if "agentview_image" in obs:
            frames.append(obs["agentview_image"][::-1])

    def gripper_abs():
        g = np.asarray(obs["robot0_gripper_qpos"], dtype=float)
        return float(np.mean(np.abs(g)))

    def debug_state(tag, target_quat=None):
        eef_pos = obs["robot0_eef_pos"]
        earbud_pos = obs["earbud_1_pos"]
        slot_pos = obs["charging_slot_1_pos"]
        grip = obs["robot0_gripper_qpos"]

        eef_obj_dist = np.linalg.norm(eef_pos - earbud_pos)
        obj_slot_xy = np.linalg.norm(earbud_pos[:2] - slot_pos[:2])
        obj_slot_z = earbud_pos[2] - slot_pos[2]

        msg = (
            f"[{tag}] "
            f"eef={np.round(eef_pos,4)} "
            f"earbud={np.round(earbud_pos,4)} "
            f"slot={np.round(slot_pos,4)} "
            f"grip={np.round(grip,4)} "
            f"grip_abs={gripper_abs():.4f} "
            f"eef_obj_dist={eef_obj_dist:.4f} "
            f"obj_slot_xy={obj_slot_xy:.4f} "
            f"obj_slot_z={obj_slot_z:.4f}"
        )

        if target_quat is not None:
            cur_q = quat_normalize(obs["robot0_eef_quat"])
            err_q = quat_mul(target_quat, quat_inv(cur_q))
            msg += f" rot_err_deg={quat_angle_deg(err_q):.2f}"

        print(msg)

    def make_pose_action(target_pos, target_quat, gripper_cmd, pos_clip=POS_CLIP, rot_clip=ROT_CLIP):
        cur_pos = obs["robot0_eef_pos"]
        cur_q = quat_normalize(obs["robot0_eef_quat"])
        tgt_q = quat_normalize(target_quat)

        pos_err = target_pos - cur_pos
        rot_err_q = quat_mul(tgt_q, quat_inv(cur_q))
        rotvec = quat_to_rotvec(rot_err_q)

        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip(KP_POS * pos_err, -pos_clip, pos_clip)
        action[3:6] = np.clip(KP_ROT * rotvec, -rot_clip, rot_clip)
        action[6] = float(np.clip(gripper_cmd, -1.0, 1.0))
        return action

    def servo_pose(target_pos, target_quat, grip_cmd, steps=MAX_SERVO_STEPS, pos_tol=0.004, rot_tol_deg=3.0):
        for _ in range(steps):
            action = make_pose_action(target_pos, target_quat, grip_cmd, pos_clip=POS_CLIP, rot_clip=ROT_CLIP)
            step_and_record(action)

            pos_err = np.linalg.norm(target_pos - obs["robot0_eef_pos"])
            cur_q = quat_normalize(obs["robot0_eef_quat"])
            rot_err_q = quat_mul(target_quat, quat_inv(cur_q))
            rot_err_deg = quat_angle_deg(rot_err_q)

            if pos_err < pos_tol and rot_err_deg < rot_tol_deg:
                break

    def servo_pose_slow(target_pos, target_quat, grip_cmd, steps=220, pos_tol=0.003, rot_tol_deg=2.0):
        for _ in range(steps):
            action = make_pose_action(
                target_pos,
                target_quat,
                grip_cmd,
                pos_clip=POS_CLIP_RELEASE,
                rot_clip=ROT_CLIP_RELEASE,
            )
            step_and_record(action)

            pos_err = np.linalg.norm(target_pos - obs["robot0_eef_pos"])
            cur_q = quat_normalize(obs["robot0_eef_quat"])
            rot_err_q = quat_mul(target_quat, quat_inv(cur_q))
            rot_err_deg = quat_angle_deg(rot_err_q)

            if pos_err < pos_tol and rot_err_deg < rot_tol_deg:
                break

    def command_gripper_to_target(target_pos, target_quat, command, target_abs, mode="open", max_steps=80):
        for _ in range(max_steps):
            action = make_pose_action(
                target_pos,
                target_quat,
                command,
                pos_clip=POS_CLIP_RELEASE,
                rot_clip=ROT_CLIP_RELEASE,
            )
            step_and_record(action)
            cur = gripper_abs()
            if mode == "open" and cur >= target_abs:
                break
            if mode == "close" and cur <= target_abs:
                break

    # ===== reset 后稳定对象 =====
    for _ in range(25):
        q_ear = get_joint_qpos(sim, earbud_joint_name)
        q_ear[:3] = earbud_stable_pos
        q_ear[3:7] = earbud_stable_quat
        set_joint_qpos(sim, earbud_joint_name, q_ear)
        set_joint_qvel_zero(sim, earbud_joint_name)

        enforce_slot()
        step_and_record(np.zeros(7, dtype=np.float32))

    frames = []
    refresh_obs()
    if "agentview_image" in obs:
        frames.append(obs["agentview_image"][::-1])

    earbud_pos0 = obs["earbud_1_pos"].copy()
    slot_pos0 = obs["charging_slot_1_pos"].copy()
    eef_pos0 = obs["robot0_eef_pos"].copy()
    eef_quat0 = quat_normalize(obs["robot0_eef_quat"])

    print("seed:", seed)
    print("rand_yaw_deg:", round(rand_yaw_deg, 3))
    print("earbud_pos0:", np.round(earbud_pos0, 6))
    print("slot_pos0:", np.round(slot_pos0, 6))
    print("eef_pos0:", np.round(eef_pos0, 6))
    print("eef_quat0:", np.round(eef_quat0, 6))

    # ===== 根据随机 yaw，先算抓取姿态 =====
    q_eef_grasp = quat_normalize(quat_mul(q_rand_yaw, eef_quat0))

    # ===== 对孔的目标姿态 =====
    q_slot_yaw = quat_from_axis_angle([0, 0, 1], SLOT_ALIGN_YAW_DEG)
    q_obj_slot = quat_normalize(quat_mul(q_slot_yaw, q_vertical))

    safe_up_pos = np.array([eef_pos0[0], eef_pos0[1], SAFE_TRAVEL_Z], dtype=float)

    safe_above_earbud = np.array([
        earbud_pos0[0] + GRASP_X_OFFSET,
        earbud_pos0[1] + GRASP_Y_OFFSET,
        SAFE_TRAVEL_Z
    ], dtype=float)

    pregrasp_pos = np.array([
        earbud_pos0[0] + GRASP_X_OFFSET,
        earbud_pos0[1] + GRASP_Y_OFFSET,
        earbud_pos0[2] + PREGRASP_Z_OFFSET
    ], dtype=float)

    preclose_pos = np.array([
        earbud_pos0[0] + GRASP_X_OFFSET,
        earbud_pos0[1] + GRASP_Y_OFFSET,
        earbud_pos0[2] + PRECLOSE_Z_OFFSET
    ], dtype=float)

    cage_pos = np.array([
        earbud_pos0[0] + GRASP_X_OFFSET,
        earbud_pos0[1] + GRASP_Y_OFFSET,
        earbud_pos0[2] + CAGE_Z_OFFSET
    ], dtype=float)

    print("\n[phase 0] rise to safe height")
    servo_pose(safe_up_pos, eef_quat0, GRIP_OPEN, steps=100, pos_tol=0.006, rot_tol_deg=2.0)
    debug_state("after_safe_up", eef_quat0)

    print("[phase 1] rotate and move above object")
    servo_pose(safe_above_earbud, q_eef_grasp, GRIP_OPEN, steps=200, pos_tol=0.005, rot_tol_deg=3.0)
    debug_state("after_safe_xy", q_eef_grasp)

    print("[phase 2] pregrasp")
    servo_pose(pregrasp_pos, q_eef_grasp, GRIP_OPEN, steps=180, pos_tol=0.004, rot_tol_deg=3.0)
    debug_state("after_pregrasp", q_eef_grasp)

    print("[phase 3] preclose height")
    servo_pose(preclose_pos, q_eef_grasp, GRIP_OPEN, steps=180, pos_tol=0.003, rot_tol_deg=2.0)
    debug_state("after_preclose_pos", q_eef_grasp)

    print("[phase 4] preclose gripper")
    for _ in range(PRECLOSE_STEPS):
        action = make_pose_action(preclose_pos, q_eef_grasp, GRIP_CLOSE)
        step_and_record(action)
    debug_state("after_preclose", q_eef_grasp)

    print("[phase 5] descend to cage")
    servo_pose(cage_pos, q_eef_grasp, GRIP_CLOSE, steps=200, pos_tol=0.002, rot_tol_deg=2.0)
    debug_state("after_descend", q_eef_grasp)

    print("[phase 6] squeeze")
    for _ in range(CLOSE_GRIPPER_STEPS):
        action = make_pose_action(cage_pos, q_eef_grasp, GRIP_CLOSE)
        step_and_record(action)
    debug_state("after_close", q_eef_grasp)

    # 抓取后计算“物体相对末端”的姿态偏移，用于后续旋转对齐孔
    q_eef_close = quat_normalize(obs["robot0_eef_quat"])
    q_obj_close = quat_normalize(obs["earbud_1_quat"])
    q_obj_in_ee = quat_normalize(quat_mul(quat_inv(q_eef_close), q_obj_close))

    print("q_obj_in_ee:", np.round(q_obj_in_ee, 6))

    print("[phase 7] lift")
    lift_pos = obs["robot0_eef_pos"].copy() + np.array([0.0, 0.0, LIFT_Z_OFFSET])
    servo_pose(lift_pos, q_eef_grasp, GRIP_CLOSE, steps=180, pos_tol=0.003, rot_tol_deg=2.0)
    debug_state("after_lift", q_eef_grasp)

    # ===== 根据 slot 目标姿态，反推末端目标姿态 =====
    q_eef_slot = quat_normalize(quat_mul(q_obj_slot, quat_inv(q_obj_in_ee)))

    earbud_pos_lift = obs["earbud_1_pos"].copy()
    eef_pos_lift = obs["robot0_eef_pos"].copy()
    obj_minus_eef = earbud_pos_lift - eef_pos_lift

    desired_obj_hover = slot_pos0.copy()
    desired_obj_hover[2] = slot_pos0[2] + SLOT_HOVER_Z_OFFSET
    desired_eef_hover = desired_obj_hover - obj_minus_eef

    safe_above_slot = np.array([
        desired_eef_hover[0],
        desired_eef_hover[1],
        SAFE_TRAVEL_Z
    ], dtype=float)

    print("[phase 8] rotate to slot yaw and move above slot")
    servo_pose(safe_above_slot, q_eef_slot, GRIP_CLOSE, steps=260, pos_tol=0.005, rot_tol_deg=3.0)
    debug_state("after_safe_slot", q_eef_slot)

    print("[phase 9] lower above slot")
    servo_pose(desired_eef_hover, q_eef_slot, GRIP_CLOSE, steps=260, pos_tol=0.004, rot_tol_deg=2.0)
    debug_state("after_slot_hover", q_eef_slot)

    print("[phase 10] hold above slot")
    for _ in range(POST_MOVE_HOLD_STEPS):
        action = make_pose_action(desired_eef_hover, q_eef_slot, GRIP_CLOSE)
        step_and_record(action)
    debug_state("after_slot_hold", q_eef_slot)

    desired_eef_release = desired_eef_hover.copy()
    desired_eef_release[2] -= RELEASE_DESCEND_Z

    print("[phase 11] descend to release")
    servo_pose_slow(desired_eef_release, q_eef_slot, GRIP_CLOSE, steps=220, pos_tol=0.003, rot_tol_deg=2.0)
    debug_state("after_release_descend", q_eef_slot)

    print("[phase 12] pre-release closed hold")
    for _ in range(PRE_RELEASE_HOLD_STEPS):
        action = make_pose_action(
            desired_eef_release,
            q_eef_slot,
            GRIP_CLOSE,
            pos_clip=POS_CLIP_RELEASE,
            rot_clip=ROT_CLIP_RELEASE,
        )
        step_and_record(action)
    debug_state("after_pre_release_hold", q_eef_slot)

    print("[phase 13] fully open gripper")
    command_gripper_to_target(
        desired_eef_release,
        q_eef_slot,
        GRIP_OPEN,
        TARGET_OPEN_ABS,
        mode="open",
        max_steps=80,
    )
    debug_state("after_open", q_eef_slot)

    print("[phase 14] wait after release")
    for _ in range(RELEASE_HOLD_STEPS):
        action = make_pose_action(
            desired_eef_release,
            q_eef_slot,
            GRIP_OPEN,
            pos_clip=POS_CLIP_RELEASE,
            rot_clip=ROT_CLIP_RELEASE,
        )
        step_and_record(action)
    debug_state("after_drop_hold", q_eef_slot)

    print("[phase 15] retreat upward")
    retreat_pos = obs["robot0_eef_pos"].copy()
    retreat_pos[2] += RETREAT_Z
    servo_pose(retreat_pos, q_eef_slot, GRIP_OPEN, steps=180, pos_tol=0.004, rot_tol_deg=3.0)
    debug_state("after_retreat", q_eef_slot)

    earbud_pos_final = obs["earbud_1_pos"].copy()
    slot_pos_final = obs["charging_slot_1_pos"].copy()
    eef_pos_final = obs["robot0_eef_pos"].copy()

    z_lift = earbud_pos_final[2] - earbud_pos0[2]
    eef_obj_dist = np.linalg.norm(eef_pos_final - earbud_pos_final)
    obj_slot_xy = np.linalg.norm(earbud_pos_final[:2] - slot_pos_final[:2])
    obj_slot_z = earbud_pos_final[2] - slot_pos_final[2]

    release_drop_success = (obj_slot_xy < 0.02) and (obj_slot_z < 0.03)

    print(f"\nfinal release_drop_success={release_drop_success}")
    print(f"earbud_z_initial={earbud_pos0[2]:.4f}")
    print(f"earbud_z_final={earbud_pos_final[2]:.4f}")
    print(f"z_lift_vs_initial={z_lift:.4f}")
    print(f"eef_obj_dist={eef_obj_dist:.4f}")
    print(f"obj_slot_xy={obj_slot_xy:.4f}")
    print(f"obj_slot_z={obj_slot_z:.4f}")

    init_path = os.path.join(out_dir, f"init_seed{seed}_{ts}.png")
    video_path = os.path.join(out_dir, f"full_chain_pick_align_descend_release_v3_seed{seed}_{ts}.mp4")

    imageio.imwrite(init_path, frames[0])
    imageio.mimwrite(video_path, frames, fps=20)

    print("saved:", init_path)
    print("saved:", video_path)

    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--yaw_min", type=float, default=DEFAULT_YAW_MIN)
    parser.add_argument("--yaw_max", type=float, default=DEFAULT_YAW_MAX)
    parser.add_argument("--xy_jitter", type=float, default=0.0)
    args = parser.parse_args()

    rollout(
        level=args.level,
        seed=args.seed,
        yaw_min=args.yaw_min,
        yaw_max=args.yaw_max,
        xy_jitter=args.xy_jitter,
    )


if __name__ == "__main__":
    main()
