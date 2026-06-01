import os
from datetime import datetime
import argparse

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

CAMERA_SIZE = 512

# ===== 真实抓取基线参数 =====
KP_POS = 2.0
POS_CLIP = 0.08

SAFE_TRAVEL_Z = 0.62
PREGRASP_Z_OFFSET = 0.08
PRECLOSE_Z_OFFSET = 0.028
CAGE_Z_OFFSET = 0.014
LIFT_Z_OFFSET = 0.10

GRASP_X_OFFSET = -0.010
GRASP_Y_OFFSET = 0.000

MAX_SERVO_STEPS = 180
PRECLOSE_STEPS = 28
CLOSE_GRIPPER_STEPS = 40
POST_HOVER_HOLD_STEPS = 8
PRE_RELEASE_HOLD_STEPS = 8
POST_RELEASE_STEPS = 40

GRIP_OPEN = -1.0
GRIP_CLOSE = 1.0

# ===== 已校准的固定 slot pose =====
SLOT_FIXED_POS = np.array([0.15016507, -0.11357928, 0.481122], dtype=float)
SLOT_FIXED_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

# ===== 插入阶段参数 =====
SLOT_HOVER_Z_OFFSET = 0.08
PARTIAL_INSERT_DELTA_Z = 0.050

# ===== 几何判定 =====
EARBUD_HALF_LENGTH_Z = 0.028
SLOT_HEIGHT = 0.045
BOTTOM_TOL = 0.003


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


def rollout(level: str):
    cfg = LEVEL_CFG[level]

    env = OffScreenRenderEnv(
        bddl_file_name=cfg["bddl"],
        camera_heights=CAMERA_SIZE,
        camera_widths=CAMERA_SIZE,
        ignore_done=True,
    )
    env.seed(0)

    if hasattr(env, "env") and hasattr(env.env, "_check_success"):
        env.env._check_success = lambda: False

    obs = env.reset()
    sim = get_sim(env)

    earbud_joint_name = get_joint_name(env, "earbud_1")
    slot_joint_name = get_joint_name(env, "charging_slot_1")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"/root/autodl-tmp/openpi_earbud_proto/full_chain_pick_insert_drop_physical/{level}"
    os.makedirs(out_dir, exist_ok=True)

    frames = []

    # 初始阶段只固定 slot；earbud 先保持在稳定桌面初始姿态，直到开始真实抓取
    earbud_init_pos = obs["earbud_1_pos"].copy()
    earbud_init_pos[2] = 0.4435
    earbud_init_quat = np.array([0.70710678, 0.0, -0.70710678, 0.0], dtype=float)

    pin_slot = True
    pin_earbud_init = True

    def enforce_slot():
        if not pin_slot:
            return
        q_slot = get_joint_qpos(sim, slot_joint_name)
        q_slot[:3] = SLOT_FIXED_POS
        q_slot[3:7] = SLOT_FIXED_QUAT_WXYZ
        set_joint_qpos(sim, slot_joint_name, q_slot)
        set_joint_qvel_zero(sim, slot_joint_name)

    def enforce_earbud_init():
        if not pin_earbud_init:
            return
        q_ear = get_joint_qpos(sim, earbud_joint_name)
        q_ear[:3] = earbud_init_pos
        q_ear[3:7] = earbud_init_quat
        set_joint_qpos(sim, earbud_joint_name, q_ear)
        set_joint_qvel_zero(sim, earbud_joint_name)

    def step_and_record(action):
        nonlocal obs
        enforce_slot()
        enforce_earbud_init()
        obs, reward, done, info = env.step(action)
        enforce_slot()
        enforce_earbud_init()
        if "agentview_image" in obs:
            frames.append(obs["agentview_image"][::-1])

    def make_pos_only_action(target_pos, gripper_cmd):
        cur_pos = obs["robot0_eef_pos"]
        pos_err = target_pos - cur_pos

        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip(KP_POS * pos_err, -POS_CLIP, POS_CLIP)
        action[3:6] = 0.0
        action[6] = float(np.clip(gripper_cmd, -1.0, 1.0))
        return action

    def servo_to_pos(target_pos, grip_cmd, steps=MAX_SERVO_STEPS, pos_tol=0.004):
        for _ in range(steps):
            action = make_pos_only_action(target_pos, grip_cmd)
            step_and_record(action)
            pos_err = np.linalg.norm(target_pos - obs["robot0_eef_pos"])
            if pos_err < pos_tol:
                break

    def debug_state(tag):
        eef_pos = obs["robot0_eef_pos"]
        earbud_pos = obs["earbud_1_pos"]
        slot_pos = SLOT_FIXED_POS.copy()

        eef_obj_dist = np.linalg.norm(eef_pos - earbud_pos)
        obj_slot_xy = np.linalg.norm(earbud_pos[:2] - slot_pos[:2])
        obj_slot_z = earbud_pos[2] - slot_pos[2]

        slot_top_z = slot_pos[2] + SLOT_HEIGHT / 2.0
        slot_bottom_z = slot_pos[2] - SLOT_HEIGHT / 2.0
        earbud_bottom_z = earbud_pos[2] - EARBUD_HALF_LENGTH_Z

        entered_hole = earbud_bottom_z <= slot_top_z
        bottom_insert_success = (obj_slot_xy < 0.015) and (earbud_bottom_z <= slot_bottom_z + BOTTOM_TOL)

        print(
            f"[{tag}] "
            f"eef={np.round(eef_pos,4)} "
            f"earbud={np.round(earbud_pos,4)} "
            f"eef_obj_dist={eef_obj_dist:.4f} "
            f"xy={obj_slot_xy:.4f} "
            f"zrel={obj_slot_z:.4f} "
            f"earbud_bottom_z={earbud_bottom_z:.4f} "
            f"slot_bottom_z={slot_bottom_z:.4f} "
            f"entered_hole={entered_hole} "
            f"bottom_insert_success={bottom_insert_success}"
        )

    # 初始化稳定
    for _ in range(25):
        step_and_record(np.zeros(7, dtype=np.float32))

    frames = []
    if "agentview_image" in obs:
        frames.append(obs["agentview_image"][::-1])

    earbud_pos0 = obs["earbud_1_pos"].copy()
    eef_pos0 = obs["robot0_eef_pos"].copy()

    print("earbud_pos0:", np.round(earbud_pos0, 6))
    print("eef_pos0:", np.round(eef_pos0, 6))
    print("slot_fixed_pos:", np.round(SLOT_FIXED_POS, 6))

    # ===== 第一阶段：真实抓取 =====
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
    servo_to_pos(safe_up_pos, GRIP_OPEN, steps=100, pos_tol=0.006)
    debug_state("after_safe_up")

    print("[phase 1] move above object")
    servo_to_pos(safe_above_earbud, GRIP_OPEN, steps=160, pos_tol=0.005)
    debug_state("after_safe_xy")

    print("[phase 2] pregrasp")
    servo_to_pos(pregrasp_pos, GRIP_OPEN, steps=160, pos_tol=0.004)
    debug_state("after_pregrasp")

    print("[phase 3] preclose height")
    servo_to_pos(preclose_pos, GRIP_OPEN, steps=160, pos_tol=0.003)
    debug_state("after_preclose_pos")

    print("[phase 4] preclose gripper")
    for _ in range(PRECLOSE_STEPS):
        step_and_record(make_pos_only_action(preclose_pos, GRIP_CLOSE))
    debug_state("after_preclose")

    # 解除 earbud 初始固定，开始真实接触抓取
    pin_earbud_init = False

    print("[phase 5] descend to cage")
    servo_to_pos(cage_pos, GRIP_CLOSE, steps=180, pos_tol=0.002)
    debug_state("after_descend")

    print("[phase 6] squeeze")
    for _ in range(CLOSE_GRIPPER_STEPS):
        step_and_record(make_pos_only_action(cage_pos, GRIP_CLOSE))
    debug_state("after_close")

    print("[phase 7] lift")
    lift_pos = obs["robot0_eef_pos"].copy() + np.array([0.0, 0.0, LIFT_Z_OFFSET])
    servo_to_pos(lift_pos, GRIP_CLOSE, steps=180, pos_tol=0.003)
    debug_state("after_lift")

    # ===== 第二阶段：真实搬运到 slot 上方 =====
    # 使用真实抓取后的当前物体相对位姿，不再人工改写 earbud
    eef_pos_lift = obs["robot0_eef_pos"].copy()
    earbud_pos_lift = obs["earbud_1_pos"].copy()
    obj_minus_eef = earbud_pos_lift - eef_pos_lift

    desired_obj_hover = SLOT_FIXED_POS.copy()
    desired_obj_hover[2] = SLOT_FIXED_POS[2] + SLOT_HOVER_Z_OFFSET
    desired_eef_hover = desired_obj_hover - obj_minus_eef

    safe_above_slot = np.array([
        desired_eef_hover[0],
        desired_eef_hover[1],
        SAFE_TRAVEL_Z
    ], dtype=float)

    print("[phase 8] move to safe above slot")
    servo_to_pos(safe_above_slot, GRIP_CLOSE, steps=260, pos_tol=0.005)
    debug_state("after_safe_slot")

    print("[phase 9] lower to hover")
    servo_to_pos(desired_eef_hover, GRIP_CLOSE, steps=260, pos_tol=0.004)
    debug_state("after_hover")

    for _ in range(POST_HOVER_HOLD_STEPS):
        step_and_record(make_pos_only_action(desired_eef_hover, GRIP_CLOSE))
    debug_state("after_hover_hold")

    # ===== 第三阶段：真实部分插入 =====
    desired_eef_insert = desired_eef_hover.copy()
    desired_eef_insert[2] -= PARTIAL_INSERT_DELTA_Z

    print("[phase 10] descend partially into hole")
    servo_to_pos(desired_eef_insert, GRIP_CLOSE, steps=260, pos_tol=0.003)
    debug_state("after_partial_insert")

    for _ in range(PRE_RELEASE_HOLD_STEPS):
        step_and_record(make_pos_only_action(desired_eef_insert, GRIP_CLOSE))
    debug_state("after_pre_release_hold")

    # ===== 第四阶段：真实松手，自然下落 =====
    print("[phase 11] open gripper and release")
    for _ in range(20):
        step_and_record(make_pos_only_action(desired_eef_insert, GRIP_OPEN))
    debug_state("after_release")

    print("[phase 12] keep still and let it fall")
    for _ in range(POST_RELEASE_STEPS):
        step_and_record(make_pos_only_action(desired_eef_insert, GRIP_OPEN))
    debug_state("after_drop_settle")

    # ===== 最终评估 =====
    earbud_pos = obs["earbud_1_pos"].copy()
    slot_pos = SLOT_FIXED_POS.copy()
    eef_pos = obs["robot0_eef_pos"].copy()

    eef_obj_dist = np.linalg.norm(eef_pos - earbud_pos)
    obj_slot_xy = np.linalg.norm(earbud_pos[:2] - slot_pos[:2])
    obj_slot_z = earbud_pos[2] - slot_pos[2]

    slot_top_z = slot_pos[2] + SLOT_HEIGHT / 2.0
    slot_bottom_z = slot_pos[2] - SLOT_HEIGHT / 2.0
    earbud_bottom_z = earbud_pos[2] - EARBUD_HALF_LENGTH_Z

    entered_hole = earbud_bottom_z <= slot_top_z
    bottom_insert_success = (obj_slot_xy < 0.015) and (earbud_bottom_z <= slot_bottom_z + BOTTOM_TOL)

    print(f"\nfinal entered_hole={entered_hole}")
    print(f"final bottom_insert_success={bottom_insert_success}")
    print(f"eef_obj_dist={eef_obj_dist:.4f}")
    print(f"obj_slot_xy={obj_slot_xy:.4f}")
    print(f"obj_slot_z={obj_slot_z:.4f}")
    print(f"earbud_bottom_z={earbud_bottom_z:.4f}")
    print(f"slot_bottom_z={slot_bottom_z:.4f}")

    png_path = os.path.join(out_dir, f"full_chain_pick_insert_drop_physical_{ts}.png")
    mp4_path = os.path.join(out_dir, f"full_chain_pick_insert_drop_physical_{ts}.mp4")

    imageio.imwrite(png_path, frames[0])
    imageio.mimwrite(mp4_path, frames, fps=10)

    print("saved:", png_path)
    print("saved:", mp4_path)

    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["easy", "medium", "hard"], default="easy")
    args = parser.parse_args()
    rollout(args.level)


if __name__ == "__main__":
    main()
