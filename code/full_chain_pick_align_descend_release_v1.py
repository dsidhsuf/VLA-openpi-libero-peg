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

# ===== 直接沿用你成功基线的核心参数 =====
KP_POS = 2.0
POS_CLIP = 0.08

SAFE_TRAVEL_Z = 0.62
PREGRASP_Z_OFFSET = 0.08
PRECLOSE_Z_OFFSET = 0.028
CAGE_Z_OFFSET = 0.014
LIFT_Z_OFFSET = 0.10

SLOT_HOVER_Z_OFFSET = 0.08

GRASP_X_OFFSET = -0.010
GRASP_Y_OFFSET = 0.000

MAX_SERVO_STEPS = 160
PRECLOSE_STEPS = 28
CLOSE_GRIPPER_STEPS = 40
POST_MOVE_HOLD_STEPS = 16

GRIP_OPEN = -1.0
GRIP_CLOSE = 1.0

# ===== 新增：释放阶段参数 =====
RELEASE_DESCEND_Z = 0.006      # 从 hover 再往下走 1.2 cm
PRE_RELEASE_HOLD_STEPS = 8     # 松开前先稳一下
RELEASE_HOLD_STEPS = 30        # 松开后等待自然下落
RETREAT_Z = 0.05               # 释放后夹爪上抬撤离
TARGET_OPEN_ABS = 0.039        # 张开到接近成功版的开口

# ===== 采用你校准后的 slot 补偿 =====
SLOT_Y_DEG = -12.0


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

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"/root/autodl-tmp/openpi_earbud_proto/full_chain_release_v1/{level}"
    os.makedirs(out_dir, exist_ok=True)

    obs = env.reset()
    print("reset ok")

    sim = get_sim(env)
    earbud_joint_name = get_joint_name(env, "earbud_1")
    slot_joint_name = get_joint_name(env, "charging_slot_1")

    frames = []

    # ===== 成功基线的初始化 =====
    earbud_stable_pos = obs["earbud_1_pos"].copy()
    earbud_stable_pos[2] = 0.4435
    earbud_stable_quat = np.array([0.70710678, 0.0, -0.70710678, 0.0], dtype=float)

    slot_stable_pos = obs["charging_slot_1_pos"].copy()
    slot_stable_pos[2] = max(slot_stable_pos[2], 0.4680)

    theta = np.deg2rad(SLOT_Y_DEG)
    # qpos 使用 wxyz
    slot_stable_quat = np.array([np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0], dtype=float)

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

    def debug_state(tag):
        eef_pos = obs["robot0_eef_pos"]
        earbud_pos = obs["earbud_1_pos"]
        slot_pos = obs["charging_slot_1_pos"]
        grip = obs["robot0_gripper_qpos"]

        eef_obj_dist = np.linalg.norm(eef_pos - earbud_pos)
        obj_slot_xy = np.linalg.norm(earbud_pos[:2] - slot_pos[:2])
        obj_slot_z = earbud_pos[2] - slot_pos[2]

        print(
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

    def command_gripper_to_target(target_pos, command, target_abs, mode="open", max_steps=80):
        for _ in range(max_steps):
            action = make_pos_only_action(target_pos, command)
            step_and_record(action)
            cur = gripper_abs()
            if mode == "open" and cur >= target_abs:
                break
            if mode == "close" and cur <= target_abs:
                break

    # ===== reset 后固定 earbud 和 slot，和基线一致 =====
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

    print("earbud_pos0:", np.round(earbud_pos0, 6))
    print("slot_pos0:", np.round(slot_pos0, 6))
    print("eef_pos0:", np.round(eef_pos0, 6))

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

    print("[phase 0.5] air close")
    for _ in range(18):
        step_and_record(make_pos_only_action(obs["robot0_eef_pos"], GRIP_CLOSE))
    debug_state("after_air_close")

    print("[phase 0.6] air open")
    for _ in range(18):
        step_and_record(make_pos_only_action(obs["robot0_eef_pos"], GRIP_OPEN))
    debug_state("after_air_open")

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

    # ===== 搬运到槽口上方：完全沿用成功基线 =====
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

    print("[phase 8] move to safe above slot")
    servo_to_pos(safe_above_slot, GRIP_CLOSE, steps=260, pos_tol=0.005)
    debug_state("after_safe_slot")

    print("[phase 9] lower above slot")
    servo_to_pos(desired_eef_hover, GRIP_CLOSE, steps=260, pos_tol=0.004)
    debug_state("after_slot_hover")

    print("[phase 10] hold above slot")
    for _ in range(POST_MOVE_HOLD_STEPS):
        step_and_record(make_pos_only_action(desired_eef_hover, GRIP_CLOSE))
    debug_state("after_slot_hold")

    # ===== 新增：下降到释放高度 =====
    desired_eef_release = desired_eef_hover.copy()
    desired_eef_release[2] -= RELEASE_DESCEND_Z

    print("[phase 11] descend to release")
    servo_to_pos(desired_eef_release, GRIP_CLOSE, steps=180, pos_tol=0.0035)
    debug_state("after_release_descend")

    print("[phase 12] pre-release hold")
    for _ in range(PRE_RELEASE_HOLD_STEPS):
        step_and_record(make_pos_only_action(desired_eef_release, GRIP_CLOSE))
    debug_state("after_pre_release_hold")

    print("[phase 13] half open gripper")
    command_gripper_to_target(desired_eef_release, GRIP_OPEN, 0.022, mode="open", max_steps=50)
    debug_state("after_half_open")

    print("[phase 14] half-open hold")
    for _ in range(8):
        step_and_record(make_pos_only_action(desired_eef_release, GRIP_OPEN))
    debug_state("after_half_open_hold")

    print("[phase 15] fully open gripper")
    command_gripper_to_target(desired_eef_release, GRIP_OPEN, TARGET_OPEN_ABS, mode="open", max_steps=80)
    debug_state("after_open")

    print("[phase 16] wait for natural drop")
    for _ in range(RELEASE_HOLD_STEPS):
        step_and_record(make_pos_only_action(desired_eef_release, GRIP_OPEN))
    debug_state("after_drop_hold")

    print("[phase 17] retreat upward")
    retreat_pos = obs["robot0_eef_pos"].copy()
    retreat_pos[2] += RETREAT_Z
    servo_to_pos(retreat_pos, GRIP_OPEN, steps=160, pos_tol=0.004)
    debug_state("after_retreat")

    # ===== 最终指标 =====
    earbud_pos_final = obs["earbud_1_pos"].copy()
    slot_pos_final = obs["charging_slot_1_pos"].copy()
    eef_pos_final = obs["robot0_eef_pos"].copy()

    z_lift = earbud_pos_final[2] - earbud_pos0[2]
    eef_obj_dist = np.linalg.norm(eef_pos_final - earbud_pos_final)
    obj_slot_xy = np.linalg.norm(earbud_pos_final[:2] - slot_pos_final[:2])
    obj_slot_z = earbud_pos_final[2] - slot_pos_final[2]

    # 这里只做一个宽松的“释放到槽口区域”判定
    release_drop_success = (obj_slot_xy < 0.02) and (obj_slot_z < 0.06)

    print(f"\nfinal release_drop_success={release_drop_success}")
    print(f"earbud_z_initial={earbud_pos0[2]:.4f}")
    print(f"earbud_z_final={earbud_pos_final[2]:.4f}")
    print(f"z_lift_vs_initial={z_lift:.4f}")
    print(f"eef_obj_dist={eef_obj_dist:.4f}")
    print(f"obj_slot_xy={obj_slot_xy:.4f}")
    print(f"obj_slot_z={obj_slot_z:.4f}")

    init_path = os.path.join(out_dir, f"init_{ts}.png")
    video_path = os.path.join(out_dir, f"full_chain_pick_align_descend_release_v1_{ts}.mp4")

    imageio.imwrite(init_path, frames[0])
    imageio.mimwrite(video_path, frames, fps=20)

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
