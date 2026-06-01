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

KP_POS = 2.0
POS_CLIP = 0.08

SAFE_TRAVEL_Z = 0.62
PREGRASP_Z_OFFSET = 0.09
PRECLOSE_Z_OFFSET = 0.035
CAGE_Z_OFFSET = 0.020
LIFT_Z_OFFSET = 0.11

SLOT_HOVER_Z_OFFSET = 0.09
SLOT_INSERT_Z_OFFSET = 0.020

GRASP_X_OFFSET = -0.008
GRASP_Y_OFFSET = 0.000

MAX_SERVO_STEPS = 180
PRECLOSE_STEPS = 28
CLOSE_GRIPPER_STEPS = 40
INSERT_HOLD_STEPS = 18

GRIP_OPEN = -1.0
GRIP_CLOSE = 1.0


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
    out_dir = f"/root/autodl-tmp/openpi_earbud_proto/robot_pick_insert_demo_success/{level}"
    os.makedirs(out_dir, exist_ok=True)

    obs = env.reset()
    print("reset ok")

    sim = get_sim(env)
    earbud_joint_name = get_joint_name(env, "earbud_1")
    slot_joint_name = get_joint_name(env, "charging_slot_1")

    frames = []

    # 固定的初始姿态
    earbud_stable_pos = obs["earbud_1_pos"].copy()
    earbud_stable_pos[2] = 0.4635
    earbud_stable_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # 竖直细柱

    slot_stable_pos = obs["charging_slot_1_pos"].copy()
    slot_stable_pos[2] = 0.4680
    slot_stable_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)    # 竖直孔

    pin_earbud = True
    pin_slot = True

    def enforce_objects():
        nonlocal obs
        if pin_earbud:
            q_ear = get_joint_qpos(sim, earbud_joint_name)
            q_ear[:3] = earbud_stable_pos
            q_ear[3:7] = earbud_stable_quat
            set_joint_qpos(sim, earbud_joint_name, q_ear)

        if pin_slot:
            q_slot = get_joint_qpos(sim, slot_joint_name)
            q_slot[:3] = slot_stable_pos
            q_slot[3:7] = slot_stable_quat
            set_joint_qpos(sim, slot_joint_name, q_slot)

    def step_and_record(action):
        nonlocal obs
        enforce_objects()
        obs, reward, done, info = env.step(action)
        enforce_objects()
        if "agentview_image" in obs:
            frames.append(obs["agentview_image"][::-1])

    def debug_state(tag):
        eef_pos = obs["robot0_eef_pos"]
        earbud_pos = obs["earbud_1_pos"]
        slot_pos = obs["charging_slot_1_pos"]
        grip = obs["robot0_gripper_qpos"]
        dist = np.linalg.norm(eef_pos - earbud_pos)
        obj_slot_xy = np.linalg.norm(earbud_pos[:2] - slot_pos[:2])
        obj_slot_z = earbud_pos[2] - slot_pos[2]

        print(
            f"[{tag}] "
            f"eef={np.round(eef_pos,4)} "
            f"earbud={np.round(earbud_pos,4)} "
            f"slot={np.round(slot_pos,4)} "
            f"grip={np.round(grip,4)} "
            f"eef_obj_dist={dist:.4f} "
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

    # 初始化 25 步，稳定物体
    for _ in range(25):
        step_and_record(np.zeros(7, dtype=np.float32))

    frames = []
    if "agentview_image" in obs:
        frames.append(obs["agentview_image"][::-1])

    earbud_pos0 = obs["earbud_1_pos"].copy()
    slot_pos0 = obs["charging_slot_1_pos"].copy()
    eef_pos0 = obs["robot0_eef_pos"].copy()

    print("earbud_pos0:", earbud_pos0)
    print("slot_pos0:", slot_pos0)
    print("eef_pos0:", eef_pos0)

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
    servo_to_pos(safe_above_earbud, GRIP_OPEN, steps=180, pos_tol=0.005)
    debug_state("after_safe_xy")

    print("[phase 2] pregrasp")
    servo_to_pos(pregrasp_pos, GRIP_OPEN, steps=180, pos_tol=0.004)
    debug_state("after_pregrasp")

    print("[phase 3] preclose height")
    servo_to_pos(preclose_pos, GRIP_OPEN, steps=180, pos_tol=0.003)
    debug_state("after_preclose_pos")

    print("[phase 4] preclose gripper")
    for _ in range(PRECLOSE_STEPS):
        step_and_record(make_pos_only_action(preclose_pos, GRIP_CLOSE))
    debug_state("after_preclose")

    # 解除 earbud 固定，开始真实抓取
    pin_earbud = False

    print("[phase 5] descend to cage")
    servo_to_pos(cage_pos, GRIP_CLOSE, steps=220, pos_tol=0.002)
    debug_state("after_descend")

    print("[phase 6] squeeze")
    for _ in range(CLOSE_GRIPPER_STEPS):
        step_and_record(make_pos_only_action(cage_pos, GRIP_CLOSE))
    debug_state("after_close")

    print("[phase 7] lift")
    lift_pos = obs["robot0_eef_pos"].copy() + np.array([0.0, 0.0, LIFT_Z_OFFSET])
    servo_to_pos(lift_pos, GRIP_CLOSE, steps=220, pos_tol=0.003)
    debug_state("after_lift")

    # 相对位姿
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

    # 关键：继续下插
    desired_obj_insert = slot_pos0.copy()
    desired_obj_insert[2] = slot_pos0[2] + SLOT_INSERT_Z_OFFSET
    desired_eef_insert = desired_obj_insert - obj_minus_eef

    print("[phase 10] descend to insert")
    servo_to_pos(desired_eef_insert, GRIP_CLOSE, steps=280, pos_tol=0.003)
    debug_state("after_insert_descend")

    print("[phase 11] hold in slot")
    for _ in range(INSERT_HOLD_STEPS):
        step_and_record(make_pos_only_action(desired_eef_insert, GRIP_CLOSE))
    debug_state("after_insert_hold")

    earbud_pos_final = obs["earbud_1_pos"].copy()
    eef_pos_final = obs["robot0_eef_pos"].copy()

    z_lift = earbud_pos_final[2] - earbud_pos0[2]
    eef_obj_dist = np.linalg.norm(eef_pos_final - earbud_pos_final)
    obj_slot_xy = np.linalg.norm(earbud_pos_final[:2] - slot_pos0[:2])
    obj_slot_z = earbud_pos_final[2] - slot_pos0[2]

    grasp_success = (z_lift > 0.03) and (eef_obj_dist < 0.08)
    carry_success = grasp_success and (obj_slot_xy < 0.04)
    insert_success = carry_success and (obj_slot_xy < 0.015) and (abs(obj_slot_z - SLOT_INSERT_Z_OFFSET) < 0.02)

    print(f"\nfinal grasp_success={grasp_success}")
    print(f"final carry_success={carry_success}")
    print(f"final insert_success={insert_success}")
    print(f"earbud_z_initial={earbud_pos0[2]:.4f}")
    print(f"earbud_z_final={earbud_pos_final[2]:.4f}")
    print(f"z_lift={z_lift:.4f}")
    print(f"eef_obj_dist={eef_obj_dist:.4f}")
    print(f"obj_slot_xy={obj_slot_xy:.4f}")
    print(f"obj_slot_z={obj_slot_z:.4f}")

    init_path = os.path.join(out_dir, f"init_{ts}.png")
    video_path = os.path.join(out_dir, f"robot_pick_insert_demo_success_{ts}.mp4")

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
