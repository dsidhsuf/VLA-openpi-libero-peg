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

# 直接沿用 baseline 成功版的控制参数
KP_POS = 2.0
POS_CLIP = 0.08
SAFE_TRAVEL_Z = 0.62
SLOT_HOVER_Z_OFFSET = 0.08

MAX_SERVO_STEPS = 180
POST_HOVER_HOLD_STEPS = 10
POST_INSERT_HOLD_STEPS = 14

GRIP_OPEN = -1.0
GRIP_CLOSE = 1.0

# slot 校准参数：沿用你 v2 脚本的思路
SLOT_Y_DEG = -12.0

# 关键：用“成功 carry 版本”中 after_slot_hover 的 gripper-object 相对位姿近似值
# 来模拟“物体已被夹住并随 gripper 携带”
OBJ_MINUS_EEF = np.array([0.0077, -0.0001, -0.0192], dtype=float)

# 插入时的小步下探量
INSERT_DELTA_Z = 0.010  # 1 cm


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
    out_dir = f"/root/autodl-tmp/openpi_earbud_proto/insert_hover_robot/{level}"
    os.makedirs(out_dir, exist_ok=True)

    obs = env.reset()
    print("reset ok")

    sim = get_sim(env)
    earbud_joint_name = get_joint_name(env, "earbud_1")
    slot_joint_name = get_joint_name(env, "charging_slot_1")

    frames = []

    # ---------- 固定 slot ----------
    slot_stable_pos = obs["charging_slot_1_pos"].copy()
    slot_stable_pos[2] = max(slot_stable_pos[2], 0.4680)

    theta = np.deg2rad(SLOT_Y_DEG)
    # qpos 用 wxyz
    slot_stable_quat = np.array([np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0], dtype=float)

    # earbud 作为“已抓取载荷”跟随 gripper
    # 这里的姿态沿用你之前成功搬运版里稳定的横向姿态
    earbud_carry_quat = np.array([0.70710678, 0.0, -0.70710678, 0.0], dtype=float)

    def enforce_slot():
        q_slot = get_joint_qpos(sim, slot_joint_name)
        q_slot[:3] = slot_stable_pos
        q_slot[3:7] = slot_stable_quat
        set_joint_qpos(sim, slot_joint_name, q_slot)
        set_joint_qvel_zero(sim, slot_joint_name)

    def enforce_earbud_follow_eef():
        eef_pos = obs["robot0_eef_pos"].copy()
        earbud_pos = eef_pos + OBJ_MINUS_EEF

        q_ear = get_joint_qpos(sim, earbud_joint_name)
        q_ear[:3] = earbud_pos
        q_ear[3:7] = earbud_carry_quat
        set_joint_qpos(sim, earbud_joint_name, q_ear)
        set_joint_qvel_zero(sim, earbud_joint_name)

    def step_and_record(action):
        nonlocal obs
        enforce_slot()
        enforce_earbud_follow_eef()
        obs, reward, done, info = env.step(action)
        enforce_slot()
        enforce_earbud_follow_eef()
        if "agentview_image" in obs:
            frames.append(obs["agentview_image"][::-1])

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

    # 初始化几步，让 slot 和 payload 稳定
    for _ in range(20):
        step_and_record(np.zeros(7, dtype=np.float32))

    frames = []
    if "agentview_image" in obs:
        frames.append(obs["agentview_image"][::-1])

    slot_pos0 = obs["charging_slot_1_pos"].copy()
    eef_pos0 = obs["robot0_eef_pos"].copy()

    desired_obj_hover = slot_pos0.copy()
    desired_obj_hover[2] = slot_pos0[2] + SLOT_HOVER_Z_OFFSET

    desired_eef_hover = desired_obj_hover - OBJ_MINUS_EEF

    safe_above_slot = np.array([
        desired_eef_hover[0],
        desired_eef_hover[1],
        SAFE_TRAVEL_Z
    ], dtype=float)

    print("slot_pos0:", np.round(slot_pos0, 6))
    print("eef_pos0:", np.round(eef_pos0, 6))
    print("desired_eef_hover:", np.round(desired_eef_hover, 6))

    print("\n[phase 0] close gripper in air")
    for _ in range(20):
        step_and_record(make_pos_only_action(obs["robot0_eef_pos"], GRIP_CLOSE))
    debug_state("after_air_close")

    print("[phase 1] move to safe above slot")
    servo_to_pos(safe_above_slot, GRIP_CLOSE, steps=240, pos_tol=0.005)
    debug_state("after_safe_slot")

    print("[phase 2] lower to hover")
    servo_to_pos(desired_eef_hover, GRIP_CLOSE, steps=240, pos_tol=0.004)
    debug_state("after_hover")

    print("[phase 3] hold at hover")
    for _ in range(POST_HOVER_HOLD_STEPS):
        step_and_record(make_pos_only_action(desired_eef_hover, GRIP_CLOSE))
    debug_state("after_hover_hold")

    desired_eef_insert = desired_eef_hover.copy()
    desired_eef_insert[2] -= INSERT_DELTA_Z

    print("[phase 4] descend to insert")
    servo_to_pos(desired_eef_insert, GRIP_CLOSE, steps=220, pos_tol=0.003)
    debug_state("after_insert")

    print("[phase 5] hold after insert")
    for _ in range(POST_INSERT_HOLD_STEPS):
        step_and_record(make_pos_only_action(desired_eef_insert, GRIP_CLOSE))
    debug_state("after_insert_hold")

    earbud_pos_final = obs["earbud_1_pos"].copy()
    slot_pos_final = obs["charging_slot_1_pos"].copy()
    eef_pos_final = obs["robot0_eef_pos"].copy()

    eef_obj_dist = np.linalg.norm(eef_pos_final - earbud_pos_final)
    obj_slot_xy = np.linalg.norm(earbud_pos_final[:2] - slot_pos_final[:2])
    obj_slot_z = earbud_pos_final[2] - slot_pos_final[2]

    hover_insert_success = (obj_slot_xy < 0.015) and (obj_slot_z < SLOT_HOVER_Z_OFFSET - 0.005)

    print(f"\nfinal hover_insert_success={hover_insert_success}")
    print(f"eef_obj_dist={eef_obj_dist:.4f}")
    print(f"obj_slot_xy={obj_slot_xy:.4f}")
    print(f"obj_slot_z={obj_slot_z:.4f}")

    init_path = os.path.join(out_dir, f"init_{ts}.png")
    video_path = os.path.join(out_dir, f"insert_from_hover_robot_v1_{ts}.mp4")

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
