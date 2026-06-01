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

KP_POS = 2.2
POS_CLIP = 0.08

SAFE_TRAVEL_Z = 0.62
PREGRASP_Z_OFFSET = 0.08
PRECLOSE_Z_OFFSET = 0.028
CAGE_Z_OFFSET = 0.014
LIFT_Z_OFFSET = 0.10

GRASP_X_OFFSET = -0.010
GRASP_Y_OFFSET = 0.000

MAX_SERVO_STEPS = 140
PRECLOSE_STEPS = 28
CLOSE_GRIPPER_STEPS = 40

# 这组方向是你当前日志里已经验证过能收紧的
GRIP_OPEN = -1.0
GRIP_CLOSE = 1.0


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


def rollout(level: str):
    cfg = LEVEL_CFG[level]

    env = OffScreenRenderEnv(
        bddl_file_name=cfg["bddl"],
        camera_heights=CAMERA_SIZE,
        camera_widths=CAMERA_SIZE,
    )
    env.seed(0)

    if hasattr(env, "env") and hasattr(env.env, "_check_success"):
        env.env._check_success = lambda: False

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"/root/autodl-tmp/openpi_earbud_proto/robot_grasp_demo_success/{level}"
    os.makedirs(out_dir, exist_ok=True)

    obs = env.reset()
    print("reset ok")

    sim = get_sim(env)
    earbud_joint_name = get_earbud_joint_name(env)

    frames = []
    episode_done = False

    def step_and_record(action):
        nonlocal obs, episode_done
        if episode_done:
            return
        try:
            obs, reward, done, info = env.step(action)
            episode_done = bool(done)
        except ValueError as e:
            if "terminated episode" in str(e):
                episode_done = True
                return
            raise
        if "agentview_image" in obs:
            frames.append(obs["agentview_image"][::-1])

    def debug_state(tag):
        eef_pos = obs["robot0_eef_pos"]
        earbud_pos = obs["earbud_1_pos"]
        grip = obs["robot0_gripper_qpos"]
        dist = np.linalg.norm(eef_pos - earbud_pos)
        print(
            f"[{tag}] "
            f"eef={np.round(eef_pos,4)} "
            f"earbud={np.round(earbud_pos,4)} "
            f"grip={np.round(grip,4)} "
            f"eef_obj_dist={dist:.4f}"
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
        nonlocal episode_done
        for _ in range(steps):
            if episode_done:
                break
            action = make_pos_only_action(target_pos, grip_cmd)
            step_and_record(action)
            if episode_done:
                break
            pos_err = np.linalg.norm(target_pos - obs["robot0_eef_pos"])
            if pos_err < pos_tol:
                break

    # ---------- 强制稳定耳机初始状态 ----------
    qpos = get_joint_qpos(sim, earbud_joint_name)

    stable_pos = obs["earbud_1_pos"].copy()
    stable_pos[2] = 0.4435

    # 这一组姿态是你当前已验证“不提前倒”的
    stable_quat = np.array([0.70710678, 0.0, -0.70710678, 0.0], dtype=float)

    for _ in range(20):
        qpos = get_joint_qpos(sim, earbud_joint_name)
        qpos[:3] = stable_pos
        qpos[3:7] = stable_quat
        set_joint_qpos(sim, earbud_joint_name, qpos)
        step_and_record(np.zeros(7, dtype=np.float32))

    frames = []
    if "agentview_image" in obs:
        frames.append(obs["agentview_image"][::-1])

    earbud_pos0 = obs["earbud_1_pos"].copy()
    eef_pos0 = obs["robot0_eef_pos"].copy()

    print("earbud_pos0:", earbud_pos0)
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

    print("[phase 0.5] air close")
    for _ in range(18):
        if episode_done:
            break
        step_and_record(make_pos_only_action(obs["robot0_eef_pos"], GRIP_CLOSE))
    debug_state("after_air_close")

    print("[phase 0.6] air open")
    for _ in range(18):
        if episode_done:
            break
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
        if episode_done:
            break
        step_and_record(make_pos_only_action(preclose_pos, GRIP_CLOSE))
    debug_state("after_preclose")

    print("[phase 5] descend to cage")
    servo_to_pos(cage_pos, GRIP_CLOSE, steps=180, pos_tol=0.002)
    debug_state("after_descend")

    print("[phase 6] squeeze")
    for _ in range(CLOSE_GRIPPER_STEPS):
        if episode_done:
            break
        step_and_record(make_pos_only_action(cage_pos, GRIP_CLOSE))
    debug_state("after_close")

    print("[phase 7] lift")
    lift_pos = obs["robot0_eef_pos"].copy() + np.array([0.0, 0.0, LIFT_Z_OFFSET])
    servo_to_pos(lift_pos, GRIP_CLOSE, steps=180, pos_tol=0.003)
    debug_state("after_lift")

    # 不再继续 hold，抓起来就退出并保存
    earbud_pos_final = obs["earbud_1_pos"].copy()
    z_lift = earbud_pos_final[2] - earbud_pos0[2]
    eef_obj_dist = np.linalg.norm(obs["robot0_eef_pos"] - earbud_pos_final)

    grasp_success = (z_lift > 0.03) and (eef_obj_dist < 0.08)

    print(f"\nepisode_done={episode_done}")
    print(f"final grasp_success={grasp_success}")
    print(f"earbud_z_initial={earbud_pos0[2]:.4f}")
    print(f"earbud_z_final={earbud_pos_final[2]:.4f}")
    print(f"z_lift={z_lift:.4f}")
    print(f"eef_obj_dist={eef_obj_dist:.4f}")

    init_path = os.path.join(out_dir, f"init_{ts}.png")
    video_path = os.path.join(out_dir, f"robot_grasp_demo_success_{ts}.mp4")

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
