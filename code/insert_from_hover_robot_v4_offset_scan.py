import os
from datetime import datetime
import csv

import numpy as np
import imageio

from libero.libero.envs import OffScreenRenderEnv

BASE_DIR = "/root/autodl-tmp/openpi_earbud_proto/third_party/libero/libero/libero/bddl_files/libero_90"
BDDL = os.path.join(BASE_DIR, "LIVING_ROOM_SCENE1_insert_the_earbud_into_the_charging_slot_easy.bddl")

CAMERA_SIZE = 512

KP_POS = 2.0
POS_CLIP = 0.08
SAFE_TRAVEL_Z = 0.62
SLOT_HOVER_Z_OFFSET = 0.08

MAX_SERVO_STEPS = 180
POST_HOVER_HOLD_STEPS = 10
POST_INSERT_HOLD_STEPS = 16

GRIP_CLOSE = 1.0

# 用你 v3 校准成功后的 slot pose
SLOT_FIXED_POS = np.array([0.15016507, -0.11357928, 0.481122], dtype=float)
SLOT_FIXED_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

# 当前代理几何
EARBUD_HALF_LENGTH_Z = 0.028
SLOT_HEIGHT = 0.045
BOTTOM_TOL = 0.003

# 固定当前最优下降量
INSERT_DELTA_Z = 0.050

# 扫描“夹爪抓得更靠上”对应的物体悬挂高度
OFFSET_Z_LIST = [-0.0192, -0.0292, -0.0392, -0.0492, -0.0592]


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


def run_one(offset_z: float, out_root: str):
    env = OffScreenRenderEnv(
        bddl_file_name=BDDL,
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

    frames = []

    OBJ_MINUS_EEF = np.array([0.0077, -0.0001, offset_z], dtype=float)
    earbud_carry_quat = np.array([0.70710678, 0.0, -0.70710678, 0.0], dtype=float)

    def enforce_slot():
        q_slot = get_joint_qpos(sim, slot_joint_name)
        q_slot[:3] = SLOT_FIXED_POS
        q_slot[3:7] = SLOT_FIXED_QUAT_WXYZ
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

    # 初始化稳定
    for _ in range(20):
        step_and_record(np.zeros(7, dtype=np.float32))

    frames = []
    if "agentview_image" in obs:
        frames.append(obs["agentview_image"][::-1])

    slot_pos0 = SLOT_FIXED_POS.copy()

    desired_obj_hover = slot_pos0.copy()
    desired_obj_hover[2] = slot_pos0[2] + SLOT_HOVER_Z_OFFSET
    desired_eef_hover = desired_obj_hover - OBJ_MINUS_EEF

    safe_above_slot = np.array([
        desired_eef_hover[0],
        desired_eef_hover[1],
        SAFE_TRAVEL_Z
    ], dtype=float)

    # 夹爪闭合
    for _ in range(20):
        step_and_record(make_pos_only_action(obs["robot0_eef_pos"], GRIP_CLOSE))

    servo_to_pos(safe_above_slot, GRIP_CLOSE, steps=240, pos_tol=0.005)
    servo_to_pos(desired_eef_hover, GRIP_CLOSE, steps=240, pos_tol=0.004)

    for _ in range(POST_HOVER_HOLD_STEPS):
        step_and_record(make_pos_only_action(desired_eef_hover, GRIP_CLOSE))

    desired_eef_insert = desired_eef_hover.copy()
    desired_eef_insert[2] -= INSERT_DELTA_Z

    servo_to_pos(desired_eef_insert, GRIP_CLOSE, steps=240, pos_tol=0.003)

    for _ in range(POST_INSERT_HOLD_STEPS):
        step_and_record(make_pos_only_action(desired_eef_insert, GRIP_CLOSE))

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

    tag = f"offset_{int(abs(offset_z)*1000):03d}mm"
    png_path = os.path.join(out_root, f"{tag}.png")
    mp4_path = os.path.join(out_root, f"{tag}.mp4")

    imageio.imwrite(png_path, frames[0])
    imageio.mimwrite(mp4_path, frames, fps=10)

    env.close()

    return {
        "offset_z": offset_z,
        "insert_delta_z": INSERT_DELTA_Z,
        "eef_obj_dist": float(eef_obj_dist),
        "obj_slot_xy": float(obj_slot_xy),
        "obj_slot_z": float(obj_slot_z),
        "slot_top_z": float(slot_top_z),
        "slot_bottom_z": float(slot_bottom_z),
        "earbud_bottom_z": float(earbud_bottom_z),
        "entered_hole": bool(entered_hole),
        "bottom_insert_success": bool(bottom_insert_success),
        "png_path": png_path,
        "mp4_path": mp4_path,
    }


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = f"/root/autodl-tmp/openpi_earbud_proto/insert_hover_robot_offset_scan/{ts}"
    os.makedirs(out_root, exist_ok=True)

    rows = []
    for oz in OFFSET_Z_LIST:
        print(f"\n=== testing offset_z = {oz:.4f} m ===")
        result = run_one(oz, out_root)
        rows.append(result)

        print(
            f"xy={result['obj_slot_xy']:.4f} "
            f"zrel={result['obj_slot_z']:.4f} "
            f"earbud_bottom_z={result['earbud_bottom_z']:.4f} "
            f"slot_bottom_z={result['slot_bottom_z']:.4f} "
            f"entered_hole={result['entered_hole']} "
            f"bottom_insert_success={result['bottom_insert_success']}"
        )

    csv_path = os.path.join(out_root, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "offset_z",
                "insert_delta_z",
                "eef_obj_dist",
                "obj_slot_xy",
                "obj_slot_z",
                "slot_top_z",
                "slot_bottom_z",
                "earbud_bottom_z",
                "entered_hole",
                "bottom_insert_success",
                "png_path",
                "mp4_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("\nsaved summary:", csv_path)
    print("done")


if __name__ == "__main__":
    main()
