import os
from datetime import datetime
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
PRE_DROP_HOLD_STEPS = 8
POST_RELEASE_STEPS = 40

GRIP_OPEN = -1.0
GRIP_CLOSE = 1.0

# 直接使用已经校准成功的 slot pose
SLOT_FIXED_POS = np.array([0.15016507, -0.11357928, 0.481122], dtype=float)
SLOT_FIXED_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

# 用你当前扫描里最稳的一组
OBJ_MINUS_EEF = np.array([0.0077, -0.0001, -0.0492], dtype=float)

# 先送入孔口一段距离，再松手
INSERT_DELTA_Z = 0.050

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


def main():
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
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = "/root/autodl-tmp/openpi_earbud_proto/insert_drop"
    os.makedirs(out_dir, exist_ok=True)

    earbud_carry_quat = np.array([0.70710678, 0.0, -0.70710678, 0.0], dtype=float)

    # release 前 earbud 跟随夹爪；release 后不再强制跟随
    follow_payload = True

    def enforce_slot():
        q_slot = get_joint_qpos(sim, slot_joint_name)
        q_slot[:3] = SLOT_FIXED_POS
        q_slot[3:7] = SLOT_FIXED_QUAT_WXYZ
        set_joint_qpos(sim, slot_joint_name, q_slot)
        set_joint_qvel_zero(sim, slot_joint_name)

    def enforce_earbud_follow_eef():
        if not follow_payload:
            return
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
            f"xy={obj_slot_xy:.4f} "
            f"zrel={obj_slot_z:.4f} "
            f"earbud_bottom_z={earbud_bottom_z:.4f} "
            f"slot_bottom_z={slot_bottom_z:.4f} "
            f"entered_hole={entered_hole} "
            f"bottom_insert_success={bottom_insert_success}"
        )

    for _ in range(20):
        step_and_record(np.zeros(7, dtype=np.float32))

    frames = []
    if "agentview_image" in obs:
        frames.append(obs["agentview_image"][::-1])

    desired_obj_hover = SLOT_FIXED_POS.copy()
    desired_obj_hover[2] = SLOT_FIXED_POS[2] + SLOT_HOVER_Z_OFFSET
    desired_eef_hover = desired_obj_hover - OBJ_MINUS_EEF

    safe_above_slot = np.array([
        desired_eef_hover[0],
        desired_eef_hover[1],
        SAFE_TRAVEL_Z
    ], dtype=float)

    print("slot_fixed_pos:", np.round(SLOT_FIXED_POS, 6))
    print("desired_eef_hover:", np.round(desired_eef_hover, 6))

    print("\n[phase 0] close gripper")
    for _ in range(20):
        step_and_record(make_pos_only_action(obs["robot0_eef_pos"], GRIP_CLOSE))
    debug_state("after_air_close")

    print("[phase 1] move to safe above slot")
    servo_to_pos(safe_above_slot, GRIP_CLOSE, steps=240, pos_tol=0.005)
    debug_state("after_safe_slot")

    print("[phase 2] lower to hover")
    servo_to_pos(desired_eef_hover, GRIP_CLOSE, steps=240, pos_tol=0.004)
    debug_state("after_hover")

    desired_eef_insert = desired_eef_hover.copy()
    desired_eef_insert[2] -= INSERT_DELTA_Z

    print("[phase 3] descend partially into hole")
    servo_to_pos(desired_eef_insert, GRIP_CLOSE, steps=240, pos_tol=0.003)
    debug_state("after_partial_insert")

    print("[phase 4] hold before release")
    for _ in range(PRE_DROP_HOLD_STEPS):
        step_and_record(make_pos_only_action(desired_eef_insert, GRIP_CLOSE))
    debug_state("after_pre_release_hold")

    print("[phase 5] open gripper and release")
    follow_payload = False
    for _ in range(20):
        step_and_record(make_pos_only_action(desired_eef_insert, GRIP_OPEN))
    debug_state("after_release")

    print("[phase 6] keep still and let it fall")
    for _ in range(POST_RELEASE_STEPS):
        step_and_record(make_pos_only_action(desired_eef_insert, GRIP_OPEN))
    debug_state("after_drop_settle")

    earbud_pos = obs["earbud_1_pos"].copy()
    slot_pos = SLOT_FIXED_POS.copy()

    obj_slot_xy = np.linalg.norm(earbud_pos[:2] - slot_pos[:2])
    obj_slot_z = earbud_pos[2] - slot_pos[2]

    slot_top_z = slot_pos[2] + SLOT_HEIGHT / 2.0
    slot_bottom_z = slot_pos[2] - SLOT_HEIGHT / 2.0
    earbud_bottom_z = earbud_pos[2] - EARBUD_HALF_LENGTH_Z

    entered_hole = earbud_bottom_z <= slot_top_z
    bottom_insert_success = (obj_slot_xy < 0.015) and (earbud_bottom_z <= slot_bottom_z + BOTTOM_TOL)

    print(f"\nfinal entered_hole={entered_hole}")
    print(f"final bottom_insert_success={bottom_insert_success}")
    print(f"obj_slot_xy={obj_slot_xy:.4f}")
    print(f"obj_slot_z={obj_slot_z:.4f}")
    print(f"earbud_bottom_z={earbud_bottom_z:.4f}")
    print(f"slot_bottom_z={slot_bottom_z:.4f}")

    png_path = os.path.join(out_dir, f"insert_drop_{ts}.png")
    mp4_path = os.path.join(out_dir, f"insert_drop_{ts}.mp4")

    imageio.imwrite(png_path, frames[0])
    imageio.mimwrite(mp4_path, frames, fps=10)

    print("saved:", png_path)
    print("saved:", mp4_path)

    env.close()


if __name__ == "__main__":
    main()
