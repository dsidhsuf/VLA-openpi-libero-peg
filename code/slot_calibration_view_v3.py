import os
from datetime import datetime
import numpy as np
import imageio

from libero.libero.envs import OffScreenRenderEnv

BASE_DIR = "/root/autodl-tmp/openpi_earbud_proto/third_party/libero/libero/libero/bddl_files/libero_90"
BDDL = os.path.join(BASE_DIR, "LIVING_ROOM_SCENE1_insert_the_earbud_into_the_charging_slot_easy.bddl")

CAMERA_SIZE = 512

# 直接写死一个绝对 slot pose，不再从 obs copy 后微调
SLOT_FIXED_POS = np.array([0.15016507, -0.11357928, 0.4925], dtype=float)
SLOT_FIXED_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

EARBUD_FIXED_POS = np.array([0.07318314, -0.10582792, 0.4435], dtype=float)
EARBUD_FIXED_QUAT_WXYZ = np.array([0.70710678, 0.0, -0.70710678, 0.0], dtype=float)


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

    slot_joint = get_joint_name(env, "charging_slot_1")
    earbud_joint = get_joint_name(env, "earbud_1")

    frames = []

    def enforce():
        q_slot = get_joint_qpos(sim, slot_joint)
        q_slot[:3] = SLOT_FIXED_POS
        q_slot[3:7] = SLOT_FIXED_QUAT_WXYZ
        set_joint_qpos(sim, slot_joint, q_slot)
        set_joint_qvel_zero(sim, slot_joint)

        q_ear = get_joint_qpos(sim, earbud_joint)
        q_ear[:3] = EARBUD_FIXED_POS
        q_ear[3:7] = EARBUD_FIXED_QUAT_WXYZ
        set_joint_qpos(sim, earbud_joint, q_ear)
        set_joint_qvel_zero(sim, earbud_joint)

    for _ in range(80):
        enforce()
        obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
        enforce()
        frames.append(obs["agentview_image"][::-1])

    print("slot_pos:", np.round(obs["charging_slot_1_pos"], 6))
    print("slot_quat:", np.round(obs["charging_slot_1_quat"], 6))
    print("earbud_pos:", np.round(obs["earbud_1_pos"], 6))
    print("earbud_quat:", np.round(obs["earbud_1_quat"], 6))

    out_dir = "/root/autodl-tmp/openpi_earbud_proto/slot_calib"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    png_path = os.path.join(out_dir, f"slot_calib_v3_{ts}.png")
    mp4_path = os.path.join(out_dir, f"slot_calib_v3_{ts}.mp4")

    imageio.imwrite(png_path, frames[0])
    imageio.mimwrite(mp4_path, frames, fps=10)

    print("saved:", png_path)
    print("saved:", mp4_path)
    env.close()


if __name__ == "__main__":
    main()
