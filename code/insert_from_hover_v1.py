import os
from datetime import datetime
import numpy as np
import imageio

from libero.libero.envs import OffScreenRenderEnv

BASE_DIR = "/root/autodl-tmp/openpi_earbud_proto/third_party/libero/libero/libero/bddl_files/libero_90"
BDDL = os.path.join(BASE_DIR, "LIVING_ROOM_SCENE1_insert_the_earbud_into_the_charging_slot_easy.bddl")

CAMERA_SIZE = 512
HOVER_Z_OFFSET = 0.08
DESCEND_TOTAL = 0.03
DESCEND_STEPS = 30

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

    slot_pos = obs["charging_slot_1_pos"].copy()
    slot_pos[2] = max(slot_pos[2], 0.4680)
    slot_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    hover_pos = slot_pos.copy()
    hover_pos[2] = slot_pos[2] + HOVER_Z_OFFSET
    hover_quat = np.array([0.70710678, 0.0, -0.70710678, 0.0], dtype=float)

    frames = []

    def enforce(slot_pose=True, ear_pose=None):
        if slot_pose:
            q_slot = get_joint_qpos(sim, slot_joint)
            q_slot[:3] = slot_pos
            q_slot[3:7] = slot_quat
            set_joint_qpos(sim, slot_joint, q_slot)
            set_joint_qvel_zero(sim, slot_joint)

        if ear_pose is not None:
            q_ear = get_joint_qpos(sim, earbud_joint)
            q_ear[:3] = ear_pose[:3]
            q_ear[3:7] = hover_quat
            set_joint_qpos(sim, earbud_joint, q_ear)
            set_joint_qvel_zero(sim, earbud_joint)

    # 初始化到 hover 位
    for _ in range(15):
        enforce(slot_pose=True, ear_pose=hover_pos)
        obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
        enforce(slot_pose=True, ear_pose=hover_pos)
        frames.append(obs["agentview_image"][::-1])

    print("slot_pos:", np.round(slot_pos, 6))
    print("hover_pos:", np.round(hover_pos, 6))

    # 逐步下探
    cur = hover_pos.copy()
    dz = DESCEND_TOTAL / DESCEND_STEPS

    for i in range(DESCEND_STEPS):
        cur[2] -= dz
        enforce(slot_pose=True, ear_pose=cur)
        obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
        enforce(slot_pose=True, ear_pose=cur)
        frames.append(obs["agentview_image"][::-1])

        earbud_pos = obs["earbud_1_pos"]
        slot_now = obs["charging_slot_1_pos"]
        xy = np.linalg.norm(earbud_pos[:2] - slot_now[:2])
        zrel = earbud_pos[2] - slot_now[2]
        print(f"step={i:02d} xy={xy:.4f} zrel={zrel:.4f}")

    out_dir = "/root/autodl-tmp/openpi_earbud_proto/insert_hover"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    png_path = os.path.join(out_dir, f"insert_hover_{ts}.png")
    mp4_path = os.path.join(out_dir, f"insert_hover_{ts}.mp4")

    imageio.imwrite(png_path, frames[0])
    imageio.mimwrite(mp4_path, frames, fps=10)

    print("saved:", png_path)
    print("saved:", mp4_path)
    env.close()

if __name__ == "__main__":
    main()
