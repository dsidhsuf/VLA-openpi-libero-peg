import os
import numpy as np
import imageio

from libero.libero.envs import OffScreenRenderEnv

BDDL = "/root/autodl-tmp/openpi_earbud_proto/third_party/libero/libero/libero/bddl_files/libero_90/LIVING_ROOM_SCENE1_insert_the_earbud_into_the_charging_slot.bddl"
OUT_DIR = "/root/autodl-tmp/openpi_earbud_proto/custom_task_scripted"

os.makedirs(OUT_DIR, exist_ok=True)

env_args = {
    "bddl_file_name": BDDL,
    "camera_heights": 768,
    "camera_widths": 768,
}

def custom_success(obs, xy_thresh=0.03, z_thresh=0.025):
    ep = obs["earbud_1_pos"]
    sp = obs["charging_slot_1_pos"]
    xy_dist = np.linalg.norm(ep[:2] - sp[:2])
    z_dist = abs(ep[2] - sp[2])
    success = (xy_dist < xy_thresh) and (z_dist < z_thresh)
    return success, xy_dist, z_dist

def get_sim(env):
    return env.env.sim if hasattr(env, "env") else env.sim

def get_earbud_joint_name(env):
    obj = env.env.objects_dict["earbud_1"] if hasattr(env, "env") else env.objects_dict["earbud_1"]
    joints = getattr(obj, "joints", None)
    if joints and len(joints) > 0:
        return joints[0]
    raise RuntimeError("Could not find free joint for earbud_1")

def get_joint_qpos(sim, joint_name):
    if hasattr(sim.data, "get_joint_qpos"):
        return np.array(sim.data.get_joint_qpos(joint_name), dtype=float)
    raise RuntimeError("sim.data.get_joint_qpos not available")

def set_joint_qpos(sim, joint_name, qpos):
    if hasattr(sim.data, "set_joint_qpos"):
        sim.data.set_joint_qpos(joint_name, qpos)
        sim.forward()
        return
    raise RuntimeError("sim.data.set_joint_qpos not available")

env = OffScreenRenderEnv(**env_args)
env.seed(0)

# 关闭旧的 LIBERO contain-region 成功判定
if hasattr(env, "env") and hasattr(env.env, "_check_success"):
    env.env._check_success = lambda: False

obs = env.reset()
print("reset ok")

sim = get_sim(env)
joint_name = get_earbud_joint_name(env)
print("earbud joint:", joint_name)

slot_pos = obs["charging_slot_1_pos"].copy()
earbud_pos = obs["earbud_1_pos"].copy()

print("start earbud_pos:", earbud_pos)
print("slot_pos:", slot_pos)

# 轨迹设计：
# phase 1: 静止几帧
# phase 2: 水平移动到槽正上方
# phase 3: 缓慢下降到槽口附近
# phase 4: 保持几帧
hover_pos = slot_pos.copy()
hover_pos[2] = max(earbud_pos[2], slot_pos[2] + 0.05)

insert_pos = slot_pos.copy()
insert_pos[2] = slot_pos[2] + 0.01

frames = []
img = obs["agentview_image"][::-1]
frames.extend([img] * 4)
imageio.imwrite(os.path.join(OUT_DIR, "init_scripted.png"), img)

dummy_action = np.zeros(7, dtype=np.float32)

def record_current_frame(obs):
    if "agentview_image" in obs:
        frames.append(obs["agentview_image"][::-1])

def move_earbud_to(target_pos, n_steps):
    qpos = get_joint_qpos(sim, joint_name)
    cur = qpos[:3].copy()
    for i in range(n_steps):
        alpha = (i + 1) / n_steps
        new_pos = (1 - alpha) * cur + alpha * target_pos
        qpos2 = get_joint_qpos(sim, joint_name)
        qpos2[:3] = new_pos
        set_joint_qpos(sim, joint_name, qpos2)

        obs2, reward, done, info = env.step(dummy_action)
        s, xy, zz = custom_success(obs2)
        print(f"move step success={s} xy_dist={xy:.4f} z_dist={zz:.4f}")
        record_current_frame(obs2)

# phase 1: hold
for _ in range(4):
    obs, reward, done, info = env.step(dummy_action)
    record_current_frame(obs)

# phase 2: move above slot
move_earbud_to(hover_pos, n_steps=18)

# phase 3: move downward into slot
move_earbud_to(insert_pos, n_steps=16)

# phase 4: hold
for _ in range(8):
    obs, reward, done, info = env.step(dummy_action)
    s, xy, zz = custom_success(obs)
    print(f"hold step success={s} xy_dist={xy:.4f} z_dist={zz:.4f}")
    record_current_frame(obs)

success, xy_dist, z_dist = custom_success(obs)
print(f"final success={success} xy_dist={xy_dist:.4f} z_dist={z_dist:.4f}")

mp4_path = os.path.join(OUT_DIR, "scripted_rollout.mp4")
imageio.mimwrite(mp4_path, frames, fps=10)
print("saved:", mp4_path)

env.close()
print("done")
