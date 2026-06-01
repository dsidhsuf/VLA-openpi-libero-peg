import os
import numpy as np
import imageio

from libero.libero.envs import OffScreenRenderEnv

BDDL = "/root/autodl-tmp/openpi_earbud_proto/third_party/libero/libero/libero/bddl_files/libero_90/LIVING_ROOM_SCENE1_insert_the_earbud_into_the_charging_slot.bddl"
OUT_DIR = "/root/autodl-tmp/openpi_earbud_proto/custom_task_eval"

os.makedirs(OUT_DIR, exist_ok=True)

env_args = {
    "bddl_file_name": BDDL,
    "camera_heights": 768,
    "camera_widths": 768,
}

def custom_success(obs, xy_thresh=0.06, z_thresh=0.05):
    ep = obs["earbud_1_pos"]
    sp = obs["charging_slot_1_pos"]

    xy_dist = np.linalg.norm(ep[:2] - sp[:2])
    z_dist = abs(ep[2] - sp[2])

    success = (xy_dist < xy_thresh) and (z_dist < z_thresh)
    return success, xy_dist, z_dist

env = OffScreenRenderEnv(**env_args)
env.seed(0)

# 关闭 LIBERO 内部旧的 contain-region 成功判定
if hasattr(env, "env") and hasattr(env.env, "_check_success"):
    env.env._check_success = lambda: False

obs = env.reset()
print("reset ok")

frames = []
img = obs["agentview_image"][::-1]
frames.extend([img] * 6)
imageio.imwrite(os.path.join(OUT_DIR, "init_eval.png"), img)

dummy_action = np.zeros(7, dtype=np.float32)

for i in range(36):
    success, xy_dist, z_dist = custom_success(obs)
    if i % 3 == 0:
        print(f"step={i:02d} success={success} xy_dist={xy_dist:.4f} z_dist={z_dist:.4f}")

    obs, reward, done, info = env.step(dummy_action)

    if (i % 2 == 0) and ("agentview_image" in obs):
        frames.append(obs["agentview_image"][::-1])

# 最后再检查一次
success, xy_dist, z_dist = custom_success(obs)
print(f"final success={success} xy_dist={xy_dist:.4f} z_dist={z_dist:.4f}")

mp4_path = os.path.join(OUT_DIR, "eval_rollout.mp4")
imageio.mimwrite(mp4_path, frames, fps=10)
print("saved:", mp4_path)

env.close()
print("done")
