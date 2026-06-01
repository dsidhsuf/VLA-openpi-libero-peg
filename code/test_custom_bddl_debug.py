import os
import numpy as np
import imageio

from libero.libero.envs import OffScreenRenderEnv

BDDL = "/root/autodl-tmp/openpi_earbud_proto/third_party/libero/libero/libero/bddl_files/libero_90/LIVING_ROOM_SCENE1_insert_the_earbud_into_the_charging_slot.bddl"
OUT_DIR = "/root/autodl-tmp/openpi_earbud_proto/custom_task_debug"

os.makedirs(OUT_DIR, exist_ok=True)

env_args = {
    "bddl_file_name": BDDL,
    "camera_heights": 512,
    "camera_widths": 512,
}

env = OffScreenRenderEnv(**env_args)
env.seed(0)

# 关闭 success/reward 里的自定义判定，先只看几何和运动
if hasattr(env, "env") and hasattr(env.env, "_check_success"):
    env.env._check_success = lambda: False

obs = env.reset()
print("reset ok")
print("obs keys:", list(obs.keys()) if isinstance(obs, dict) else type(obs))

frame = None
if isinstance(obs, dict):
    for k, v in obs.items():
        if "image" in k and hasattr(v, "shape"):
            frame = v
            print("using image key:", k, "shape:", v.shape)
            break

if frame is None:
    raise RuntimeError("No image found in reset obs")

frame = frame[::-1]
png_path = os.path.join(OUT_DIR, "init_debug.png")
imageio.imwrite(png_path, frame)
print("saved:", png_path)

frames = [frame]
dummy_action = np.zeros(7, dtype=np.float32)

for i in range(20):
    obs, reward, done, info = env.step(dummy_action)
    cur = None
    if isinstance(obs, dict):
        for k, v in obs.items():
            if "image" in k and hasattr(v, "shape"):
                cur = v[::-1]
                break
    if cur is not None:
        frames.append(cur)

mp4_path = os.path.join(OUT_DIR, "noop_rollout_debug.mp4")
imageio.mimwrite(mp4_path, frames, fps=10)
print("saved:", mp4_path)

env.close()
print("done")
