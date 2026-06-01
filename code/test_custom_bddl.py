import os
import numpy as np
import imageio

from libero.libero.envs import OffScreenRenderEnv

BDDL = "/root/autodl-tmp/openpi_earbud_proto/third_party/libero/libero/libero/bddl_files/libero_90/LIVING_ROOM_SCENE1_insert_the_earbud_into_the_charging_slot.bddl"
OUT_DIR = "/root/autodl-tmp/openpi_earbud_proto/custom_task_debug"

os.makedirs(OUT_DIR, exist_ok=True)

env_args = {
    "bddl_file_name": BDDL,
    "camera_heights": 128,
    "camera_widths": 128,
}

env = OffScreenRenderEnv(**env_args)
env.seed(0)

obs = env.reset()
print("reset ok")
print("obs keys:", list(obs.keys()) if isinstance(obs, dict) else type(obs))

# 尝试从 observation 里找图像
frame = None
if isinstance(obs, dict):
    for k, v in obs.items():
        if "image" in k and hasattr(v, "shape"):
            frame = v
            print("using image key:", k, "shape:", v.shape)
            break

# 如果 reset 没拿到图像，就 step 一下
if frame is None:
    dummy_action = np.zeros(7, dtype=np.float32)
    obs, reward, done, info = env.step(dummy_action)
    print("step once ok, reward:", reward, "done:", done)
    if isinstance(obs, dict):
        for k, v in obs.items():
            if "image" in k and hasattr(v, "shape"):
                frame = v
                print("using image key after step:", k, "shape:", v.shape)
                break

if frame is None:
    raise RuntimeError("No image found in observations")

# robosuite / mujoco 图像常常上下颠倒，翻一下更自然
frame = frame[::-1]

png_path = os.path.join(OUT_DIR, "init.png")
imageio.imwrite(png_path, frame)
print("saved:", png_path)

# 再录一个很短的 no-op rollout，纯粹验证任务能正常跑
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

mp4_path = os.path.join(OUT_DIR, "noop_rollout.mp4")
imageio.mimwrite(mp4_path, frames, fps=10)
print("saved:", mp4_path)

env.close()
print("done")
