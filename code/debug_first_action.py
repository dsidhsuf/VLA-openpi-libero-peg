import argparse
import base64
import io
import json
import urllib.request

import numpy as np
from PIL import Image

from earbud_benchmark_v1 import (
    get_task_specs,
    build_env,
    load_init_states,
    set_state_qpos_qvel,
)

def quat_xyzw_to_axis_angle(quat_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    x, y, z, w = q
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
    s = np.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-8 or angle < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = np.array([x, y, z], dtype=np.float64) / s
    return (axis * angle).astype(np.float32)

def get_latest_obs(env):
    base = env.env if hasattr(env, "env") else env
    if hasattr(base, "_get_observations"):
        try:
            return base._get_observations(force_update=True)
        except TypeError:
            return base._get_observations()
    raise RuntimeError("Environment does not expose _get_observations().")

def encode_image_to_b64(img: np.ndarray, quality: int = 90) -> str:
    pil_img = Image.fromarray(img.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def make_payload(obs, instruction: str):
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    eef_quat_xyzw = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
    eef_aa = quat_xyzw_to_axis_angle(eef_quat_xyzw)
    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
    state = np.concatenate([eef_pos, eef_aa, gripper_qpos], axis=0).astype(np.float32)

    return {
        "task": instruction,
        "observation.state": state.tolist(),
        "observation.images.image": encode_image_to_b64(obs["agentview_image"]),
        "observation.images.image2": encode_image_to_b64(obs["robot0_eye_in_hand_image"]),
    }

def http_json_request(url: str, obj=None, timeout: int = 1800):
    if obj is None:
        req = urllib.request.Request(url, method="GET")
    else:
        data = json.dumps(obj).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--task_name", type=str, default="earbud_insert_easy_upright")
    parser.add_argument("--episode_id", type=int, default=0)
    parser.add_argument("--camera_size", type=int, default=512)
    args = parser.parse_args()

    task = None
    for t in get_task_specs():
        if t.name == args.task_name:
            task = t
            break
    if task is None:
        raise ValueError(f"task not found: {args.task_name}")

    env = build_env(task, camera_size=args.camera_size)
    init_states = load_init_states(task)

    env.reset()
    set_state_qpos_qvel(env, init_states[args.episode_id])
    obs = get_latest_obs(env)

    payload = make_payload(obs, task.language)

    print("health:", http_json_request(args.server.rstrip("/") + "/health"))
    print("reset:", http_json_request(args.server.rstrip("/") + "/reset", obj={}))
    out = http_json_request(args.server.rstrip("/") + "/infer", obj=payload)
    action = np.asarray(out["action"], dtype=np.float32)

    print("first_action:", action)
    print("action_norm:", float(np.linalg.norm(action)))
    env.close()

if __name__ == "__main__":
    main()
