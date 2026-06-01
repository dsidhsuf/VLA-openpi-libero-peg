import argparse
import base64
import io
import json
import urllib.request
import urllib.error

import numpy as np
from PIL import Image

from earbud_benchmark_v1 import (
    get_task_specs,
    build_env,
    load_init_states,
    set_state_qpos_qvel,
    compute_success,
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
    aa = axis * angle
    return aa.astype(np.float32)


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
    assert state.shape[0] == 8, f"Expected 8-dim state, got {state.shape}"

    return {
        "task": instruction,
        "observation.state": state.tolist(),
        "observation.images.image": encode_image_to_b64(obs["agentview_image"]),
        "observation.images.image2": encode_image_to_b64(obs["robot0_eye_in_hand_image"]),
    }


def http_json_request(url: str, obj=None, timeout: int = 1800):
    try:
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
            body = resp.read().decode("utf-8")
            return json.loads(body)

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code} from {url}\n{body}") from e


def reset_policy_server(server_url: str):
    return http_json_request(server_url.rstrip("/") + "/reset", obj={})


def infer_action(server_url: str, payload):
    out = http_json_request(server_url.rstrip("/") + "/infer", obj=payload)
    action = np.asarray(out["action"], dtype=np.float32)
    return action


def check_server(server_url: str):
    return http_json_request(server_url.rstrip("/") + "/health", obj=None)


def run_one_episode(env, init_state, instruction, max_steps, server_url: str):
    env.reset()
    set_state_qpos_qvel(env, init_state)
    obs = get_latest_obs(env)

    reset_policy_server(server_url)

    for step in range(max_steps):
        payload = make_payload(obs, instruction)
        action = infer_action(server_url, payload)

        obs, _, _, _ = env.step(action.astype(np.float32))
        metrics = compute_success(obs)

        if metrics["success"] > 0.5:
            return True, step + 1, metrics

    metrics = compute_success(obs)
    return False, max_steps, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--camera_size", type=int, default=512)
    parser.add_argument("--output_json", type=str, default="earbud_pi_eval_results.json")
    args = parser.parse_args()

    health = check_server(args.server)
    print("[client] server health:", health)

    all_results = {}

    for task in get_task_specs():
        print(f"\n=== evaluating: {task.name} ===")
        env = build_env(task, camera_size=args.camera_size)
        init_states = load_init_states(task)

        success_count = 0
        episode_logs = []

        for i, init_state in enumerate(init_states):
            success, steps, metrics = run_one_episode(
                env=env,
                init_state=init_state,
                instruction=task.language,
                max_steps=task.max_steps,
                server_url=args.server,
            )

            success_count += int(success)
            log = {
                "episode_id": i,
                "success": bool(success),
                "steps": int(steps),
                "obj_slot_xy": float(metrics["obj_slot_xy"]),
                "obj_slot_z": float(metrics["obj_slot_z"]),
            }
            episode_logs.append(log)
            print(log)

        success_rate = success_count / len(init_states)
        all_results[task.name] = {
            "success_rate": success_rate,
            "num_episodes": len(init_states),
            "episodes": episode_logs,
        }

        env.close()

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n[saved] {args.output_json}")


if __name__ == "__main__":
    main()
