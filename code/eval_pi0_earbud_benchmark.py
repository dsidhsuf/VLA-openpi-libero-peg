import os
import json
import math
import argparse
from typing import Dict, Any

import numpy as np
import torch

from earbud_benchmark_v1 import (
    get_task_specs,
    build_env,
    load_init_states,
    set_state_qpos_qvel,
    compute_success,
)

# ===== 兼容不同 lerobot 版本的 PI0Policy 导入 =====
PI0Policy = None
_import_errors = []

for mod_path, cls_name in [
    ("lerobot.policies.pi0.modeling_pi0", "PI0Policy"),
    ("lerobot.policies.pi0", "PI0Policy"),
    ("lerobot.policies.pi0", "Pi0Policy"),
]:
    try:
        module = __import__(mod_path, fromlist=[cls_name])
        PI0Policy = getattr(module, cls_name)
        break
    except Exception as e:
        _import_errors.append(f"{mod_path}.{cls_name}: {repr(e)}")

if PI0Policy is None:
    raise ImportError(
        "Could not import PI0Policy. Tried:\n" + "\n".join(_import_errors)
    )


def quat_xyzw_to_axis_angle(quat_xyzw: np.ndarray) -> np.ndarray:
    """
    输入: xyzw
    输出: 3 维 axis-angle 向量
    """
    q = np.asarray(quat_xyzw, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)

    x, y, z, w = q
    # 统一到 w >= 0，减少角度跳变
    if w < 0:
        x, y, z, w = -x, -y, -z, -w

    angle = 2.0 * math.acos(max(min(w, 1.0), -1.0))
    s = math.sqrt(max(1.0 - w * w, 0.0))

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


def make_policy_obs(obs: Dict[str, Any], instruction: str, device: torch.device):
    # 官方 LeRobot LIBERO 文档要求：
    # observation.state = 8-dim = eef position(3) + axis-angle(3) + gripper qpos(2)
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    eef_quat_xyzw = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
    eef_aa = quat_xyzw_to_axis_angle(eef_quat_xyzw)
    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)

    state = np.concatenate([eef_pos, eef_aa, gripper_qpos], axis=0).astype(np.float32)
    assert state.shape[0] == 8, f"Expected 8-dim state, got {state.shape}"

    image = np.asarray(obs["agentview_image"], dtype=np.uint8)
    image2 = np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8)

    batch = {
        "observation.state": torch.from_numpy(state).to(device).unsqueeze(0),
        "observation.images.image": torch.from_numpy(image).to(device).unsqueeze(0),
        "observation.images.image2": torch.from_numpy(image2).to(device).unsqueeze(0),
        "task": [instruction],
    }
    return batch


def load_policy(policy_path: str, device: torch.device):
    policy = PI0Policy.from_pretrained(policy_path)
    policy = policy.to(device)
    policy.eval()
    return policy


@torch.inference_mode()
def run_one_episode(policy, env, init_state, instruction, max_steps, device):
    env.reset()
    set_state_qpos_qvel(env, init_state)
    obs = get_latest_obs(env)

    for step in range(max_steps):
        batch = make_policy_obs(obs, instruction, device)

        # 尽量兼容不同版本 policy.select_action 返回类型
        action = policy.select_action(batch)

        if isinstance(action, torch.Tensor):
            action = action.squeeze(0).detach().cpu().numpy()
        elif isinstance(action, dict):
            if "action" in action:
                a = action["action"]
                action = a.squeeze(0).detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
            else:
                raise RuntimeError(f"Unknown dict output from policy.select_action: {list(action.keys())}")
        else:
            action = np.asarray(action)

        action = action.astype(np.float32)
        obs, _, _, _ = env.step(action)
        metrics = compute_success(obs)

        if metrics["success"] > 0.5:
            return True, step + 1, metrics

    metrics = compute_success(obs)
    return False, max_steps, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", type=str, required=True,
                        help="HF model id or local checkpoint path of the base pi0_libero policy")
    parser.add_argument("--camera_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, default="earbud_pi0_eval_results.json")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[info] using device: {device}")

    policy = load_policy(args.policy_path, device)

    all_results = {}

    for task in get_task_specs():
        print(f"\n=== evaluating: {task.name} ===")
        env = build_env(task, camera_size=args.camera_size)
        init_states = load_init_states(task)

        success_count = 0
        episode_logs = []

        for i, init_state in enumerate(init_states):
            success, steps, metrics = run_one_episode(
                policy=policy,
                env=env,
                init_state=init_state,
                instruction=task.language,
                max_steps=task.max_steps,
                device=device,
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
