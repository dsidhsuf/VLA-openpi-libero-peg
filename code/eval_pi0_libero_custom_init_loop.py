from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import pathlib
import sys
import time
from typing import Any

import imageio
import numpy as np
import tqdm

from custom_libero_suite import CustomLiberoSuite, CustomTaskSpec


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def _inject_openpi_python_paths(openpi_root: str | None) -> None:
    if openpi_root is None:
        return
    root = pathlib.Path(openpi_root).expanduser().resolve()
    candidates = [
        root,
        root / "src",
        root / "packages" / "openpi-client" / "src",
        root / "third_party" / "libero",
    ]
    for path in candidates:
        if path.exists():
            p = str(path)
            if p not in sys.path:
                sys.path.insert(0, p)


def wrap_deg(x: float) -> float:
    return float((x + 180.0) % 360.0 - 180.0)


def canonical_axis_deg(angle_deg: float) -> float:
    return float(angle_deg % 180.0)


def wrap_axis_err_deg(target_deg: float, current_deg: float) -> float:
    return float((target_deg - current_deg + 90.0) % 180.0 - 90.0)


def quat_to_rotmat_wxyz(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def projected_axis_heading_deg_from_quat_wxyz(q, local_axis) -> float:
    rot = quat_to_rotmat_wxyz(q)
    axis_world = rot @ np.asarray(local_axis, dtype=float)
    axis_xy = axis_world[:2]
    norm_xy = np.linalg.norm(axis_xy)
    if norm_xy < 1e-8:
        return 0.0
    heading = np.rad2deg(np.arctan2(axis_xy[1], axis_xy[0]))
    return canonical_axis_deg(float(heading))


def get_sim(env):
    return env.env.sim if hasattr(env, "env") else env.sim


def get_joint_name(env, object_name: str) -> str:
    base = env.env if hasattr(env, "env") else env
    obj = base.objects_dict[object_name]
    joints = getattr(obj, "joints", None)
    if joints and len(joints) > 0:
        return joints[0]
    raise RuntimeError(f"Could not find free joint for object '{object_name}'")


def _sim_data(sim):
    if hasattr(sim, "data"):
        return sim.data
    if hasattr(sim, "_data"):
        return sim._data
    return None


def get_joint_qpos(sim, joint_name: str) -> np.ndarray:
    data = _sim_data(sim)
    if data is not None and hasattr(data, "get_joint_qpos"):
        return np.asarray(data.get_joint_qpos(joint_name), dtype=float)
    if hasattr(sim, "get_joint_qpos"):
        return np.asarray(sim.get_joint_qpos(joint_name), dtype=float)
    raise AttributeError(
        "Could not read joint qpos from sim. Expected sim.data/sim._data.get_joint_qpos or sim.get_joint_qpos."
    )


def quat_xyzw_to_axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float).copy()
    # robosuite / LIBERO convention in obs is xyzw.
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=float)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate local pi0_libero checkpoint on a custom LIBERO suite with explicit fixed-init-state loops."
        )
    )
    parser.add_argument("--suite-json", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--openpi-config", type=str, default="pi0_libero")
    parser.add_argument("--openpi-root", type=str, default=None)
    parser.add_argument("--default-prompt", type=str, default=None)

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)

    parser.add_argument("--num-trials", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--num-steps-wait", type=int, default=-1)

    parser.add_argument("--allow-env-done-success", action="store_true")
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--video-dir", type=str, default="./custom_eval_videos")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--results-json", type=str, default="./custom_eval_results.json")
    return parser.parse_args()


def evaluate_success(
    obs: dict[str, Any],
    env,
    task: CustomTaskSpec,
    joint_cache: dict[str, str],
) -> tuple[bool, dict[str, float]]:
    cfg = dict(task.success or {})
    mode = str(cfg.get("type", "env_check_success")).strip().lower()
    metrics: dict[str, float] = {}

    if mode in {"env_check_success", "env_success", "env_done"}:
        if hasattr(env, "check_success"):
            success = bool(env.check_success())
        else:
            base = env.env if hasattr(env, "env") else env
            success = bool(base._check_success())
        return success, metrics

    if mode not in {"pose_threshold", "pose", "insert_pose_threshold"}:
        raise ValueError(
            f"Unsupported success.type='{cfg.get('type')}' for task '{task.name}'"
        )

    object_name = str(cfg["object_name"])
    target_name = str(cfg["target_name"])

    object_pos_key = str(cfg.get("object_pos_key", f"{object_name}_pos"))
    target_pos_key = str(cfg.get("target_pos_key", f"{target_name}_pos"))
    obj_pos = np.asarray(obs[object_pos_key], dtype=float)
    tgt_pos = np.asarray(obs[target_pos_key], dtype=float)

    xy_err = float(np.linalg.norm(obj_pos[:2] - tgt_pos[:2]))
    z_above = float(obj_pos[2] - tgt_pos[2])
    metrics["xy_err"] = xy_err
    metrics["z_above"] = z_above

    success = True
    success = success and xy_err <= float(cfg.get("xy_thresh", 0.02))
    success = success and z_above <= float(cfg.get("z_above_max", 0.03))
    if "z_above_min" in cfg:
        success = success and z_above >= float(cfg["z_above_min"])

    if bool(cfg.get("require_gripper_open", False)):
        grip_abs = float(np.mean(np.abs(np.asarray(obs["robot0_gripper_qpos"], dtype=float))))
        metrics["gripper_abs"] = grip_abs
        success = success and grip_abs >= float(cfg.get("gripper_abs_min", 0.03))

    if bool(cfg.get("require_eef_obj_distance", False)):
        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=float)
        eef_obj_dist = float(np.linalg.norm(eef_pos - obj_pos))
        metrics["eef_obj_dist"] = eef_obj_dist
        success = success and eef_obj_dist >= float(cfg.get("eef_obj_dist_min", 0.035))

    yaw_thresh = cfg.get("yaw_thresh_deg", None)
    if yaw_thresh is not None:
        sim = get_sim(env)
        obj_joint_key = f"joint::{object_name}"
        tgt_joint_key = f"joint::{target_name}"
        if obj_joint_key not in joint_cache:
            joint_cache[obj_joint_key] = get_joint_name(env, object_name)
        if tgt_joint_key not in joint_cache:
            joint_cache[tgt_joint_key] = get_joint_name(env, target_name)

        q_obj = get_joint_qpos(sim, joint_cache[obj_joint_key])[3:7]
        q_tgt = get_joint_qpos(sim, joint_cache[tgt_joint_key])[3:7]
        obj_axis_local = np.asarray(cfg.get("object_long_axis_local", [0.0, 0.0, 1.0]), dtype=float)
        tgt_axis_local = np.asarray(cfg.get("target_long_axis_local", [0.0, 0.0, 1.0]), dtype=float)
        target_axis_yaw_offset_deg = float(cfg.get("target_axis_yaw_offset_deg", 0.0))

        obj_axis_deg = projected_axis_heading_deg_from_quat_wxyz(q_obj, obj_axis_local)
        tgt_axis_deg = projected_axis_heading_deg_from_quat_wxyz(q_tgt, tgt_axis_local)
        target_axis_deg = canonical_axis_deg(tgt_axis_deg + target_axis_yaw_offset_deg)
        axis_err_deg = abs(wrap_axis_err_deg(target_axis_deg, obj_axis_deg))
        metrics["axis_err_deg"] = float(axis_err_deg)
        success = success and axis_err_deg <= float(yaw_thresh)

    return bool(success), metrics


def make_policy_input(
    obs: dict[str, Any],
    prompt: str,
    resize_size: int,
    image_tools_module,
) -> dict[str, Any]:
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    if "robot0_eye_in_hand_image" in obs:
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    else:
        wrist_img = np.zeros_like(img)

    img = image_tools_module.convert_to_uint8(
        image_tools_module.resize_with_pad(img, resize_size, resize_size)
    )
    wrist_img = image_tools_module.convert_to_uint8(
        image_tools_module.resize_with_pad(wrist_img, resize_size, resize_size)
    )

    state = np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=float),
            quat_xyzw_to_axisangle(np.asarray(obs["robot0_eef_quat"], dtype=float)),
            np.asarray(obs["robot0_gripper_qpos"], dtype=float),
        ),
        axis=0,
    )

    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "prompt": prompt,
    }


def main() -> None:
    args = parse_args()
    _inject_openpi_python_paths(args.openpi_root)

    from libero.libero.envs import OffScreenRenderEnv
    from openpi.policies import policy_config as policy_config
    from openpi.training import config as train_config
    from openpi_client import image_tools

    suite = CustomLiberoSuite.from_json(args.suite_json)
    checkpoint_dir = pathlib.Path(args.checkpoint_dir).expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint-dir does not exist: {checkpoint_dir}")

    policy = policy_config.create_trained_policy(
        train_config.get_config(args.openpi_config),
        checkpoint_dir,
        default_prompt=args.default_prompt,
    )
    logging.info(
        "Loaded policy config=%s checkpoint=%s", args.openpi_config, checkpoint_dir
    )
    logging.info("Loaded custom suite=%s n_tasks=%d", suite.suite_name, suite.n_tasks)

    video_dir = pathlib.Path(args.video_dir).expanduser().resolve()
    results_json_path = pathlib.Path(args.results_json).expanduser().resolve()
    if args.save_videos:
        video_dir.mkdir(parents=True, exist_ok=True)
    results_json_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []
    total_episodes = 0
    total_successes = 0
    global_start = time.time()

    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        n_total_inits = int(init_states.shape[0])
        n_eval = n_total_inits if args.num_trials <= 0 else min(args.num_trials, n_total_inits)

        task_max_steps = task.max_steps if args.max_steps < 0 else args.max_steps
        task_wait_steps = task.num_wait_steps if args.num_steps_wait < 0 else args.num_steps_wait

        logging.info(
            "Task %d/%d: %s | inits=%d eval=%d max_steps=%d wait_steps=%d",
            task_id + 1,
            suite.n_tasks,
            task.name,
            n_total_inits,
            n_eval,
            task_max_steps,
            task_wait_steps,
        )

        env = OffScreenRenderEnv(
            bddl_file_name=task.bddl_file,
            camera_heights=args.camera_size,
            camera_widths=args.camera_size,
            ignore_done=True,
        )
        env.seed(args.seed + task_id)

        task_episodes = 0
        task_successes = 0
        task_start = time.time()

        for ep_idx in tqdm.tqdm(range(n_eval), desc=f"{task.name}"):
            init_state = np.asarray(init_states[ep_idx], dtype=np.float64)
            obs = env.reset()
            obs = env.set_init_state(init_state)

            action_plan: collections.deque[np.ndarray] = collections.deque()
            joint_cache: dict[str, str] = {}
            replay_frames: list[np.ndarray] = []

            if args.save_videos and "agentview_image" in obs:
                replay_frames.append(np.ascontiguousarray(obs["agentview_image"][::-1]))

            step_count = 0
            env_done = False
            custom_success = False
            final_metrics: dict[str, float] = {}

            for t in range(task_max_steps + task_wait_steps):
                if t < task_wait_steps:
                    obs, _, env_done, _ = env.step(LIBERO_DUMMY_ACTION)
                    if args.save_videos and "agentview_image" in obs:
                        replay_frames.append(np.ascontiguousarray(obs["agentview_image"][::-1]))
                    step_count = t + 1
                    continue

                if not action_plan:
                    policy_input = make_policy_input(
                        obs=obs,
                        prompt=str(task.language),
                        resize_size=args.resize_size,
                        image_tools_module=image_tools,
                    )
                    infer_out = policy.infer(policy_input)
                    action_chunk = np.asarray(infer_out["actions"], dtype=np.float32)
                    if action_chunk.ndim == 1:
                        action_chunk = action_chunk[None, :]
                    if action_chunk.shape[0] < args.replan_steps:
                        raise RuntimeError(
                            f"Policy predicted only {action_chunk.shape[0]} actions, "
                            f"but replan_steps={args.replan_steps}"
                        )
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = np.asarray(action_plan.popleft(), dtype=np.float32)
                obs, _, env_done, _ = env.step(action.tolist())
                if args.save_videos and "agentview_image" in obs:
                    replay_frames.append(np.ascontiguousarray(obs["agentview_image"][::-1]))

                custom_success, final_metrics = evaluate_success(
                    obs=obs,
                    env=env,
                    task=task,
                    joint_cache=joint_cache,
                )
                step_count = t + 1
                if custom_success:
                    break
                if args.allow_env_done_success and bool(env_done):
                    custom_success = True
                    break

            suffix = "success" if custom_success else "failure"
            if args.save_videos and replay_frames:
                safe_task_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in task.name)
                video_path = (
                    video_dir
                    / f"task{task_id:02d}_{safe_task_name}_ep{ep_idx:03d}_{suffix}.mp4"
                )
                imageio.mimwrite(video_path, replay_frames, fps=args.video_fps)

            task_episodes += 1
            total_episodes += 1
            if custom_success:
                task_successes += 1
                total_successes += 1

            all_results.append(
                {
                    "task_id": task_id,
                    "task_name": task.name,
                    "episode_idx": ep_idx,
                    "success": bool(custom_success),
                    "steps": int(step_count),
                    "env_done": bool(env_done),
                    "metrics": {k: float(v) for k, v in final_metrics.items()},
                }
            )

        env.close()
        task_rate = (task_successes / task_episodes) if task_episodes > 0 else 0.0
        elapsed = time.time() - task_start
        logging.info(
            "Task done: %s | success=%d/%d (%.2f%%) | elapsed=%.1fs",
            task.name,
            task_successes,
            task_episodes,
            task_rate * 100.0,
            elapsed,
        )

    overall_rate = (total_successes / total_episodes) if total_episodes > 0 else 0.0
    payload = {
        "suite_name": suite.suite_name,
        "checkpoint_dir": str(checkpoint_dir),
        "openpi_config": args.openpi_config,
        "seed": int(args.seed),
        "total_episodes": int(total_episodes),
        "total_successes": int(total_successes),
        "overall_success_rate": float(overall_rate),
        "elapsed_sec": float(time.time() - global_start),
        "results": all_results,
    }
    results_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logging.info(
        "Finished eval | success=%d/%d (%.2f%%) | saved=%s",
        total_successes,
        total_episodes,
        overall_rate * 100.0,
        results_json_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
