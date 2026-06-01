#!/usr/bin/env python3
"""Evaluate a PI0.5 policy server on official LIBERO benchmark suites.

This client is intentionally separate from the custom red-peg benchmark client.
It uses LIBERO's official benchmark API and talks to policy_server_pi05.py over
HTTP. Run the policy server in a LeRobot environment and this client in the
LIBERO/robosuite environment.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from libero.libero.envs import OffScreenRenderEnv


SUITE_ALIASES = {
    "libero-90": "libero_90",
    "libero90": "libero_90",
    "libero_90": "libero_90",
    "libero-10": "libero_10",
    "libero10": "libero_10",
    "libero_10": "libero_10",
    "libero-object": "libero_object",
    "libero_object": "libero_object",
    "liberoobject": "libero_object",
}


@dataclass
class EvalTask:
    task_id: int
    language: str
    bddl_file: str
    init_states: list[Any]


def normalize_suite_name(name: str) -> str:
    key = str(name).strip().lower().replace(" ", "_")
    if key not in SUITE_ALIASES:
        raise ValueError(
            f"Unsupported suite {name!r}. Use one of: libero_90, libero_10, libero_object."
        )
    return SUITE_ALIASES[key]


def get_benchmark_suite(suite_name: str):
    from libero.libero import benchmark

    suite_name = normalize_suite_name(suite_name)
    if hasattr(benchmark, "get_benchmark_dict"):
        bench_dict = benchmark.get_benchmark_dict()
        if suite_name not in bench_dict:
            raise KeyError(f"{suite_name!r} not found in LIBERO benchmark dict: {list(bench_dict)}")
        return bench_dict[suite_name]()
    if hasattr(benchmark, "get_benchmark"):
        return benchmark.get_benchmark(suite_name)()
    raise RuntimeError("Could not find LIBERO benchmark factory.")


def get_num_tasks(task_suite) -> int:
    for attr in ("get_num_tasks", "num_tasks"):
        value = getattr(task_suite, attr, None)
        if callable(value):
            return int(value())
        if value is not None:
            return int(value)
    tasks = getattr(task_suite, "tasks", None)
    if tasks is not None:
        return len(tasks)
    raise RuntimeError("Could not infer number of tasks from LIBERO task suite.")


def get_task_language(task) -> str:
    for attr in ("language", "language_instruction", "task_description", "description"):
        value = getattr(task, attr, None)
        if value:
            return str(value)
    if isinstance(task, dict):
        for key in ("language", "language_instruction", "task_description", "description"):
            if task.get(key):
                return str(task[key])
    return ""


def get_task_bddl_file(task) -> str:
    for attr in ("bddl_file", "bddl_file_name", "bddl_path"):
        value = getattr(task, attr, None)
        if value:
            return str(value)
    if isinstance(task, dict):
        for key in ("bddl_file", "bddl_file_name", "bddl_path"):
            if task.get(key):
                return str(task[key])
    raise RuntimeError(f"Could not find bddl file from task object: {task!r}")


def get_init_states(task_suite, task_id: int) -> list[Any]:
    for name in ("get_task_init_states", "get_init_states"):
        fn = getattr(task_suite, name, None)
        if callable(fn):
            states = fn(task_id)
            return normalize_init_states(states)

    # Some LIBERO versions expose a path instead of loading the npy array.
    for name in ("get_task_init_states_path", "get_init_states_path"):
        fn = getattr(task_suite, name, None)
        if callable(fn):
            path = fn(task_id)
            return normalize_init_states(np.load(path, allow_pickle=True))

    raise RuntimeError("Could not load official LIBERO initial states.")


def normalize_init_states(states) -> list[Any]:
    if isinstance(states, np.ndarray):
        if states.dtype == object:
            return list(states)
        return [states[i] for i in range(len(states))]
    if isinstance(states, (list, tuple)):
        return list(states)
    return [states]


def make_eval_task(task_suite, task_id: int) -> EvalTask:
    task = task_suite.get_task(task_id)
    return EvalTask(
        task_id=task_id,
        language=get_task_language(task),
        bddl_file=get_task_bddl_file(task),
        init_states=get_init_states(task_suite, task_id),
    )



def resolve_bddl_path(bddl_file: str) -> str:
    """Resolve a LIBERO bddl filename to an absolute path."""
    p = Path(str(bddl_file)).expanduser()
    if p.is_file():
        return str(p)

    roots = []

    env_root = os.environ.get("LIBERO_BDDL_ROOT", "")
    if env_root:
        roots.append(Path(env_root).expanduser())

    roots.extend([
        Path("/root/autodl-tmp/openpi_earbud_proto/third_party/libero/libero/libero/bddl_files"),
        Path("/root/autodl-tmp/openpi/third_party/libero/libero/libero/bddl_files"),
    ])

    suites = [
        "libero_object",
        "libero_10",
        "libero_90",
        "libero_spatial",
        "libero_goal",
    ]

    checked = []
    for root in roots:
        for suite in suites:
            cand = root / suite / str(bddl_file)
            checked.append(str(cand))
            if cand.is_file():
                return str(cand)

        cand = root / str(bddl_file)
        checked.append(str(cand))
        if cand.is_file():
            return str(cand)

    raise FileNotFoundError(
        "Could not resolve LIBERO bddl file: "
        + str(bddl_file)
        + "\nChecked:\n  "
        + "\n  ".join(checked[:40])
    )


def build_env(task: EvalTask, camera_size: int, render_gpu_device_id: int):
    resolved_bddl = resolve_bddl_path(task.bddl_file)
    print(f"[bddl] {task.bddl_file} -> {resolved_bddl}")
    kwargs = dict(
        bddl_file_name=resolved_bddl,
        camera_heights=camera_size,
        camera_widths=camera_size,
        ignore_done=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
    )
    if render_gpu_device_id >= 0:
        kwargs["render_gpu_device_id"] = int(render_gpu_device_id)
    try:
        return OffScreenRenderEnv(**kwargs)
    except TypeError:
        kwargs.pop("camera_names", None)
        return OffScreenRenderEnv(**kwargs)


def try_seed_env(env, seed: int):
    for obj in (env, getattr(env, "env", None)):
        if obj is None:
            continue
        seed_fn = getattr(obj, "seed", None)
        if callable(seed_fn):
            try:
                seed_fn(seed)
                return
            except Exception:
                pass


def set_init_state(env, init_state):
    obs = env.reset()
    if hasattr(env, "set_init_state"):
        return env.set_init_state(init_state)

    base = env.env if hasattr(env, "env") else env
    sim = base.sim
    if isinstance(init_state, dict) and "qpos" in init_state and "qvel" in init_state:
        sim.data.qpos[:] = np.asarray(init_state["qpos"], dtype=np.float64)
        sim.data.qvel[:] = np.asarray(init_state["qvel"], dtype=np.float64)
        sim.forward()
        return get_latest_obs(env)

    arr = np.asarray(init_state)
    if hasattr(sim, "set_state_from_flattened"):
        sim.set_state_from_flattened(arr.astype(np.float64))
        sim.forward()
        return get_latest_obs(env)

    return obs


def get_latest_obs(env):
    base = env.env if hasattr(env, "env") else env
    if hasattr(base, "_get_observations"):
        try:
            return base._get_observations(force_update=True)
        except TypeError:
            return base._get_observations()
    raise RuntimeError("Environment does not expose _get_observations().")


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError(f"Expected quaternion dim=4, got shape={q.shape}")
    out = np.asarray([q[3], q[0], q[1], q[2]], dtype=np.float32)
    return out / (np.linalg.norm(out) + 1e-12)


def get_eef_quat_wxyz(obs) -> np.ndarray:
    if "robot0_eef_quat_wxyz" in obs:
        q = np.asarray(obs["robot0_eef_quat_wxyz"], dtype=np.float32).reshape(4)
        return q / (np.linalg.norm(q) + 1e-12)
    if "robot0_eef_quat" in obs:
        return quat_xyzw_to_wxyz(obs["robot0_eef_quat"])
    raise KeyError("Observation has neither robot0_eef_quat_wxyz nor robot0_eef_quat.")


def encode_image_to_b64(img: np.ndarray, quality: int = 90) -> str:
    pil_img = Image.fromarray(np.asarray(img, dtype=np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=int(quality))
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_image(obs, camera_name: str, flip: bool) -> np.ndarray:
    key = f"{camera_name}_image"
    if key not in obs:
        raise KeyError(f"Missing camera image {key!r}. Available image keys: {[k for k in obs if k.endswith('_image')]}")
    img = np.asarray(obs[key], dtype=np.uint8)
    return img[::-1] if flip else img


def make_payload(obs, instruction: str, flip_images: bool, jpeg_quality: int):
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(3)
    eef_quat = get_eef_quat_wxyz(obs)
    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    gripper_mean = np.asarray([float(np.mean(np.abs(gripper_qpos)))], dtype=np.float32)
    state = np.concatenate([eef_pos, eef_quat, gripper_mean], axis=0).astype(np.float32)

    return {
        "task": instruction,
        "observation.state": state.tolist(),
        "observation.images.image": encode_image_to_b64(
            get_image(obs, "agentview", flip_images), quality=jpeg_quality
        ),
        "observation.images.image2": encode_image_to_b64(
            get_image(obs, "robot0_eye_in_hand", flip_images), quality=jpeg_quality
        ),
    }


def http_json_request(url: str, obj=None, timeout: int = 1800):
    try:
        if obj is None:
            req = urllib.request.Request(url, method="GET")
        else:
            data = json.dumps(obj).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code} from {url}\n{body}") from e


def check_server(server_url: str):
    return http_json_request(server_url.rstrip("/") + "/health", obj=None)


def reset_policy_server(server_url: str):
    return http_json_request(server_url.rstrip("/") + "/reset", obj={})


def infer_action_chunk(server_url: str, payload: dict) -> np.ndarray:
    out = http_json_request(server_url.rstrip("/") + "/infer", obj=payload)
    action = np.asarray(out["action"], dtype=np.float32)
    if action.ndim == 1:
        action = action[None, :]
    elif action.ndim == 2 and action.shape[0] == 7 and action.shape[1] != 7:
        action = action.T
    if action.ndim != 2 or action.shape[1] < 7:
        raise ValueError(f"Unexpected action shape from server: {action.shape}")
    return action[:, :7].astype(np.float32)


def clip_action(
    action: np.ndarray,
    enabled: bool,
    pos_clip: float,
    rot_xy_clip: float,
    rot_z_clip: float,
    gripper_clip: float,
) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).copy()
    if not enabled:
        return action
    action[:3] = np.clip(action[:3], -pos_clip, pos_clip)
    action[3:5] = np.clip(action[3:5], -rot_xy_clip, rot_xy_clip)
    action[5] = np.clip(action[5], -rot_z_clip, rot_z_clip)
    action[6] = np.clip(action[6], -gripper_clip, gripper_clip)
    return action


def check_success(env, info: Optional[dict] = None) -> bool:
    if isinstance(info, dict):
        for key in ("success", "is_success"):
            if key in info:
                return bool(info[key])
    for obj in (env, getattr(env, "env", None)):
        if obj is not None and hasattr(obj, "_check_success"):
            try:
                return bool(obj._check_success())
            except Exception:
                pass
    return False


def capture_video_frame(obs, camera_names: list[str], frames: dict[str, list[np.ndarray]], flip: bool):
    for cam in camera_names:
        key = f"{cam}_image"
        if key not in obs:
            continue
        img = np.asarray(obs[key], dtype=np.uint8)
        frames[cam].append(img[::-1] if flip else img)


def save_videos(frames: dict[str, list[np.ndarray]], out_dir: Path, fps: int) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for cam, cam_frames in frames.items():
        if not cam_frames:
            continue
        path = out_dir / f"{cam}.mp4"
        imageio.mimsave(path, cam_frames, fps=int(fps), macro_block_size=1)
        paths[cam] = str(path)
    return paths


def rollout_one(
    env,
    task: EvalTask,
    trial_id: int,
    args,
    action_trace_rows: Optional[list[dict[str, Any]]] = None,
    record_video: bool = False,
    video_out_dir: Optional[Path] = None,
):
    init_state = task.init_states[trial_id % len(task.init_states)]
    obs = set_init_state(env, init_state)

    zero_action = np.zeros(7, dtype=np.float32)
    for _ in range(max(0, int(args.warmup_steps))):
        obs, _, _, _ = env.step(zero_action)

    reset_policy_server(args.server)

    pending_actions: list[np.ndarray] = []
    frames = {cam: [] for cam in args.video_cameras}
    if record_video:
        capture_video_frame(obs, args.video_cameras, frames, flip=args.render_flip_images)

    success = False
    final_info = {}
    steps = 0
    first_chunk_info = None

    for step in range(int(args.max_steps)):
        if not pending_actions:
            payload = make_payload(
                obs,
                task.language,
                flip_images=not args.no_flip_images,
                jpeg_quality=args.jpeg_quality,
            )
            chunk = infer_action_chunk(args.server, payload)
            use_len = max(1, min(int(args.exec_horizon), int(chunk.shape[0])))
            pending_actions = [chunk[i] for i in range(use_len)]
            if first_chunk_info is None:
                first_chunk_info = {
                    "chunk_len": int(chunk.shape[0]),
                    "use_len": int(use_len),
                    "first_gripper": float(chunk[0, -1]),
                    "last_used_gripper": float(chunk[use_len - 1, -1]),
                }
                if args.debug_action_trace:
                    print(
                        "[debug_action_trace] "
                        f"task={task.task_id} trial={trial_id} step={step} "
                        f"chunk_len={chunk.shape[0]} use_len={use_len} "
                        f"first_gripper={chunk[0, -1]:+.4f} "
                        f"last_used_gripper={chunk[use_len - 1, -1]:+.4f}"
                    )

        action = clip_action(
            pending_actions.pop(0),
            enabled=not args.disable_action_clip,
            pos_clip=float(args.pos_action_clip),
            rot_xy_clip=float(args.rot_xy_action_clip),
            rot_z_clip=float(args.rot_z_action_clip),
            gripper_clip=float(args.gripper_action_clip),
        )

        if args.debug_action_trace and (
            step < 10 or (args.debug_action_trace_steps > 0 and step % args.debug_action_trace_steps == 0)
        ):
            print(
                "[action_trace] "
                f"task={task.task_id} trial={trial_id} step={step} "
                f"xyz=({action[0]:+.4f},{action[1]:+.4f},{action[2]:+.4f}) "
                f"rot=({action[3]:+.4f},{action[4]:+.4f},{action[5]:+.4f}) "
                f"gripper={action[6]:+.4f}"
            )

        if action_trace_rows is not None:
            action_trace_rows.append(
                {
                    "suite": args.suite,
                    "task_id": task.task_id,
                    "trial_id": trial_id,
                    "step": step,
                    "a0": float(action[0]),
                    "a1": float(action[1]),
                    "a2": float(action[2]),
                    "a3": float(action[3]),
                    "a4": float(action[4]),
                    "a5": float(action[5]),
                    "gripper": float(action[6]),
                    "language": task.language,
                }
            )

        obs, reward, done, info = env.step(action)
        steps = step + 1
        final_info = dict(info) if isinstance(info, dict) else {}
        if record_video:
            capture_video_frame(obs, args.video_cameras, frames, flip=args.render_flip_images)

        success = check_success(env, info)
        if success:
            break

    videos = {}
    if record_video and video_out_dir is not None:
        videos = save_videos(frames, video_out_dir, fps=args.video_fps)

    return {
        "task_id": task.task_id,
        "trial_id": trial_id,
        "success": bool(success),
        "steps": int(steps),
        "language": task.language,
        "bddl_file": task.bddl_file,
        "first_chunk": first_chunk_info,
        "final_info": final_info,
        "videos": videos,
    }


def write_summary_csv(path: Path, rows: list[dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "suite",
        "task_id",
        "language",
        "num_trials",
        "num_success",
        "success_rate",
        "avg_steps_success",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=str, required=True, help="libero_90, libero_10, or libero_object")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--summary_csv", type=str, default="")
    parser.add_argument("--action_trace_csv", type=str, default="")
    parser.add_argument("--camera_size", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=520)
    parser.add_argument("--num_trials_per_task", type=int, default=50)
    parser.add_argument("--start_task", type=int, default=0)
    parser.add_argument("--max_tasks", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exec_horizon", type=int, default=10)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--jpeg_quality", type=int, default=90)
    parser.add_argument("--render_gpu_device_id", type=int, default=-1)

    parser.add_argument("--no_flip_images", action="store_true", help="Do not flip images before sending to policy.")
    parser.add_argument(
        "--render_flip_images",
        action="store_true",
        help="Flip saved video frames vertically. Usually keep false for human-readable videos.",
    )

    parser.add_argument("--disable_action_clip", action="store_true")
    parser.add_argument("--pos_action_clip", type=float, default=0.08)
    parser.add_argument("--rot_xy_action_clip", type=float, default=0.10)
    parser.add_argument("--rot_z_action_clip", type=float, default=0.08)
    parser.add_argument("--gripper_action_clip", type=float, default=1.0)

    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_dir", type=str, default="")
    parser.add_argument("--video_cameras", type=str, default="agentview")
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--record_limit_per_task", type=int, default=1)
    parser.add_argument("--record_success", action="store_true", help="Also record successful trials.")

    parser.add_argument("--debug_action_trace", action="store_true")
    parser.add_argument("--debug_action_trace_steps", type=int, default=50)
    parser.add_argument("--continue_on_error", action="store_true")
    args = parser.parse_args()

    args.suite = normalize_suite_name(args.suite)
    args.video_cameras = [x.strip() for x in args.video_cameras.split(",") if x.strip()]

    print("[client] server health:", check_server(args.server))
    task_suite = get_benchmark_suite(args.suite)
    num_tasks_total = get_num_tasks(task_suite)
    task_ids = list(range(args.start_task, num_tasks_total))
    if args.max_tasks is not None and int(args.max_tasks) > 0:
        task_ids = task_ids[: int(args.max_tasks)]

    print(
        f"[suite] {args.suite}: total_tasks={num_tasks_total}, "
        f"selected_tasks={len(task_ids)}, trials_per_task={args.num_trials_per_task}"
    )

    all_results = {
        "_meta": {
            "suite": args.suite,
            "server": args.server,
            "camera_size": args.camera_size,
            "max_steps": args.max_steps,
            "num_trials_per_task": args.num_trials_per_task,
            "exec_horizon": args.exec_horizon,
            "flip_images_for_policy": not args.no_flip_images,
            "action_clip": not args.disable_action_clip,
            "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "tasks": [],
    }
    summary_rows = []
    action_trace_rows = [] if args.action_trace_csv else None

    rng = np.random.default_rng(args.seed)

    for task_id in task_ids:
        task = make_eval_task(task_suite, task_id)
        print(f"\n=== {args.suite} task_id={task_id} ===")
        print(f"language: {task.language}")
        print(f"bddl: {task.bddl_file}")
        print(f"init_states: {len(task.init_states)}")

        env = None
        task_results = []
        record_count = 0
        try:
            env = build_env(task, camera_size=args.camera_size, render_gpu_device_id=args.render_gpu_device_id)
            try_seed_env(env, int(args.seed + task_id))

            trial_indices = np.arange(len(task.init_states))
            if len(trial_indices) > 0:
                rng.shuffle(trial_indices)

            for trial_i in range(int(args.num_trials_per_task)):
                init_index = int(trial_indices[trial_i % len(trial_indices)]) if len(trial_indices) else trial_i
                should_record = False
                if args.record_video and record_count < int(args.record_limit_per_task):
                    should_record = True

                video_out_dir = None
                if should_record and args.video_dir:
                    video_out_dir = (
                        Path(args.video_dir)
                        / args.suite
                        / f"task_{task_id:03d}"
                        / f"trial_{trial_i:03d}_init_{init_index:03d}"
                    )

                result = rollout_one(
                    env,
                    task,
                    init_index,
                    args,
                    action_trace_rows=action_trace_rows,
                    record_video=should_record,
                    video_out_dir=video_out_dir,
                )
                result["eval_trial_id"] = int(trial_i)
                result["init_state_index"] = int(init_index)

                # If we only want to save failures and this was successful,
                # delete its video unless --record_success was requested.
                if should_record:
                    if result["success"] and not args.record_success:
                        for p in result.get("videos", {}).values():
                            try:
                                Path(p).unlink(missing_ok=True)
                            except Exception:
                                pass
                        result["videos"] = {}
                    else:
                        record_count += 1

                task_results.append(result)
                print(
                    f"[trial] task={task_id:03d} trial={trial_i:03d} init={init_index:03d} "
                    f"success={result['success']} steps={result['steps']}"
                )

        except Exception as e:
            tb = traceback.format_exc()
            print(f"[error] task_id={task_id}: {e}")
            print(tb)
            if not args.continue_on_error:
                raise
            task_results.append(
                {
                    "task_id": task_id,
                    "success": False,
                    "error": repr(e),
                    "traceback": tb,
                    "language": task.language,
                    "bddl_file": task.bddl_file,
                }
            )
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass

        successes = [r for r in task_results if r.get("success")]
        success_rate = len(successes) / max(1, len(task_results))
        avg_steps_success = float(np.mean([r["steps"] for r in successes])) if successes else float("nan")

        print(
            f"[task_summary] task={task_id:03d} success={len(successes)}/{len(task_results)} "
            f"rate={success_rate:.3f} avg_steps_success={avg_steps_success}"
        )

        all_results["tasks"].append(
            {
                "task_id": task_id,
                "language": task.language,
                "bddl_file": task.bddl_file,
                "num_trials": len(task_results),
                "num_success": len(successes),
                "success_rate": success_rate,
                "avg_steps_success": avg_steps_success,
                "episodes": task_results,
            }
        )
        summary_rows.append(
            {
                "suite": args.suite,
                "task_id": task_id,
                "language": task.language,
                "num_trials": len(task_results),
                "num_success": len(successes),
                "success_rate": success_rate,
                "avg_steps_success": avg_steps_success,
            }
        )

    total_trials = sum(t["num_trials"] for t in all_results["tasks"])
    total_success = sum(t["num_success"] for t in all_results["tasks"])
    all_results["_summary"] = {
        "num_tasks": len(all_results["tasks"]),
        "total_trials": int(total_trials),
        "total_success": int(total_success),
        "success_rate": float(total_success / max(1, total_trials)),
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] output_json: {out_json}")
    print("[summary]", all_results["_summary"])

    if args.summary_csv:
        write_summary_csv(Path(args.summary_csv), summary_rows)
        print(f"[saved] summary_csv: {args.summary_csv}")

    if action_trace_rows is not None:
        path = Path(args.action_trace_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "suite",
            "task_id",
            "trial_id",
            "step",
            "a0",
            "a1",
            "a2",
            "a3",
            "a4",
            "a5",
            "gripper",
            "language",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(action_trace_rows)
        print(f"[saved] action_trace_csv: {path}")


if __name__ == "__main__":
    main()
