#!/usr/bin/env python3
"""
HTTP client for evaluating a PI0/PI0.5 policy server on official LIBERO suites.

This script is intended for the case where:
  - the policy/model runs in a LeRobot environment;
  - LIBERO + robosuite run in another environment;
  - the two parts communicate through HTTP: /health, /reset, /infer.

Supported suites:
  libero_spatial, libero_object, libero_goal, libero_10, libero_90
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import random
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from libero.libero.envs import OffScreenRenderEnv


@dataclass
class OfficialTask:
    suite: str
    task_id: int
    name: str
    language: str
    bddl_file: str


def http_json(url: str, obj: Optional[dict] = None, timeout: int = 1800) -> dict:
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
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code} from {url}\n{body}") from e


def check_server(server: str) -> dict:
    return http_json(server.rstrip("/") + "/health")


def reset_server(server: str) -> dict:
    return http_json(server.rstrip("/") + "/reset", obj={})


def infer_action_chunk(server: str, payload: dict) -> np.ndarray:
    out = http_json(server.rstrip("/") + "/infer", obj=payload)
    action = np.asarray(out["action"], dtype=np.float32)

    # Server may return (7,), (T, 7), or accidentally (7, T).
    if action.ndim == 1:
        if action.shape[0] != 7:
            raise ValueError(f"Unexpected action shape: {action.shape}")
        return action[None, :]
    if action.ndim == 2:
        if action.shape[1] == 7:
            return action
        if action.shape[0] == 7:
            return action.T
    raise ValueError(f"Unexpected action shape from server: {action.shape}")


def encode_image_to_b64(img: np.ndarray, quality: int = 90) -> str:
    img = np.asarray(img, dtype=np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(quality))
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    if q.shape[0] != 4:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.asarray([q[3], q[0], q[1], q[2]], dtype=np.float32)


def rotmat_to_quat_wxyz(mat: np.ndarray) -> np.ndarray:
    m = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    tr = np.trace(m)
    if tr > 0.0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12))
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12))
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12))
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.asarray([w, x, y, z], dtype=np.float32)
    return q / (np.linalg.norm(q) + 1e-12)


def get_base_env(env):
    return env.env if hasattr(env, "env") else env


def resolve_eef_site_id(env) -> int:
    base = get_base_env(env)
    sim = base.sim

    robots = getattr(base, "robots", None)
    if robots:
        robot = robots[0]
        for attr in ("eef_site_id", "grip_site_id"):
            if hasattr(robot, attr):
                val = getattr(robot, attr)
                if isinstance(val, (int, np.integer)):
                    return int(val)
                if isinstance(val, dict) and val:
                    return int(list(val.values())[0])

    for name in ("gripper0_grip_site", "robot0_grip_site", "eef_site", "grip_site"):
        try:
            return int(sim.model.site_name2id(name))
        except Exception:
            continue
    return -1


def get_eef_quat_wxyz(env, obs: dict, eef_site_id: int) -> np.ndarray:
    if "robot0_eef_quat_wxyz" in obs:
        q = np.asarray(obs["robot0_eef_quat_wxyz"], dtype=np.float32).reshape(-1)
        if q.shape[0] == 4:
            return q / (np.linalg.norm(q) + 1e-12)
    if "robot0_eef_quat" in obs:
        return quat_xyzw_to_wxyz(np.asarray(obs["robot0_eef_quat"], dtype=np.float32))
    if eef_site_id >= 0:
        base = get_base_env(env)
        xmat = np.asarray(base.sim.data.site_xmat[eef_site_id], dtype=np.float64).reshape(3, 3)
        return rotmat_to_quat_wxyz(xmat)
    return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def get_latest_obs(env) -> dict:
    base = get_base_env(env)
    if hasattr(base, "_get_observations"):
        try:
            return base._get_observations(force_update=True)
        except TypeError:
            return base._get_observations()
    raise RuntimeError("LIBERO env does not expose _get_observations().")


def build_payload(
    env,
    obs: dict,
    instruction: str,
    *,
    image_key: str,
    image2_key: str,
    flip_images: bool,
    jpeg_quality: int,
    policy_step: int,
    max_steps: int,
    eef_site_id: int,
) -> dict:
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)[:3]
    eef_quat = get_eef_quat_wxyz(env, obs, eef_site_id).reshape(-1)[:4]
    gripper_qpos = np.asarray(obs.get("robot0_gripper_qpos", np.zeros(2)), dtype=np.float32).reshape(-1)
    gripper_mean = np.asarray([float(np.mean(np.abs(gripper_qpos)))], dtype=np.float32)
    state = np.concatenate([eef_pos, eef_quat, gripper_mean], axis=0).astype(np.float32)

    img1 = np.asarray(obs[image_key], dtype=np.uint8)
    img2 = np.asarray(obs[image2_key], dtype=np.uint8)
    if flip_images:
        img1 = img1[::-1]
        img2 = img2[::-1]

    return {
        "task": instruction,
        "observation.state": state.tolist(),
        "observation.images.image": encode_image_to_b64(img1, quality=jpeg_quality),
        "observation.images.image2": encode_image_to_b64(img2, quality=jpeg_quality),
        "policy_step": int(policy_step),
        "policy_progress": float(policy_step) / float(max(1, max_steps)),
    }


def clip_action(
    action: np.ndarray,
    *,
    enabled: bool,
    action_clip: float,
    pos_clip: float,
    rot_clip: float,
    gripper_clip: float,
) -> np.ndarray:
    a = np.asarray(action, dtype=np.float32).copy()
    if not enabled:
        return a
    if action_clip > 0:
        a = np.clip(a, -float(action_clip), float(action_clip))
    if pos_clip > 0:
        a[:3] = np.clip(a[:3], -float(pos_clip), float(pos_clip))
    if rot_clip > 0:
        a[3:6] = np.clip(a[3:6], -float(rot_clip), float(rot_clip))
    if gripper_clip > 0:
        a[6] = np.clip(a[6], -float(gripper_clip), float(gripper_clip))
    return a


def check_success(env) -> bool:
    for obj in (env, get_base_env(env)):
        if obj is not None and hasattr(obj, "_check_success"):
            try:
                return bool(obj._check_success())
            except Exception:
                pass
        if obj is not None and hasattr(obj, "check_success"):
            try:
                return bool(obj.check_success())
            except Exception:
                pass
    return False


def get_task_language(task: Any) -> str:
    for attr in ("language", "task_language", "instruction", "task_description", "description"):
        val = getattr(task, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return str(task)


def get_task_name(task: Any, task_id: int) -> str:
    for attr in ("name", "task_name", "problem_name"):
        val = getattr(task, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return f"task_{task_id:03d}"


def resolve_bddl_file(task: Any) -> str:
    raw_bddl = getattr(task, "bddl_file", "")
    if raw_bddl and os.path.isabs(str(raw_bddl)) and os.path.exists(str(raw_bddl)):
        return str(raw_bddl)

    problem_folder = getattr(task, "problem_folder", "")
    candidates: list[Path] = []
    if raw_bddl:
        candidates.append(Path(str(raw_bddl)))

    try:
        from libero.libero import get_libero_path

        root = Path(get_libero_path("bddl_files"))
        if raw_bddl and problem_folder:
            candidates.append(root / str(problem_folder) / str(raw_bddl))
        if raw_bddl:
            candidates.append(root / str(raw_bddl))
    except Exception:
        pass

    env_root = os.environ.get("LIBERO_BDDL_ROOT", "")
    if env_root and raw_bddl:
        root = Path(env_root)
        if problem_folder:
            candidates.append(root / str(problem_folder) / str(raw_bddl))
        candidates.append(root / str(raw_bddl))

    for c in candidates:
        if c.exists():
            return str(c.resolve())

    raise FileNotFoundError(
        f"Could not resolve BDDL for task={task!r}, bddl_file={raw_bddl!r}, problem_folder={problem_folder!r}. "
        "Set LIBERO_BDDL_ROOT to your bddl_files directory if needed."
    )


def get_benchmark_suite(suite_name: str):
    from libero.libero import benchmark

    if hasattr(benchmark, "get_benchmark_dict"):
        d = benchmark.get_benchmark_dict()
        if suite_name not in d:
            raise KeyError(f"{suite_name!r} not found in LIBERO benchmark dict. Keys={list(d.keys())}")
        return d[suite_name]()
    if hasattr(benchmark, "get_benchmark"):
        return benchmark.get_benchmark(suite_name)()
    raise RuntimeError("Cannot find LIBERO benchmark API: get_benchmark_dict/get_benchmark missing.")


def get_init_states(task_suite, task_id: int):
    if hasattr(task_suite, "get_task_init_states"):
        return task_suite.get_task_init_states(task_id)
    if hasattr(task_suite, "get_task_demonstration"):
        demo = task_suite.get_task_demonstration(task_id)
        if isinstance(demo, dict) and "init_states" in demo:
            return demo["init_states"]
    return None


def build_official_tasks(suite_name: str) -> tuple[Any, list[OfficialTask]]:
    task_suite = get_benchmark_suite(suite_name)
    n_tasks = int(getattr(task_suite, "n_tasks", 0))
    if n_tasks <= 0 and hasattr(task_suite, "tasks"):
        n_tasks = len(task_suite.tasks)
    if n_tasks <= 0:
        raise RuntimeError(f"Could not infer number of tasks for suite {suite_name}")

    tasks = []
    for task_id in range(n_tasks):
        task = task_suite.get_task(task_id)
        tasks.append(
            OfficialTask(
                suite=suite_name,
                task_id=task_id,
                name=get_task_name(task, task_id),
                language=get_task_language(task),
                bddl_file=resolve_bddl_file(task),
            )
        )
    return task_suite, tasks


def build_env(bddl_file: str, camera_size: int):
    kwargs = dict(
        bddl_file_name=bddl_file,
        camera_heights=int(camera_size),
        camera_widths=int(camera_size),
        ignore_done=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
    )
    try:
        return OffScreenRenderEnv(**kwargs)
    except TypeError:
        kwargs.pop("camera_names", None)
        return OffScreenRenderEnv(**kwargs)


def apply_init_state(env, init_state):
    if init_state is not None and hasattr(env, "set_init_state"):
        return env.set_init_state(init_state)

    if init_state is not None:
        base = get_base_env(env)
        arr = np.asarray(init_state, dtype=np.float64).reshape(-1)
        if arr.size == base.sim.data.qpos.size:
            base.sim.data.qpos[:] = arr
            base.sim.data.qvel[:] = 0
            base.sim.forward()
            return get_latest_obs(env)

    return get_latest_obs(env)


def capture(obs: dict, cameras: list[str], frames: dict[str, list[np.ndarray]], flip: bool):
    for cam in cameras:
        key = f"{cam}_image"
        if key in obs:
            img = np.asarray(obs[key], dtype=np.uint8)
            frames[cam].append(img[::-1] if flip else img)


def save_episode_videos(
    video_root: Path,
    suite: str,
    task_id: int,
    episode_id: int,
    success: bool,
    frames: dict[str, list[np.ndarray]],
    fps: int,
):
    out = {}
    ep_dir = video_root / suite / f"task_{task_id:03d}" / f"episode_{episode_id:03d}_{'succ' if success else 'fail'}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    for cam, vals in frames.items():
        if not vals:
            continue
        path = ep_dir / f"{cam}.mp4"
        imageio.mimwrite(str(path), vals, fps=max(1, int(fps)))
        out[cam] = str(path)
    return out


def run_one_episode(
    *,
    env,
    init_state,
    instruction: str,
    server: str,
    max_steps: int,
    exec_horizon: int,
    image_key: str,
    image2_key: str,
    flip_images: bool,
    jpeg_quality: int,
    clip_actions: bool,
    action_clip: float,
    pos_clip: float,
    rot_clip: float,
    gripper_clip: float,
    record_video: bool,
    video_cameras: list[str],
    video_fps: int,
):
    env.reset()
    obs = apply_init_state(env, init_state)
    eef_site_id = resolve_eef_site_id(env)

    reset_server(server)

    pending: list[np.ndarray] = []
    frames = {c: [] for c in video_cameras}
    if record_video:
        capture(obs, video_cameras, frames, flip=flip_images)

    for step in range(int(max_steps)):
        if not pending:
            payload = build_payload(
                env,
                obs,
                instruction,
                image_key=image_key,
                image2_key=image2_key,
                flip_images=flip_images,
                jpeg_quality=jpeg_quality,
                policy_step=step,
                max_steps=max_steps,
                eef_site_id=eef_site_id,
            )
            chunk = infer_action_chunk(server, payload)
            use_len = chunk.shape[0] if exec_horizon <= 0 else min(int(exec_horizon), chunk.shape[0])
            pending = [chunk[i] for i in range(use_len)]

        action = clip_action(
            pending.pop(0),
            enabled=clip_actions,
            action_clip=action_clip,
            pos_clip=pos_clip,
            rot_clip=rot_clip,
            gripper_clip=gripper_clip,
        )
        obs, _, _, _ = env.step(action)

        if record_video:
            capture(obs, video_cameras, frames, flip=flip_images)

        if check_success(env):
            return True, step + 1, frames

    return bool(check_success(env)), int(max_steps), frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, help="libero_object/libero_10/libero_90/libero_spatial/libero_goal")
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-end", type=int, default=-1, help="exclusive; -1 means all")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--camera-size", type=int, default=224)
    parser.add_argument("--exec-horizon", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--image-key", default="agentview_image")
    parser.add_argument("--image2-key", default="robot0_eye_in_hand_image")
    parser.add_argument("--no-flip-images", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=90)

    parser.add_argument("--clip-actions", action="store_true")
    parser.add_argument("--action-clip", type=float, default=1.0)
    parser.add_argument("--pos-clip", type=float, default=0.08)
    parser.add_argument("--rot-clip", type=float, default=0.10)
    parser.add_argument("--gripper-clip", type=float, default=1.0)

    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--record-limit-per-task", type=int, default=1, help="0 means record all")
    parser.add_argument("--video-dir", default="./official_libero_eval_videos")
    parser.add_argument("--video-cameras", default="agentview")
    parser.add_argument("--video-fps", type=int, default=20)

    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    health = check_server(args.server)
    print("[client] server health:", json.dumps(health, indent=2, ensure_ascii=False))

    task_suite, tasks = build_official_tasks(args.suite)
    task_start = max(0, int(args.task_start))
    task_end = len(tasks) if args.task_end < 0 else min(len(tasks), int(args.task_end))
    selected_tasks = tasks[task_start:task_end]
    if not selected_tasks:
        raise RuntimeError(f"No tasks selected: start={task_start}, end={task_end}, total={len(tasks)}")

    print(f"[client] suite={args.suite} selected_tasks={task_start}:{task_end} n={len(selected_tasks)}")

    video_cameras = [x.strip() for x in args.video_cameras.split(",") if x.strip()]
    video_root = Path(args.video_dir).resolve()
    result = {
        "_meta": {
            "suite": args.suite,
            "server": args.server,
            "episodes_per_task": args.episodes_per_task,
            "max_steps": args.max_steps,
            "camera_size": args.camera_size,
            "exec_horizon": args.exec_horizon,
            "flip_images": not args.no_flip_images,
            "health": health,
        },
        "tasks": [],
    }
    csv_rows = []

    for task in selected_tasks:
        print(f"\n=== {args.suite} task_id={task.task_id} ===")
        print("name:", task.name)
        print("language:", task.language)
        print("bddl:", task.bddl_file)

        init_states = get_init_states(task_suite, task.task_id)
        n_init = len(init_states) if init_states is not None else 0
        env = build_env(task.bddl_file, camera_size=args.camera_size)

        ep_logs = []
        succ = 0

        for ep in range(int(args.episodes_per_task)):
            init_state = init_states[ep % n_init] if init_states is not None and n_init > 0 else None
            record_this = args.record_video and (
                args.record_limit_per_task <= 0 or ep < args.record_limit_per_task
            )

            try:
                success, steps, frames = run_one_episode(
                    env=env,
                    init_state=init_state,
                    instruction=task.language,
                    server=args.server,
                    max_steps=args.max_steps,
                    exec_horizon=args.exec_horizon,
                    image_key=args.image_key,
                    image2_key=args.image2_key,
                    flip_images=not args.no_flip_images,
                    jpeg_quality=args.jpeg_quality,
                    clip_actions=args.clip_actions,
                    action_clip=args.action_clip,
                    pos_clip=args.pos_clip,
                    rot_clip=args.rot_clip,
                    gripper_clip=args.gripper_clip,
                    record_video=record_this,
                    video_cameras=video_cameras,
                    video_fps=args.video_fps,
                )
                err = ""
            except Exception as e:
                success = False
                steps = 0
                frames = {c: [] for c in video_cameras}
                err = repr(e)
                print("[episode error]", err)

            succ += int(success)
            ep_log = {
                "episode_id": ep,
                "success": bool(success),
                "steps": int(steps),
            }
            if err:
                ep_log["error"] = err
            if record_this:
                ep_log["videos"] = save_episode_videos(
                    video_root, args.suite, task.task_id, ep, bool(success), frames, args.video_fps
                )
            ep_logs.append(ep_log)
            csv_rows.append(
                {
                    "suite": args.suite,
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "language": task.language,
                    "episode_id": ep,
                    "success": int(success),
                    "steps": int(steps),
                    "error": err,
                }
            )
            print(f"[episode] task={task.task_id:03d} ep={ep:03d} success={success} steps={steps}")

        env.close()

        task_sr = succ / max(1, int(args.episodes_per_task))
        result["tasks"].append(
            {
                "suite": args.suite,
                "task_id": task.task_id,
                "task_name": task.name,
                "language": task.language,
                "bddl_file": task.bddl_file,
                "success_rate": task_sr,
                "success_count": succ,
                "num_episodes": int(args.episodes_per_task),
                "episodes": ep_logs,
            }
        )
        print(f"[task summary] task={task.task_id:03d} SR={task_sr:.3f}")

    total_success = sum(t["success_count"] for t in result["tasks"])
    total_episodes = sum(t["num_episodes"] for t in result["tasks"])
    result["summary"] = {
        "success_count": int(total_success),
        "num_episodes": int(total_episodes),
        "success_rate": float(total_success / max(1, total_episodes)),
        "num_tasks": len(result["tasks"]),
    }

    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved json] {output_json}")
    print("[summary]", result["summary"])

    if args.output_csv:
        output_csv = Path(args.output_csv).resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["suite", "task_id", "task_name", "language", "episode_id", "success", "steps", "error"],
            )
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[saved csv] {output_csv}")


if __name__ == "__main__":
    main()
