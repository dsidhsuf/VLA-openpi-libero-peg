from __future__ import annotations
import argparse
import base64
import io
import json
import re
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path

import imageio
import numpy as np
from PIL import Image
from libero.libero.envs import OffScreenRenderEnv


@dataclass
class TaskSpec:
    name: str
    level: str
    bddl_file: str
    language: str
    max_steps: int
    state_file: Path


def safe_name(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", s).strip("_")


def parse_camera_names(raw: str):
    cams = [x.strip() for x in raw.split(",") if x.strip()]
    return cams if cams else ["agentview", "robot0_eye_in_hand"]


def load_task_specs(assets_dir: Path):
    tasks_json = assets_dir / "tasks.json"
    if not tasks_json.exists():
        raise FileNotFoundError(f"Missing benchmark tasks.json: {tasks_json}")

    with tasks_json.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    tasks = []
    levels = raw.get("levels", {})
    for level in ("easy", "medium", "hard"):
        if level not in levels:
            continue
        t = levels[level]
        state_file = Path(t["state_file"])
        if not state_file.is_absolute():
            state_file = (assets_dir / state_file).resolve()

        tasks.append(
            TaskSpec(
                name=str(t["name"]),
                level=level,
                bddl_file=str(t["bddl"]),
                language=str(t["language"]),
                max_steps=int(t["max_steps"]),
                state_file=state_file,
            )
        )
    return tasks


def build_env(task: TaskSpec, camera_size: int, camera_names):
    kwargs = dict(
        bddl_file_name=task.bddl_file,
        camera_heights=camera_size,
        camera_widths=camera_size,
        ignore_done=True,
        camera_names=list(camera_names),
    )
    try:
        env = OffScreenRenderEnv(**kwargs)
    except TypeError:
        kwargs.pop("camera_names", None)
        env = OffScreenRenderEnv(**kwargs)

    base = env.env if hasattr(env, "env") else env
    if hasattr(base, "_check_success"):
        base._check_success = lambda: False
    return env


def load_init_states(task: TaskSpec):
    if not task.state_file.exists():
        raise FileNotFoundError(f"Missing state file: {task.state_file}")
    pack = np.load(task.state_file, allow_pickle=True)
    qpos = np.asarray(pack["qpos"], dtype=np.float32)
    qvel = np.asarray(pack["qvel"], dtype=np.float32)
    return [{"qpos": qpos[i], "qvel": qvel[i]} for i in range(len(qpos))]


def set_state_qpos_qvel(env, init_state):
    base = env.env if hasattr(env, "env") else env
    sim = base.sim
    sim.data.qpos[:] = np.asarray(init_state["qpos"], dtype=np.float64)
    sim.data.qvel[:] = np.asarray(init_state["qvel"], dtype=np.float64)
    sim.forward()


def compute_success(obs):
    obj = np.asarray(obs["earbud_1_pos"], dtype=np.float32)
    slot = np.asarray(obs["charging_slot_1_pos"], dtype=np.float32)
    obj_slot_xy = float(np.linalg.norm(obj[:2] - slot[:2]))
    obj_slot_z = float(obj[2] - slot[2])
    success = float((obj_slot_xy < 0.02) and (obj_slot_z < 0.03))
    return {"success": success, "obj_slot_xy": obj_slot_xy, "obj_slot_z": obj_slot_z}


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
    return np.asarray(out["action"], dtype=np.float32)


def check_server(server_url: str):
    return http_json_request(server_url.rstrip("/") + "/health", obj=None)


class EpisodeVideoRecorder:
    def __init__(self, video_root: Path, task_name: str, level: str, episode_id: int, cameras, fps: int, flip_ud: bool):
        self.video_root = video_root
        self.task_name = safe_name(task_name)
        self.level = safe_name(level)
        self.episode_id = episode_id
        self.cameras = list(cameras)
        self.fps = int(max(1, fps))
        self.flip_ud = bool(flip_ud)
        self.writers = {}
        self.paths = {}

    def write_obs(self, obs):
        task_dir = self.video_root / self.level / self.task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        for cam in self.cameras:
            key = f"{cam}_image"
            if key not in obs:
                continue
            frame = np.asarray(obs[key], dtype=np.uint8)
            if self.flip_ud:
                frame = frame[::-1]

            if cam not in self.writers:
                out_path = task_dir / f"ep_{self.episode_id:03d}_{cam}.mp4"
                writer = imageio.get_writer(
                    str(out_path),
                    fps=self.fps,
                    codec="libx264",
                    quality=8,
                    macro_block_size=None,
                )
                self.writers[cam] = writer
                self.paths[cam] = str(out_path)

            self.writers[cam].append_data(frame)

    def close(self):
        for w in self.writers.values():
            try:
                w.close()
            except Exception:
                pass
        return dict(self.paths)


def run_one_episode(
    env,
    init_state,
    instruction,
    max_steps,
    server_url: str,
    task_name: str,
    level: str,
    episode_id: int,
    video_root: Path | None,
    video_cameras,
    video_fps: int,
    video_flip_ud: bool,
):
    env.reset()
    base = env.env if hasattr(env, "env") else env
    if hasattr(base, "_check_success"):
        base._check_success = lambda: False

    set_state_qpos_qvel(env, init_state)
    obs = get_latest_obs(env)
    reset_policy_server(server_url)

    recorder = None
    if video_root is not None:
        recorder = EpisodeVideoRecorder(
            video_root=video_root,
            task_name=task_name,
            level=level,
            episode_id=episode_id,
            cameras=video_cameras,
            fps=video_fps,
            flip_ud=video_flip_ud,
        )
        recorder.write_obs(obs)

    success = False
    steps = max_steps
    metrics = compute_success(obs)

    try:
        for step in range(max_steps):
            payload = make_payload(obs, instruction)
            action = infer_action(server_url, payload)

            obs, _, _, _ = env.step(action.astype(np.float32))
            if recorder is not None:
                recorder.write_obs(obs)

            metrics = compute_success(obs)
            if metrics["success"] > 0.5:
                success = True
                steps = step + 1
                break
    finally:
        video_paths = recorder.close() if recorder is not None else {}

    return success, steps, metrics, video_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--camera_size", type=int, default=512)
    parser.add_argument("--output_json", type=str, default="earbud_pi_eval_results.json")
    parser.add_argument("--benchmark_assets", type=str, required=True)
    parser.add_argument("--expected_per_level", type=int, default=20)
    parser.add_argument("--no_strict_count", action="store_true")

    parser.add_argument("--save_videos", action="store_true")
    parser.add_argument("--video_root", type=str, default="/root/autodl-tmp/openpi_earbud_proto/benchmark_eval_videos")
    parser.add_argument("--video_cameras", type=str, default="agentview,robot0_eye_in_hand")
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--no_video_flip_ud", action="store_true")
    parser.add_argument("--record_max_episodes_per_level", type=int, default=0, help="0 means all episodes")

    args = parser.parse_args()

    assets_dir = Path(args.benchmark_assets).resolve()
    tasks = load_task_specs(assets_dir)
    video_cameras = parse_camera_names(args.video_cameras)
    video_root = Path(args.video_root).resolve() if args.save_videos else None
    video_flip_ud = not args.no_video_flip_ud

    required_cams = {"agentview", "robot0_eye_in_hand"}
    env_cameras = sorted(required_cams.union(video_cameras if args.save_videos else []))

    health = check_server(args.server)
    print("[client] server health:", health)

    all_results = {
        "_meta": {
            "benchmark_assets": str(assets_dir),
            "expected_per_level": int(args.expected_per_level),
            "strict_count": bool(not args.no_strict_count),
            "save_videos": bool(args.save_videos),
            "video_root": str(video_root) if video_root else None,
            "video_cameras": video_cameras if args.save_videos else [],
            "video_fps": int(args.video_fps),
            "video_flip_ud": bool(video_flip_ud),
        }
    }

    for task in tasks:
        print(f"\n=== evaluating: {task.name} ({task.level}) ===")
        env = build_env(task, camera_size=args.camera_size, camera_names=env_cameras)
        init_states = load_init_states(task)
        print(f"[count] {task.level}: {len(init_states)} episodes")

        if (not args.no_strict_count) and (len(init_states) != args.expected_per_level):
            raise RuntimeError(
                f"{task.level} count mismatch: got {len(init_states)}, expected {args.expected_per_level}"
            )

        success_count = 0
        episode_logs = []

        for i, init_state in enumerate(init_states):
            need_video = args.save_videos and (
                args.record_max_episodes_per_level <= 0 or i < args.record_max_episodes_per_level
            )

            success, steps, metrics, video_paths = run_one_episode(
                env=env,
                init_state=init_state,
                instruction=task.language,
                max_steps=task.max_steps,
                server_url=args.server,
                task_name=task.name,
                level=task.level,
                episode_id=i,
                video_root=video_root if need_video else None,
                video_cameras=video_cameras,
                video_fps=args.video_fps,
                video_flip_ud=video_flip_ud,
            )

            success_count += int(success)
            log = {
                "episode_id": i,
                "success": bool(success),
                "steps": int(steps),
                "obj_slot_xy": float(metrics["obj_slot_xy"]),
                "obj_slot_z": float(metrics["obj_slot_z"]),
                "video_paths": video_paths,
            }
            episode_logs.append(log)
            print(log)

        success_rate = success_count / len(init_states) if init_states else 0.0
        all_results[task.name] = {
            "level": task.level,
            "success_rate": success_rate,
            "num_episodes": len(init_states),
            "episodes": episode_logs,
        }

        env.close()

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n[saved] {args.output_json}")
    if args.save_videos:
        print(f"[saved videos] {video_root}")


if __name__ == "__main__":
    main()
