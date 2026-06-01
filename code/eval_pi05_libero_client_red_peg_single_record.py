import argparse
import base64
import csv
import io
import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import imageio.v2 as imageio
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


def parse_levels(levels_arg: str):
    levels_arg = str(levels_arg).strip().lower()
    if levels_arg in ("", "all", "*"):
        return ["easy", "medium", "hard"]
    levels = [x.strip() for x in levels_arg.split(",") if x.strip()]
    valid = {"easy", "medium", "hard"}
    bad = [x for x in levels if x not in valid]
    if bad:
        raise ValueError(f"Unsupported level(s): {bad}. Use easy, medium, hard, or all.")
    return levels


def load_task_specs(assets_dir: Path, levels):
    tasks_json = assets_dir / "tasks.json"
    with tasks_json.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    tasks = []
    raw_levels = raw.get("levels", {})
    for level in levels:
        if level not in raw_levels:
            print(f"[skip] level '{level}' is not present in {tasks_json}")
            continue
        t = raw_levels[level]
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
    if not tasks:
        raise RuntimeError(f"No tasks selected from {tasks_json}. Requested levels: {levels}")
    return tasks


def build_env(task: TaskSpec, camera_size: int = 512):
    kwargs = dict(
        bddl_file_name=task.bddl_file,
        camera_heights=camera_size,
        camera_widths=camera_size,
        ignore_done=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
    )
    try:
        env = OffScreenRenderEnv(**kwargs)
    except TypeError:
        kwargs.pop("camera_names", None)
        env = OffScreenRenderEnv(**kwargs)

    for obj in (env, getattr(env, "env", None)):
        if obj is not None and hasattr(obj, "_check_success"):
            obj._check_success = lambda: False
    return env


def load_init_states(task: TaskSpec):
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


def normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError(f"Expected quaternion dim=4, got shape={q.shape}")
    return q / (np.linalg.norm(q) + 1e-12)


def quat_angle_rad(q1: Optional[np.ndarray], q2: Optional[np.ndarray]) -> float:
    if q1 is None or q2 is None:
        return float("nan")
    a = normalize_quat_wxyz(q1)
    b = normalize_quat_wxyz(q2)
    dot = float(abs(np.dot(a, b)))
    return float(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def obs_quat_wxyz(obs, object_name: str) -> Optional[np.ndarray]:
    for key in (f"{object_name}_quat_wxyz", f"{object_name}_quat"):
        if key not in obs:
            continue
        q = np.asarray(obs[key], dtype=np.float32).reshape(-1)
        if q.shape[0] != 4:
            continue
        if key.endswith("_quat_wxyz"):
            return normalize_quat_wxyz(q)
        # LIBERO / robosuite observations usually expose object quats as xyzw.
        return normalize_quat_wxyz(quat_xyzw_to_wxyz(q))
    return None


def compute_success(
    obs,
    initial_slot_pos: Optional[np.ndarray] = None,
    initial_slot_quat_wxyz: Optional[np.ndarray] = None,
    initial_peg_quat_wxyz: Optional[np.ndarray] = None,
    strict: bool = False,
    strict_xy: float = 0.015,
    strict_z_min: float = -0.005,
    strict_z_max: float = 0.035,
    strict_slot_pos: float = 0.006,
    strict_slot_angle: float = 0.20,
    strict_peg_angle: float = 0.55,
):
    obj = np.asarray(obs["earbud_1_pos"], dtype=np.float32)
    slot = np.asarray(obs["charging_slot_1_pos"], dtype=np.float32)
    obj_slot_xy = float(np.linalg.norm(obj[:2] - slot[:2]))
    obj_slot_z = float(obj[2] - slot[2])

    slot_pos_drift = float("nan")
    if initial_slot_pos is not None:
        slot_pos_drift = float(np.linalg.norm(slot - np.asarray(initial_slot_pos, dtype=np.float32).reshape(3)))
    slot_quat = obs_quat_wxyz(obs, "charging_slot_1")
    peg_quat = obs_quat_wxyz(obs, "earbud_1")
    slot_angle = quat_angle_rad(slot_quat, initial_slot_quat_wxyz)
    peg_angle = quat_angle_rad(peg_quat, initial_peg_quat_wxyz)

    naive_success = float((obj_slot_xy < 0.02) and (obj_slot_z < 0.03))
    strict_success = float(
        (obj_slot_xy < strict_xy)
        and (strict_z_min <= obj_slot_z <= strict_z_max)
        and (np.isnan(slot_pos_drift) or slot_pos_drift < strict_slot_pos)
        and (np.isnan(slot_angle) or slot_angle < strict_slot_angle)
        and (np.isnan(peg_angle) or peg_angle < strict_peg_angle)
    )
    success = strict_success if strict else naive_success
    return {
        "success": success,
        "naive_success": naive_success,
        "strict_success": strict_success,
        "obj_slot_xy": obj_slot_xy,
        "obj_slot_z": obj_slot_z,
        "slot_pos_drift": slot_pos_drift,
        "slot_quat_angle": slot_angle,
        "peg_quat_angle_from_init": peg_angle,
    }


def quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float32).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError(f"Expected quaternion dim=4, got shape={q.shape}")
    return np.asarray([q[3], q[0], q[1], q[2]], dtype=np.float32)


def get_quat_wxyz_from_obs(obs, stem: str) -> Optional[np.ndarray]:
    key_wxyz = f"{stem}_quat_wxyz"
    if key_wxyz in obs:
        q = np.asarray(obs[key_wxyz], dtype=np.float32).reshape(-1)
        if q.shape[0] == 4:
            return q
    key_xyzw = f"{stem}_quat"
    if key_xyzw in obs:
        q = np.asarray(obs[key_xyzw], dtype=np.float32).reshape(-1)
        if q.shape[0] == 4:
            return quat_xyzw_to_wxyz(q)
    return None


def rotmat_to_quat_wxyz(mat: np.ndarray) -> np.ndarray:
    m = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    trace = np.trace(m)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
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


def resolve_eef_site_id(env, sim) -> int:
    base = env.env if hasattr(env, "env") else env
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
    raise RuntimeError("Could not resolve end-effector site id.")


def resolve_object_joint_name(env, object_name: str) -> str:
    base = env.env if hasattr(env, "env") else env
    obj = getattr(base, "objects_dict", {}).get(object_name)
    if obj is None:
        raise RuntimeError(f"Could not find object {object_name!r} in env.objects_dict.")
    joints = getattr(obj, "joints", None)
    if joints and len(joints) > 0:
        return str(joints[0])
    raise RuntimeError(f"Could not find free joint for object {object_name!r}.")


def zero_joint_qvel(sim, joint_name: str) -> None:
    try:
        qvel = np.asarray(sim.data.get_joint_qvel(joint_name), dtype=np.float64)
        sim.data.set_joint_qvel(joint_name, np.zeros_like(qvel))
    except Exception:
        pass


def set_free_joint_pose(sim, joint_name: str, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
    q = np.asarray(sim.data.get_joint_qpos(joint_name), dtype=np.float64).copy()
    q[:3] = np.asarray(pos, dtype=np.float64).reshape(3)
    q[3:7] = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    sim.data.set_joint_qpos(joint_name, q)
    zero_joint_qvel(sim, joint_name)
    sim.forward()


def load_raw_object_pose(raw_episode: str, object_name: str, raw_index: int) -> Tuple[np.ndarray, np.ndarray]:
    raw = np.load(Path(raw_episode) / "trajectory.npz", allow_pickle=True)
    pos_key = f"{object_name}_pos"
    quat_key = f"{object_name}_quat_wxyz"
    if pos_key not in raw or quat_key not in raw:
        raise KeyError(f"Raw episode is missing {pos_key!r} or {quat_key!r}.")
    n = raw[pos_key].shape[0]
    idx = int(np.clip(raw_index, 0, max(0, n - 1)))
    return (
        np.asarray(raw[pos_key][idx], dtype=np.float32),
        np.asarray(raw[quat_key][idx], dtype=np.float32),
    )


def get_eef_quat_wxyz_from_sim(sim, eef_site_id: int) -> np.ndarray:
    xmat = np.asarray(sim.data.site_xmat[eef_site_id], dtype=np.float64).reshape(3, 3)
    return rotmat_to_quat_wxyz(xmat)


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


def make_payload(
    obs,
    instruction: str,
    eef_quat_wxyz: Optional[np.ndarray] = None,
    policy_step: Optional[int] = None,
    max_steps: Optional[int] = None,
):
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    if eef_quat_wxyz is not None:
        eef_quat_wxyz = np.asarray(eef_quat_wxyz, dtype=np.float32)
    elif "robot0_eef_quat_wxyz" in obs:
        eef_quat_wxyz = np.asarray(obs["robot0_eef_quat_wxyz"], dtype=np.float32)
    else:
        eef_quat_xyzw = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
        eef_quat_wxyz = quat_xyzw_to_wxyz(eef_quat_xyzw)
    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
    gripper_mean = np.asarray([float(np.mean(np.abs(gripper_qpos)))], dtype=np.float32)
    # Keep state encoding identical to training conversion:
    # [eef_pos(3), eef_quat_wxyz(4), gripper_mean(1)] => 8 dims.
    state = np.concatenate([eef_pos, eef_quat_wxyz, gripper_mean], axis=0).astype(np.float32)
    payload = {
        "task": instruction,
        "observation.state": state.tolist(),
        # Training videos were recorded with obs[cam][::-1], so keep inference images identical.
        "observation.images.image": encode_image_to_b64(obs["agentview_image"][::-1]),
        "observation.images.image2": encode_image_to_b64(obs["robot0_eye_in_hand_image"][::-1]),
    }
    if policy_step is not None:
        payload["policy_step"] = int(policy_step)
        denom = max(1, int(max_steps)) if max_steps is not None else 500
        payload["policy_progress"] = float(policy_step) / float(denom)
    # Privileged sim-only fields for diagnostic controllers. Normal policies
    # safely ignore these keys, but waypoint debuggers can use them to separate
    # action semantics from contact geometry.
    if "earbud_1_pos" in obs:
        payload["observation.privileged.peg_pos"] = np.asarray(obs["earbud_1_pos"], dtype=np.float32).tolist()
    if "charging_slot_1_pos" in obs:
        payload["observation.privileged.slot_pos"] = np.asarray(obs["charging_slot_1_pos"], dtype=np.float32).tolist()
    return payload


def http_json_request(url: str, obj=None, timeout: int = 1800):
    try:
        if obj is None:
            req = urllib.request.Request(url, method="GET")
        else:
            data = json.dumps(obj).encode("utf-8")
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}, method="POST"
            )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code} from {url}\n{body}") from e


def reset_policy_server(server_url: str):
    return http_json_request(server_url.rstrip("/") + "/reset", obj={})


def infer_action_chunk(server_url: str, payload):
    out = http_json_request(server_url.rstrip("/") + "/infer", obj=payload)
    action = np.asarray(out["action"], dtype=np.float32)
    # Server may return either (7,) or (T, 7). Normalize to (T, 7).
    if action.ndim == 1:
        if action.shape[0] != 7:
            raise ValueError(f"Unexpected action dim from server: {action.shape}")
        action = action[None, :]
    elif action.ndim == 2:
        if action.shape[1] != 7:
            # Fallback for accidental transposed shape (7, T).
            if action.shape[0] == 7:
                action = action.T
            else:
                raise ValueError(f"Unexpected action shape from server: {action.shape}")
    else:
        raise ValueError(f"Unexpected action shape from server: {action.shape}")
    return action


def check_server(server_url: str):
    return http_json_request(server_url.rstrip("/") + "/health", obj=None)


def capture_frames(obs, cameras, frames):
    for cam in cameras:
        key = f"{cam}_image"
        if key in obs:
            frame = np.asarray(obs[key], dtype=np.uint8)
            frames[cam].append(frame[::-1])


def save_videos(video_dir: Path, task: TaskSpec, episode_id: int, success: bool, frames, fps: int):
    out = {}
    ep_dir = video_dir / task.level / task.name / f"episode_{episode_id:03d}_{'succ' if success else 'fail'}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    for cam, cam_frames in frames.items():
        if len(cam_frames) == 0:
            continue
        vp = ep_dir / f"{cam}.mp4"
        imageio.mimwrite(str(vp), cam_frames, fps=max(1, int(fps)))
        out[cam] = str(vp)
    return out


def clip_policy_action(
    action,
    enabled: bool = True,
    pos_clip: float = 0.08,
    rot_xy_clip: float = 0.10,
    rot_z_clip: float = 0.08,
    gripper_clip: float = 1.0,
) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).copy()
    if not enabled:
        return action

    action[:3] = np.clip(action[:3], -float(pos_clip), float(pos_clip))
    action[3:5] = np.clip(action[3:5], -float(rot_xy_clip), float(rot_xy_clip))
    action[5] = np.clip(action[5], -float(rot_z_clip), float(rot_z_clip))
    action[6] = np.clip(action[6], -float(gripper_clip), float(gripper_clip))
    return action


def run_one_episode(
    env,
    init_state,
    instruction,
    max_steps,
    server_url: str,
    record=False,
    video_cameras=None,
    exec_horizon: int = 50,
    debug_action_trace: bool = False,
    debug_action_trace_steps: int = 0,
    debug_action_trace_start: int = 0,
    debug_action_trace_end: int = -1,
    debug_state_trace_steps: int = 0,
    debug_state_trace_start: int = 0,
    debug_state_trace_end: int = -1,
    action_trace_rows=None,
    state_trace_rows=None,
    task_name: str = "",
    level: str = "",
    episode_id: int = -1,
    clip_actions: bool = True,
    pos_action_clip: float = 0.08,
    rot_xy_action_clip: float = 0.10,
    rot_z_action_clip: float = 0.08,
    gripper_action_clip: float = 1.0,
    settle_after_set_state: int = 0,
    stop_on_success: bool = True,
    pin_slot_name: str = "",
    pin_slot_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None,
):
    env.reset()
    base = env.env if hasattr(env, "env") else env
    if hasattr(base, "_check_success"):
        base._check_success = lambda: False
    sim = base.sim
    eef_site_id = resolve_eef_site_id(env, sim)
    pin_slot_joint = None
    if pin_slot_pose is not None:
        pin_slot_joint = resolve_object_joint_name(env, pin_slot_name or "charging_slot_1")

    set_state_qpos_qvel(env, init_state)
    if pin_slot_joint is not None and pin_slot_pose is not None:
        set_free_joint_pose(sim, pin_slot_joint, pin_slot_pose[0], pin_slot_pose[1])
    obs = get_latest_obs(env)
    for _ in range(max(0, int(settle_after_set_state))):
        obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
        if pin_slot_joint is not None and pin_slot_pose is not None:
            set_free_joint_pose(sim, pin_slot_joint, pin_slot_pose[0], pin_slot_pose[1])
    if settle_after_set_state > 0:
        obs = get_latest_obs(env)
    reset_policy_server(server_url)

    frames = {c: [] for c in (video_cameras or [])} if record else None
    if record:
        capture_frames(obs, video_cameras, frames)

    pending_actions = []
    printed_debug = False
    ever_success = False
    last_success_metrics = None

    for step in range(max_steps):
        if pin_slot_joint is not None and pin_slot_pose is not None:
            set_free_joint_pose(sim, pin_slot_joint, pin_slot_pose[0], pin_slot_pose[1])
            obs = get_latest_obs(env)
        if not pending_actions:
            payload = make_payload(
                obs,
                instruction,
                eef_quat_wxyz=get_eef_quat_wxyz_from_sim(sim, eef_site_id),
                policy_step=step,
                max_steps=max_steps,
            )
            action_chunk = infer_action_chunk(server_url, payload)  # (T, 7)
            chunk_len = int(action_chunk.shape[0])
            use_len = chunk_len if exec_horizon <= 0 else min(int(exec_horizon), chunk_len)
            pending_actions = [action_chunk[i] for i in range(use_len)]

            if debug_action_trace and not printed_debug:
                print(
                    "[debug_action_trace] "
                    f"step={step} chunk_len={chunk_len} use_len={use_len} "
                    f"first_gripper={float(action_chunk[0, -1]):.4f} "
                    f"last_gripper={float(action_chunk[use_len - 1, -1]):.4f}"
                )
                printed_debug = True

        action = clip_policy_action(
            pending_actions.pop(0),
            enabled=clip_actions,
            pos_clip=pos_action_clip,
            rot_xy_clip=rot_xy_action_clip,
            rot_z_clip=rot_z_action_clip,
            gripper_clip=gripper_action_clip,
        )
        if debug_action_trace_steps > 0:
            in_action_window = step >= int(debug_action_trace_start) and (
                int(debug_action_trace_end) < 0 or step <= int(debug_action_trace_end)
            )
        else:
            in_action_window = False
        if debug_action_trace_steps > 0 and in_action_window and (
            step < 10 or (step - int(debug_action_trace_start)) % int(debug_action_trace_steps) == 0
        ):
            print(
                "[action_trace] "
                f"level={level} episode={episode_id} step={step} "
                f"xyz=({float(action[0]):+.4f},{float(action[1]):+.4f},{float(action[2]):+.4f}) "
                f"rot=({float(action[3]):+.4f},{float(action[4]):+.4f},{float(action[5]):+.4f}) "
                f"gripper={float(action[6]):+.4f}"
            )
        if action_trace_rows is not None:
            action_trace_rows.append(
                {
                    "task": task_name,
                    "level": level,
                    "episode_id": episode_id,
                    "step": step,
                    "a0": float(action[0]),
                    "a1": float(action[1]),
                    "a2": float(action[2]),
                    "a3": float(action[3]),
                    "a4": float(action[4]),
                    "a5": float(action[5]),
                    "gripper": float(action[6]),
                }
            )
        obs, _, _, _ = env.step(action)
        if pin_slot_joint is not None and pin_slot_pose is not None:
            set_free_joint_pose(sim, pin_slot_joint, pin_slot_pose[0], pin_slot_pose[1])
            obs = get_latest_obs(env)
        need_state_values = state_trace_rows is not None or debug_state_trace_steps > 0
        if need_state_values:
            eef_pos = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32).reshape(-1)[:3]
            eef_quat = get_eef_quat_wxyz_from_sim(sim, eef_site_id)
            peg_pos = np.asarray(obs.get("earbud_1_pos", np.zeros(3)), dtype=np.float32).reshape(-1)[:3]
            peg_quat = get_quat_wxyz_from_obs(obs, "earbud_1")
            slot_pos = np.asarray(obs.get("charging_slot_1_pos", np.zeros(3)), dtype=np.float32).reshape(-1)[:3]
            slot_quat = get_quat_wxyz_from_obs(obs, "charging_slot_1")
            gripper_qpos = np.asarray(obs.get("robot0_gripper_qpos", np.zeros(2)), dtype=np.float32).reshape(-1)
            gripper_mean_abs = float(np.mean(np.abs(gripper_qpos))) if gripper_qpos.size else 0.0
            eef_obj = peg_pos - eef_pos
            obj_slot = peg_pos - slot_pos
            if peg_quat is None:
                peg_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            if slot_quat is None:
                slot_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        if state_trace_rows is not None:
            state_trace_rows.append(
                {
                    "task": task_name,
                    "level": level,
                    "episode_id": episode_id,
                    "step": step,
                    "eef_x": float(eef_pos[0]),
                    "eef_y": float(eef_pos[1]),
                    "eef_z": float(eef_pos[2]),
                    "eef_qw": float(eef_quat[0]),
                    "eef_qx": float(eef_quat[1]),
                    "eef_qy": float(eef_quat[2]),
                    "eef_qz": float(eef_quat[3]),
                    "peg_x": float(peg_pos[0]),
                    "peg_y": float(peg_pos[1]),
                    "peg_z": float(peg_pos[2]),
                    "peg_qw": float(peg_quat[0]),
                    "peg_qx": float(peg_quat[1]),
                    "peg_qy": float(peg_quat[2]),
                    "peg_qz": float(peg_quat[3]),
                    "slot_x": float(slot_pos[0]),
                    "slot_y": float(slot_pos[1]),
                    "slot_z": float(slot_pos[2]),
                    "slot_qw": float(slot_quat[0]),
                    "slot_qx": float(slot_quat[1]),
                    "slot_qy": float(slot_quat[2]),
                    "slot_qz": float(slot_quat[3]),
                    "gripper_qpos_mean_abs": gripper_mean_abs,
                    "eef_obj_dx": float(eef_obj[0]),
                    "eef_obj_dy": float(eef_obj[1]),
                    "eef_obj_dz": float(eef_obj[2]),
                    "eef_obj_dist": float(np.linalg.norm(eef_obj)),
                    "obj_slot_xy": float(np.linalg.norm(obj_slot[:2])),
                    "obj_slot_z": float(obj_slot[2]),
                }
            )
        if debug_state_trace_steps > 0:
            in_window = step >= int(debug_state_trace_start) and (
                int(debug_state_trace_end) < 0 or step <= int(debug_state_trace_end)
            )
            should_print = in_window and (
                step < 10 or (step - int(debug_state_trace_start)) % int(debug_state_trace_steps) == 0
            )
            if should_print:
                print(
                    "[state_trace] "
                    f"level={level} episode={episode_id} step={step} "
                    f"eef=({float(eef_pos[0]):+.5f},{float(eef_pos[1]):+.5f},{float(eef_pos[2]):+.5f}) "
                    f"peg=({float(peg_pos[0]):+.5f},{float(peg_pos[1]):+.5f},{float(peg_pos[2]):+.5f}) "
                    f"peg-eef=({float(eef_obj[0]):+.5f},{float(eef_obj[1]):+.5f},{float(eef_obj[2]):+.5f}) "
                    f"dist={float(np.linalg.norm(eef_obj)):.5f} "
                    f"slot_xy={float(np.linalg.norm(obj_slot[:2])):.5f} "
                    f"slot_z={float(obj_slot[2]):+.5f} "
                    f"grip_q={gripper_mean_abs:.5f}"
                )
        if record:
            capture_frames(obs, video_cameras, frames)

        metrics = compute_success(obs)
        if metrics["success"] > 0.5:
            ever_success = True
            last_success_metrics = metrics
            if stop_on_success:
                return True, step + 1, metrics, frames

    metrics = compute_success(obs)
    if ever_success:
        return True, max_steps, last_success_metrics or metrics, frames
    return False, max_steps, metrics, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--camera_size", type=int, default=256)
    parser.add_argument("--output_json", type=str, default="earbud_pi05_eval_results.json")
    parser.add_argument("--benchmark_assets", type=str, required=True)
    parser.add_argument("--levels", type=str, default="easy", help="Comma-separated levels to evaluate: easy,medium,hard or all.")
    parser.add_argument("--expected_per_level", type=int, default=1)
    parser.add_argument("--no_strict_count", action="store_true")
    parser.add_argument("--episode_start", type=int, default=0, help="Start index inside each selected level.")
    parser.add_argument("--max_episodes_per_level", type=int, default=1, help="0 means evaluate all episodes.")

    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_dir", type=str, default="/root/autodl-tmp/openpi_earbud_proto/benchmark_eval_videos")
    parser.add_argument("--video_cameras", type=str, default="agentview")
    parser.add_argument("--video_fps", type=int, default=20)
    parser.add_argument("--record_limit_per_level", type=int, default=1)  # 0=all
    parser.add_argument("--exec_horizon", type=int, default=50)
    parser.add_argument("--disable_action_clip", action="store_true")
    parser.add_argument("--pos_action_clip", type=float, default=0.08)
    parser.add_argument("--rot_xy_action_clip", type=float, default=0.10)
    parser.add_argument("--rot_z_action_clip", type=float, default=0.08)
    parser.add_argument("--gripper_action_clip", type=float, default=1.0)
    parser.add_argument(
        "--settle_after_set_state",
        type=int,
        default=0,
        help="Run this many zero-action env steps after setting qpos/qvel and before policy reset.",
    )
    parser.add_argument(
        "--no_stop_on_success",
        action="store_true",
        help="Keep stepping after success so diagnostic traces cover the full horizon.",
    )
    parser.add_argument(
        "--pin_slot_to_raw_episode",
        type=str,
        default="",
        help="Diagnostic only: keep the slot free joint pinned to this raw episode pose every step.",
    )
    parser.add_argument("--pin_slot_name", type=str, default="charging_slot_1")
    parser.add_argument("--pin_slot_raw_index", type=int, default=0)
    parser.add_argument("--debug_action_trace", action="store_true")
    parser.add_argument("--debug_action_trace_steps", type=int, default=0)
    parser.add_argument("--debug_action_trace_start", type=int, default=0)
    parser.add_argument("--debug_action_trace_end", type=int, default=-1)
    parser.add_argument(
        "--debug_state_trace_steps",
        type=int,
        default=0,
        help="Print state trace every N steps inside [debug_state_trace_start, debug_state_trace_end].",
    )
    parser.add_argument("--debug_state_trace_start", type=int, default=0)
    parser.add_argument("--debug_state_trace_end", type=int, default=-1)
    parser.add_argument("--action_trace_csv", type=str, default="")
    parser.add_argument("--state_trace_csv", type=str, default="")
    parser.add_argument(
        "--task_override",
        type=str,
        default="",
        help="Temporarily override benchmark language instruction for prompt-sensitivity checks.",
    )
    parser.add_argument(
        "--require_adapter_loaded",
        action="store_true",
        help="Fail fast if the policy server health does not report adapter_loaded=true.",
    )

    args = parser.parse_args()

    assets_dir = Path(args.benchmark_assets).resolve()
    selected_levels = parse_levels(args.levels)
    tasks = load_task_specs(assets_dir, selected_levels)
    video_cameras = [x.strip() for x in args.video_cameras.split(",") if x.strip()]
    video_dir = Path(args.video_dir).resolve()

    health = check_server(args.server)
    print("[client] server health:", health)
    if args.require_adapter_loaded and not bool(health.get("adapter_loaded", False)):
        raise RuntimeError(
            "Server did not report adapter_loaded=true. "
            "For LoRA evaluation, start policy_server_pi05.py with --base_policy_path "
            "and the patched LoRA adapter-loading server."
        )

    all_results = {"_meta": {"benchmark_assets": str(assets_dir), "server": args.server}}
    action_trace_rows = [] if args.action_trace_csv else None
    state_trace_rows = [] if args.state_trace_csv else None
    pin_slot_pose = None
    if args.pin_slot_to_raw_episode:
        pin_slot_pose = load_raw_object_pose(
            args.pin_slot_to_raw_episode,
            object_name=args.pin_slot_name,
            raw_index=args.pin_slot_raw_index,
        )
        print(
            "[pin_slot] "
            f"name={args.pin_slot_name} raw_episode={args.pin_slot_to_raw_episode} "
            f"raw_index={args.pin_slot_raw_index} pos={pin_slot_pose[0].round(5).tolist()}"
        )

    for task in tasks:
        print(f"\n=== evaluating: {task.name} ({task.level}) ===")
        env = build_env(task, camera_size=args.camera_size)
        init_states = load_init_states(task)
        total_init_states = len(init_states)
        start = max(0, int(args.episode_start))
        end = None if args.max_episodes_per_level <= 0 else start + int(args.max_episodes_per_level)
        init_states = init_states[start:end]
        print(f"[count] {task.level}: selected {len(init_states)} / total {total_init_states} episodes")

        if (not args.no_strict_count) and (len(init_states) != args.expected_per_level):
            raise RuntimeError(f"{task.level} count mismatch: got {len(init_states)}, expected {args.expected_per_level}")

        success_count = 0
        episode_logs = []

        for local_i, init_state in enumerate(init_states):
            i = start + local_i
            record_this = args.record_video and (
                args.record_limit_per_level <= 0 or local_i < args.record_limit_per_level
            )
            instruction = args.task_override.strip() or task.language
            success, steps, metrics, frames = run_one_episode(
                env=env,
                init_state=init_state,
                instruction=instruction,
                max_steps=task.max_steps,
                server_url=args.server,
                record=record_this,
                video_cameras=video_cameras,
                exec_horizon=args.exec_horizon,
                debug_action_trace=args.debug_action_trace,
                debug_action_trace_steps=args.debug_action_trace_steps,
                debug_action_trace_start=args.debug_action_trace_start,
                debug_action_trace_end=args.debug_action_trace_end,
                debug_state_trace_steps=args.debug_state_trace_steps,
                debug_state_trace_start=args.debug_state_trace_start,
                debug_state_trace_end=args.debug_state_trace_end,
                action_trace_rows=action_trace_rows,
                state_trace_rows=state_trace_rows,
                task_name=task.name,
                level=task.level,
                episode_id=i,
                clip_actions=not args.disable_action_clip,
                pos_action_clip=args.pos_action_clip,
                rot_xy_action_clip=args.rot_xy_action_clip,
                rot_z_action_clip=args.rot_z_action_clip,
                gripper_action_clip=args.gripper_action_clip,
                settle_after_set_state=args.settle_after_set_state,
                stop_on_success=not args.no_stop_on_success,
                pin_slot_name=args.pin_slot_name,
                pin_slot_pose=pin_slot_pose,
            )
            success_count += int(success)

            log = {
                "episode_id": i,
                "success": bool(success),
                "steps": int(steps),
                "obj_slot_xy": float(metrics["obj_slot_xy"]),
                "obj_slot_z": float(metrics["obj_slot_z"]),
            }

            if record_this and frames is not None:
                log["videos"] = save_videos(video_dir, task, i, bool(success), frames, args.video_fps)

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

    if action_trace_rows is not None:
        trace_path = Path(args.action_trace_csv).resolve()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["task", "level", "episode_id", "step", "a0", "a1", "a2", "a3", "a4", "a5", "gripper"],
            )
            writer.writeheader()
            writer.writerows(action_trace_rows)
        print(f"[saved] action_trace_csv: {trace_path}")

    if state_trace_rows is not None:
        trace_path = Path(args.state_trace_csv).resolve()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "task",
            "level",
            "episode_id",
            "step",
            "eef_x",
            "eef_y",
            "eef_z",
            "eef_qw",
            "eef_qx",
            "eef_qy",
            "eef_qz",
            "peg_x",
            "peg_y",
            "peg_z",
            "peg_qw",
            "peg_qx",
            "peg_qy",
            "peg_qz",
            "slot_x",
            "slot_y",
            "slot_z",
            "slot_qw",
            "slot_qx",
            "slot_qy",
            "slot_qz",
            "gripper_qpos_mean_abs",
            "eef_obj_dx",
            "eef_obj_dy",
            "eef_obj_dz",
            "eef_obj_dist",
            "obj_slot_xy",
            "obj_slot_z",
        ]
        with trace_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(state_trace_rows)
        print(f"[saved] state_trace_csv: {trace_path}")


if __name__ == "__main__":
    main()
