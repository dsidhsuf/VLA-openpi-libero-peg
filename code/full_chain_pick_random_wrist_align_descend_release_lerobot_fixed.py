import os
import json
from datetime import datetime
import argparse
from pathlib import Path

import numpy as np
import imageio

from libero.libero.envs import OffScreenRenderEnv

BASE_DIR = "/root/autodl-tmp/openpi_earbud_proto/third_party/libero/libero/libero/bddl_files/libero_90"

CAMERA_SIZE = 768
DEFAULT_CONTROL_HZ = 20
DEFAULT_VIDEO_FPS = 0
DEFAULT_FRAME_STRIDE = 2
DEFAULT_MAX_VIDEO_FRAMES = 0
DEFAULT_PLAYBACK_SPEED = 1.0
DEFAULT_CAMERA_NAMES = ("agentview",)

# Keep the successful baseline position pipeline intact.
KP_POS = 2.0
POS_CLIP = 0.08
POS_CLIP_RELEASE = 0.02

SAFE_TRAVEL_Z = 0.62
PREGRASP_Z_OFFSET = 0.08
PRECLOSE_Z_OFFSET = 0.028
CAGE_Z_OFFSET = 0.014
LIFT_Z_OFFSET = 0.10
SLOT_HOVER_Z_OFFSET = 0.08
PRE_INSERT_OBJ_Z_OFFSET = 0.030
FINAL_INSERT_OBJ_Z_OFFSET = 0.012
PREGRASP_Z_OFFSET_FLAT = 0.05
PRECLOSE_Z_OFFSET_FLAT = 0.018
CAGE_Z_OFFSET_FLAT = 0.010
LIFT_Z_OFFSET_FLAT = 0.12

GRASP_X_OFFSET = -0.010
GRASP_Y_OFFSET = 0.000
GRASP_X_OFFSET_FLAT = 0.000
GRASP_Y_OFFSET_FLAT = 0.000
FLAT_SIDE_APPROACH_DIST = 0.040
FLAT_SIDE_CONTACT_DIST = 0.010
FLAT_SIDE_GRASP_Z_OFFSET = 0.020
FLAT_GRASP_ROLL_DEG = 0.0
FLAT_GRASP_PITCH_DEG = -90.0

MAX_SERVO_STEPS = 160
PRECLOSE_STEPS = 28
CLOSE_GRIPPER_STEPS = 40
POST_MOVE_HOLD_STEPS = 16

GRIP_OPEN = -1.0
GRIP_CLOSE = 1.0

RELEASE_DESCEND_Z = 0.030
PRE_RELEASE_HOLD_STEPS = 16
RELEASE_HOLD_STEPS = 24
RETREAT_Z = 0.06
TARGET_OPEN_ABS = 0.039

# Slot pose stays fixed as in the baseline.
SLOT_Y_DEG = -12.0

# New: random initial object yaw plus explicit wrist rotation later.
RANDOM_YAW_MIN_DEG = -90.0
RANDOM_YAW_MAX_DEG = 90.0
FLAT_REST_PROB = 0.5
FLAT_REST_ROLL_DEG = 90.0
EARBUD_EDGE_REST_Z = 0.4435
EARBUD_FLAT_REST_Z = 0.4435

# New: align wrist to the object before grasping.
# In your successful baseline, eef yaw is about 90 deg when the earbud is also
# effectively aligned at about 90 deg, so the default offset is 0 deg.
GRASP_EEF_YAW_OFFSET_DEG = 0.0
GRASP_EEF_YAW_OFFSET_DEG_FLAT = 90.0

# Use projected local long-axis directions instead of raw rigid-body yaw.
# For this task, the long rectangular insert feature is usually along local z.
EARBUD_LONG_AXIS_LOCAL = np.array([0.0, 0.0, 1.0], dtype=float)
SLOT_LONG_AXIS_LOCAL = np.array([0.0, 0.0, 1.0], dtype=float)

# The slot body has nearly zero world yaw in this setup.
# The earbud long-edge target is the slot long-edge yaw, which in your baseline
# is effectively "slot yaw + 90 deg".
SLOT_LONG_AXIS_YAW_OFFSET_DEG = 90.0

# Wrist rotation controller. These only affect the new rotate-in-air phase.
KP_YAW = 2.2
YAW_CLIP = 0.08
KP_ROT = 2.0
ROT_CLIP = 0.10
YAW_TOL_DEG = 2.0
RP_TOL_DEG = 4.0
MAX_ROTATE_STEPS = 200
ROTATE_SETTLE_STEPS = 16
MAX_OBJECT_YAW_ALIGN_STEPS = 420


def quat_wxyz_from_axis_angle(axis, deg):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    theta = np.deg2rad(deg)
    w = np.cos(theta / 2.0)
    xyz = axis * np.sin(theta / 2.0)
    return np.array([w, xyz[0], xyz[1], xyz[2]], dtype=float)


def quat_mul_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=float)


def quat_to_rotmat_wxyz(q):
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)


def rotmat_to_quat_wxyz(m):
    m = np.asarray(m, dtype=float)
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
    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)


def yaw_deg_from_quat_wxyz(q):
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.rad2deg(np.arctan2(siny_cosp, cosy_cosp))


def rpy_deg_from_quat_wxyz(q):
    rot = quat_to_rotmat_wxyz(q)
    sy = np.sqrt(rot[0, 0] * rot[0, 0] + rot[1, 0] * rot[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(rot[2, 1], rot[2, 2])
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
    else:
        roll = np.arctan2(-rot[1, 2], rot[1, 1])
        pitch = np.arctan2(-rot[2, 0], sy)
        yaw = 0.0
    return np.rad2deg(np.array([roll, pitch, yaw], dtype=float))


def rotmat_from_rpy_deg(rpy_deg):
    roll, pitch, yaw = np.deg2rad(np.asarray(rpy_deg, dtype=float))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def rotvec_from_rotmat(rot):
    rot = np.asarray(rot, dtype=float)
    trace = np.trace(rot)
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3, dtype=float)
    wx = rot[2, 1] - rot[1, 2]
    wy = rot[0, 2] - rot[2, 0]
    wz = rot[1, 0] - rot[0, 1]
    axis = np.array([wx, wy, wz], dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        return np.zeros(3, dtype=float)
    axis = axis / axis_norm
    return axis * theta


def wrap_deg(x):
    return (x + 180.0) % 360.0 - 180.0


def canonical_axis_deg(angle_deg):
    return angle_deg % 180.0


def wrap_axis_err_deg(target_deg, current_deg):
    return (target_deg - current_deg + 90.0) % 180.0 - 90.0


def projected_axis_heading_deg_from_quat_wxyz(q, local_axis):
    rot = quat_to_rotmat_wxyz(q)
    axis_world = rot @ np.asarray(local_axis, dtype=float)
    axis_xy = axis_world[:2]
    norm_xy = np.linalg.norm(axis_xy)
    if norm_xy < 1e-8:
        return 0.0
    heading = np.rad2deg(np.arctan2(axis_xy[1], axis_xy[0]))
    return canonical_axis_deg(heading)


def get_sim(env):
    return env.env.sim if hasattr(env, "env") else env.sim


def get_joint_name(env, object_name):
    obj = env.env.objects_dict[object_name] if hasattr(env, "env") else env.objects_dict[object_name]
    joints = getattr(obj, "joints", None)
    if joints and len(joints) > 0:
        return joints[0]
    raise RuntimeError(f"Could not find free joint for {object_name}")


def get_joint_qpos(sim, joint_name):
    return np.array(sim.data.get_joint_qpos(joint_name), dtype=float)


def set_joint_qpos(sim, joint_name, qpos):
    sim.data.set_joint_qpos(joint_name, qpos)
    sim.forward()


def set_joint_qvel_zero(sim, joint_name):
    try:
        qvel = np.array(sim.data.get_joint_qvel(joint_name), dtype=float)
        sim.data.set_joint_qvel(joint_name, np.zeros_like(qvel))
    except Exception:
        pass


def resolve_eef_site_id(env, sim):
    base = env.env if hasattr(env, "env") else env
    robots = getattr(base, "robots", None)
    if robots:
        robot = robots[0]
        for attr in ("eef_site_id", "grip_site_id"):
            if hasattr(robot, attr):
                val = getattr(robot, attr)
                if isinstance(val, (int, np.integer)):
                    return int(val)
        if hasattr(robot, "eef_site_id") and isinstance(robot.eef_site_id, dict):
            vals = list(robot.eef_site_id.values())
            if vals:
                return int(vals[0])

    for name in (
        "gripper0_grip_site",
        "robot0_grip_site",
        "eef_site",
        "grip_site",
    ):
        try:
            return sim.model.site_name2id(name)
        except Exception:
            continue

    raise RuntimeError("Could not resolve end-effector site id for yaw readout")


def get_eef_quat_wxyz(sim, eef_site_id):
    xmat = np.array(sim.data.site_xmat[eef_site_id], dtype=float).reshape(3, 3)
    return rotmat_to_quat_wxyz(xmat)


def parse_camera_names(raw_camera_names):
    if raw_camera_names is None:
        return list(DEFAULT_CAMERA_NAMES)
    if isinstance(raw_camera_names, str):
        names = [name.strip() for name in raw_camera_names.split(",")]
        names = [name for name in names if name]
        return names if names else list(DEFAULT_CAMERA_NAMES)
    if isinstance(raw_camera_names, (list, tuple)):
        names = [str(name).strip() for name in raw_camera_names if str(name).strip()]
        return names if names else list(DEFAULT_CAMERA_NAMES)
    return list(DEFAULT_CAMERA_NAMES)


def build_level_cfg(bddl_base_dir):
    return {
        "easy": {
            "bddl": os.path.join(bddl_base_dir, "LIVING_ROOM_SCENE1_insert_the_earbud_into_the_charging_slot_easy.bddl"),
        },
        "medium": {
            "bddl": os.path.join(bddl_base_dir, "LIVING_ROOM_SCENE1_insert_the_earbud_into_the_charging_slot_medium.bddl"),
        },
        "hard": {
            "bddl": os.path.join(bddl_base_dir, "LIVING_ROOM_SCENE1_insert_the_earbud_into_the_charging_slot_hard.bddl"),
        },
    }


def make_offscreen_env(bddl_file, camera_size, camera_names):
    kwargs = dict(
        bddl_file_name=bddl_file,
        camera_heights=camera_size,
        camera_widths=camera_size,
        ignore_done=True,
    )
    if camera_names:
        kwargs["camera_names"] = list(camera_names)

    try:
        return OffScreenRenderEnv(**kwargs)
    except TypeError:
        kwargs.pop("camera_names", None)
        return OffScreenRenderEnv(**kwargs)


def _to_f32(x):
    return np.asarray(x, dtype=np.float32)


class EpisodeRecorder:
    def __init__(
        self,
        requested_camera_names,
        frame_stride=DEFAULT_FRAME_STRIDE,
        video_fps=DEFAULT_VIDEO_FPS,
        max_video_frames=DEFAULT_MAX_VIDEO_FRAMES,
        control_hz=DEFAULT_CONTROL_HZ,
    ):
        self.requested_camera_names = list(requested_camera_names)
        self.frame_stride = int(max(1, frame_stride))
        self.video_fps = int(max(1, video_fps))
        self.max_video_frames = None if max_video_frames is None or max_video_frames <= 0 else int(max_video_frames)
        self.control_hz = float(max(1, control_hz))

        self.selected_camera_names = []
        self.frames = {}
        self.phase = "init"
        self.step_count = 0
        self.captured_frame_count = 0
        self.video_cap_hit = False

        self.records = {
            "step": [],
            "timestamp_s": [],
            "phase": [],
            "action": [],
            "reward": [],
            "done": [],
            "robot0_eef_pos": [],
            "robot0_eef_quat_wxyz": [],
            "robot0_gripper_qpos": [],
            "earbud_1_pos": [],
            "earbud_1_quat_wxyz": [],
            "charging_slot_1_pos": [],
            "charging_slot_1_quat_wxyz": [],
            "observation_state": [],
        }

    def set_phase(self, phase_name):
        self.phase = str(phase_name)

    def _discover_cameras(self, obs):
        if self.selected_camera_names:
            return
        available = [k[:-6] for k in obs.keys() if k.endswith("_image")]
        requested = [c for c in self.requested_camera_names if c in available]
        selected = requested if requested else available
        if not selected:
            selected = []
        self.selected_camera_names = selected
        self.frames = {cam: [] for cam in self.selected_camera_names}
        print(f"selected_cameras={self.selected_camera_names}")

    def _capture_frames(self, obs, force=False):
        self._discover_cameras(obs)
        if not self.selected_camera_names:
            return
        if (not force) and (self.step_count % self.frame_stride != 0):
            return
        if (not force) and self.max_video_frames is not None and self.captured_frame_count >= self.max_video_frames:
            self.video_cap_hit = True
            return

        captured = False
        for cam in self.selected_camera_names:
            key = f"{cam}_image"
            if key in obs:
                frame = np.asarray(obs[key], dtype=np.uint8)[::-1]
                self.frames[cam].append(frame)
                captured = True
        if captured:
            self.captured_frame_count += 1

    def capture_initial(self, obs):
        self._capture_frames(obs, force=True)

    def log_step(self, obs, action, reward, done, sim, eef_site_id, earbud_joint_name, slot_joint_name):
        self.step_count += 1
        self._capture_frames(obs)

        eef_pos = _to_f32(obs["robot0_eef_pos"])
        eef_quat = _to_f32(get_eef_quat_wxyz(sim, eef_site_id))
        grip = _to_f32(obs["robot0_gripper_qpos"])
        earbud_q = _to_f32(get_joint_qpos(sim, earbud_joint_name))
        slot_q = _to_f32(get_joint_qpos(sim, slot_joint_name))

        observation_state = np.concatenate(
            [
                eef_pos,
                eef_quat,
                grip,
                earbud_q[:3],
                earbud_q[3:7],
                slot_q[:3],
                slot_q[3:7],
            ],
            axis=0,
        ).astype(np.float32)

        self.records["step"].append(self.step_count)
        self.records["timestamp_s"].append(self.step_count / self.control_hz)
        self.records["phase"].append(self.phase)
        self.records["action"].append(_to_f32(action))
        self.records["reward"].append(float(reward))
        self.records["done"].append(bool(done))
        self.records["robot0_eef_pos"].append(eef_pos)
        self.records["robot0_eef_quat_wxyz"].append(eef_quat)
        self.records["robot0_gripper_qpos"].append(grip)
        self.records["earbud_1_pos"].append(earbud_q[:3])
        self.records["earbud_1_quat_wxyz"].append(earbud_q[3:7])
        self.records["charging_slot_1_pos"].append(slot_q[:3])
        self.records["charging_slot_1_quat_wxyz"].append(slot_q[3:7])
        self.records["observation_state"].append(observation_state)

    def _as_array(self, key, dtype=np.float32):
        vals = self.records[key]
        if len(vals) == 0:
            return np.zeros((0,), dtype=dtype)
        return np.asarray(vals, dtype=dtype)

    def finalize(self, episode_dir, episode_meta):
        episode_dir = Path(episode_dir)
        episode_dir.mkdir(parents=True, exist_ok=True)
        video_dir = episode_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        traj_npz = episode_dir / "trajectory.npz"
        np.savez_compressed(
            traj_npz,
            step=self._as_array("step", dtype=np.int32),
            timestamp_s=self._as_array("timestamp_s", dtype=np.float32),
            action=self._as_array("action", dtype=np.float32),
            reward=self._as_array("reward", dtype=np.float32),
            done=self._as_array("done", dtype=np.bool_),
            robot0_eef_pos=self._as_array("robot0_eef_pos", dtype=np.float32),
            robot0_eef_quat_wxyz=self._as_array("robot0_eef_quat_wxyz", dtype=np.float32),
            robot0_gripper_qpos=self._as_array("robot0_gripper_qpos", dtype=np.float32),
            earbud_1_pos=self._as_array("earbud_1_pos", dtype=np.float32),
            earbud_1_quat_wxyz=self._as_array("earbud_1_quat_wxyz", dtype=np.float32),
            charging_slot_1_pos=self._as_array("charging_slot_1_pos", dtype=np.float32),
            charging_slot_1_quat_wxyz=self._as_array("charging_slot_1_quat_wxyz", dtype=np.float32),
            observation_state=self._as_array("observation_state", dtype=np.float32),
        )

        phases_path = episode_dir / "phases.json"
        with phases_path.open("w", encoding="utf-8") as f:
            json.dump(self.records["phase"], f, ensure_ascii=False)

        step_jsonl = episode_dir / "steps.jsonl"
        with step_jsonl.open("w", encoding="utf-8") as f:
            for i in range(len(self.records["step"])):
                row = {
                    "step": int(self.records["step"][i]),
                    "timestamp_s": float(self.records["timestamp_s"][i]),
                    "phase": self.records["phase"][i],
                    "reward": float(self.records["reward"][i]),
                    "done": bool(self.records["done"][i]),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        video_paths = {}
        for cam in self.selected_camera_names:
            cam_frames = self.frames.get(cam, [])
            if not cam_frames:
                continue
            video_path = video_dir / f"{cam}.mp4"
            imageio.mimwrite(str(video_path), cam_frames, fps=self.video_fps)
            video_paths[cam] = str(video_path)

        init_png_path = None
        preview_path = None
        if self.selected_camera_names:
            preferred = "agentview" if "agentview" in self.selected_camera_names else self.selected_camera_names[0]
            preferred_frames = self.frames.get(preferred, [])
            if preferred_frames:
                init_png = episode_dir / f"init_{preferred}.png"
                imageio.imwrite(str(init_png), preferred_frames[0])
                init_png_path = str(init_png)

                preview = episode_dir / "preview.mp4"
                imageio.mimwrite(str(preview), preferred_frames, fps=self.video_fps)
                preview_path = str(preview)

        metadata = dict(episode_meta)
        metadata.update(
            {
                "num_steps": int(len(self.records["step"])),
                "selected_cameras": self.selected_camera_names,
                "video_fps": self.video_fps,
                "frame_stride": self.frame_stride,
                "control_hz": self.control_hz,
                "video_cap_hit": bool(self.video_cap_hit),
                "paths": {
                    "trajectory_npz": str(traj_npz),
                    "phases_json": str(phases_path),
                    "steps_jsonl": str(step_jsonl),
                    "videos": video_paths,
                    "preview_video": preview_path,
                    "init_png": init_png_path,
                },
                "lerobot_columns_hint": {
                    "observation.state": "trajectory.npz:observation_state",
                    "action": "trajectory.npz:action",
                },
            }
        )

        metadata_path = episode_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        metadata["paths"]["metadata_json"] = str(metadata_path)
        return metadata


def append_jsonl(path, payload):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def rollout(
    level: str,
    seed: int,
    random_yaw_min_deg: float,
    random_yaw_max_deg: float,
    flat_rest_prob: float,
    out_root: str,
    bddl_base_dir: str,
    camera_size: int,
    camera_names,
    video_fps: int,
    frame_stride: int,
    max_video_frames: int,
    control_hz: int,
    playback_speed: float,
    compact_mode: bool,
):
    level_cfg = build_level_cfg(bddl_base_dir)
    cfg = level_cfg[level]
    if not os.path.exists(cfg["bddl"]):
        raise FileNotFoundError(f"BDDL file not found: {cfg['bddl']}")

    camera_names = parse_camera_names(camera_names)
    frame_stride = int(max(1, frame_stride))
    control_hz = int(max(1, control_hz))
    playback_speed = float(max(0.1, playback_speed))
    capture_hz = float(control_hz) / float(frame_stride)
    base_video_fps = capture_hz if video_fps <= 0 else float(video_fps)
    effective_video_fps = int(max(1, round(base_video_fps * playback_speed)))
    print(
        f"video_config: capture_hz={capture_hz:.2f}, "
        f"playback_speed={playback_speed:.2f}, "
        f"base_video_fps={base_video_fps:.2f}, "
        f"effective_video_fps={effective_video_fps}, "
        f"max_video_frames={max_video_frames}"
    )

    env = make_offscreen_env(
        bddl_file=cfg["bddl"],
        camera_size=int(camera_size),
        camera_names=camera_names,
    )
    env.seed(seed)
    rng = np.random.RandomState(seed)

    if hasattr(env, "env") and hasattr(env.env, "_check_success"):
        env.env._check_success = lambda: False

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    episode_tag = f"episode_seed{seed}_{ts}"
    episode_dir = Path(out_root) / level / episode_tag
    episode_dir.mkdir(parents=True, exist_ok=True)
    print(f"episode_dir={episode_dir}")

    obs = env.reset()
    print("reset ok")

    sim = get_sim(env)
    eef_site_id = resolve_eef_site_id(env, sim)
    earbud_joint_name = get_joint_name(env, "earbud_1")
    slot_joint_name = get_joint_name(env, "charging_slot_1")
    recorder = EpisodeRecorder(
        requested_camera_names=camera_names,
        frame_stride=frame_stride,
        video_fps=effective_video_fps,
        max_video_frames=max_video_frames,
        control_hz=control_hz,
    )
    record_enabled = False

    earbud_stable_pos = obs["earbud_1_pos"].copy()
    q_vertical = np.array([0.70710678, 0.0, -0.70710678, 0.0], dtype=float)
    q_flat_roll_local = quat_wxyz_from_axis_angle([0, 0, 1], FLAT_REST_ROLL_DEG)
    random_yaw_deg = rng.uniform(random_yaw_min_deg, random_yaw_max_deg)
    q_random_yaw = quat_wxyz_from_axis_angle([0, 0, 1], random_yaw_deg)
    rest_pose_mode = "flat" if rng.rand() < flat_rest_prob else "edge"
    if rest_pose_mode == "flat":
        earbud_stable_pos[2] = EARBUD_FLAT_REST_Z
        q_rest_base = quat_mul_wxyz(q_vertical, q_flat_roll_local)
    else:
        earbud_stable_pos[2] = EARBUD_EDGE_REST_Z
        q_rest_base = q_vertical
    earbud_stable_quat = quat_mul_wxyz(q_random_yaw, q_rest_base)

    slot_stable_pos = obs["charging_slot_1_pos"].copy()
    slot_stable_pos[2] = max(slot_stable_pos[2], 0.4680)
    slot_stable_quat = quat_wxyz_from_axis_angle([0, 1, 0], SLOT_Y_DEG)

    slot_body_yaw_deg = yaw_deg_from_quat_wxyz(slot_stable_quat)
    slot_long_axis_deg = projected_axis_heading_deg_from_quat_wxyz(slot_stable_quat, SLOT_LONG_AXIS_LOCAL)
    target_earbud_axis_deg = canonical_axis_deg(slot_long_axis_deg + SLOT_LONG_AXIS_YAW_OFFSET_DEG)

    print(f"random_yaw_deg={random_yaw_deg:.2f}")
    print(f"rest_pose_mode={rest_pose_mode}")
    print(f"slot_body_yaw_deg={slot_body_yaw_deg:.2f}")
    print(f"slot_long_axis_deg={slot_long_axis_deg:.2f}")
    print(f"target_earbud_axis_deg={target_earbud_axis_deg:.2f}")

    def refresh_obs():
        nonlocal obs
        base = env.env if hasattr(env, "env") else env
        if hasattr(base, "_get_observations"):
            try:
                obs = base._get_observations(force_update=True)
            except TypeError:
                obs = base._get_observations()

    def enforce_slot():
        q_slot = get_joint_qpos(sim, slot_joint_name)
        q_slot[:3] = slot_stable_pos
        q_slot[3:7] = slot_stable_quat
        set_joint_qpos(sim, slot_joint_name, q_slot)
        set_joint_qvel_zero(sim, slot_joint_name)

    def step_and_record(action):
        nonlocal obs
        enforce_slot()
        obs, reward, done, info = env.step(action)
        enforce_slot()
        refresh_obs()
        if record_enabled:
            recorder.log_step(
                obs=obs,
                action=action,
                reward=reward,
                done=done,
                sim=sim,
                eef_site_id=eef_site_id,
                earbud_joint_name=earbud_joint_name,
                slot_joint_name=slot_joint_name,
            )

    def set_phase(name):
        recorder.set_phase(name)

    def gripper_abs():
        g = np.asarray(obs["robot0_gripper_qpos"], dtype=float)
        return float(np.mean(np.abs(g)))

    def current_eef_yaw_deg():
        return yaw_deg_from_quat_wxyz(get_eef_quat_wxyz(sim, eef_site_id))

    def current_eef_rpy_deg():
        return rpy_deg_from_quat_wxyz(get_eef_quat_wxyz(sim, eef_site_id))

    def current_eef_rotmat():
        return quat_to_rotmat_wxyz(get_eef_quat_wxyz(sim, eef_site_id))

    def current_earbud_yaw_deg():
        q_ear = get_joint_qpos(sim, earbud_joint_name)
        return yaw_deg_from_quat_wxyz(q_ear[3:7])

    def current_earbud_axis_deg():
        q_ear = get_joint_qpos(sim, earbud_joint_name)
        return projected_axis_heading_deg_from_quat_wxyz(q_ear[3:7], EARBUD_LONG_AXIS_LOCAL)

    def debug_state(tag):
        eef_pos = obs["robot0_eef_pos"]
        earbud_pos = obs["earbud_1_pos"]
        slot_pos = obs["charging_slot_1_pos"]
        grip = obs["robot0_gripper_qpos"]

        eef_obj_dist = np.linalg.norm(eef_pos - earbud_pos)
        obj_slot_xy = np.linalg.norm(earbud_pos[:2] - slot_pos[:2])
        obj_slot_z = earbud_pos[2] - slot_pos[2]
        eef_yaw = current_eef_yaw_deg()
        earbud_yaw = current_earbud_yaw_deg()
        earbud_axis = current_earbud_axis_deg()
        yaw_err = wrap_axis_err_deg(target_earbud_axis_deg, earbud_axis)

        print(
            f"[{tag}] "
            f"eef={np.round(eef_pos,4)} "
            f"earbud={np.round(earbud_pos,4)} "
            f"slot={np.round(slot_pos,4)} "
            f"grip={np.round(grip,4)} "
            f"grip_abs={gripper_abs():.4f} "
            f"eef_yaw_deg={eef_yaw:.2f} "
            f"earbud_yaw_deg={earbud_yaw:.2f} "
            f"earbud_axis_deg={earbud_axis:.2f} "
            f"axis_err_deg={yaw_err:.2f} "
            f"eef_obj_dist={eef_obj_dist:.4f} "
            f"obj_slot_xy={obj_slot_xy:.4f} "
            f"obj_slot_z={obj_slot_z:.4f}"
        )

    def make_pose_action(target_pos, gripper_cmd, rot_cmd=None, rot_cmd_z=0.0, clip_val=POS_CLIP):
        cur_pos = obs["robot0_eef_pos"]
        pos_err = target_pos - cur_pos

        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip(KP_POS * pos_err, -clip_val, clip_val)
        if rot_cmd is None:
            action[3:5] = 0.0
            action[5] = float(np.clip(rot_cmd_z, -YAW_CLIP, YAW_CLIP))
        else:
            rot_cmd = np.asarray(rot_cmd, dtype=float)
            action[3] = float(np.clip(rot_cmd[0], -ROT_CLIP, ROT_CLIP))
            action[4] = float(np.clip(rot_cmd[1], -ROT_CLIP, ROT_CLIP))
            action[5] = float(np.clip(rot_cmd[2], -YAW_CLIP, YAW_CLIP))
        action[6] = float(np.clip(gripper_cmd, -1.0, 1.0))
        return action

    def servo_to_pos(target_pos, grip_cmd, steps=MAX_SERVO_STEPS, pos_tol=0.004):
        for _ in range(steps):
            action = make_pose_action(target_pos, grip_cmd, rot_cmd_z=0.0, clip_val=POS_CLIP)
            step_and_record(action)
            pos_err = np.linalg.norm(target_pos - obs["robot0_eef_pos"])
            if pos_err < pos_tol:
                break

    def servo_to_pos_slow(target_pos, grip_cmd, steps=220, pos_tol=0.003):
        for _ in range(steps):
            action = make_pose_action(target_pos, grip_cmd, rot_cmd_z=0.0, clip_val=POS_CLIP_RELEASE)
            step_and_record(action)
            pos_err = np.linalg.norm(target_pos - obs["robot0_eef_pos"])
            if pos_err < pos_tol:
                break

    def servo_yaw_hold_pos(target_pos, target_eef_yaw_deg, grip_cmd, steps=MAX_ROTATE_STEPS):
        for _ in range(steps):
            eef_yaw = current_eef_yaw_deg()
            yaw_err_deg = wrap_deg(target_eef_yaw_deg - eef_yaw)
            rot_cmd = np.clip(KP_YAW * np.deg2rad(yaw_err_deg), -YAW_CLIP, YAW_CLIP)
            action = make_pose_action(target_pos, grip_cmd, rot_cmd_z=rot_cmd, clip_val=POS_CLIP_RELEASE)
            step_and_record(action)
            pos_err = np.linalg.norm(target_pos - obs["robot0_eef_pos"])
            if abs(yaw_err_deg) < YAW_TOL_DEG and pos_err < 0.004:
                break

    def servo_rpy_hold_pos(target_pos, target_rpy_deg, grip_cmd, steps=MAX_ROTATE_STEPS, clip_val=POS_CLIP_RELEASE):
        target_rpy_deg = np.asarray(target_rpy_deg, dtype=float)
        target_rot = rotmat_from_rpy_deg(target_rpy_deg)
        for _ in range(steps):
            cur_rot = current_eef_rotmat()
            rot_err_mat = target_rot @ cur_rot.T
            rotvec_err = rotvec_from_rotmat(rot_err_mat)
            rot_cmd = np.clip(KP_ROT * rotvec_err, -ROT_CLIP, ROT_CLIP)
            action = make_pose_action(target_pos, grip_cmd, rot_cmd=rot_cmd, clip_val=clip_val)
            step_and_record(action)
            pos_err = np.linalg.norm(target_pos - obs["robot0_eef_pos"])
            cur_rpy_deg = current_eef_rpy_deg()
            rpy_err_deg = np.array([
                wrap_deg(target_rpy_deg[0] - cur_rpy_deg[0]),
                wrap_deg(target_rpy_deg[1] - cur_rpy_deg[1]),
                wrap_deg(target_rpy_deg[2] - cur_rpy_deg[2]),
            ], dtype=float)
            if np.max(np.abs(rpy_err_deg[:2])) < RP_TOL_DEG and abs(rpy_err_deg[2]) < YAW_TOL_DEG and pos_err < 0.004:
                break

    def servo_object_yaw_hold_pos(target_pos, target_object_yaw_deg, grip_cmd, steps=MAX_OBJECT_YAW_ALIGN_STEPS):
        for _ in range(steps):
            object_yaw = current_earbud_axis_deg()
            yaw_err_deg = wrap_axis_err_deg(target_object_yaw_deg, object_yaw)
            rot_cmd = np.clip(KP_YAW * np.deg2rad(yaw_err_deg), -YAW_CLIP, YAW_CLIP)
            action = make_pose_action(target_pos, grip_cmd, rot_cmd_z=rot_cmd, clip_val=POS_CLIP_RELEASE)
            step_and_record(action)
            pos_err = np.linalg.norm(target_pos - obs["robot0_eef_pos"])
            if abs(yaw_err_deg) < YAW_TOL_DEG and pos_err < 0.004:
                break

    def servo_object_pose_to_target(
        target_object_pos,
        target_object_yaw_deg,
        grip_cmd,
        steps=260,
        pos_tol_xy=0.0025,
        pos_tol_z=0.003,
        yaw_tol_deg=6.0,
        clip_val=POS_CLIP_RELEASE,
    ):
        for _ in range(steps):
            earbud_pos = obs["earbud_1_pos"].copy()
            eef_pos = obs["robot0_eef_pos"].copy()
            obj_minus_eef = earbud_pos - eef_pos
            desired_eef_pos = target_object_pos - obj_minus_eef

            object_yaw = current_earbud_axis_deg()
            yaw_err_deg = wrap_axis_err_deg(target_object_yaw_deg, object_yaw)
            rot_cmd = np.clip(KP_YAW * np.deg2rad(yaw_err_deg), -YAW_CLIP, YAW_CLIP)

            action = make_pose_action(desired_eef_pos, grip_cmd, rot_cmd_z=rot_cmd, clip_val=clip_val)
            step_and_record(action)

            earbud_pos = obs["earbud_1_pos"].copy()
            xy_err = np.linalg.norm(earbud_pos[:2] - target_object_pos[:2])
            z_err = abs(earbud_pos[2] - target_object_pos[2])
            object_yaw = current_earbud_axis_deg()
            yaw_err_deg = abs(wrap_axis_err_deg(target_object_yaw_deg, object_yaw))

            if xy_err < pos_tol_xy and z_err < pos_tol_z and yaw_err_deg < yaw_tol_deg:
                break

    def command_gripper_to_target(target_pos, command, target_abs, mode="open", max_steps=80):
        for _ in range(max_steps):
            action = make_pose_action(target_pos, command, rot_cmd_z=0.0, clip_val=POS_CLIP_RELEASE)
            step_and_record(action)
            cur = gripper_abs()
            if mode == "open" and cur >= target_abs:
                break
            if mode == "close" and cur <= target_abs:
                break

    # Stabilize the manually placed object and slot after reset.
    for _ in range(25):
        q_ear = get_joint_qpos(sim, earbud_joint_name)
        q_ear[:3] = earbud_stable_pos
        q_ear[3:7] = earbud_stable_quat
        set_joint_qpos(sim, earbud_joint_name, q_ear)
        set_joint_qvel_zero(sim, earbud_joint_name)

        enforce_slot()
        step_and_record(np.zeros(7, dtype=np.float32))

    refresh_obs()
    record_enabled = True
    recorder.set_phase("start")
    recorder.capture_initial(obs)

    earbud_pos0 = obs["earbud_1_pos"].copy()
    slot_pos0 = obs["charging_slot_1_pos"].copy()
    eef_pos0 = obs["robot0_eef_pos"].copy()

    if rest_pose_mode == "flat":
        pregrasp_z_offset = PREGRASP_Z_OFFSET_FLAT
        preclose_z_offset = PRECLOSE_Z_OFFSET_FLAT
        cage_z_offset = CAGE_Z_OFFSET_FLAT
        lift_z_offset = LIFT_Z_OFFSET_FLAT
        grasp_eef_yaw_offset_deg = GRASP_EEF_YAW_OFFSET_DEG_FLAT
        grasp_x_offset = GRASP_X_OFFSET_FLAT
        grasp_y_offset = GRASP_Y_OFFSET_FLAT
    else:
        pregrasp_z_offset = PREGRASP_Z_OFFSET
        preclose_z_offset = PRECLOSE_Z_OFFSET
        cage_z_offset = CAGE_Z_OFFSET
        lift_z_offset = LIFT_Z_OFFSET
        grasp_eef_yaw_offset_deg = GRASP_EEF_YAW_OFFSET_DEG
        grasp_x_offset = GRASP_X_OFFSET
        grasp_y_offset = GRASP_Y_OFFSET

    print("earbud_pos0:", np.round(earbud_pos0, 6))
    print("slot_pos0:", np.round(slot_pos0, 6))
    print("eef_pos0:", np.round(eef_pos0, 6))
    print(f"earbud_yaw0_deg={current_earbud_yaw_deg():.2f}")
    print(f"earbud_axis0_deg={current_earbud_axis_deg():.2f}")
    print(f"eef_yaw0_deg={current_eef_yaw_deg():.2f}")
    print(f"grasp_eef_yaw_offset_deg={grasp_eef_yaw_offset_deg:.2f}")
    if rest_pose_mode == "flat":
        print(f"flat_grasp_rpy_deg={[FLAT_GRASP_ROLL_DEG, FLAT_GRASP_PITCH_DEG]}")
    print(f"pregrasp_z_offset={pregrasp_z_offset:.4f}")
    print(f"preclose_z_offset={preclose_z_offset:.4f}")
    print(f"cage_z_offset={cage_z_offset:.4f}")
    print(f"lift_z_offset={lift_z_offset:.4f}")

    safe_up_pos = np.array([eef_pos0[0], eef_pos0[1], SAFE_TRAVEL_Z], dtype=float)

    safe_above_earbud = np.array([
        earbud_pos0[0] + grasp_x_offset,
        earbud_pos0[1] + grasp_y_offset,
        SAFE_TRAVEL_Z,
    ], dtype=float)

    pregrasp_pos = np.array([
        earbud_pos0[0] + grasp_x_offset,
        earbud_pos0[1] + grasp_y_offset,
        earbud_pos0[2] + pregrasp_z_offset,
    ], dtype=float)

    preclose_pos = np.array([
        earbud_pos0[0] + grasp_x_offset,
        earbud_pos0[1] + grasp_y_offset,
        earbud_pos0[2] + preclose_z_offset,
    ], dtype=float)

    cage_pos = np.array([
        earbud_pos0[0] + grasp_x_offset,
        earbud_pos0[1] + grasp_y_offset,
        earbud_pos0[2] + cage_z_offset,
    ], dtype=float)

    post_move_hold_steps = POST_MOVE_HOLD_STEPS if not compact_mode else max(6, POST_MOVE_HOLD_STEPS // 2)
    pre_release_hold_steps = PRE_RELEASE_HOLD_STEPS if not compact_mode else max(6, PRE_RELEASE_HOLD_STEPS // 2)
    release_hold_steps = RELEASE_HOLD_STEPS if not compact_mode else max(8, RELEASE_HOLD_STEPS // 2)

    set_phase("rise_safe")
    print("\n[phase 0] rise to safe height")
    servo_to_pos(safe_up_pos, GRIP_OPEN, steps=100, pos_tol=0.006)
    debug_state("after_safe_up")

    if not compact_mode:
        set_phase("air_close")
        print("[phase 0.5] air close")
        for _ in range(18):
            step_and_record(make_pose_action(obs["robot0_eef_pos"], GRIP_CLOSE))
        debug_state("after_air_close")

        set_phase("air_open")
        print("[phase 0.6] air open")
        for _ in range(18):
            step_and_record(make_pose_action(obs["robot0_eef_pos"], GRIP_OPEN))
        debug_state("after_air_open")

    set_phase("move_above_object")
    print("[phase 1] move above object")
    servo_to_pos(safe_above_earbud, GRIP_OPEN, steps=160, pos_tol=0.005)
    debug_state("after_safe_xy")

    if rest_pose_mode == "flat":
        set_phase("flat_align_wrist")
        print("[phase 1.5] rotate wrist to flat grasp pose")
        object_yaw_for_grasp_deg = current_earbud_axis_deg()
        target_eef_yaw_for_grasp_deg = wrap_deg(object_yaw_for_grasp_deg + grasp_eef_yaw_offset_deg)
        flat_target_rpy_deg = np.array([FLAT_GRASP_ROLL_DEG, FLAT_GRASP_PITCH_DEG, target_eef_yaw_for_grasp_deg], dtype=float)
        print(f"object_axis_for_grasp_deg={object_yaw_for_grasp_deg:.2f}")
        print(f"target_eef_yaw_for_grasp_deg={target_eef_yaw_for_grasp_deg:.2f}")
        servo_rpy_hold_pos(safe_above_earbud, flat_target_rpy_deg, GRIP_OPEN, steps=MAX_ROTATE_STEPS)
        debug_state("after_grasp_pose_align")

        set_phase("flat_side_pregrasp")
        print("[phase 1.6] move to flat side pregrasp")
        side_heading_rad = np.deg2rad(target_eef_yaw_for_grasp_deg)
        side_dir = np.array([np.cos(side_heading_rad), np.sin(side_heading_rad)], dtype=float)
        side_pregrasp = np.array([
            earbud_pos0[0] - FLAT_SIDE_APPROACH_DIST * side_dir[0],
            earbud_pos0[1] - FLAT_SIDE_APPROACH_DIST * side_dir[1],
            earbud_pos0[2] + FLAT_SIDE_GRASP_Z_OFFSET,
        ], dtype=float)
        servo_rpy_hold_pos(side_pregrasp, flat_target_rpy_deg, GRIP_OPEN, steps=200)
        debug_state("after_flat_side_pregrasp")

        set_phase("flat_side_contact")
        print("[phase 2] move to flat side contact")
        side_contact = np.array([
            earbud_pos0[0] - FLAT_SIDE_CONTACT_DIST * side_dir[0],
            earbud_pos0[1] - FLAT_SIDE_CONTACT_DIST * side_dir[1],
            earbud_pos0[2] + FLAT_SIDE_GRASP_Z_OFFSET,
        ], dtype=float)
        servo_rpy_hold_pos(side_contact, flat_target_rpy_deg, GRIP_OPEN, steps=200)
        debug_state("after_flat_side_contact")

        set_phase("flat_preclose")
        print("[phase 3] preclose gripper at flat contact")
        for _ in range(PRECLOSE_STEPS):
            step_and_record(make_pose_action(side_contact, GRIP_CLOSE, rot_cmd=np.array([0.0, 0.0, 0.0]), clip_val=POS_CLIP_RELEASE))
        debug_state("after_preclose")

        set_phase("flat_slide_center")
        print("[phase 4] slide into flat grasp center")
        flat_center = np.array([
            earbud_pos0[0] + grasp_x_offset,
            earbud_pos0[1] + grasp_y_offset,
            earbud_pos0[2] + FLAT_SIDE_GRASP_Z_OFFSET,
        ], dtype=float)
        servo_rpy_hold_pos(flat_center, flat_target_rpy_deg, GRIP_CLOSE, steps=220)
        debug_state("after_descend")

        set_phase("flat_squeeze")
        print("[phase 5] squeeze")
        for _ in range(CLOSE_GRIPPER_STEPS):
            step_and_record(make_pose_action(flat_center, GRIP_CLOSE, rot_cmd=np.array([0.0, 0.0, 0.0]), clip_val=POS_CLIP_RELEASE))
        debug_state("after_close")

        set_phase("flat_lift")
        print("[phase 6] lift")
        lift_pos = obs["robot0_eef_pos"].copy() + np.array([0.0, 0.0, lift_z_offset])
        servo_rpy_hold_pos(lift_pos, flat_target_rpy_deg, GRIP_CLOSE, steps=220)
        debug_state("after_lift")
    else:
        set_phase("align_wrist_yaw")
        print("[phase 1.5] rotate wrist to object yaw for grasp")
        object_yaw_for_grasp_deg = current_earbud_axis_deg()
        target_eef_yaw_for_grasp_deg = wrap_deg(object_yaw_for_grasp_deg + grasp_eef_yaw_offset_deg)
        print(f"object_axis_for_grasp_deg={object_yaw_for_grasp_deg:.2f}")
        print(f"target_eef_yaw_for_grasp_deg={target_eef_yaw_for_grasp_deg:.2f}")
        servo_yaw_hold_pos(safe_above_earbud, target_eef_yaw_for_grasp_deg, GRIP_OPEN, steps=MAX_ROTATE_STEPS)
        debug_state("after_grasp_yaw_align")

        set_phase("grasp_yaw_settle")
        print("[phase 1.6] settle after grasp yaw align")
        for _ in range(ROTATE_SETTLE_STEPS):
            step_and_record(make_pose_action(safe_above_earbud, GRIP_OPEN, rot_cmd_z=0.0, clip_val=POS_CLIP_RELEASE))
        debug_state("after_grasp_yaw_settle")

        set_phase("pregrasp")
        print("[phase 2] pregrasp")
        servo_to_pos(pregrasp_pos, GRIP_OPEN, steps=160, pos_tol=0.004)
        debug_state("after_pregrasp")

        set_phase("preclose_height")
        print("[phase 3] preclose height")
        servo_to_pos(preclose_pos, GRIP_OPEN, steps=160, pos_tol=0.003)
        debug_state("after_preclose_pos")

        set_phase("preclose_gripper")
        print("[phase 4] preclose gripper")
        for _ in range(PRECLOSE_STEPS):
            step_and_record(make_pose_action(preclose_pos, GRIP_CLOSE))
        debug_state("after_preclose")

        set_phase("descend_to_cage")
        print("[phase 5] descend to cage")
        servo_to_pos(cage_pos, GRIP_CLOSE, steps=180, pos_tol=0.002)
        debug_state("after_descend")

        set_phase("squeeze")
        print("[phase 6] squeeze")
        for _ in range(CLOSE_GRIPPER_STEPS):
            step_and_record(make_pose_action(cage_pos, GRIP_CLOSE))
        debug_state("after_close")

        set_phase("lift")
        print("[phase 7] lift")
        lift_pos = obs["robot0_eef_pos"].copy() + np.array([0.0, 0.0, lift_z_offset])
        servo_to_pos(lift_pos, GRIP_CLOSE, steps=180, pos_tol=0.003)
        debug_state("after_lift")

    if rest_pose_mode == "flat":
        set_phase("flat_restore_upright")
        print("[phase 6.5] restore upright wrist after flat lift")
        restore_pos = obs["robot0_eef_pos"].copy()
        restore_rpy_deg = np.array([0.0, 0.0, current_eef_yaw_deg()], dtype=float)
        servo_rpy_hold_pos(restore_pos, restore_rpy_deg, GRIP_CLOSE, steps=MAX_ROTATE_STEPS)
        debug_state("after_flat_restore_upright")

        set_phase("flat_restore_settle")
        print("[phase 6.6] settle after flat restore")
        for _ in range(ROTATE_SETTLE_STEPS):
            step_and_record(make_pose_action(restore_pos, GRIP_CLOSE, rot_cmd=np.array([0.0, 0.0, 0.0]), clip_val=POS_CLIP_RELEASE))
        debug_state("after_flat_restore_settle")

    # New explicit wrist rotation phase.
    set_phase("object_axis_align")
    print("[phase 7.5] explicit wrist rotation to align long axis")
    earbud_yaw_before_rotate = current_earbud_axis_deg()
    eef_yaw_before_rotate = current_eef_yaw_deg()
    yaw_err_before_deg = wrap_axis_err_deg(target_earbud_axis_deg, earbud_yaw_before_rotate)
    rotate_anchor_pos = obs["robot0_eef_pos"].copy()

    print(f"earbud_axis_before_rotate_deg={earbud_yaw_before_rotate:.2f}")
    print(f"eef_yaw_before_rotate_deg={eef_yaw_before_rotate:.2f}")
    print(f"yaw_err_before_rotate_deg={yaw_err_before_deg:.2f}")
    print(f"target_earbud_axis_deg={target_earbud_axis_deg:.2f}")

    servo_object_yaw_hold_pos(rotate_anchor_pos, target_earbud_axis_deg, GRIP_CLOSE, steps=MAX_OBJECT_YAW_ALIGN_STEPS)
    debug_state("after_rotate_align")

    set_phase("object_axis_settle")
    print("[phase 7.6] settle after wrist rotation")
    for _ in range(ROTATE_SETTLE_STEPS):
        step_and_record(make_pose_action(rotate_anchor_pos, GRIP_CLOSE, rot_cmd_z=0.0, clip_val=POS_CLIP_RELEASE))
    debug_state("after_rotate_settle")

    # Recompute carried offset after rotation instead of reusing the pre-rotate one.
    earbud_pos_lift = obs["earbud_1_pos"].copy()
    eef_pos_lift = obs["robot0_eef_pos"].copy()
    obj_minus_eef = earbud_pos_lift - eef_pos_lift

    desired_obj_hover = slot_pos0.copy()
    desired_obj_hover[2] = slot_pos0[2] + SLOT_HOVER_Z_OFFSET
    desired_eef_hover = desired_obj_hover - obj_minus_eef

    safe_above_slot = np.array([
        desired_eef_hover[0],
        desired_eef_hover[1],
        SAFE_TRAVEL_Z,
    ], dtype=float)

    set_phase("move_safe_above_slot")
    print("[phase 8] move to safe above slot")
    servo_to_pos(safe_above_slot, GRIP_CLOSE, steps=260, pos_tol=0.005)
    debug_state("after_safe_slot")

    set_phase("slot_hover")
    print("[phase 9] lower above slot")
    servo_object_pose_to_target(
        desired_obj_hover,
        target_earbud_axis_deg,
        GRIP_CLOSE,
        steps=320,
        pos_tol_xy=0.003,
        pos_tol_z=0.004,
        yaw_tol_deg=5.0,
        clip_val=POS_CLIP_RELEASE,
    )
    debug_state("after_slot_hover")

    set_phase("slot_fine_rotate")
    print("[phase 9.5] fine rotate above slot")
    slot_rotate_anchor_pos = obs["robot0_eef_pos"].copy()
    servo_object_yaw_hold_pos(slot_rotate_anchor_pos, target_earbud_axis_deg, GRIP_CLOSE, steps=MAX_OBJECT_YAW_ALIGN_STEPS)
    debug_state("after_slot_fine_rotate")

    set_phase("slot_fine_rotate_settle")
    print("[phase 9.6] settle after fine rotate")
    for _ in range(ROTATE_SETTLE_STEPS):
        step_and_record(make_pose_action(slot_rotate_anchor_pos, GRIP_CLOSE, rot_cmd_z=0.0, clip_val=POS_CLIP_RELEASE))
    debug_state("after_slot_fine_rotate_settle")

    set_phase("slot_hover_hold")
    print("[phase 10] hold above slot")
    for _ in range(post_move_hold_steps):
        step_and_record(make_pose_action(obs["robot0_eef_pos"].copy(), GRIP_CLOSE))
    debug_state("after_slot_hold")

    target_obj_pre_insert = slot_pos0.copy()
    target_obj_pre_insert[2] = slot_pos0[2] + PRE_INSERT_OBJ_Z_OFFSET

    target_obj_final_insert = slot_pos0.copy()
    target_obj_final_insert[2] = slot_pos0[2] + FINAL_INSERT_OBJ_Z_OFFSET

    set_phase("descend_pre_insert")
    print("[phase 11] descend to pre-insert")
    servo_object_pose_to_target(
        target_obj_pre_insert,
        target_earbud_axis_deg,
        GRIP_CLOSE,
        steps=320,
        pos_tol_xy=0.0025,
        pos_tol_z=0.004,
        yaw_tol_deg=6.0,
        clip_val=POS_CLIP_RELEASE,
    )
    debug_state("after_pre_insert_descend")

    set_phase("descend_final_insert")
    print("[phase 11.5] descend to final insert depth")
    servo_object_pose_to_target(
        target_obj_final_insert,
        target_earbud_axis_deg,
        GRIP_CLOSE,
        steps=360,
        pos_tol_xy=0.002,
        pos_tol_z=0.003,
        yaw_tol_deg=7.0,
        clip_val=POS_CLIP_RELEASE,
    )
    debug_state("after_release_descend")

    desired_eef_release = obs["robot0_eef_pos"].copy()

    set_phase("pre_release_hold")
    print("[phase 12] pre-release closed hold")
    for _ in range(pre_release_hold_steps):
        step_and_record(make_pose_action(desired_eef_release, GRIP_CLOSE, clip_val=POS_CLIP_RELEASE))
    debug_state("after_pre_release_hold")

    set_phase("open_gripper")
    print("[phase 13] fully open gripper")
    command_gripper_to_target(
        desired_eef_release,
        GRIP_OPEN,
        TARGET_OPEN_ABS,
        mode="open",
        max_steps=80,
    )
    debug_state("after_open")

    set_phase("post_release_wait")
    print("[phase 14] wait after release")
    for _ in range(release_hold_steps):
        step_and_record(make_pose_action(desired_eef_release, GRIP_OPEN, clip_val=POS_CLIP_RELEASE))
    debug_state("after_drop_hold")

    set_phase("retreat")
    print("[phase 15] retreat upward")
    retreat_pos = obs["robot0_eef_pos"].copy()
    retreat_pos[2] += RETREAT_Z
    servo_to_pos(retreat_pos, GRIP_OPEN, steps=160, pos_tol=0.004)
    debug_state("after_retreat")

    earbud_pos_final = obs["earbud_1_pos"].copy()
    slot_pos_final = obs["charging_slot_1_pos"].copy()
    eef_pos_final = obs["robot0_eef_pos"].copy()

    z_lift = earbud_pos_final[2] - earbud_pos0[2]
    eef_obj_dist = np.linalg.norm(eef_pos_final - earbud_pos_final)
    obj_slot_xy = np.linalg.norm(earbud_pos_final[:2] - slot_pos_final[:2])
    obj_slot_z = earbud_pos_final[2] - slot_pos_final[2]
    yaw_err_final_deg = wrap_axis_err_deg(target_earbud_axis_deg, current_earbud_axis_deg())

    release_drop_success = (obj_slot_xy < 0.02) and (obj_slot_z < 0.03)

    print(f"\nfinal release_drop_success={release_drop_success}")
    print(f"earbud_z_initial={earbud_pos0[2]:.4f}")
    print(f"earbud_z_final={earbud_pos_final[2]:.4f}")
    print(f"z_lift_vs_initial={z_lift:.4f}")
    print(f"eef_obj_dist={eef_obj_dist:.4f}")
    print(f"obj_slot_xy={obj_slot_xy:.4f}")
    print(f"obj_slot_z={obj_slot_z:.4f}")
    print(f"yaw_err_final_deg={yaw_err_final_deg:.2f}")

    episode_meta = {
        "timestamp": ts,
        "level": level,
        "seed": int(seed),
        "bddl_file": cfg["bddl"],
        "random_yaw_deg": float(random_yaw_deg),
        "rest_pose_mode": rest_pose_mode,
        "target_earbud_axis_deg": float(target_earbud_axis_deg),
        "release_drop_success": bool(release_drop_success),
        "earbud_z_initial": float(earbud_pos0[2]),
        "earbud_z_final": float(earbud_pos_final[2]),
        "z_lift_vs_initial": float(z_lift),
        "eef_obj_dist": float(eef_obj_dist),
        "obj_slot_xy": float(obj_slot_xy),
        "obj_slot_z": float(obj_slot_z),
        "yaw_err_final_deg": float(yaw_err_final_deg),
        "compact_mode": bool(compact_mode),
        "capture_hz": float(capture_hz),
        "playback_speed": float(playback_speed),
        "requested_video_fps": int(video_fps),
        "effective_video_fps": int(effective_video_fps),
        "max_video_frames": int(max_video_frames),
    }
    metadata = recorder.finalize(episode_dir=episode_dir, episode_meta=episode_meta)
    print("saved metadata:", metadata["paths"]["metadata_json"])
    print("saved preview:", metadata["paths"]["preview_video"])
    print("saved videos:", metadata["paths"]["videos"])
    if metadata.get("video_cap_hit", False):
        print("warning: video hit max_video_frames and may be truncated")

    env.close()
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--random-yaw-min-deg", type=float, default=RANDOM_YAW_MIN_DEG)
    parser.add_argument("--random-yaw-max-deg", type=float, default=RANDOM_YAW_MAX_DEG)
    parser.add_argument("--flat-rest-prob", type=float, default=FLAT_REST_PROB)
    parser.add_argument("--out-root", type=str, default="./libero_lerobot_dataset")
    parser.add_argument("--bddl-base-dir", type=str, default=BASE_DIR)
    parser.add_argument("--camera-size", type=int, default=CAMERA_SIZE)
    parser.add_argument("--camera-names", type=str, default=",".join(DEFAULT_CAMERA_NAMES))
    parser.add_argument(
        "--video-fps",
        type=int,
        default=DEFAULT_VIDEO_FPS,
        help="Base output video fps. <=0 means auto from capture_hz. Final fps = base_video_fps * playback_speed.",
    )
    parser.add_argument("--frame-stride", type=int, default=DEFAULT_FRAME_STRIDE)
    parser.add_argument(
        "--max-video-frames",
        type=int,
        default=DEFAULT_MAX_VIDEO_FRAMES,
        help="Cap video frames for each episode. <=0 means no cap.",
    )
    parser.add_argument("--control-hz", type=int, default=DEFAULT_CONTROL_HZ)
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=DEFAULT_PLAYBACK_SPEED,
        help="Playback multiplier applied to final encoded fps.",
    )
    parser.add_argument(
        "--compact-video",
        action="store_true",
        help="Enable compact demo mode (shorter holds and skipping air close/open rehearsal).",
    )
    parser.add_argument(
        "--full-video",
        action="store_true",
        help="Deprecated compatibility flag. Full baseline flow is now the default.",
    )
    args = parser.parse_args()
    if args.compact_video and args.full_video:
        print("warning: both --compact-video and --full-video are set; using full baseline flow")

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.jsonl"

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_results = []
    for ep_idx in range(int(max(1, args.episodes))):
        ep_seed = args.seed + ep_idx * args.seed_step
        print(f"\n=== episode {ep_idx + 1}/{args.episodes} seed={ep_seed} ===")
        metadata = rollout(
            level=args.level,
            seed=ep_seed,
            random_yaw_min_deg=args.random_yaw_min_deg,
            random_yaw_max_deg=args.random_yaw_max_deg,
            flat_rest_prob=args.flat_rest_prob,
            out_root=str(out_root),
            bddl_base_dir=args.bddl_base_dir,
            camera_size=args.camera_size,
            camera_names=args.camera_names,
            video_fps=args.video_fps,
            frame_stride=args.frame_stride,
            max_video_frames=args.max_video_frames,
            control_hz=args.control_hz,
            playback_speed=args.playback_speed,
            compact_mode=bool(args.compact_video and (not args.full_video)),
        )
        metadata["episode_index"] = ep_idx
        append_jsonl(manifest_path, metadata)
        run_results.append(metadata)

    run_report = out_root / f"run_{run_ts}.json"
    with run_report.open("w", encoding="utf-8") as f:
        json.dump({"run_timestamp": run_ts, "episodes": run_results}, f, ensure_ascii=False, indent=2)

    print("\ncollection finished")
    print(f"manifest: {manifest_path}")
    print(f"run report: {run_report}")


if __name__ == "__main__":
    main()
