import os
import json
import shutil
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
DEFAULT_TARGET_VIDEO_DURATION_SEC = 60.0
DEFAULT_CAMERA_NAMES = ("agentview", "frontview", "robot0_eye_in_hand")
DEFAULT_QUALITY_MODE = "vla"
DEFAULT_RECORD_MODE = "all"

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

GRASP_X_OFFSET = -0.010
GRASP_Y_OFFSET = 0.000

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
TARGET_PRECLOSE_ABS = 0.0065
TARGET_CLOSE_ABS = 0.0055

# Slot pose stays fixed as in the baseline.
SLOT_Y_DEG = -12.0

# New: random initial object yaw plus explicit wrist rotation later.
RANDOM_YAW_MIN_DEG = -90.0
RANDOM_YAW_MAX_DEG = 90.0
# This optimized collector is upright edge-only.
EARBUD_EDGE_REST_Z = 0.4435

# New: align wrist to the object before grasping.
# In your successful baseline, eef yaw is about 90 deg when the earbud is also
# effectively aligned at about 90 deg, so the default offset is 0 deg.
GRASP_EEF_YAW_OFFSET_DEG = 0.0

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
    alias = {
        "agent_view": "agentview",
        "agent-view": "agentview",
        "agentview": "agentview",
        "front_view": "frontview",
        "front-view": "frontview",
        "frontview": "frontview",
        "robot0_eye_in_hand": "robot0_eye_in_hand",
        "robot0-eye-in-hand": "robot0_eye_in_hand",
        "robot0_eye_in_hand3": "robot0_eye_in_hand",
        "eye_in_hand": "robot0_eye_in_hand",
    }

    def normalize(name):
        key = str(name).strip().lower()
        if not key:
            return ""
        return alias.get(key, str(name).strip())

    if raw_camera_names is None:
        return list(DEFAULT_CAMERA_NAMES)
    if isinstance(raw_camera_names, str):
        raw = [name.strip() for name in raw_camera_names.split(",")]
        names = [normalize(name) for name in raw if normalize(name)]
    elif isinstance(raw_camera_names, (list, tuple)):
        names = [normalize(name) for name in raw_camera_names if normalize(name)]
    else:
        names = []

    if not names:
        return list(DEFAULT_CAMERA_NAMES)

    # Keep order and remove duplicates.
    deduped = []
    seen = set()
    for name in names:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


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
        target_video_duration_sec=DEFAULT_TARGET_VIDEO_DURATION_SEC,
        record_mode=DEFAULT_RECORD_MODE,
    ):
        self.requested_camera_names = list(requested_camera_names)
        self.frame_stride = int(max(1, frame_stride))
        self.requested_frame_stride = self.frame_stride
        self.video_fps = int(max(1, video_fps))
        self.max_video_frames = None if max_video_frames is None or max_video_frames <= 0 else int(max_video_frames)
        self.control_hz = float(max(1, control_hz))
        self.target_video_duration_sec = None if target_video_duration_sec is None or target_video_duration_sec <= 0 else float(target_video_duration_sec)
        self.resolved_video_fps = self.video_fps
        self.record_mode = str(record_mode or DEFAULT_RECORD_MODE).lower()
        if self.record_mode not in ("all", "keyframe", "keyframe_critical"):
            raise ValueError(f"Unsupported record_mode={self.record_mode!r}; use all, keyframe, or keyframe_critical.")
        # Keyframe mode keeps physics/control dense but writes fewer synchronized
        # image/state/action samples for BC fine-tuning. The converter expects
        # one frame per stored action plus the initial frame, so expose stride=1.
        if self.record_mode != "all":
            self.frame_stride = 1

        self.selected_camera_names = []
        self.frames = {}
        self.phase = "init"
        self.step_count = 0
        self.phase_step_count = 0
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
        self.phase_step_count = 0

    def _phase_record_stride(self):
        hard_critical = {
            "descend_to_cage",
            "lift",
            "descend_pre_insert",
            "descend_final_insert",
            "open_gripper",
        }
        soft_align = {"object_axis_align", "slot_hover", "slot_fine_rotate"}
        grip_hold = {"preclose_gripper", "squeeze"}
        critical = hard_critical | soft_align
        sparse = {
            "rise_safe",
            "move_above_object",
            "align_wrist_yaw",
            "pregrasp",
            "preclose_height",
            "move_safe_above_slot",
            "retreat",
        }
        hold_like = (
            "settle" in self.phase
            or "hold" in self.phase
            or "wait" in self.phase
            or self.phase in {"start", "pre_release_hold", "post_release_wait"}
        )
        if self.record_mode == "keyframe_critical":
            if self.phase in hard_critical:
                return 1
            if self.phase in grip_hold:
                return 4
            if self.phase == "object_axis_align":
                return 2
            if self.phase == "slot_hover":
                return 8
            if self.phase == "slot_fine_rotate":
                return 6
            if self.phase in sparse:
                return 5
            if hold_like:
                return 12
            return 4
        if self.phase in critical:
            return 1
        if self.phase in sparse:
            return 3
        if hold_like:
            return 8
        return 2

    def _should_record_step(self, action):
        if self.record_mode == "all":
            return True
        if self.phase_step_count <= 2:
            return True
        action = np.asarray(action, dtype=np.float32)
        if self.record_mode == "keyframe_critical":
            hard_critical = {
                "descend_to_cage",
                "lift",
                "descend_pre_insert",
                "descend_final_insert",
                "open_gripper",
            }
            soft_align = {"object_axis_align", "slot_hover", "slot_fine_rotate"}
            if self.phase in hard_critical:
                return True
            # In critical-focused mode, ordinary long moves are sampled by phase
            # stride rather than kept almost fully because the position action is saturated.
            if np.linalg.norm(action[:3]) >= 0.078 or abs(float(action[5])) >= 0.078:
                sparse = {
                    "rise_safe",
                    "move_above_object",
                    "align_wrist_yaw",
                    "pregrasp",
                    "preclose_height",
                    "move_safe_above_slot",
                    "retreat",
                }
                hold_like = (
                    "settle" in self.phase
                    or "hold" in self.phase
                    or "wait" in self.phase
                    or self.phase in {"start", "pre_release_hold", "post_release_wait"}
                )
                if self.phase not in sparse and self.phase not in soft_align and not hold_like:
                    return True
            stride = self._phase_record_stride()
            return (self.phase_step_count % stride) == 0
        # Keep saturated/high-intent controls even outside explicitly critical phases.
        if np.linalg.norm(action[:3]) >= 0.06 or abs(float(action[5])) >= 0.06:
            return True
        stride = self._phase_record_stride()
        return (self.phase_step_count % stride) == 0

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
        self.phase_step_count += 1
        if not self._should_record_step(action):
            return
        self._capture_frames(obs, force=(self.record_mode == "keyframe"))

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

        max_frame_count = 0
        for cam in self.selected_camera_names:
            max_frame_count = max(max_frame_count, len(self.frames.get(cam, [])))
        resolved_video_fps = self.video_fps
        if self.target_video_duration_sec is not None and max_frame_count > 0:
            target_fps = int(max(1, round(max_frame_count / self.target_video_duration_sec)))
            resolved_video_fps = max(resolved_video_fps, target_fps)
        self.resolved_video_fps = int(max(1, resolved_video_fps))

        video_paths = {}
        for cam in self.selected_camera_names:
            cam_frames = self.frames.get(cam, [])
            if not cam_frames:
                continue
            video_path = video_dir / f"{cam}.mp4"
            imageio.mimwrite(str(video_path), cam_frames, fps=self.resolved_video_fps)
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
                imageio.mimwrite(str(preview), preferred_frames, fps=self.resolved_video_fps)
                preview_path = str(preview)

        metadata = dict(episode_meta)
        metadata.update(
            {
                "num_steps": int(len(self.records["step"])),
                "selected_cameras": self.selected_camera_names,
                "video_fps_base": self.video_fps,
                "video_fps_resolved": self.resolved_video_fps,
                "frame_stride": self.frame_stride,
                "requested_frame_stride": self.requested_frame_stride,
                "record_mode": self.record_mode,
                "control_hz": self.control_hz,
                "target_video_duration_sec": self.target_video_duration_sec,
                "max_recorded_frames": int(max_frame_count),
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
    out_root: str,
    bddl_base_dir: str,
    camera_size: int,
    camera_names,
    video_fps: int,
    frame_stride: int,
    max_video_frames: int,
    control_hz: int,
    playback_speed: float,
    target_video_duration_sec: float,
    compact_mode: bool,
    quality_mode: str,
    record_mode: str,
):
    quality_mode = str(quality_mode or DEFAULT_QUALITY_MODE).lower()
    if quality_mode not in (
        "train",
        "train_fast",
        "train_insert_fast",
        "train_insert_faster",
        "vla",
        "balanced",
        "compact",
        "full",
    ):
        raise ValueError(
            f"Unsupported quality_mode={quality_mode!r}; use train, train_fast, train_insert_fast, "
            "train_insert_faster, vla, balanced, compact, or full."
        )
    compact_mode = bool(compact_mode or quality_mode == "compact")
    record_mode = str(record_mode or DEFAULT_RECORD_MODE).lower()
    if record_mode not in ("all", "keyframe", "keyframe_critical"):
        raise ValueError(f"Unsupported record_mode={record_mode!r}; use all, keyframe, or keyframe_critical.")

    def quality_steps(
        full_steps,
        train_steps=None,
        train_insert_fast_steps=None,
        train_insert_faster_steps=None,
        vla_steps=None,
        balanced_steps=None,
        compact_steps=None,
        minimum=1,
    ):
        """Shorten non-informative repeated phases without changing the task path."""
        full_steps = int(full_steps)
        if quality_mode == "full":
            return full_steps
        if quality_mode == "train":
            if train_steps is not None:
                return int(max(minimum, train_steps))
            if balanced_steps is not None:
                return int(max(minimum, balanced_steps))
            return int(max(minimum, round(full_steps * 0.80)))
        if quality_mode == "train_fast":
            if train_steps is not None and vla_steps is not None:
                return int(max(minimum, round(0.5 * float(train_steps) + 0.5 * float(vla_steps))))
            if train_steps is not None:
                return int(max(minimum, round(float(train_steps) * 0.75)))
            if balanced_steps is not None:
                return int(max(minimum, round(float(balanced_steps) * 0.85)))
            return int(max(minimum, round(full_steps * 0.65)))
        if quality_mode == "train_insert_fast":
            if train_insert_fast_steps is not None:
                return int(max(minimum, train_insert_fast_steps))
            # Default to the already-successful train_fast schedule. Only phases
            # that pass train_insert_fast_steps are compressed more aggressively.
            if train_steps is not None and vla_steps is not None:
                return int(max(minimum, round(0.5 * float(train_steps) + 0.5 * float(vla_steps))))
            if train_steps is not None:
                return int(max(minimum, round(float(train_steps) * 0.75)))
            if balanced_steps is not None:
                return int(max(minimum, round(float(balanced_steps) * 0.85)))
            return int(max(minimum, round(full_steps * 0.65)))
        if quality_mode == "train_insert_faster":
            if train_insert_faster_steps is not None:
                return int(max(minimum, train_insert_faster_steps))
            if train_insert_fast_steps is not None:
                return int(max(minimum, train_insert_fast_steps))
            if train_steps is not None and vla_steps is not None:
                return int(max(minimum, round(0.5 * float(train_steps) + 0.5 * float(vla_steps))))
            if train_steps is not None:
                return int(max(minimum, round(float(train_steps) * 0.75)))
            if balanced_steps is not None:
                return int(max(minimum, round(float(balanced_steps) * 0.85)))
            return int(max(minimum, round(full_steps * 0.65)))
        if quality_mode == "compact":
            if compact_steps is not None:
                return int(max(minimum, compact_steps))
            return int(max(minimum, round(full_steps * 0.50)))
        if quality_mode == "vla":
            if vla_steps is not None:
                return int(max(minimum, vla_steps))
            if balanced_steps is not None:
                return int(max(minimum, round(float(balanced_steps) * 0.75)))
            return int(max(minimum, round(full_steps * 0.45)))
        if balanced_steps is not None:
            return int(max(minimum, balanced_steps))
        return int(max(minimum, round(full_steps * 0.35)))

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
        f"target_video_duration_sec={target_video_duration_sec}, "
        f"max_video_frames={max_video_frames}, "
        f"record_mode={record_mode}"
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
        target_video_duration_sec=target_video_duration_sec,
        record_mode=record_mode,
    )
    record_enabled = False

    earbud_stable_pos = obs["earbud_1_pos"].copy()
    q_vertical = np.array([0.70710678, 0.0, -0.70710678, 0.0], dtype=float)
    random_yaw_deg = rng.uniform(random_yaw_min_deg, random_yaw_max_deg)
    q_random_yaw = quat_wxyz_from_axis_angle([0, 0, 1], random_yaw_deg)
    rest_pose_mode = "edge"
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

    # The old script spent many frames on repeated gripper commands and holds.
    # For BC fine-tuning these frames dominate the loss while adding little new
    # information, so balanced/compact modes keep only a short, purposeful tail.
    preclose_steps = quality_steps(
        PRECLOSE_STEPS, train_steps=18, train_insert_faster_steps=8, balanced_steps=12, compact_steps=8
    )
    close_gripper_steps = quality_steps(
        CLOSE_GRIPPER_STEPS, train_steps=24, train_insert_faster_steps=8, balanced_steps=14, compact_steps=10
    )
    rotate_settle_steps = quality_steps(
        ROTATE_SETTLE_STEPS, train_steps=2, train_insert_faster_steps=0, balanced_steps=2, compact_steps=1, minimum=0
    )
    post_move_hold_steps = quality_steps(
        POST_MOVE_HOLD_STEPS, train_steps=2, train_insert_faster_steps=0, balanced_steps=2, compact_steps=1, minimum=0
    )
    pre_release_hold_steps = quality_steps(PRE_RELEASE_HOLD_STEPS, train_steps=3, balanced_steps=3, compact_steps=1)
    release_hold_steps = quality_steps(RELEASE_HOLD_STEPS, train_steps=4, balanced_steps=4, compact_steps=2)
    safe_up_steps = quality_steps(100, train_steps=100, vla_steps=45, balanced_steps=65, compact_steps=35)
    move_above_object_steps = quality_steps(160, train_steps=145, vla_steps=65, balanced_steps=95, compact_steps=45)
    grasp_yaw_steps = quality_steps(MAX_ROTATE_STEPS, train_steps=180, vla_steps=90, balanced_steps=130, compact_steps=70)
    pregrasp_steps = quality_steps(160, train_steps=145, vla_steps=60, balanced_steps=90, compact_steps=45)
    preclose_height_steps = quality_steps(160, train_steps=150, vla_steps=55, balanced_steps=85, compact_steps=40)
    descend_cage_steps = quality_steps(180, train_steps=170, vla_steps=80, balanced_steps=110, compact_steps=60)
    lift_steps = quality_steps(
        180, train_steps=160, train_insert_faster_steps=30, vla_steps=70, balanced_steps=105, compact_steps=55
    )
    object_axis_align_steps = quality_steps(
        MAX_OBJECT_YAW_ALIGN_STEPS,
        train_steps=320,
        train_insert_faster_steps=30,
        vla_steps=140,
        balanced_steps=220,
        compact_steps=100,
    )
    move_safe_slot_steps = quality_steps(
        260,
        train_steps=220,
        train_insert_fast_steps=120,
        train_insert_faster_steps=35,
        vla_steps=95,
        balanced_steps=150,
        compact_steps=75,
    )
    slot_hover_steps = quality_steps(
        320,
        train_steps=260,
        train_insert_fast_steps=100,
        train_insert_faster_steps=10,
        vla_steps=140,
        balanced_steps=200,
        compact_steps=110,
    )
    slot_fine_rotate_steps = quality_steps(
        MAX_OBJECT_YAW_ALIGN_STEPS,
        train_steps=260,
        train_insert_fast_steps=55,
        train_insert_faster_steps=2,
        vla_steps=100,
        balanced_steps=180,
        compact_steps=80,
    )
    descend_pre_insert_steps = quality_steps(
        320,
        train_steps=260,
        train_insert_fast_steps=110,
        train_insert_faster_steps=75,
        vla_steps=150,
        balanced_steps=220,
        compact_steps=120,
    )
    descend_final_insert_steps = quality_steps(
        360,
        train_steps=300,
        train_insert_fast_steps=120,
        train_insert_faster_steps=85,
        vla_steps=180,
        balanced_steps=260,
        compact_steps=140,
    )
    retreat_steps = quality_steps(
        160,
        train_steps=110,
        train_insert_fast_steps=55,
        train_insert_faster_steps=40,
        vla_steps=55,
        balanced_steps=90,
        compact_steps=45,
    )

    if quality_mode != "full":
        print(
            "quality_mode="
            f"{quality_mode}: preclose={preclose_steps}, close={close_gripper_steps}, "
            f"settle={rotate_settle_steps}, slot_hold={post_move_hold_steps}, "
            f"pre_release_hold={pre_release_hold_steps}, release_hold={release_hold_steps}, "
            f"safe_up={safe_up_steps}, move_above={move_above_object_steps}, "
            f"pregrasp={pregrasp_steps}, preclose_height={preclose_height_steps}, "
            f"descend_cage={descend_cage_steps}, lift={lift_steps}, object_align={object_axis_align_steps}, "
            f"move_slot={move_safe_slot_steps}, "
            f"slot_hover={slot_hover_steps}, fine_rotate={slot_fine_rotate_steps}, "
            f"insert=({descend_pre_insert_steps},{descend_final_insert_steps}), retreat={retreat_steps}"
        )

    set_phase("rise_safe")
    print("\n[phase 0] rise to safe height")
    servo_to_pos(safe_up_pos, GRIP_OPEN, steps=safe_up_steps, pos_tol=0.006)
    debug_state("after_safe_up")

    if quality_mode == "full":
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
    else:
        print("[skip] air close/open rehearsal skipped in optimized edge-only dataset")

    set_phase("move_above_object")
    print("[phase 1] move above object")
    servo_to_pos(safe_above_earbud, GRIP_OPEN, steps=move_above_object_steps, pos_tol=0.005)
    debug_state("after_safe_xy")

    set_phase("align_wrist_yaw")
    print("[phase 1.5] rotate wrist to object yaw for grasp")
    object_yaw_for_grasp_deg = current_earbud_axis_deg()
    target_eef_yaw_for_grasp_deg = wrap_deg(object_yaw_for_grasp_deg + grasp_eef_yaw_offset_deg)
    print(f"object_axis_for_grasp_deg={object_yaw_for_grasp_deg:.2f}")
    print(f"target_eef_yaw_for_grasp_deg={target_eef_yaw_for_grasp_deg:.2f}")
    servo_yaw_hold_pos(safe_above_earbud, target_eef_yaw_for_grasp_deg, GRIP_OPEN, steps=grasp_yaw_steps)
    debug_state("after_grasp_yaw_align")

    set_phase("grasp_yaw_settle")
    print("[phase 1.6] settle after grasp yaw align")
    for _ in range(rotate_settle_steps):
        step_and_record(make_pose_action(safe_above_earbud, GRIP_OPEN, rot_cmd_z=0.0, clip_val=POS_CLIP_RELEASE))
    debug_state("after_grasp_yaw_settle")

    set_phase("pregrasp")
    print("[phase 2] pregrasp")
    servo_to_pos(pregrasp_pos, GRIP_OPEN, steps=pregrasp_steps, pos_tol=0.004)
    debug_state("after_pregrasp")

    set_phase("preclose_height")
    print("[phase 3] preclose height")
    servo_to_pos(preclose_pos, GRIP_OPEN, steps=preclose_height_steps, pos_tol=0.003)
    debug_state("after_preclose_pos")

    set_phase("preclose_gripper")
    print("[phase 4] preclose gripper")
    command_gripper_to_target(
        preclose_pos,
        GRIP_CLOSE,
        TARGET_PRECLOSE_ABS,
        mode="close",
        max_steps=preclose_steps,
    )
    debug_state("after_preclose")

    set_phase("descend_to_cage")
    print("[phase 5] descend to cage")
    servo_to_pos(cage_pos, GRIP_CLOSE, steps=descend_cage_steps, pos_tol=0.002)
    debug_state("after_descend")

    set_phase("squeeze")
    print("[phase 6] squeeze")
    if gripper_abs() > TARGET_CLOSE_ABS:
        command_gripper_to_target(
            cage_pos,
            GRIP_CLOSE,
            TARGET_CLOSE_ABS,
            mode="close",
            max_steps=close_gripper_steps,
        )
    else:
        print(f"[skip] squeeze skipped because gripper_abs={gripper_abs():.4f} <= target={TARGET_CLOSE_ABS:.4f}")
    debug_state("after_close")

    set_phase("lift")
    print("[phase 7] lift")
    lift_pos = obs["robot0_eef_pos"].copy() + np.array([0.0, 0.0, lift_z_offset])
    servo_to_pos(lift_pos, GRIP_CLOSE, steps=lift_steps, pos_tol=0.003)
    debug_state("after_lift")

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

    servo_object_yaw_hold_pos(rotate_anchor_pos, target_earbud_axis_deg, GRIP_CLOSE, steps=object_axis_align_steps)
    debug_state("after_rotate_align")

    set_phase("object_axis_settle")
    print("[phase 7.6] settle after wrist rotation")
    for _ in range(rotate_settle_steps):
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
    servo_to_pos(safe_above_slot, GRIP_CLOSE, steps=move_safe_slot_steps, pos_tol=0.005)
    debug_state("after_safe_slot")

    set_phase("slot_hover")
    print("[phase 9] lower above slot")
    servo_object_pose_to_target(
        desired_obj_hover,
        target_earbud_axis_deg,
        GRIP_CLOSE,
        steps=slot_hover_steps,
        pos_tol_xy=0.003,
        pos_tol_z=0.004,
        yaw_tol_deg=5.0,
        clip_val=POS_CLIP_RELEASE,
    )
    debug_state("after_slot_hover")

    set_phase("slot_fine_rotate")
    print("[phase 9.5] fine rotate above slot")
    slot_rotate_anchor_pos = obs["robot0_eef_pos"].copy()
    servo_object_yaw_hold_pos(slot_rotate_anchor_pos, target_earbud_axis_deg, GRIP_CLOSE, steps=slot_fine_rotate_steps)
    debug_state("after_slot_fine_rotate")

    set_phase("slot_fine_rotate_settle")
    print("[phase 9.6] settle after fine rotate")
    for _ in range(rotate_settle_steps):
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
        steps=descend_pre_insert_steps,
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
        steps=descend_final_insert_steps,
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
        max_steps=quality_steps(80, balanced_steps=18, compact_steps=10),
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
    servo_to_pos(retreat_pos, GRIP_OPEN, steps=retreat_steps, pos_tol=0.004)
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
        "quality_mode": str(quality_mode),
        "edge_only_optimized": True,
        "capture_hz": float(capture_hz),
        "playback_speed": float(playback_speed),
        "requested_video_fps": float(video_fps),
        "base_video_fps": float(base_video_fps),
        "effective_video_fps": int(effective_video_fps),
        "target_video_duration_sec": float(target_video_duration_sec),
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
    parser.add_argument("--out-root", type=str, default="./libero_lerobot_dataset")
    parser.add_argument("--bddl-base-dir", type=str, default=BASE_DIR)
    parser.add_argument("--camera-size", type=int, default=CAMERA_SIZE)
    parser.add_argument(
        "--camera-names",
        type=str,
        default=",".join(DEFAULT_CAMERA_NAMES),
        help="Comma-separated camera names. Recommended: agentview,frontview,robot0_eye_in_hand.",
    )
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
        "--target-video-duration-sec",
        type=float,
        default=DEFAULT_TARGET_VIDEO_DURATION_SEC,
        help="If >0, auto speed up encoded videos so each episode is about this duration in seconds.",
    )
    parser.add_argument(
        "--compact-video",
        action="store_true",
        help="Enable compact demo mode (shorter holds and skipping air close/open rehearsal).",
    )
    parser.add_argument(
        "--quality-mode",
        choices=[
            "train",
            "train_fast",
            "train_insert_fast",
            "train_insert_faster",
            "vla",
            "balanced",
            "compact",
            "full",
        ],
        default=DEFAULT_QUALITY_MODE,
        help=(
            "train is safer for BC data; train_fast mildly compresses the successful path; "
            "train_insert_fast keeps the successful grasp path but compresses slot/insert phases; "
            "train_insert_faster also speeds lift/object alignment and makes descent more decisive; "
            "vla is concise; balanced is safer; compact is shorter; full keeps the old long flow."
        ),
    )
    parser.add_argument(
        "--record-mode",
        choices=["all", "keyframe", "keyframe_critical"],
        default=DEFAULT_RECORD_MODE,
        help=(
            "all records every control step; keyframe records fewer synchronized samples; "
            "keyframe_critical increases the ratio of grasp/insert/release samples."
        ),
    )
    parser.add_argument(
        "--require-success",
        action="store_true",
        help="Move failed episodes to _failed/ so they will not be picked up by category-based conversion.",
    )
    parser.add_argument(
        "--full-video",
        action="store_true",
        help="Deprecated compatibility flag. Full baseline flow is now the default.",
    )
    args = parser.parse_args()
    if args.compact_video and args.full_video:
        print("warning: both --compact-video and --full-video are set; using full baseline flow")
    quality_mode = str(args.quality_mode)
    if args.compact_video and not args.full_video:
        quality_mode = "compact"
    if args.full_video:
        quality_mode = "full"

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
            out_root=str(out_root),
            bddl_base_dir=args.bddl_base_dir,
            camera_size=args.camera_size,
            camera_names=args.camera_names,
            video_fps=args.video_fps,
            frame_stride=args.frame_stride,
            max_video_frames=args.max_video_frames,
            control_hz=args.control_hz,
            playback_speed=args.playback_speed,
            target_video_duration_sec=args.target_video_duration_sec,
            compact_mode=bool(args.compact_video and (not args.full_video)),
            quality_mode=quality_mode,
            record_mode=args.record_mode,
        )
        if args.require_success and not bool(metadata.get("release_drop_success", False)):
            metadata_path = metadata.get("paths", {}).get("metadata_json", "")
            episode_dir = Path(metadata_path).parent if metadata_path else None
            if episode_dir is not None and episode_dir.exists():
                failed_root = out_root / "_failed" / args.level
                failed_root.mkdir(parents=True, exist_ok=True)
                failed_dir = failed_root / episode_dir.name
                if failed_dir.exists():
                    shutil.rmtree(failed_dir)
                shutil.move(str(episode_dir), str(failed_dir))
                metadata["moved_to_failed_dir"] = str(failed_dir)
                print(f"[skip failed] moved episode to: {failed_dir}")
            append_jsonl(out_root / "failed_manifest.jsonl", metadata)
            continue
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
