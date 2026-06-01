import os
from datetime import datetime
import argparse

import numpy as np
import imageio

from libero.libero.envs import OffScreenRenderEnv
from demo_recorder import DemoRecorder

BASE_DIR = "/root/autodl-tmp/openpi_earbud_proto/third_party/libero/libero/libero/bddl_files/libero_90"

LEVEL_CFG = {
    "easy": {
        "bddl": os.path.join(BASE_DIR, "LIVING_ROOM_SCENE1_insert_the_earbud_into_the_charging_slot_easy.bddl"),
    },
    "medium": {
        "bddl": os.path.join(BASE_DIR, "LIVING_ROOM_SCENE1_insert_the_earbud_into_the_charging_slot_medium.bddl"),
    },
    "hard": {
        "bddl": os.path.join(BASE_DIR, "LIVING_ROOM_SCENE1_insert_the_earbud_into_the_charging_slot_hard.bddl"),
    },
}

CAMERA_SIZE = 768

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


def rollout(level: str, seed: int, random_yaw_min_deg: float, random_yaw_max_deg: float, flat_rest_prob: float, save_demo: bool = False, demo_dir: str = "/root/autodl-tmp/openpi_earbud_proto/demo_examples"):
    cfg = LEVEL_CFG[level]

    env = OffScreenRenderEnv(
        bddl_file_name=cfg["bddl"],
        camera_heights=CAMERA_SIZE,
        camera_widths=CAMERA_SIZE,
        ignore_done=True,
    )
    env.seed(seed)
    rng = np.random.RandomState(seed)

    if hasattr(env, "env") and hasattr(env.env, "_check_success"):
        env.env._check_success = lambda: False

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"/root/autodl-tmp/openpi_earbud_proto/full_chain_release_random_wrist_align/{level}"
    os.makedirs(out_dir, exist_ok=True)

    obs = env.reset()
    print("reset ok")

    task_name = "earbud_insert"
    task_text = "insert the earbud into the charging slot"
    episode_id = seed
    recorder = DemoRecorder(
        save_dir=demo_dir,
        task_name=task_name,
        level=level,
        episode_id=episode_id,
        task_text=task_text,
    ) if save_demo else None

    sim = get_sim(env)
    eef_site_id = resolve_eef_site_id(env, sim)
    earbud_joint_name = get_joint_name(env, "earbud_1")
    slot_joint_name = get_joint_name(env, "charging_slot_1")

    frames = []

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
        if recorder is not None:
            recorder.record_step(obs, action)
        if "agentview_image" in obs:
            frames.append(obs["agentview_image"][::-1])

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

    frames = []
    refresh_obs()
    if "agentview_image" in obs:
        frames.append(obs["agentview_image"][::-1])

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

    print("\n[phase 0] rise to safe height")
    servo_to_pos(safe_up_pos, GRIP_OPEN, steps=100, pos_tol=0.006)
    debug_state("after_safe_up")

    print("[phase 0.5] air close")
    for _ in range(18):
        step_and_record(make_pose_action(obs["robot0_eef_pos"], GRIP_CLOSE))
    debug_state("after_air_close")

    print("[phase 0.6] air open")
    for _ in range(18):
        step_and_record(make_pose_action(obs["robot0_eef_pos"], GRIP_OPEN))
    debug_state("after_air_open")

    print("[phase 1] move above object")
    servo_to_pos(safe_above_earbud, GRIP_OPEN, steps=160, pos_tol=0.005)
    debug_state("after_safe_xy")

    if rest_pose_mode == "flat":
        print("[phase 1.5] rotate wrist to flat grasp pose")
        object_yaw_for_grasp_deg = current_earbud_axis_deg()
        target_eef_yaw_for_grasp_deg = wrap_deg(object_yaw_for_grasp_deg + grasp_eef_yaw_offset_deg)
        flat_target_rpy_deg = np.array([FLAT_GRASP_ROLL_DEG, FLAT_GRASP_PITCH_DEG, target_eef_yaw_for_grasp_deg], dtype=float)
        print(f"object_axis_for_grasp_deg={object_yaw_for_grasp_deg:.2f}")
        print(f"target_eef_yaw_for_grasp_deg={target_eef_yaw_for_grasp_deg:.2f}")
        servo_rpy_hold_pos(safe_above_earbud, flat_target_rpy_deg, GRIP_OPEN, steps=MAX_ROTATE_STEPS)
        debug_state("after_grasp_pose_align")

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

        print("[phase 2] move to flat side contact")
        side_contact = np.array([
            earbud_pos0[0] - FLAT_SIDE_CONTACT_DIST * side_dir[0],
            earbud_pos0[1] - FLAT_SIDE_CONTACT_DIST * side_dir[1],
            earbud_pos0[2] + FLAT_SIDE_GRASP_Z_OFFSET,
        ], dtype=float)
        servo_rpy_hold_pos(side_contact, flat_target_rpy_deg, GRIP_OPEN, steps=200)
        debug_state("after_flat_side_contact")

        print("[phase 3] preclose gripper at flat contact")
        for _ in range(PRECLOSE_STEPS):
            step_and_record(make_pose_action(side_contact, GRIP_CLOSE, rot_cmd=np.array([0.0, 0.0, 0.0]), clip_val=POS_CLIP_RELEASE))
        debug_state("after_preclose")

        print("[phase 4] slide into flat grasp center")
        flat_center = np.array([
            earbud_pos0[0] + grasp_x_offset,
            earbud_pos0[1] + grasp_y_offset,
            earbud_pos0[2] + FLAT_SIDE_GRASP_Z_OFFSET,
        ], dtype=float)
        servo_rpy_hold_pos(flat_center, flat_target_rpy_deg, GRIP_CLOSE, steps=220)
        debug_state("after_descend")

        print("[phase 5] squeeze")
        for _ in range(CLOSE_GRIPPER_STEPS):
            step_and_record(make_pose_action(flat_center, GRIP_CLOSE, rot_cmd=np.array([0.0, 0.0, 0.0]), clip_val=POS_CLIP_RELEASE))
        debug_state("after_close")

        print("[phase 6] lift")
        lift_pos = obs["robot0_eef_pos"].copy() + np.array([0.0, 0.0, lift_z_offset])
        servo_rpy_hold_pos(lift_pos, flat_target_rpy_deg, GRIP_CLOSE, steps=220)
        debug_state("after_lift")
    else:
        print("[phase 1.5] rotate wrist to object yaw for grasp")
        object_yaw_for_grasp_deg = current_earbud_axis_deg()
        target_eef_yaw_for_grasp_deg = wrap_deg(object_yaw_for_grasp_deg + grasp_eef_yaw_offset_deg)
        print(f"object_axis_for_grasp_deg={object_yaw_for_grasp_deg:.2f}")
        print(f"target_eef_yaw_for_grasp_deg={target_eef_yaw_for_grasp_deg:.2f}")
        servo_yaw_hold_pos(safe_above_earbud, target_eef_yaw_for_grasp_deg, GRIP_OPEN, steps=MAX_ROTATE_STEPS)
        debug_state("after_grasp_yaw_align")

        print("[phase 1.6] settle after grasp yaw align")
        for _ in range(ROTATE_SETTLE_STEPS):
            step_and_record(make_pose_action(safe_above_earbud, GRIP_OPEN, rot_cmd_z=0.0, clip_val=POS_CLIP_RELEASE))
        debug_state("after_grasp_yaw_settle")

        print("[phase 2] pregrasp")
        servo_to_pos(pregrasp_pos, GRIP_OPEN, steps=160, pos_tol=0.004)
        debug_state("after_pregrasp")

        print("[phase 3] preclose height")
        servo_to_pos(preclose_pos, GRIP_OPEN, steps=160, pos_tol=0.003)
        debug_state("after_preclose_pos")

        print("[phase 4] preclose gripper")
        for _ in range(PRECLOSE_STEPS):
            step_and_record(make_pose_action(preclose_pos, GRIP_CLOSE))
        debug_state("after_preclose")

        print("[phase 5] descend to cage")
        servo_to_pos(cage_pos, GRIP_CLOSE, steps=180, pos_tol=0.002)
        debug_state("after_descend")

        print("[phase 6] squeeze")
        for _ in range(CLOSE_GRIPPER_STEPS):
            step_and_record(make_pose_action(cage_pos, GRIP_CLOSE))
        debug_state("after_close")

        print("[phase 7] lift")
        lift_pos = obs["robot0_eef_pos"].copy() + np.array([0.0, 0.0, lift_z_offset])
        servo_to_pos(lift_pos, GRIP_CLOSE, steps=180, pos_tol=0.003)
        debug_state("after_lift")

    if rest_pose_mode == "flat":
        print("[phase 6.5] restore upright wrist after flat lift")
        restore_pos = obs["robot0_eef_pos"].copy()
        restore_rpy_deg = np.array([0.0, 0.0, current_eef_yaw_deg()], dtype=float)
        servo_rpy_hold_pos(restore_pos, restore_rpy_deg, GRIP_CLOSE, steps=MAX_ROTATE_STEPS)
        debug_state("after_flat_restore_upright")

        print("[phase 6.6] settle after flat restore")
        for _ in range(ROTATE_SETTLE_STEPS):
            step_and_record(make_pose_action(restore_pos, GRIP_CLOSE, rot_cmd=np.array([0.0, 0.0, 0.0]), clip_val=POS_CLIP_RELEASE))
        debug_state("after_flat_restore_settle")

    # New explicit wrist rotation phase.
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

    print("[phase 8] move to safe above slot")
    servo_to_pos(safe_above_slot, GRIP_CLOSE, steps=260, pos_tol=0.005)
    debug_state("after_safe_slot")

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

    print("[phase 9.5] fine rotate above slot")
    slot_rotate_anchor_pos = obs["robot0_eef_pos"].copy()
    servo_object_yaw_hold_pos(slot_rotate_anchor_pos, target_earbud_axis_deg, GRIP_CLOSE, steps=MAX_OBJECT_YAW_ALIGN_STEPS)
    debug_state("after_slot_fine_rotate")

    print("[phase 9.6] settle after fine rotate")
    for _ in range(ROTATE_SETTLE_STEPS):
        step_and_record(make_pose_action(slot_rotate_anchor_pos, GRIP_CLOSE, rot_cmd_z=0.0, clip_val=POS_CLIP_RELEASE))
    debug_state("after_slot_fine_rotate_settle")

    print("[phase 10] hold above slot")
    for _ in range(POST_MOVE_HOLD_STEPS):
        step_and_record(make_pose_action(obs["robot0_eef_pos"].copy(), GRIP_CLOSE))
    debug_state("after_slot_hold")

    target_obj_pre_insert = slot_pos0.copy()
    target_obj_pre_insert[2] = slot_pos0[2] + PRE_INSERT_OBJ_Z_OFFSET

    target_obj_final_insert = slot_pos0.copy()
    target_obj_final_insert[2] = slot_pos0[2] + FINAL_INSERT_OBJ_Z_OFFSET

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

    print("[phase 12] pre-release closed hold")
    for _ in range(PRE_RELEASE_HOLD_STEPS):
        step_and_record(make_pose_action(desired_eef_release, GRIP_CLOSE, clip_val=POS_CLIP_RELEASE))
    debug_state("after_pre_release_hold")

    print("[phase 13] fully open gripper")
    command_gripper_to_target(
        desired_eef_release,
        GRIP_OPEN,
        TARGET_OPEN_ABS,
        mode="open",
        max_steps=80,
    )
    debug_state("after_open")

    print("[phase 14] wait after release")
    for _ in range(RELEASE_HOLD_STEPS):
        step_and_record(make_pose_action(desired_eef_release, GRIP_OPEN, clip_val=POS_CLIP_RELEASE))
    debug_state("after_drop_hold")

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

    init_path = os.path.join(out_dir, f"init_{ts}.png")
    video_path = os.path.join(out_dir, f"full_chain_pick_random_wrist_align_descend_release_{ts}.mp4")

    imageio.imwrite(init_path, frames[0])
    imageio.mimwrite(video_path, frames, fps=20)

    print("saved:", init_path)
    print("saved:", video_path)

    if recorder is not None:
        recorder.save(
            success=release_drop_success,
            metrics={
                "obj_slot_xy": float(obj_slot_xy),
                "obj_slot_z": float(obj_slot_z),
                "yaw_err_final_deg": float(yaw_err_final_deg),
                "eef_obj_dist": float(eef_obj_dist),
                "z_lift_vs_initial": float(z_lift),
            },
        )

    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-yaw-min-deg", type=float, default=RANDOM_YAW_MIN_DEG)
    parser.add_argument("--random-yaw-max-deg", type=float, default=RANDOM_YAW_MAX_DEG)
    parser.add_argument("--flat-rest-prob", type=float, default=FLAT_REST_PROB)
    parser.add_argument("--save-demo", action="store_true")
    parser.add_argument("--demo-dir", type=str, default="/root/autodl-tmp/openpi_earbud_proto/demo_examples")
    args = parser.parse_args()
    rollout(
        args.level,
        args.seed,
        args.random_yaw_min_deg,
        args.random_yaw_max_deg,
        args.flat_rest_prob,
        save_demo=args.save_demo,
        demo_dir=args.demo_dir,
    )


if __name__ == "__main__":
    main()
