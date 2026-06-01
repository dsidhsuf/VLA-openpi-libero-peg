import os
import json
from datetime import datetime
import argparse

import numpy as np
import imageio

import full_chain_pick_random_wrist_align_descend_release as teacher
from libero.libero.envs import OffScreenRenderEnv


def collect_one_episode(level, seed, yaw_min_deg, yaw_max_deg, out_root):
    cfg = teacher.LEVEL_CFG[level]

    env = OffScreenRenderEnv(
        bddl_file_name=cfg["bddl"],
        camera_heights=teacher.CAMERA_SIZE,
        camera_widths=teacher.CAMERA_SIZE,
        ignore_done=True,
    )
    env.seed(seed)
    rng = np.random.RandomState(seed)

    if hasattr(env, "env") and hasattr(env.env, "_check_success"):
        env.env._check_success = lambda: False

    sim = teacher.get_sim(env)
    earbud_joint_name = teacher.get_joint_name(env, "earbud_1")
    slot_joint_name = teacher.get_joint_name(env, "charging_slot_1")

    obs = env.reset()

    def _get_joint_qpos(joint_name):
        if hasattr(sim, "data") and hasattr(sim.data, "get_joint_qpos"):
            return np.array(sim.data.get_joint_qpos(joint_name), dtype=float)
        if hasattr(sim, "get_joint_qpos"):
            return np.array(sim.get_joint_qpos(joint_name), dtype=float)
        raise AttributeError("sim has neither data.get_joint_qpos nor get_joint_qpos")

    def _set_joint_qpos(joint_name, qpos):
        if hasattr(sim, "data") and hasattr(sim.data, "set_joint_qpos"):
            sim.data.set_joint_qpos(joint_name, qpos)
        elif hasattr(sim, "set_joint_qpos"):
            sim.set_joint_qpos(joint_name, qpos)
        else:
            raise AttributeError("sim has neither data.set_joint_qpos nor set_joint_qpos")
        sim.forward()

    def _set_joint_qvel_zero(joint_name):
        try:
            if hasattr(sim, "data") and hasattr(sim.data, "get_joint_qvel"):
                qvel = np.array(sim.data.get_joint_qvel(joint_name), dtype=float)
                sim.data.set_joint_qvel(joint_name, np.zeros_like(qvel))
            elif hasattr(sim, "get_joint_qvel"):
                qvel = np.array(sim.get_joint_qvel(joint_name), dtype=float)
                sim.set_joint_qvel(joint_name, np.zeros_like(qvel))
        except Exception:
            pass

    # ========= edge-only =========
    rest_pose_mode = "edge"
    random_yaw_deg = rng.uniform(yaw_min_deg, yaw_max_deg)

    earbud_stable_pos = obs["earbud_1_pos"].copy()
    earbud_stable_pos[2] = teacher.EARBUD_EDGE_REST_Z

    q_vertical = np.array([0.70710678, 0.0, -0.70710678, 0.0], dtype=float)
    q_random_yaw = teacher.quat_wxyz_from_axis_angle([0, 0, 1], random_yaw_deg)
    earbud_stable_quat = teacher.quat_mul_wxyz(q_random_yaw, q_vertical)

    slot_stable_pos = obs["charging_slot_1_pos"].copy()
    slot_stable_pos[2] = max(slot_stable_pos[2], 0.4680)
    slot_stable_quat = teacher.quat_wxyz_from_axis_angle([0, 1, 0], teacher.SLOT_Y_DEG)

    slot_long_axis_deg = teacher.projected_axis_heading_deg_from_quat_wxyz(
        slot_stable_quat, teacher.SLOT_LONG_AXIS_LOCAL
    )
    target_earbud_axis_deg = teacher.canonical_axis_deg(
        slot_long_axis_deg + teacher.SLOT_LONG_AXIS_YAW_OFFSET_DEG
    )

    episode = {
        "images_agentview": [],
        "state": [],
        "action": [],
        "phase": [],
    }

    def refresh_obs():
        nonlocal obs
        base = env.env if hasattr(env, "env") else env
        if hasattr(base, "_get_observations"):
            try:
                obs = base._get_observations(force_update=True)
            except TypeError:
                obs = base._get_observations()

    def enforce_slot():
        q_slot = _get_joint_qpos(slot_joint_name)
        q_slot[:3] = slot_stable_pos
        q_slot[3:7] = slot_stable_quat
        _set_joint_qpos(slot_joint_name, q_slot)
        _set_joint_qvel_zero(slot_joint_name)

    def step_and_record(action, phase_name):
        nonlocal obs
        enforce_slot()
        obs, reward, done, info = env.step(action)
        enforce_slot()
        refresh_obs()

        episode["images_agentview"].append(obs["agentview_image"].copy())
        episode["state"].append(obs["robot0_proprio-state"].copy())
        episode["action"].append(action.copy())
        episode["phase"].append(phase_name)

    def gripper_abs():
        g = np.asarray(obs["robot0_gripper_qpos"], dtype=float)
        return float(np.mean(np.abs(g)))

    def current_eef_yaw_deg():
        return teacher.yaw_deg_from_quat_wxyz(obs["robot0_eef_quat"])

    def current_earbud_axis_deg():
        q_ear = _get_joint_qpos(earbud_joint_name)
        return teacher.projected_axis_heading_deg_from_quat_wxyz(q_ear[3:7], teacher.EARBUD_LONG_AXIS_LOCAL)

    def make_pose_action(target_pos, gripper_cmd, rot_cmd=None, rot_cmd_z=0.0, clip_val=teacher.POS_CLIP):
        cur_pos = obs["robot0_eef_pos"]
        pos_err = target_pos - cur_pos

        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip(teacher.KP_POS * pos_err, -clip_val, clip_val)

        if rot_cmd is None:
            action[3:5] = 0.0
            action[5] = float(np.clip(rot_cmd_z, -teacher.YAW_CLIP, teacher.YAW_CLIP))
        else:
            rot_cmd = np.asarray(rot_cmd, dtype=float)
            action[3] = float(np.clip(rot_cmd[0], -teacher.ROT_CLIP, teacher.ROT_CLIP))
            action[4] = float(np.clip(rot_cmd[1], -teacher.ROT_CLIP, teacher.ROT_CLIP))
            action[5] = float(np.clip(rot_cmd[2], -teacher.YAW_CLIP, teacher.YAW_CLIP))

        action[6] = float(np.clip(gripper_cmd, -1.0, 1.0))
        return action

    def servo_to_pos(target_pos, grip_cmd, phase_name, steps=teacher.MAX_SERVO_STEPS, pos_tol=0.004):
        for _ in range(steps):
            action = make_pose_action(target_pos, grip_cmd, rot_cmd_z=0.0, clip_val=teacher.POS_CLIP)
            step_and_record(action, phase_name)
            pos_err = np.linalg.norm(target_pos - obs["robot0_eef_pos"])
            if pos_err < pos_tol:
                break

    def servo_to_pos_slow(target_pos, grip_cmd, phase_name, steps=220, pos_tol=0.003):
        for _ in range(steps):
            action = make_pose_action(target_pos, grip_cmd, rot_cmd_z=0.0, clip_val=teacher.POS_CLIP_RELEASE)
            step_and_record(action, phase_name)
            pos_err = np.linalg.norm(target_pos - obs["robot0_eef_pos"])
            if pos_err < pos_tol:
                break

    def servo_yaw_hold_pos(target_pos, target_eef_yaw_deg, grip_cmd, phase_name, steps=teacher.MAX_ROTATE_STEPS):
        for _ in range(steps):
            eef_yaw = current_eef_yaw_deg()
            yaw_err_deg = teacher.wrap_deg(target_eef_yaw_deg - eef_yaw)
            rot_cmd = np.clip(teacher.KP_YAW * np.deg2rad(yaw_err_deg), -teacher.YAW_CLIP, teacher.YAW_CLIP)
            action = make_pose_action(target_pos, grip_cmd, rot_cmd_z=rot_cmd, clip_val=teacher.POS_CLIP_RELEASE)
            step_and_record(action, phase_name)
            pos_err = np.linalg.norm(target_pos - obs["robot0_eef_pos"])
            if abs(yaw_err_deg) < teacher.YAW_TOL_DEG and pos_err < 0.004:
                break

    def servo_object_yaw_hold_pos(target_pos, target_object_yaw_deg, grip_cmd, phase_name, steps=teacher.MAX_OBJECT_YAW_ALIGN_STEPS):
        for _ in range(steps):
            object_yaw = current_earbud_axis_deg()
            yaw_err_deg = teacher.wrap_axis_err_deg(target_object_yaw_deg, object_yaw)
            rot_cmd = np.clip(teacher.KP_YAW * np.deg2rad(yaw_err_deg), -teacher.YAW_CLIP, teacher.YAW_CLIP)
            action = make_pose_action(target_pos, grip_cmd, rot_cmd_z=rot_cmd, clip_val=teacher.POS_CLIP_RELEASE)
            step_and_record(action, phase_name)
            pos_err = np.linalg.norm(target_pos - obs["robot0_eef_pos"])
            if abs(yaw_err_deg) < teacher.YAW_TOL_DEG and pos_err < 0.004:
                break

    def servo_object_pose_to_target(
        target_object_pos,
        target_object_yaw_deg,
        grip_cmd,
        phase_name,
        steps=260,
        pos_tol_xy=0.0025,
        pos_tol_z=0.003,
        yaw_tol_deg=6.0,
        clip_val=teacher.POS_CLIP_RELEASE,
    ):
        for _ in range(steps):
            earbud_pos = obs["earbud_1_pos"].copy()
            eef_pos = obs["robot0_eef_pos"].copy()
            obj_minus_eef = earbud_pos - eef_pos
            desired_eef_pos = target_object_pos - obj_minus_eef

            object_yaw = current_earbud_axis_deg()
            yaw_err_deg = teacher.wrap_axis_err_deg(target_object_yaw_deg, object_yaw)
            rot_cmd = np.clip(teacher.KP_YAW * np.deg2rad(yaw_err_deg), -teacher.YAW_CLIP, teacher.YAW_CLIP)

            action = make_pose_action(desired_eef_pos, grip_cmd, rot_cmd_z=rot_cmd, clip_val=clip_val)
            step_and_record(action, phase_name)

            earbud_pos = obs["earbud_1_pos"].copy()
            xy_err = np.linalg.norm(earbud_pos[:2] - target_object_pos[:2])
            z_err = abs(earbud_pos[2] - target_object_pos[2])
            object_yaw = current_earbud_axis_deg()
            yaw_err_deg = abs(teacher.wrap_axis_err_deg(target_object_yaw_deg, object_yaw))

            if xy_err < pos_tol_xy and z_err < pos_tol_z and yaw_err_deg < yaw_tol_deg:
                break

    def command_gripper_to_target(target_pos, command, target_abs, mode, phase_name, max_steps=80):
        for _ in range(max_steps):
            action = make_pose_action(target_pos, command, rot_cmd_z=0.0, clip_val=teacher.POS_CLIP_RELEASE)
            step_and_record(action, phase_name)
            cur = gripper_abs()
            if mode == "open" and cur >= target_abs:
                break
            if mode == "close" and cur <= target_abs:
                break

    # stabilize
    for _ in range(25):
        q_ear = _get_joint_qpos(earbud_joint_name)
        q_ear[:3] = earbud_stable_pos
        q_ear[3:7] = earbud_stable_quat
        _set_joint_qpos(earbud_joint_name, q_ear)
        _set_joint_qvel_zero(earbud_joint_name)
        enforce_slot()
        step_and_record(np.zeros(7, dtype=np.float32), "stabilize")

    earbud_pos0 = obs["earbud_1_pos"].copy()
    slot_pos0 = obs["charging_slot_1_pos"].copy()
    eef_pos0 = obs["robot0_eef_pos"].copy()

    pregrasp_z_offset = teacher.PREGRASP_Z_OFFSET
    preclose_z_offset = teacher.PRECLOSE_Z_OFFSET
    cage_z_offset = teacher.CAGE_Z_OFFSET
    lift_z_offset = teacher.LIFT_Z_OFFSET
    grasp_eef_yaw_offset_deg = teacher.GRASP_EEF_YAW_OFFSET_DEG
    grasp_x_offset = teacher.GRASP_X_OFFSET
    grasp_y_offset = teacher.GRASP_Y_OFFSET

    safe_up_pos = np.array([eef_pos0[0], eef_pos0[1], teacher.SAFE_TRAVEL_Z], dtype=float)
    safe_above_earbud = np.array([earbud_pos0[0] + grasp_x_offset, earbud_pos0[1] + grasp_y_offset, teacher.SAFE_TRAVEL_Z], dtype=float)
    pregrasp_pos = np.array([earbud_pos0[0] + grasp_x_offset, earbud_pos0[1] + grasp_y_offset, earbud_pos0[2] + pregrasp_z_offset], dtype=float)
    preclose_pos = np.array([earbud_pos0[0] + grasp_x_offset, earbud_pos0[1] + grasp_y_offset, earbud_pos0[2] + preclose_z_offset], dtype=float)
    cage_pos = np.array([earbud_pos0[0] + grasp_x_offset, earbud_pos0[1] + grasp_y_offset, earbud_pos0[2] + cage_z_offset], dtype=float)

    servo_to_pos(safe_up_pos, teacher.GRIP_OPEN, "safe_up", 100, 0.006)
    for _ in range(18):
        step_and_record(make_pose_action(obs["robot0_eef_pos"], teacher.GRIP_CLOSE), "air_close")
    for _ in range(18):
        step_and_record(make_pose_action(obs["robot0_eef_pos"], teacher.GRIP_OPEN), "air_open")

    servo_to_pos(safe_above_earbud, teacher.GRIP_OPEN, "safe_xy", 160, 0.005)

    object_yaw_for_grasp_deg = current_earbud_axis_deg()
    target_eef_yaw_for_grasp_deg = teacher.wrap_deg(object_yaw_for_grasp_deg + grasp_eef_yaw_offset_deg)
    servo_yaw_hold_pos(safe_above_earbud, target_eef_yaw_for_grasp_deg, teacher.GRIP_OPEN, "grasp_yaw_align")

    for _ in range(teacher.ROTATE_SETTLE_STEPS):
        step_and_record(make_pose_action(safe_above_earbud, teacher.GRIP_OPEN, rot_cmd_z=0.0, clip_val=teacher.POS_CLIP_RELEASE), "grasp_yaw_settle")

    servo_to_pos(pregrasp_pos, teacher.GRIP_OPEN, "pregrasp", 160, 0.004)
    servo_to_pos(preclose_pos, teacher.GRIP_OPEN, "preclose_pos", 160, 0.003)

    for _ in range(teacher.PRECLOSE_STEPS):
        step_and_record(make_pose_action(preclose_pos, teacher.GRIP_CLOSE), "preclose")

    servo_to_pos(cage_pos, teacher.GRIP_CLOSE, "descend", 180, 0.002)

    for _ in range(teacher.CLOSE_GRIPPER_STEPS):
        step_and_record(make_pose_action(cage_pos, teacher.GRIP_CLOSE), "close")

    lift_pos = obs["robot0_eef_pos"].copy() + np.array([0.0, 0.0, lift_z_offset])
    servo_to_pos(lift_pos, teacher.GRIP_CLOSE, "lift", 180, 0.003)

    rotate_anchor_pos = obs["robot0_eef_pos"].copy()
    servo_object_yaw_hold_pos(rotate_anchor_pos, target_earbud_axis_deg, teacher.GRIP_CLOSE, "rotate_align")

    for _ in range(teacher.ROTATE_SETTLE_STEPS):
        step_and_record(make_pose_action(rotate_anchor_pos, teacher.GRIP_CLOSE, rot_cmd_z=0.0, clip_val=teacher.POS_CLIP_RELEASE), "rotate_settle")

    earbud_pos_lift = obs["earbud_1_pos"].copy()
    eef_pos_lift = obs["robot0_eef_pos"].copy()
    obj_minus_eef = earbud_pos_lift - eef_pos_lift

    desired_obj_hover = slot_pos0.copy()
    desired_obj_hover[2] = slot_pos0[2] + teacher.SLOT_HOVER_Z_OFFSET
    desired_eef_hover = desired_obj_hover - obj_minus_eef

    safe_above_slot = np.array([desired_eef_hover[0], desired_eef_hover[1], teacher.SAFE_TRAVEL_Z], dtype=float)

    servo_to_pos(safe_above_slot, teacher.GRIP_CLOSE, "safe_slot", 260, 0.005)

    servo_object_pose_to_target(
        desired_obj_hover,
        target_earbud_axis_deg,
        teacher.GRIP_CLOSE,
        "slot_hover",
        steps=320,
        pos_tol_xy=0.003,
        pos_tol_z=0.004,
        yaw_tol_deg=5.0,
        clip_val=teacher.POS_CLIP_RELEASE,
    )

    slot_rotate_anchor_pos = obs["robot0_eef_pos"].copy()
    servo_object_yaw_hold_pos(slot_rotate_anchor_pos, target_earbud_axis_deg, teacher.GRIP_CLOSE, "slot_fine_rotate")

    for _ in range(teacher.ROTATE_SETTLE_STEPS):
        step_and_record(make_pose_action(slot_rotate_anchor_pos, teacher.GRIP_CLOSE, rot_cmd_z=0.0, clip_val=teacher.POS_CLIP_RELEASE), "slot_fine_rotate_settle")

    for _ in range(teacher.POST_MOVE_HOLD_STEPS):
        step_and_record(make_pose_action(obs["robot0_eef_pos"].copy(), teacher.GRIP_CLOSE), "slot_hold")

    target_obj_pre_insert = slot_pos0.copy()
    target_obj_pre_insert[2] = slot_pos0[2] + teacher.PRE_INSERT_OBJ_Z_OFFSET

    target_obj_final_insert = slot_pos0.copy()
    target_obj_final_insert[2] = slot_pos0[2] + teacher.FINAL_INSERT_OBJ_Z_OFFSET

    servo_object_pose_to_target(
        target_obj_pre_insert,
        target_earbud_axis_deg,
        teacher.GRIP_CLOSE,
        "pre_insert_descend",
        steps=320,
        pos_tol_xy=0.0025,
        pos_tol_z=0.004,
        yaw_tol_deg=6.0,
        clip_val=teacher.POS_CLIP_RELEASE,
    )

    servo_object_pose_to_target(
        target_obj_final_insert,
        target_earbud_axis_deg,
        teacher.GRIP_CLOSE,
        "final_insert_descend",
        steps=360,
        pos_tol_xy=0.002,
        pos_tol_z=0.003,
        yaw_tol_deg=7.0,
        clip_val=teacher.POS_CLIP_RELEASE,
    )

    desired_eef_release = obs["robot0_eef_pos"].copy()

    for _ in range(teacher.PRE_RELEASE_HOLD_STEPS):
        step_and_record(make_pose_action(desired_eef_release, teacher.GRIP_CLOSE, clip_val=teacher.POS_CLIP_RELEASE), "pre_release_hold")

    command_gripper_to_target(desired_eef_release, teacher.GRIP_OPEN, teacher.TARGET_OPEN_ABS, "open", "open")

    for _ in range(teacher.RELEASE_HOLD_STEPS):
        step_and_record(make_pose_action(desired_eef_release, teacher.GRIP_OPEN, clip_val=teacher.POS_CLIP_RELEASE), "drop_hold")

    retreat_pos = obs["robot0_eef_pos"].copy()
    retreat_pos[2] += teacher.RETREAT_Z
    servo_to_pos(retreat_pos, teacher.GRIP_OPEN, "retreat", 160, 0.004)

    earbud_pos_final = obs["earbud_1_pos"].copy()
    slot_pos_final = obs["charging_slot_1_pos"].copy()
    eef_pos_final = obs["robot0_eef_pos"].copy()

    z_lift = earbud_pos_final[2] - earbud_pos0[2]
    eef_obj_dist = np.linalg.norm(eef_pos_final - earbud_pos_final)
    obj_slot_xy = np.linalg.norm(earbud_pos_final[:2] - slot_pos_final[:2])
    obj_slot_z = earbud_pos_final[2] - slot_pos_final[2]
    yaw_err_final_deg = teacher.wrap_axis_err_deg(target_earbud_axis_deg, current_earbud_axis_deg())

    release_drop_success = (obj_slot_xy < 0.02) and (obj_slot_z < 0.03)

    episode["images_agentview"] = np.stack(episode["images_agentview"], axis=0).astype(np.uint8)
    episode["state"] = np.stack(episode["state"], axis=0).astype(np.float32)
    episode["action"] = np.stack(episode["action"], axis=0).astype(np.float32)
    episode["phase"] = np.asarray(episode["phase"])

    meta = {
        "level": level,
        "seed": seed,
        "rest_pose_mode": rest_pose_mode,
        "random_yaw_deg": float(random_yaw_deg),
        "target_earbud_axis_deg": float(target_earbud_axis_deg),
        "release_drop_success": bool(release_drop_success),
        "earbud_z_initial": float(earbud_pos0[2]),
        "earbud_z_final": float(earbud_pos_final[2]),
        "z_lift_vs_initial": float(z_lift),
        "eef_obj_dist": float(eef_obj_dist),
        "obj_slot_xy": float(obj_slot_xy),
        "obj_slot_z": float(obj_slot_z),
        "yaw_err_final_deg": float(yaw_err_final_deg),
        "language": "insert the earbud into the charging slot",
        "num_steps": int(len(episode["action"])),
    }

    env.close()

    if not release_drop_success:
        return False, meta, None

    episode_dir = os.path.join(out_root, f"ep_seed{seed:04d}")
    os.makedirs(episode_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(episode_dir, "episode.npz"),
        images_agentview=episode["images_agentview"],
        state=episode["state"],
        action=episode["action"],
        phase=episode["phase"],
    )
    with open(os.path.join(episode_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    imageio.mimwrite(
        os.path.join(episode_dir, "preview.mp4"),
        episode["images_agentview"],
        fps=20,
    )

    return True, meta, episode_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--num_episodes", type=int, default=6)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--yaw_min", type=float, default=-30.0)
    parser.add_argument("--yaw_max", type=float, default=30.0)
    parser.add_argument("--out_root", type=str, default="/root/autodl-tmp/openpi_earbud_proto/small_edge_dataset")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(args.out_root, f"{args.level}_edge_yaw_{ts}")
    os.makedirs(out_root, exist_ok=True)

    summary = []
    success_count = 0
    seed = args.seed_start

    while success_count < args.num_episodes:
        ok, meta, episode_dir = collect_one_episode(
            level=args.level,
            seed=seed,
            yaw_min_deg=args.yaw_min,
            yaw_max_deg=args.yaw_max,
            out_root=out_root,
        )

        row = dict(meta)
        row["episode_dir"] = episode_dir
        summary.append(row)

        print(
            f"[seed={seed}] "
            f"success={ok} "
            f"yaw={meta['random_yaw_deg']:.2f} "
            f"xy={meta['obj_slot_xy']:.4f} "
            f"z={meta['obj_slot_z']:.4f}"
        )

        if ok:
            success_count += 1

        seed += 1

    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("saved dataset root:", out_root)
    print("successful episodes:", success_count)


if __name__ == "__main__":
    main()
