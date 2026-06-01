import os
import json
import glob
from pathlib import Path
from datetime import datetime
import argparse

import numpy as np
import imageio
import torch

from libero.libero.envs import OffScreenRenderEnv

from lerobot.policies.pi0 import PI0Policy
from lerobot.policies.factory import make_pre_post_processors

import full_chain_pick_random_wrist_align_descend_release as teacher


def quat_wxyz_to_axis_angle(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    w = np.clip(q[0], -1.0, 1.0)
    xyz = q[1:]
    s = np.linalg.norm(xyz)
    if s < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(s, w)
    axis = xyz / s
    return (axis * angle).astype(np.float32)


def build_libero_state(obs):
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)  # teacher code uses wxyz
    eef_axis_angle = quat_wxyz_to_axis_angle(eef_quat)
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
    state = np.concatenate([eef_pos, eef_axis_angle, gripper], axis=0).astype(np.float32)
    return state


def resolve_dataset_run_root(root):
    root = Path(root)
    if (root / "summary.json").exists():
        return root
    cands = sorted([p for p in root.iterdir() if p.is_dir() and (p / "summary.json").exists()])
    if not cands:
        raise FileNotFoundError(f"No summary.json found under {root}")
    return cands[-1]


def load_seed_records(dataset_root):
    dataset_root = resolve_dataset_run_root(dataset_root)
    with open(dataset_root / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    records = []
    for row in summary:
        if row.get("success", True):
            records.append(row)

    if not records:
        raise RuntimeError(f"No successful records found in {dataset_root / 'summary.json'}")
    return dataset_root, records


def load_policy(model_id, device):
    policy = PI0Policy.from_pretrained(model_id).to(device).eval()

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocess, postprocess


def parse_action(pred_action):
    if isinstance(pred_action, dict):
        if "action" in pred_action:
            pred_action = pred_action["action"]
        else:
            pred_action = next(iter(pred_action.values()))

    if torch.is_tensor(pred_action):
        pred_action = pred_action.detach().cpu().numpy()

    pred_action = np.asarray(pred_action)
    pred_action = np.squeeze(pred_action)

    if pred_action.ndim != 1:
        raise ValueError(f"Unexpected action shape after squeeze: {pred_action.shape}")

    return pred_action.astype(np.float32)


def build_env_from_seed(level, seed, yaw_min, yaw_max, flat_rest_prob):
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

    obs = env.reset()

    sim = teacher.get_sim(env)
    earbud_joint_name = teacher.get_joint_name(env, "earbud_1")
    slot_joint_name = teacher.get_joint_name(env, "charging_slot_1")

    earbud_stable_pos = obs["earbud_1_pos"].copy()
    q_vertical = np.array([0.70710678, 0.0, -0.70710678, 0.0], dtype=float)
    q_flat_roll_local = teacher.quat_wxyz_from_axis_angle([0, 0, 1], teacher.FLAT_REST_ROLL_DEG)
    random_yaw_deg = rng.uniform(yaw_min, yaw_max)
    q_random_yaw = teacher.quat_wxyz_from_axis_angle([0, 0, 1], random_yaw_deg)

    rest_pose_mode = "flat" if rng.rand() < flat_rest_prob else "edge"
    if rest_pose_mode == "flat":
        earbud_stable_pos[2] = teacher.EARBUD_FLAT_REST_Z
        q_rest_base = teacher.quat_mul_wxyz(q_vertical, q_flat_roll_local)
    else:
        earbud_stable_pos[2] = teacher.EARBUD_EDGE_REST_Z
        q_rest_base = q_vertical

    earbud_stable_quat = teacher.quat_mul_wxyz(q_random_yaw, q_rest_base)

    slot_stable_pos = obs["charging_slot_1_pos"].copy()
    slot_stable_pos[2] = max(slot_stable_pos[2], 0.4680)
    slot_stable_quat = teacher.quat_wxyz_from_axis_angle([0, 1, 0], teacher.SLOT_Y_DEG)

    slot_long_axis_deg = teacher.projected_axis_heading_deg_from_quat_wxyz(
        slot_stable_quat, teacher.SLOT_LONG_AXIS_LOCAL
    )
    target_earbud_axis_deg = teacher.canonical_axis_deg(
        slot_long_axis_deg + teacher.SLOT_LONG_AXIS_YAW_OFFSET_DEG
    )

    def refresh_obs():
        nonlocal obs
        base = env.env if hasattr(env, "env") else env
        if hasattr(base, "_get_observations"):
            try:
                obs = base._get_observations(force_update=True)
            except TypeError:
                obs = base._get_observations()

    def enforce_slot():
        q_slot = teacher.get_joint_qpos(sim, slot_joint_name)
        q_slot[:3] = slot_stable_pos
        q_slot[3:7] = slot_stable_quat
        teacher.set_joint_qpos(sim, slot_joint_name, q_slot)
        teacher.set_joint_qvel_zero(sim, slot_joint_name)

    def step_env(action):
        nonlocal obs
        enforce_slot()
        obs, reward, done, info = env.step(action)
        enforce_slot()
        refresh_obs()
        return obs, reward, done, info

    # 与教师脚本一致：reset 后先稳定 25 步
    for _ in range(25):
        q_ear = teacher.get_joint_qpos(sim, earbud_joint_name)
        q_ear[:3] = earbud_stable_pos
        q_ear[3:7] = earbud_stable_quat
        teacher.set_joint_qpos(sim, earbud_joint_name, q_ear)
        teacher.set_joint_qvel_zero(sim, earbud_joint_name)

        step_env(np.zeros(7, dtype=np.float32))

    return env, obs, step_env, {
        "seed": seed,
        "random_yaw_deg": float(random_yaw_deg),
        "rest_pose_mode": rest_pose_mode,
        "target_earbud_axis_deg": float(target_earbud_axis_deg),
    }


def evaluate_one_seed(policy, preprocess, postprocess, device, level, seed, yaw_min, yaw_max, flat_rest_prob, max_steps, out_dir):
    env, obs, step_env, init_meta = build_env_from_seed(level, seed, yaw_min, yaw_max, flat_rest_prob)

    frames = []
    success_count = 0

    for step in range(max_steps):
        frame = {
            "observation.images.image": obs["agentview_image"][::-1].copy(),
            "observation.images.image2": obs["robot0_eye_in_hand_image"][::-1].copy(),
            "observation.state": build_libero_state(obs),
            "task": "insert the earbud into the charging slot",
        }

        batch = preprocess(frame)

        with torch.inference_mode():
            pred_action = policy.select_action(batch)
            pred_action = postprocess(pred_action)

        action = parse_action(pred_action)

        obs, reward, done, info = step_env(action)

        frames.append(obs["agentview_image"][::-1].copy())

        earbud_pos = obs["earbud_1_pos"].copy()
        slot_pos = obs["charging_slot_1_pos"].copy()
        obj_slot_xy = float(np.linalg.norm(earbud_pos[:2] - slot_pos[:2]))
        obj_slot_z = float(earbud_pos[2] - slot_pos[2])

        success = (obj_slot_xy < 0.02) and (obj_slot_z < 0.03)
        if success:
            success_count += 1
        else:
            success_count = 0

        # 连续 5 步满足就算成功并提前结束
        if success_count >= 5:
            break

    earbud_pos_final = obs["earbud_1_pos"].copy()
    slot_pos_final = obs["charging_slot_1_pos"].copy()
    eef_pos_final = obs["robot0_eef_pos"].copy()

    obj_slot_xy = float(np.linalg.norm(earbud_pos_final[:2] - slot_pos_final[:2]))
    obj_slot_z = float(earbud_pos_final[2] - slot_pos_final[2])
    eef_obj_dist = float(np.linalg.norm(eef_pos_final - earbud_pos_final))
    release_drop_success = bool((obj_slot_xy < 0.02) and (obj_slot_z < 0.03))

    result = {
        **init_meta,
        "level": level,
        "num_steps": len(frames),
        "release_drop_success": release_drop_success,
        "eef_obj_dist": eef_obj_dist,
        "obj_slot_xy": obj_slot_xy,
        "obj_slot_z": obj_slot_z,
    }

    ep_dir = out_dir / f"seed_{seed:04d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    imageio.mimwrite(ep_dir / "preview.mp4", frames, fps=20)
    with open(ep_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    env.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="/root/autodl-tmp/openpi_earbud_proto/small_edge_dataset_minimal")
    parser.add_argument("--level", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--model_id", type=str, default="lerobot/pi0_libero_base")
    parser.add_argument("--max_steps", type=int, default=220)
    parser.add_argument("--yaw_min", type=float, default=None)
    parser.add_argument("--yaw_max", type=float, default=None)
    parser.add_argument("--flat_rest_prob", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    dataset_root, records = load_seed_records(args.dataset_root)

    # 如果 summary.json 里带了 yaw 范围，就优先复用它
    if args.yaw_min is None:
        args.yaw_min = float(records[0].get("yaw_min", -30.0))
    if args.yaw_max is None:
        args.yaw_max = float(records[0].get("yaw_max", 30.0))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy, preprocess, postprocess = load_policy(args.model_id, device)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"/root/autodl-tmp/openpi_earbud_proto/pi0_base_zero_shot_eval/{args.level}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("dataset_root:", dataset_root)
    print("num_test_seeds:", len(records))
    print("model_id:", args.model_id)
    print("device:", device)
    print("yaw_min:", args.yaw_min)
    print("yaw_max:", args.yaw_max)
    print("flat_rest_prob:", args.flat_rest_prob)

    all_results = []
    success_count = 0

    for row in records:
        seed = int(row["seed"])
        print(f"\n===== evaluating seed {seed} =====")
        result = evaluate_one_seed(
            policy=policy,
            preprocess=preprocess,
            postprocess=postprocess,
            device=device,
            level=args.level,
            seed=seed,
            yaw_min=args.yaw_min,
            yaw_max=args.yaw_max,
            flat_rest_prob=args.flat_rest_prob,
            max_steps=args.max_steps,
            out_dir=out_dir,
        )
        all_results.append(result)
        if result["release_drop_success"]:
            success_count += 1

        print(
            f"seed={seed} "
            f"success={result['release_drop_success']} "
            f"xy={result['obj_slot_xy']:.4f} "
            f"z={result['obj_slot_z']:.4f} "
            f"steps={result['num_steps']}"
        )

    summary = {
        "model_id": args.model_id,
        "level": args.level,
        "num_episodes": len(all_results),
        "num_success": success_count,
        "success_rate": success_count / max(len(all_results), 1),
        "results": all_results,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== FINAL SUMMARY =====")
    print("output_dir:", out_dir)
    print("num_episodes:", summary["num_episodes"])
    print("num_success:", summary["num_success"])
    print("success_rate:", summary["success_rate"])


if __name__ == "__main__":
    main()
