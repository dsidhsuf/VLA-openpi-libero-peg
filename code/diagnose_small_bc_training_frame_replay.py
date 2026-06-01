#!/usr/bin/env python3
"""
Replay real LeRobot training frames against any compatible HTTP policy server.

This is a small-model friendly version of the PI0.5 replay diagnostic. It
prints both aggregate metrics and phase-specific groups:
  - label_g_neg / label_g_pos
  - label_z_pos / label_z_neg
  - large_xy
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_TASK = "Pick up the red rectangular peg, keep it vertical, and insert it into the rectangular slot."


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def image_to_uint8_hwc(value: Any) -> np.ndarray:
    arr = to_numpy(value)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if float(np.nanmax(arr)) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def encode_image_to_b64(img: Any, quality: int = 90) -> str:
    pil = Image.fromarray(image_to_uint8_hwc(img))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def http_json(url: str, obj: Any = None, timeout: int = 1800) -> Any:
    if obj is None:
        req = urllib.request.Request(url, method="GET")
    else:
        data = json.dumps(obj).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def infer_first_action(server: str, payload: dict[str, Any]) -> np.ndarray:
    out = http_json(server.rstrip("/") + "/infer", payload)
    # Waypoint-target servers return the actual servo action plus the raw model
    # output. For training-frame replay we want to compare labels to the model
    # output, not to the derived servo action.
    action = np.asarray(out.get("model_output", out["action"]), dtype=np.float32)
    if action.ndim == 1:
        return action.reshape(-1)[:7]
    if action.ndim == 2:
        if action.shape[1] == 7:
            return action[0].reshape(-1)[:7]
        if action.shape[0] == 7:
            return action[:, 0].reshape(-1)[:7]
    raise ValueError(f"Unexpected action shape from server: {action.shape}")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / den)


def select_indices(actions: np.ndarray, max_samples: int) -> list[int]:
    n = len(actions)
    picks: list[int] = []

    def add(indices: Any, limit: int | None = None) -> None:
        added = 0
        for idx in indices:
            idx = int(idx)
            if 0 <= idx < n and idx not in picks:
                picks.append(idx)
                added += 1
                if limit is not None and added >= limit:
                    break

    quota = max(10, max_samples // 6)
    add(range(min(20, n)), limit=20)
    grip = actions[:, -1]
    add(np.argsort(np.abs(np.diff(grip, prepend=grip[0])))[::-1], limit=quota)
    add(np.where(grip < -0.5)[0], limit=quota)
    add(np.where(grip > 0.5)[0], limit=quota)
    add(np.argsort(actions[:, 2]), limit=quota)  # strongest downward labels
    add(np.argsort(-actions[:, 2]), limit=quota)  # strongest upward labels
    add(np.argsort(-np.linalg.norm(actions[:, :2], axis=1)), limit=quota)
    add(np.linspace(0, n - 1, num=min(max_samples, n), dtype=int))
    return picks[:max_samples]


def sample_task(sample: dict[str, Any], default_task: str) -> str:
    task = sample.get("task", default_task)
    if isinstance(task, (list, tuple)) and task:
        task = task[0]
    if not isinstance(task, str):
        task = default_task
    return task.strip() or default_task


def summarize_group(name: str, rows: list[dict[str, Any]], mask: np.ndarray) -> None:
    selected = [row for row, keep in zip(rows, mask) if keep]
    if not selected:
        return
    cos_values = [r["cos_xyz"] for r in selected if not math.isnan(r["cos_xyz"])]
    print(name, "n=", len(selected))
    print("  mae_xyz:", float(np.mean([r["mae_xyz"] for r in selected])))
    print("  mae_action:", float(np.mean([r["mae_action"] for r in selected])))
    print("  cos_xyz:", float(np.mean(cos_values)) if cos_values else float("nan"))
    print("  z_sign:", float(np.mean([r["sign_z_match"] for r in selected])))
    print("  g_sign:", float(np.mean([r["sign_gripper_match"] for r in selected])))
    print("  pred_g_mean:", float(np.mean([r["pred_gripper"] for r in selected])))
    print("  pred_z_mean:", float(np.mean([r["pred_z"] for r in selected])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8001")
    parser.add_argument("--output-csv", default="training_frame_replay_small_bc.csv")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--image-key", default="observation.images.image")
    parser.add_argument("--image2-key", default="observation.images.image2")
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--default-task", default=DEFAULT_TASK)
    parser.add_argument("--task-override", default="")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    try:
        ds = LeRobotDataset(repo_id=args.repo_id, root=args.dataset_root, return_uint8=True)
    except TypeError:
        ds = LeRobotDataset(repo_id=args.repo_id, root=args.dataset_root)

    health = http_json(args.server.rstrip("/") + "/health")
    progress_denominator = float(health.get("progress_denominator") or 500.0)
    print("[server health]", health)
    print("[dataset]", args.dataset_root, "len=", len(ds))

    actions = []
    for i in range(len(ds)):
        actions.append(to_numpy(ds[i][args.action_key]).astype(np.float32).reshape(-1)[:7])
    actions_np = np.asarray(actions, dtype=np.float32)
    indices = select_indices(actions_np, max_samples=int(args.max_samples))

    rows: list[dict[str, Any]] = []
    for out_i, idx in enumerate(indices, start=1):
        sample = ds[idx]
        task = args.task_override.strip() or sample_task(sample, args.default_task)
        payload = {
            "task": task,
            args.state_key: to_numpy(sample[args.state_key]).astype(np.float32).reshape(-1).tolist(),
            args.image_key: encode_image_to_b64(sample[args.image_key], args.jpeg_quality),
            args.image2_key: encode_image_to_b64(sample[args.image2_key], args.jpeg_quality),
            "policy_step": int(idx),
            "policy_progress": float(idx) / max(1.0, progress_denominator),
        }
        pred = infer_first_action(args.server, payload).astype(np.float32)
        label = actions_np[idx]
        row = {
            "idx": int(idx),
            "policy_progress": float(idx) / max(1.0, progress_denominator),
            "task": task,
            "label_x": float(label[0]),
            "label_y": float(label[1]),
            "label_z": float(label[2]),
            "label_rx": float(label[3]),
            "label_ry": float(label[4]),
            "label_rz": float(label[5]),
            "label_gripper": float(label[6]),
            "pred_x": float(pred[0]),
            "pred_y": float(pred[1]),
            "pred_z": float(pred[2]),
            "pred_rx": float(pred[3]),
            "pred_ry": float(pred[4]),
            "pred_rz": float(pred[5]),
            "pred_gripper": float(pred[6]),
            "mae_xyz": float(np.mean(np.abs(pred[:3] - label[:3]))),
            "mae_action": float(np.mean(np.abs(pred[:7] - label[:7]))),
            "cos_xyz": cosine(pred[:3], label[:3]),
            "sign_z_match": bool(np.sign(pred[2]) == np.sign(label[2]) or abs(label[2]) < 1e-6),
            "sign_gripper_match": bool(np.sign(pred[6]) == np.sign(label[6]) or abs(label[6]) < 1e-6),
        }
        rows.append(row)
        print(
            f"[{out_i:03d}/{len(indices):03d}] idx={idx:05d} "
            f"label_xyz=({label[0]:+.3f},{label[1]:+.3f},{label[2]:+.3f}) "
            f"pred_xyz=({pred[0]:+.3f},{pred[1]:+.3f},{pred[2]:+.3f}) "
            f"label_g={label[6]:+.3f} pred_g={pred[6]:+.3f} "
            f"mae_xyz={row['mae_xyz']:.4f} cos_xyz={row['cos_xyz']:.3f}"
        )

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n[summary]")
    summarize_group("all", rows, np.ones(len(rows), dtype=bool))

    labels = np.asarray([[r["label_x"], r["label_y"], r["label_z"], r["label_gripper"]] for r in rows])
    groups = {
        "label_g_neg": labels[:, 3] < -0.5,
        "label_g_pos": labels[:, 3] > 0.5,
        "label_z_pos": labels[:, 2] > 0.01,
        "label_z_neg": labels[:, 2] < -0.01,
        "large_xy": np.linalg.norm(labels[:, :2], axis=1) > 0.05,
    }
    print("\n[groups]")
    for name, mask in groups.items():
        summarize_group(name, rows, mask)
    print("saved_csv:", out_path)


if __name__ == "__main__":
    main()
