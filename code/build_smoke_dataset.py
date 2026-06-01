import os
import re
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path("/root/autodl-tmp/openpi_earbud_proto")
RECORDER = ROOT / "full_chain_pick_random_wrist_align_descend_release_record.py"

SUCCESS_RE = re.compile(r"final release_drop_success=(True|False)")
EPISODE_DIR_RE = re.compile(r"saved_episode_dir:\s+(.*)")

def run_one(level, seed, yaw_min, yaw_max, flat_rest_prob, record_root):
    cmd = [
        "python",
        str(RECORDER),
        "--level", level,
        "--seed", str(seed),
        "--random-yaw-min-deg", str(yaw_min),
        "--random-yaw-max-deg", str(yaw_max),
        "--flat-rest-prob", str(flat_rest_prob),
        "--record-root", str(record_root),
    ]

    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    text = result.stdout
    m = SUCCESS_RE.search(text)
    success = bool(m and m.group(1) == "True")

    ep_dir = None
    m2 = EPISODE_DIR_RE.search(text)
    if m2:
        ep_dir = m2.group(1).strip()

    return success, ep_dir, text

def collect_split(split_name, level, num_success, seed_start, yaw_min, yaw_max, flat_rest_prob, split_root):
    split_root.mkdir(parents=True, exist_ok=True)

    summary = []
    success_count = 0
    seed = seed_start

    while success_count < num_success:
        ok, ep_dir, text = run_one(
            level=level,
            seed=seed,
            yaw_min=yaw_min,
            yaw_max=yaw_max,
            flat_rest_prob=flat_rest_prob,
            record_root=split_root,
        )

        row = {
            "seed": seed,
            "success": ok,
            "episode_dir": ep_dir,
        }
        summary.append(row)

        print(f"[{split_name}] seed={seed} success={ok} episode_dir={ep_dir}")

        seed += 1
        if ok:
            success_count += 1

    with open(split_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--train_num", type=int, default=24)
    parser.add_argument("--val_num", type=int, default=6)
    parser.add_argument("--yaw_min", type=float, default=-30.0)
    parser.add_argument("--yaw_max", type=float, default=30.0)
    parser.add_argument("--out_root", type=str, default="/root/autodl-tmp/openpi_earbud_proto/smoke_trainable_dataset")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.out_root) / f"{args.level}_edge_yaw_smoke_{ts}"
    train_root = root / "train"
    val_root = root / "val"

    collect_split(
        split_name="train",
        level=args.level,
        num_success=args.train_num,
        seed_start=0,
        yaw_min=args.yaw_min,
        yaw_max=args.yaw_max,
        flat_rest_prob=0.0,
        split_root=train_root,
    )

    collect_split(
        split_name="val",
        level=args.level,
        num_success=args.val_num,
        seed_start=10000,
        yaw_min=args.yaw_min,
        yaw_max=args.yaw_max,
        flat_rest_prob=0.0,
        split_root=val_root,
    )

    with open(root / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump({
            "level": args.level,
            "rest_pose_mode": "edge_only",
            "yaw_min": args.yaw_min,
            "yaw_max": args.yaw_max,
            "train_num": args.train_num,
            "val_num": args.val_num,
            "format": "npz",
            "keys": ["images_agentview", "state", "action", "phase"],
        }, f, indent=2, ensure_ascii=False)

    print("saved smoke dataset root:", root)

if __name__ == "__main__":
    main()
