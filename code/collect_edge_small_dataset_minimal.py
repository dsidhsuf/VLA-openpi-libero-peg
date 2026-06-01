import os
import re
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path("/root/autodl-tmp/openpi_earbud_proto")
TEACHER = ROOT / "full_chain_pick_random_wrist_align_descend_release.py"

SUCCESS_RE = re.compile(r"final release_drop_success=(True|False)")
FLOAT_RE = {
    "earbud_z_initial": re.compile(r"earbud_z_initial=([-+0-9.eE]+)"),
    "earbud_z_final": re.compile(r"earbud_z_final=([-+0-9.eE]+)"),
    "z_lift_vs_initial": re.compile(r"z_lift_vs_initial=([-+0-9.eE]+)"),
    "eef_obj_dist": re.compile(r"eef_obj_dist=([-+0-9.eE]+)"),
    "obj_slot_xy": re.compile(r"obj_slot_xy=([-+0-9.eE]+)"),
    "obj_slot_z": re.compile(r"obj_slot_z=([-+0-9.eE]+)"),
    "yaw_err_final_deg": re.compile(r"yaw_err_final_deg=([-+0-9.eE]+)"),
}

SAVED_RE = re.compile(r"saved:\s+(.*)")

def parse_output(text: str):
    meta = {}
    m = SUCCESS_RE.search(text)
    if not m:
        meta["release_drop_success"] = False
    else:
        meta["release_drop_success"] = (m.group(1) == "True")

    for k, pat in FLOAT_RE.items():
        m = pat.search(text)
        if m:
            meta[k] = float(m.group(1))

    saved_paths = SAVED_RE.findall(text)
    return meta, saved_paths

def run_one(level: str, seed: int, yaw_min: float, yaw_max: float, out_root: Path):
    cmd = [
        "python",
        str(TEACHER),
        "--level", level,
        "--seed", str(seed),
        "--random-yaw-min-deg", str(yaw_min),
        "--random-yaw-max-deg", str(yaw_max),
        "--flat-rest-prob", "0.0",   # edge-only
    ]

    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    stdout = result.stdout
    meta, saved_paths = parse_output(stdout)

    meta["seed"] = seed
    meta["level"] = level
    meta["yaw_min"] = yaw_min
    meta["yaw_max"] = yaw_max
    meta["rest_pose_mode"] = "edge"
    meta["returncode"] = result.returncode

    ep_dir = out_root / f"ep_seed{seed:04d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    # 保存原始日志
    (ep_dir / "run.log").write_text(stdout, encoding="utf-8")
    (ep_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # 复制教师脚本保存出来的图片 / 视频
    copied = []
    for p in saved_paths:
        src = Path(p.strip())
        if src.exists():
            dst = ep_dir / src.name
            shutil.copy2(src, dst)
            copied.append(str(dst))

    meta["copied_artifacts"] = copied
    (ep_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return meta["release_drop_success"], meta, ep_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--num_episodes", type=int, default=6)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--yaw_min", type=float, default=-30.0)
    parser.add_argument("--yaw_max", type=float, default=30.0)
    parser.add_argument("--out_root", type=str, default="/root/autodl-tmp/openpi_earbud_proto/small_edge_dataset_minimal")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) / f"{args.level}_edge_yaw_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    success_count = 0
    seed = args.seed_start

    while success_count < args.num_episodes:
        ok, meta, ep_dir = run_one(
            level=args.level,
            seed=seed,
            yaw_min=args.yaw_min,
            yaw_max=args.yaw_max,
            out_root=out_root,
        )

        summary.append({**meta, "episode_dir": str(ep_dir)})
        print(
            f"[seed={seed}] "
            f"success={ok} "
            f"xy={meta.get('obj_slot_xy')} "
            f"z={meta.get('obj_slot_z')} "
            f"yaw_err={meta.get('yaw_err_final_deg')}"
        )

        if ok:
            success_count += 1
        seed += 1

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("saved dataset root:", out_root)
    print("successful episodes:", success_count)

if __name__ == "__main__":
    main()
