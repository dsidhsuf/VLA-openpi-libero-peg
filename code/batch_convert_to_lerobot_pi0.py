#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


SUCCESS_KEYS = ("release_drop_success", "success", "is_success", "final_success")


def is_success_episode(ep_dir: Path) -> bool:
    meta_path = ep_dir / "metadata.json"
    if not meta_path.exists():
        return True
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return True

    for k in SUCCESS_KEYS:
        if k in meta:
            return bool(meta[k])

    if "obj_slot_xy" in meta and "obj_slot_z" in meta:
        try:
            return (float(meta["obj_slot_xy"]) < 0.02) and (float(meta["obj_slot_z"]) < 0.03)
        except Exception:
            return False

    return True


def collect_episodes(source_root: Path):
    items = []
    for level in ("easy", "medium", "hard"):
        level_dir = source_root / level
        if not level_dir.exists():
            continue
        for ep in sorted(level_dir.glob("episode_*")):
            if ep.is_dir():
                items.append((level, ep))
    return items


def sanitize_repo_id(x: str) -> str:
    x = re.sub(r"[^0-9A-Za-z._/-]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x


def build_cmd(job: dict):
    cmd = [
        job["python_bin"],
        str(job["converter_script"]),
        "--episode-dir", str(job["episode_dir"]),
        "--output-root", str(job["episode_out"]),
        "--repo-id", job["repo_id"],
        "--task", job["task"],
        "--camera-primary", job["camera_primary"],
        "--camera-secondary", job["camera_secondary"],
        "--image-size", str(job["image_size"]),
    ]
    if job["include_aux_camera"]:
        cmd += ["--include-aux-camera", "--camera-aux", job["camera_aux"]]
    if job["overwrite_existing"]:
        cmd += ["--force-overwrite"]
    return cmd


def run_one(job: dict):
    ep = job["episode_dir"]
    out = job["episode_out"]
    if out.exists() and not job["overwrite_existing"]:
        return {
            "episode": str(ep),
            "output": str(out),
            "status": "skipped_exists",
            "returncode": 0,
            "cmd": build_cmd(job),
            "stderr_tail": "",
        }

    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(job)
    p = subprocess.run(cmd, text=True, capture_output=True)
    ok = p.returncode == 0
    return {
        "episode": str(ep),
        "output": str(out),
        "status": "ok" if ok else "failed",
        "returncode": p.returncode,
        "cmd": cmd,
        "stderr_tail": (p.stderr or "")[-2000:],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--converter-script", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id-prefix", type=str, default="local/earbud_insert_batch")
    parser.add_argument("--task", type=str, default="Insert the earbud into the charging slot.")
    parser.add_argument("--camera-primary", type=str, default="agentview")
    parser.add_argument("--camera-secondary", type=str, default="robot0_eye_in_hand")
    parser.add_argument("--camera-aux", type=str, default="frontview")
    parser.add_argument("--include-aux-camera", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-failures", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    converter = args.converter_script.resolve()

    if not converter.exists():
        raise FileNotFoundError(f"converter script not found: {converter}")
    if not source_root.exists():
        raise FileNotFoundError(f"source root not found: {source_root}")

    all_eps = collect_episodes(source_root)
    success_eps = []
    fail_eps = []
    for level, ep in all_eps:
        if is_success_episode(ep):
            success_eps.append((level, ep))
        else:
            fail_eps.append((level, ep))

    selected = list(success_eps)
    if args.max_failures >= 0:
        selected += fail_eps[: args.max_failures]
    else:
        selected += fail_eps

    jobs = []
    for level, ep in selected:
        episode_out = output_root / level / ep.name
        repo_id = sanitize_repo_id(f"{args.repo_id_prefix}_{level}_{ep.name}")
        jobs.append({
            "python_bin": args.python_bin,
            "converter_script": converter,
            "episode_dir": ep,
            "episode_out": episode_out,
            "repo_id": repo_id,
            "task": args.task,
            "camera_primary": args.camera_primary,
            "camera_secondary": args.camera_secondary,
            "camera_aux": args.camera_aux,
            "include_aux_camera": bool(args.include_aux_camera),
            "image_size": int(args.image_size),
            "overwrite_existing": bool(args.overwrite_existing),
        })

    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[collect] total={len(all_eps)} success={len(success_eps)} fail={len(fail_eps)}")
    print(f"[select] selected={len(jobs)} max_failures={args.max_failures}")
    print(f"[run] workers={args.num_workers} image_size={args.image_size}")

    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
        fut_map = {ex.submit(run_one, j): j for j in jobs}
        done = 0
        for fut in as_completed(fut_map):
            done += 1
            r = fut.result()
            results.append(r)
            print(f"[{done}/{len(jobs)}] {r['status']} {r['episode']}")

    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped_exists"]
    failed = [r for r in results if r["status"] == "failed"]

    manifest = {
        "created_at": datetime.now().isoformat(),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "converter_script": str(converter),
        "image_size": int(args.image_size),
        "num_workers": int(args.num_workers),
        "selected_total": len(jobs),
        "ok": len(ok),
        "skipped_exists": len(skipped),
        "failed": len(failed),
        "results": results,
    }
    manifest_path = output_root / "batch_conversion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] ok={len(ok)} skipped={len(skipped)} failed={len(failed)}")
    print(f"[saved] {manifest_path}")
    if failed:
        print("[hint] some episodes failed; check manifest stderr_tail for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
