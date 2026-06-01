#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    print("\nRunning:")
    print(" \\\n  ".join(shlex.quote(x) for x in cmd))
    print("")
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LIBERO tree to one LeRobot dataset root, then start global-step fine-tuning."
    )
    parser.add_argument(
        "--src-root",
        default="/root/autodl-tmp/openpi_earbud_proto/libero_lerobot_dataset",
    )
    parser.add_argument(
        "--dataset-root",
        default="/root/autodl-tmp/openpi_earbud_proto/lerobot_ready/earbud_insert_batch_v3_global_single",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default="local/earbud_insert_batch_v3_global_single",
    )
    parser.add_argument(
        "--model-path",
        default="/root/autodl-tmp/hf_models/pi0_libero_base",
    )
    parser.add_argument(
        "--output-root",
        default="/root/autodl-tmp/openpi_earbud_proto/outputs",
    )
    parser.add_argument("--categories", nargs="+", default=["easy", "medium", "hard"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--save-freq", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--job-name", default="pi0_earbud_global_ft")
    parser.add_argument("--skip-invalid", action="store_true")
    parser.add_argument("--include-aux-camera", action="store_true")
    parser.add_argument("--force-overwrite", action="store_true")
    parser.add_argument(
        "--script-dir",
        default=str(Path(__file__).resolve().parent),
        help="Directory containing convert_libero_tree_to_single_lerobot.py and finetune_pi05_global_mixed.py",
    )
    args = parser.parse_args()

    script_dir = Path(args.script_dir).resolve()
    convert_py = script_dir / "convert_libero_tree_to_single_lerobot.py"
    finetune_py = script_dir / "finetune_pi05_global_mixed.py"

    if not convert_py.exists():
        raise FileNotFoundError(f"Missing converter script: {convert_py}")
    if not finetune_py.exists():
        raise FileNotFoundError(f"Missing finetune script: {finetune_py}")

    convert_cmd = [
        sys.executable,
        str(convert_py),
        "--src-root",
        args.src_root,
        "--output-root",
        args.dataset_root,
        "--repo-id",
        args.dataset_repo_id,
        "--categories",
        *args.categories,
        "--workers",
        str(args.workers),
        "--prefetch",
        str(args.prefetch),
    ]
    if args.skip_invalid:
        convert_cmd.append("--skip-invalid")
    if args.include_aux_camera:
        convert_cmd.append("--include-aux-camera")
    if args.force_overwrite:
        convert_cmd.append("--force-overwrite")

    finetune_cmd = [
        sys.executable,
        str(finetune_py),
        "--dataset-path",
        args.dataset_root,
        "--model-path",
        args.model_path,
        "--output-root",
        args.output_root,
        "--dataset-repo-id",
        args.dataset_repo_id,
        "--steps",
        str(args.steps),
        "--save-freq",
        str(args.save_freq),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--job-name",
        args.job_name,
    ]

    print("Step 1/2: convert dataset")
    run_cmd(convert_cmd)
    print("Step 2/2: start fine-tuning")
    run_cmd(finetune_cmd)
    print("\nAll done.")


if __name__ == "__main__":
    main()
