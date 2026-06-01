#!/usr/bin/env python3
import argparse
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

DATASET_PATH_DEFAULT = "/root/autodl-tmp/openpi_earbud_proto/lerobot_ready/earbud_insert_single_v3"
MODEL_PATH_DEFAULT = "/root/autodl-tmp/hf_models/pi05_libero_finetuned_v044"
OUTPUT_ROOT_DEFAULT = "/root/autodl-tmp/openpi_earbud_proto/outputs"

def pick_dtype(user_dtype: str) -> str:
    if user_dtype != "auto":
        return user_dtype
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return "bfloat16"
        return "float16"
    except Exception:
        return "float16"

def unique_path(p: Path) -> Path:
    if not p.exists():
        return p
    i = 1
    while True:
        cand = p.parent / f"{p.name}_r{i}"
        if not cand.exists():
            return cand
        i += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default=DATASET_PATH_DEFAULT)
    parser.add_argument("--model-path", default=MODEL_PATH_DEFAULT)
    parser.add_argument("--output-root", default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--job-name", default="pi05_earbud_single_try")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--save-freq", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).resolve()
    model_path = Path(args.model_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    dataset_repo_id = dataset_path.name
    dataset_root = str(dataset_path.parent)

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = Path(args.output_root).resolve() / f"{args.job_name}_{run_tag}"

    if not args.resume:
        output_dir = unique_path(output_dir)

    dtype = pick_dtype(args.dtype)

    if shutil.which("lerobot-train"):
        launcher = ["lerobot-train"]
    else:
        launcher = [sys.executable, "-m", "lerobot.scripts.lerobot_train"]

    cmd = launcher + [
        f"--dataset.repo_id={dataset_repo_id}",
        f"--dataset.root={dataset_root}",
        f"--policy.path={model_path}",
        f"--output_dir={output_dir}",
        f"--job_name={args.job_name}_{run_tag}",
        f"--policy.device={args.device}",
        "--policy.push_to_hub=false",
        "--wandb.enable=false",
        "--policy.gradient_checkpointing=true",
        "--policy.compile_model=false",
        f"--policy.dtype={dtype}",
        '--policy.normalization_mapping={"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}',
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--steps={args.steps}",
        f"--log_freq={args.log_freq}",
        f"--save_freq={args.save_freq}",
        "--eval_freq=0",
        f"--seed={args.seed}",
        f"--resume={'true' if args.resume else 'false'}",
    ]

    print("Running command:")
    print(" \\\n  ".join(shlex.quote(x) for x in cmd))
    print(f"\nOutput dir: {output_dir}")

    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
