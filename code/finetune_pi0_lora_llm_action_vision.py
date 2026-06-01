#!/usr/bin/env python3
"""
LoRA fine-tuning launcher for PI0/PI0-LIBERO LeRobot checkpoints.

Default behavior:
  - trains on one already-merged LeRobot dataset root
  - enables LoRA over the LLM/VLM side, PI0 action expert, and vision encoder
  - keeps the base checkpoint frozen through PEFT while training LoRA adapters
  - keeps n_action_steps=50 for chunk execution during evaluation
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ANSI_ESCAPE_RE = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")


# Broad but still targeted regex for PI0/Paligemma-style modules across
# LeRobot versions. It is meant to catch:
#   1) LLM/VLM transformer linear layers: q/k/v/o + MLP projections
#   2) action expert transformer linear layers
#   3) vision encoder attention/MLP/projection linear layers
#   4) PI0 action/state projection heads
#
# If your local LeRobot build uses different module names, run with
# --dry-run first, inspect the printed command, then override
# --lora-target-modules with a narrower regex.
DEFAULT_LORA_TARGET_MODULES = (
    r"(.*(?:paligemma|gemma|language_model|text_model|llm|vlm|"
    r"gemma_expert|action_expert|expert|vision_tower|vision_model|"
    r"vision_encoder|image_encoder|siglip|Siglip).*(?:q_proj|k_proj|v_proj|"
    r"o_proj|out_proj|gate_proj|up_proj|down_proj|fc1|fc2|proj|linear)"
    r"|model\.(?:state_proj|action_in_proj|action_out_proj|"
    r"action_time_mlp_in|action_time_mlp_out))"
)


def is_lerobot_dataset_root(path: Path) -> bool:
    return (path / "meta" / "info.json").exists() and (path / "data").exists()


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    idx = 1
    while True:
        candidate = path.parent / f"{path.name}_r{idx}"
        if not candidate.exists():
            return candidate
        idx += 1


def resolve_dataset_root(dataset_path: Path) -> Path:
    if is_lerobot_dataset_root(dataset_path):
        return dataset_path.resolve()
    roots = []
    for child in sorted(dataset_path.iterdir()):
        if child.is_dir() and is_lerobot_dataset_root(child):
            roots.append(child.resolve())
    if len(roots) == 1:
        return roots[0]
    if not roots:
        raise FileNotFoundError(
            f"No LeRobot dataset root found at {dataset_path}. "
            "Expected a folder containing meta/info.json and data/."
        )
    raise RuntimeError(
        f"Found {len(roots)} dataset roots under {dataset_path}; "
        "please pass --dataset-path to one merged LeRobot dataset root."
    )


def make_output_dirs(output_root: Path, job_name: str) -> tuple[str, Path, Path]:
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    output_dir = unique_path(output_root / f"{job_name}_{run_tag}")
    logs_dir = unique_path(output_root / "_run_logs" / f"{job_name}_{run_tag}")
    logs_dir.mkdir(parents=True, exist_ok=True)
    return run_tag, output_dir, logs_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="One merged LeRobot dataset root containing meta/info.json and data/.",
    )
    parser.add_argument(
        "--model-path",
        default="/root/autodl-tmp/hf_models/pi0_libero_base",
        help="Base PI0 checkpoint path.",
    )
    parser.add_argument(
        "--output-root",
        default="/root/autodl-tmp/openpi_earbud_proto/outputs_lora",
    )
    parser.add_argument("--dataset-repo-id", default="local/earbud_lora")
    parser.add_argument("--job-name", default="pi0_earbud_lora")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--save-freq", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-target-modules", default=DEFAULT_LORA_TARGET_MODULES)
    parser.add_argument("--n-action-steps", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--optimizer-lr", type=float, default=1e-5)
    parser.add_argument("--scheduler-warmup-steps", type=int, default=1000)
    parser.add_argument("--scheduler-decay-steps", type=int, default=30000)
    parser.add_argument("--scheduler-decay-lr", type=float, default=2.5e-6)
    parser.add_argument("--freeze-vision-encoder", choices=["true", "false"], default="true")
    parser.add_argument("--train-expert-only", choices=["true", "false"], default="true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the lerobot-train command and exit without launching training.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).resolve()
    model_path = Path(args.model_path).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    dataset_root = resolve_dataset_root(dataset_path)
    run_tag, output_dir, logs_dir = make_output_dirs(output_root, args.job_name)

    launcher = (
        ["lerobot-train"]
        if shutil.which("lerobot-train")
        else [sys.executable, "-m", "lerobot.scripts.lerobot_train"]
    )

    cmd = launcher + [
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--dataset.root={dataset_root}",
        "--dataset.revision=v3.0",
        f"--policy.path={model_path}",
        "--policy.device=cuda",
        "--policy.dtype=bfloat16",
        "--policy.gradient_checkpointing=true",
        "--policy.compile_model=false",
        f"--policy.train_expert_only={args.train_expert_only}",
        f"--policy.freeze_vision_encoder={args.freeze_vision_encoder}",
        f"--policy.chunk_size={args.chunk_size}",
        f"--policy.n_action_steps={args.n_action_steps}",
        f"--policy.optimizer_lr={args.optimizer_lr}",
        f"--policy.scheduler_warmup_steps={args.scheduler_warmup_steps}",
        f"--policy.scheduler_decay_steps={args.scheduler_decay_steps}",
        f"--policy.scheduler_decay_lr={args.scheduler_decay_lr}",
        '--policy.normalization_mapping={"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}',
        "--peft.method_type=LORA",
        f"--peft.r={args.lora_rank}",
        f"--peft.target_modules={args.lora_target_modules}",
        f"--output_dir={output_dir}",
        f"--job_name={args.job_name}_{run_tag}",
        "--policy.push_to_hub=false",
        "--wandb.enable=false",
        "--eval_freq=0",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--steps={args.steps}",
        f"--log_freq={args.log_freq}",
        f"--save_freq={args.save_freq}",
        f"--seed={args.seed}",
        "--resume=false",
    ]

    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    print("========== LoRA Training Command ==========")
    print(" \\\n  ".join(shlex.quote(x) for x in cmd))
    print("")
    print(f"dataset_root: {dataset_root}")
    print(f"base_model: {model_path}")
    print(f"output_dir: {output_dir}")
    print(f"logs_dir: {logs_dir}")
    print(f"lora_rank: {args.lora_rank}")
    print("lora_scope: LLM/VLM + action_expert + vision_encoder")
    print(f"train_expert_only: {args.train_expert_only}")
    print(f"freeze_vision_encoder: {args.freeze_vision_encoder}")
    print(f"lora_target_modules: {args.lora_target_modules}")
    print("")

    if args.dry_run:
        print("dry_run=true, skip training.")
        return

    raw_log = logs_dir / "train_stdout.log"
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    assert process.stdout is not None
    with raw_log.open("w", encoding="utf-8") as f:
        for line in process.stdout:
            print(line, end="")
            clean = ANSI_ESCAPE_RE.sub("", line.rstrip("\n"))
            f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {clean}\n")
            f.flush()

    return_code = process.wait()

    summary = {
        "return_code": return_code,
        "run_tag": run_tag,
        "dataset_root": str(dataset_root),
        "base_model": str(model_path),
        "output_dir": str(output_dir),
        "logs_dir": str(logs_dir),
        "raw_log": str(raw_log),
        "command": cmd,
        "args": vars(args),
    }
    summary_path = logs_dir / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n========== Run Summary ==========")
    print(f"return_code: {return_code}")
    print(f"output_dir: {output_dir}")
    print(f"raw_log: {raw_log}")
    print(f"summary_json: {summary_path}")
    sys.exit(return_code)


if __name__ == "__main__":
    main()
