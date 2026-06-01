#!/usr/bin/env python3
"""
LoRA fine-tuning launcher for PI0/PI0-LIBERO LeRobot checkpoints.

Default behavior:
  - trains on one already-merged LeRobot dataset root
  - freezes the vision/VLM side through PI0's train_expert_only path
  - adds LoRA to the PI0 action expert q/v projections and action/state projections
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


DEFAULT_LORA_TARGET_MODULES = (
    r"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|"
    r"model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))"
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


def assert_base_model_for_fresh_lora(model_path: Path) -> None:
    """Make sure policy.path points to a plain base model, not an adapter model."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        return

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if not bool(config.get("use_peft", False)):
        return

    raise RuntimeError(
        "Base model config has use_peft=true, but this LoRA launcher needs a plain "
        "base model config with use_peft=false. Create a full copy of the base "
        "model, patch config.json use_peft to false, then pass that copy with "
        "--model-path. See the command printed in the assistant message."
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
    parser.add_argument("--optimizer-lr", type=float, default=2.5e-5)
    parser.add_argument("--scheduler-warmup-steps", type=int, default=1000)
    parser.add_argument("--scheduler-decay-steps", type=int, default=30000)
    parser.add_argument("--scheduler-decay-lr", type=float, default=2.5e-6)
    parser.add_argument("--freeze-vision-encoder", choices=["true", "false"], default="true")
    parser.add_argument("--train-expert-only", choices=["true", "false"], default="true")
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
    assert_base_model_for_fresh_lora(model_path)
    train_model_path = model_path

    launcher = (
        ["lerobot-train"]
        if shutil.which("lerobot-train")
        else [sys.executable, "-m", "lerobot.scripts.lerobot_train"]
    )

    cmd = launcher + [
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--dataset.root={dataset_root}",
        "--dataset.revision=v3.0",
        f"--policy.path={train_model_path}",
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
    print(f"lora_target_modules: {args.lora_target_modules}")
    print("")

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
        "train_model_path": str(train_model_path),
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
