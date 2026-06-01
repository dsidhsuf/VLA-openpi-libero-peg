#!/usr/bin/env python3
"""
Action-expert-only LoRA fine-tuning launcher for PI0/PI0-LIBERO.

This script intentionally freezes the vision encoder and the main VLM side, and
only attaches LoRA adapters to PI0's action expert path:
  - gemma_expert self-attention q/v projections
  - action/state projection layers around the flow/action expert

Expected dataset format:
  A single merged LeRobot dataset root containing:
    meta/info.json
    data/
"""

from __future__ import annotations

import argparse
import inspect
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

# Only target the action expert side. This avoids LoRA on the vision encoder or
# the main PaliGemma/VLM trunk.
DEFAULT_LORA_TARGET_MODULES = (
    r"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|"
    r"model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))"
)

LEROBOT_FORCE_PROCESSOR_MARKER = "LEROBOT_FORCE_BUILD_PROCESSORS"
PI0_LOCAL_TOKENIZER_ENV = "PI0_LOCAL_TOKENIZER_PATH"


def is_lerobot_dataset_root(path: Path) -> bool:
    return (path / "meta" / "info.json").exists() and (path / "data").exists()


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


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    idx = 1
    while True:
        candidate = path.parent / f"{path.name}_r{idx}"
        if not candidate.exists():
            return candidate
        idx += 1


def prepare_plain_base_model(model_path: Path, output_root: Path) -> Path:
    """Return a base-model path that LeRobot will not mistake for a PEFT adapter."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        return model_path

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if not bool(config.get("use_peft", False)):
        return model_path

    patched_root = output_root / "_patched_base_models"
    patched_root.mkdir(parents=True, exist_ok=True)
    patched_path = patched_root / f"{model_path.name}_use_peft_false"

    if not patched_path.exists():
        print(
            "[compat] Base model config has use_peft=true, so LeRobot may treat it "
            "as a LoRA adapter. Creating a patched training copy with use_peft=false."
        )
        # Dereference symlinks. Some LeRobot processor files live behind HF-cache
        # symlinks; preserving them can create a patched copy that looks valid
        # but later falls back to hf_hub_download with a local absolute path.
        shutil.copytree(model_path, patched_path, symlinks=False)

    patched_config_path = patched_path / "config.json"
    with patched_config_path.open("r", encoding="utf-8") as f:
        patched_config = json.load(f)
    patched_config["use_peft"] = False
    with patched_config_path.open("w", encoding="utf-8") as f:
        json.dump(patched_config, f, indent=2, ensure_ascii=False)

    return patched_path


def patch_lerobot_train_for_fresh_processors() -> Path | None:
    """Patch LeRobot so old local base checkpoints do not load broken processors.

    Some locally mirrored PI0 checkpoints have policy_preprocessor.json but miss
    one or more referenced processor state files. LeRobot then falls back to
    hf_hub_download(repo_id=<local absolute path>), which raises HFValidationError.

    The patch is guarded by LEROBOT_FORCE_BUILD_PROCESSORS=1 and only changes
    behavior when that environment variable is set.
    """
    patched_paths = []

    # Patch the factory directly. This is the most stable place across train
    # script variants because all training paths eventually call this function.
    try:
        import lerobot.policies.factory as factory
    except Exception as exc:
        print(f"[compat] Could not import lerobot.policies.factory for processor patch: {exc}")
    else:
        factory_py = Path(inspect.getfile(factory)).resolve()
        text = factory_py.read_text(encoding="utf-8")
        if LEROBOT_FORCE_PROCESSOR_MARKER not in text:
            needle = "    if pretrained_path:\n"
            replacement = (
                f"    if pretrained_path and __import__('os').environ.get('{LEROBOT_FORCE_PROCESSOR_MARKER}') == '1':\n"
                "        logging.warning(\n"
                "            'LEROBOT_FORCE_BUILD_PROCESSORS=1: building policy processors from current '\n"
                "            'dataset stats instead of loading pretrained checkpoint processors.'\n"
                "        )\n"
                "        pretrained_path = None\n"
                "\n"
                "    if pretrained_path:\n"
            )
            if needle in text:
                backup = factory_py.with_suffix(factory_py.suffix + ".bak_force_build_processors")
                if not backup.exists():
                    shutil.copy2(factory_py, backup)
                factory_py.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
                print(f"[compat] Patched LeRobot processor factory: {factory_py}")
                print(f"[compat] Backup saved at: {backup}")
            else:
                print(f"[compat] Could not patch processor factory; pattern not found in {factory_py}")
        patched_paths.append(factory_py)

    # Also try the train-script patch for versions that route through an
    # intermediate processor_pretrained_path variable.
    try:
        import lerobot.scripts.lerobot_train as lerobot_train
    except Exception as exc:
        print(f"[compat] Could not import lerobot_train for processor patch: {exc}")
        return patched_paths[0] if patched_paths else None

    train_py = Path(inspect.getfile(lerobot_train)).resolve()
    text = train_py.read_text(encoding="utf-8")
    if LEROBOT_FORCE_PROCESSOR_MARKER in text:
        patched_paths.append(train_py)
        return patched_paths[0] if patched_paths else train_py

    needle = "        processor_pretrained_path = active_cfg.pretrained_path\n"
    replacement = (
        "        processor_pretrained_path = active_cfg.pretrained_path\n"
        f"        if __import__('os').environ.get('{LEROBOT_FORCE_PROCESSOR_MARKER}') == '1':\n"
        "            logging.warning(\n"
        "                'LEROBOT_FORCE_BUILD_PROCESSORS=1: building policy processors from current '\n"
        "                'dataset stats instead of loading pretrained checkpoint processors.'\n"
        "            )\n"
        "            processor_pretrained_path = None\n"
    )
    if needle not in text:
        print(f"[compat] Could not patch processor loading; pattern not found in {train_py}")
        patched_paths.append(train_py)
        return patched_paths[0] if patched_paths else train_py

    backup = train_py.with_suffix(train_py.suffix + ".bak_force_build_processors")
    if not backup.exists():
        shutil.copy2(train_py, backup)
    train_py.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
    print(f"[compat] Patched LeRobot processor loading: {train_py}")
    print(f"[compat] Backup saved at: {backup}")
    patched_paths.append(train_py)
    return patched_paths[0] if patched_paths else train_py


def patch_pi0_processor_for_local_tokenizer() -> Path | None:
    """Patch PI0 processor to allow a local PaliGemma tokenizer path."""
    try:
        import lerobot.policies.pi0.processor_pi0 as processor_pi0
    except Exception as exc:
        print(f"[compat] Could not import pi0 processor for tokenizer patch: {exc}")
        return None

    processor_py = Path(inspect.getfile(processor_pi0)).resolve()
    text = processor_py.read_text(encoding="utf-8")
    if PI0_LOCAL_TOKENIZER_ENV in text:
        return processor_py

    if "import os" not in text:
        text = text.replace("from typing import Any\n", "from typing import Any\nimport os\n", 1)

    needle = 'tokenizer_name="google/paligemma-3b-pt-224"'
    replacement = f'tokenizer_name=os.environ.get("{PI0_LOCAL_TOKENIZER_ENV}", "google/paligemma-3b-pt-224")'
    if needle not in text:
        print(f"[compat] Could not patch local tokenizer; pattern not found in {processor_py}")
        return processor_py

    backup = processor_py.with_suffix(processor_py.suffix + ".bak_local_tokenizer")
    if not backup.exists():
        shutil.copy2(processor_py, backup)
    processor_py.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
    print(f"[compat] Patched PI0 processor tokenizer path: {processor_py}")
    print(f"[compat] Backup saved at: {backup}")
    return processor_py


def make_output_dirs(output_root: Path, job_name: str) -> tuple[str, Path, Path]:
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    output_dir = unique_path(output_root / f"{job_name}_{run_tag}")
    logs_dir = unique_path(output_root / "_run_logs" / f"{job_name}_{run_tag}")
    logs_dir.mkdir(parents=True, exist_ok=True)
    return run_tag, output_dir, logs_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        default="/root/autodl-tmp/openpi_earbud_proto/lerobot_ready/earbud_better_data",
        help="One merged LeRobot dataset root containing meta/info.json and data/.",
    )
    parser.add_argument(
        "--model-path",
        default="/root/autodl-tmp/hf_models/pi0_libero_base",
        help="Base PI0 checkpoint path.",
    )
    parser.add_argument(
        "--output-root",
        default="/root/autodl-tmp/openpi_earbud_proto/outputs_lora_action_expert",
    )
    parser.add_argument("--dataset-repo-id", default="local/earbud_better_data")
    parser.add_argument("--job-name", default="pi0_lora_action_expert_better_data")

    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-freq", type=int, default=10)

    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-target-modules", default=DEFAULT_LORA_TARGET_MODULES)

    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=50)
    parser.add_argument("--optimizer-lr", type=float, default=2.5e-5)
    parser.add_argument("--scheduler-warmup-steps", type=int, default=500)
    parser.add_argument("--scheduler-decay-steps", type=int, default=20000)
    parser.add_argument("--scheduler-decay-lr", type=float, default=2.5e-6)
    parser.add_argument(
        "--tokenizer-path",
        default="/root/autodl-tmp/cache/huggingface/google/paligemma-3b-pt-224",
        help=(
            "Local PaliGemma tokenizer path. Required when force-building "
            "processors in offline mode."
        ),
    )
    parser.add_argument(
        "--offline",
        choices=["true", "false"],
        default="true",
        help="Set HuggingFace/Transformers offline environment variables.",
    )
    parser.add_argument(
        "--force-build-processors",
        choices=["true", "false"],
        default="true",
        help=(
            "Build policy pre/postprocessors from current dataset stats instead "
            "of loading checkpoint processor state. This avoids old local PI0 "
            "checkpoint processor path issues."
        ),
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
    train_model_path = prepare_plain_base_model(model_path, output_root)
    patched_train_py = None
    patched_tokenizer_py = None
    if args.force_build_processors == "true":
        patched_train_py = patch_lerobot_train_for_fresh_processors()
        patched_tokenizer_py = patch_pi0_processor_for_local_tokenizer()
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
        f"--policy.path={train_model_path}",
        "--policy.device=cuda",
        "--policy.dtype=bfloat16",
        "--policy.gradient_checkpointing=true",
        "--policy.compile_model=false",
        "--policy.train_expert_only=true",
        "--policy.freeze_vision_encoder=true",
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
    if args.offline == "true":
        env["HF_HUB_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    else:
        env.pop("HF_HUB_OFFLINE", None)
        env.pop("HF_DATASETS_OFFLINE", None)
        env.pop("TRANSFORMERS_OFFLINE", None)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.force_build_processors == "true":
        env[LEROBOT_FORCE_PROCESSOR_MARKER] = "1"
    if args.tokenizer_path:
        tokenizer_path = Path(args.tokenizer_path).expanduser()
        if tokenizer_path.exists():
            env[PI0_LOCAL_TOKENIZER_ENV] = str(tokenizer_path.resolve())
        else:
            print(f"[warn] tokenizer path does not exist, will fall back to HF name: {tokenizer_path}")

    print("========== Action-Expert LoRA Training Command ==========")
    print(" \\\n  ".join(shlex.quote(x) for x in cmd))
    print("")
    print(f"dataset_root: {dataset_root}")
    print(f"base_model: {model_path}")
    print(f"train_model_path: {train_model_path}")
    print(f"output_dir: {output_dir}")
    print(f"logs_dir: {logs_dir}")
    print(f"lora_rank: {args.lora_rank}")
    print(f"lora_target_modules: {args.lora_target_modules}")
    print("train_scope: action_expert_only")
    print(f"force_build_processors: {args.force_build_processors}")
    print(f"offline: {args.offline}")
    print(f"tokenizer_path: {env.get(PI0_LOCAL_TOKENIZER_ENV, 'google/paligemma-3b-pt-224')}")
    if patched_train_py is not None:
        print(f"patched_lerobot_train: {patched_train_py}")
    if patched_tokenizer_py is not None:
        print(f"patched_pi0_processor: {patched_tokenizer_py}")
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
        "train_scope": "action_expert_only",
        "force_build_processors": args.force_build_processors,
        "offline": args.offline,
        "tokenizer_path": env.get(PI0_LOCAL_TOKENIZER_ENV),
        "patched_lerobot_train": str(patched_train_py) if patched_train_py else None,
        "patched_pi0_processor": str(patched_tokenizer_py) if patched_tokenizer_py else None,
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
