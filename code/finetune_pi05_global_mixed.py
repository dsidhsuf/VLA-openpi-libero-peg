#!/usr/bin/env python3
"""
Global-step mixed fine-tuning for per-episode LeRobot datasets.

What this script does:
1) Resolve one LeRobot dataset root from --dataset-path.
2) Launch ONE lerobot-train process directly on that root.

Because training is launched once, --steps is GLOBAL step count.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


DEFAULT_FEATURE_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    idx = 1
    while True:
        candidate = path.parent / f"{path.name}_r{idx}"
        if not candidate.exists():
            return candidate
        idx += 1


def is_lerobot_dataset_root(path: Path) -> bool:
    return (path / "meta" / "info.json").exists() and (path / "data").exists()


def path_display(path: Path, base: Path) -> str:
    try:
        if path.is_relative_to(base):
            return str(path.relative_to(base))
    except Exception:
        pass
    return str(path)


def find_lerobot_dataset_roots(search_root: Path, max_depth: int = 3) -> list[Path]:
    roots: list[Path] = []

    def walk(base: Path, depth: int):
        if depth > max_depth:
            return
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            if is_lerobot_dataset_root(child):
                roots.append(child.resolve())
                continue
            if depth < max_depth:
                walk(child, depth + 1)

    walk(search_root, depth=1)
    deduped = sorted({p.resolve() for p in roots}, key=lambda p: str(p))
    return deduped


def discover_source_roots(dataset_path: Path, dataset_subdir: str, search_depth: int) -> list[Path]:
    if dataset_subdir:
        search_base = (dataset_path / dataset_subdir).resolve()
        if not search_base.exists() or not search_base.is_dir():
            raise FileNotFoundError(f"Dataset subdir not found: {search_base}")
    else:
        search_base = dataset_path

    if is_lerobot_dataset_root(search_base):
        return [search_base.resolve()]

    roots = find_lerobot_dataset_roots(search_base, max_depth=search_depth)
    if not roots:
        raise FileNotFoundError(
            "No valid LeRobot dataset roots found. "
            f"Checked: {search_base}, search_depth={search_depth}"
        )
    return roots


def resolve_training_dataset_root(dataset_path: Path, dataset_subdir: str, search_depth: int) -> Path:
    roots = discover_source_roots(dataset_path, dataset_subdir, search_depth)
    if len(roots) == 1:
        return roots[0]

    preview = [path_display(p, dataset_path) for p in roots[:8]]
    raise RuntimeError(
        "This script now trains directly and does not merge per-episode roots.\n"
        f"Found {len(roots)} dataset roots under {dataset_path} (showing first 8): {preview}\n"
        "Please pass --dataset-path to one already-merged LeRobot dataset root "
        "(a folder containing meta/info.json and data/)."
    )


def try_read_repo_id_from_info(dataset_root: Path) -> str | None:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return None
    try:
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
        for key in ("repo_id", "hf_repo_id", "dataset_repo_id"):
            if isinstance(info.get(key), str) and info.get(key):
                return info.get(key)
    except Exception:
        return None
    return None


def make_task_lookup(meta_tasks) -> dict[int, str]:
    lookup: dict[int, str] = {}
    if meta_tasks is None:
        return lookup

    try:
        # Expected: pandas DataFrame indexed by task string with column task_index
        if hasattr(meta_tasks, "iterrows"):
            for task_name, row in meta_tasks.iterrows():
                if "task_index" in row:
                    lookup[int(row["task_index"])] = str(task_name)
            return lookup
    except Exception:
        pass

    try:
        # Fallback: list-like records
        for row in meta_tasks:
            if isinstance(row, dict) and "task_index" in row and "task" in row:
                lookup[int(row["task_index"])] = str(row["task"])
    except Exception:
        pass

    return lookup


def to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def frame_from_sample(
    sample: dict,
    expected_feature_keys: list[str],
    task_lookup: dict[int, str],
    default_task: str,
) -> dict:
    frame = {}
    for key in expected_feature_keys:
        if key not in sample:
            raise KeyError(f"Sample missing required feature key: {key}")
        frame[key] = to_numpy(sample[key])

    task = sample.get("task", None)
    if not isinstance(task, str) or not task:
        task_index = sample.get("task_index", None)
        if task_index is not None:
            try:
                task = task_lookup.get(int(task_index), None)
            except Exception:
                task = None
    if not isinstance(task, str) or not task:
        task = default_task

    frame["task"] = task
    return frame


def feature_signature(features: dict, expected_feature_keys: list[str]) -> dict:
    sig = {}
    for key in expected_feature_keys:
        spec = features[key]
        sig[key] = {
            "dtype": spec.get("dtype"),
            "shape": tuple(spec.get("shape", [])),
            "names": tuple(spec.get("names", [])),
        }
    return sig


def merge_episode_roots_to_single_dataset(
    source_roots: list[Path],
    merged_root: Path,
    merged_repo_id: str,
    limit_episodes: int,
    default_task: str,
    vcodec: str,
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    warned_legacy_api = False

    def open_dataset_compat(repo_id: str, root: Path):
        nonlocal warned_legacy_api
        try:
            return LeRobotDataset(repo_id=repo_id, root=root, return_uint8=True)
        except TypeError as e:
            if "return_uint8" not in str(e):
                raise
            if not warned_legacy_api:
                print(
                    "[compat] LeRobotDataset(..., return_uint8=...) is unsupported in this "
                    "environment; falling back to LeRobotDataset(repo_id=..., root=...)."
                )
                warned_legacy_api = True
            return LeRobotDataset(repo_id=repo_id, root=root)

    roots = list(source_roots)
    if limit_episodes > 0:
        roots = roots[:limit_episodes]
    if not roots:
        raise RuntimeError("No source dataset roots available after filtering.")

    first_root = roots[0]
    first_repo_id = try_read_repo_id_from_info(first_root) or merged_repo_id
    first_ds = open_dataset_compat(repo_id=first_repo_id, root=first_root)

    expected_feature_keys = sorted([k for k in first_ds.features.keys() if k not in DEFAULT_FEATURE_KEYS])
    if not expected_feature_keys:
        raise RuntimeError(f"No user features found in first dataset root: {first_root}")

    user_features = {k: first_ds.features[k] for k in expected_feature_keys}
    first_sig = feature_signature(first_ds.features, expected_feature_keys)
    fps = int(first_ds.meta.fps)
    robot_type = first_ds.meta.robot_type
    use_videos = any(ft.get("dtype") == "video" for ft in user_features.values())

    if merged_root.exists():
        raise FileExistsError(f"Merged dataset root already exists: {merged_root}")

    merged_ds = LeRobotDataset.create(
        repo_id=merged_repo_id,
        root=merged_root,
        fps=fps,
        features=user_features,
        robot_type=robot_type,
        use_videos=use_videos,
        vcodec=vcodec,
    )

    merge_manifest = []
    try:
        for idx, src_root in enumerate(roots, start=1):
            src_repo_id = try_read_repo_id_from_info(src_root) or merged_repo_id
            src_ds = open_dataset_compat(repo_id=src_repo_id, root=src_root)

            src_keys = sorted([k for k in src_ds.features.keys() if k not in DEFAULT_FEATURE_KEYS])
            if src_keys != expected_feature_keys:
                raise ValueError(
                    "Feature key mismatch across roots.\n"
                    f"Expected: {expected_feature_keys}\n"
                    f"Found   : {src_keys}\n"
                    f"Root    : {src_root}"
                )
            src_sig = feature_signature(src_ds.features, expected_feature_keys)
            if src_sig != first_sig:
                raise ValueError(
                    "Feature schema mismatch across roots (dtype/shape/names).\n"
                    f"Root: {src_root}"
                )
            if int(src_ds.meta.fps) != fps:
                raise ValueError(
                    f"FPS mismatch across roots. expected={fps} found={src_ds.meta.fps} root={src_root}"
                )

            task_lookup = make_task_lookup(getattr(src_ds.meta, "tasks", None))
            num_frames = len(src_ds)

            print(
                f"[merge {idx:04d}/{len(roots):04d}] "
                f"{src_root.name} | frames={num_frames}"
            )
            start_t = time.time()
            for frame_idx in range(num_frames):
                sample = src_ds[frame_idx]
                frame = frame_from_sample(
                    sample=sample,
                    expected_feature_keys=expected_feature_keys,
                    task_lookup=task_lookup,
                    default_task=default_task,
                )
                merged_ds.add_frame(frame)
            merged_ds.save_episode(parallel_encoding=True)
            elapsed = time.time() - start_t

            merge_manifest.append(
                {
                    "order": idx,
                    "source_root": str(src_root),
                    "source_repo_id": src_repo_id,
                    "source_frames": num_frames,
                    "elapsed_sec": elapsed,
                }
            )
    finally:
        merged_ds.finalize()

    return {
        "merged_root": str(merged_root),
        "merged_repo_id": merged_repo_id,
        "episode_count": len(roots),
        "fps": fps,
        "feature_keys": expected_feature_keys,
        "items": merge_manifest,
    }


def ensure_unique_output_dir(output_root: Path, job_name: str, run_tag: str) -> Path:
    base = output_root / f"{job_name}_{run_tag}"
    return unique_path(base)


def pick_logs_dir(output_root: Path, job_name: str, run_tag: str) -> Path:
    logs_root = output_root / "_run_logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    return unique_path(logs_root / f"{job_name}_{run_tag}")


def maybe_copy_logs_into_output(logs_dir: Path, output_dir: Path):
    if not output_dir.exists():
        return
    dst = output_dir / "run_logs"
    if dst.exists():
        dst = unique_path(output_dir / "run_logs_copy")
    shutil.copytree(logs_dir, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        default="/root/autodl-tmp/openpi_earbud_proto/lerobot_ready/earbud_insert_batch_v3_shuffled",
    )
    parser.add_argument("--dataset-subdir", default="")
    parser.add_argument("--dataset-search-depth", type=int, default=3)
    parser.add_argument(
        "--model-path",
        default="/root/autodl-tmp/hf_models/pi05_libero_finetuned_v044",
    )
    parser.add_argument(
        "--output-root",
        default="/root/autodl-tmp/openpi_earbud_proto/outputs",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default="local/earbud_insert_batch_v3_shuffled_mixed",
        help="Repo id used in training config metadata.",
    )
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-freq", type=int, default=10000)
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument(
        "--train-expert-only",
        choices=("true", "false"),
        default="true",
        help="Whether to train only the action expert/projections.",
    )
    parser.add_argument(
        "--freeze-vision-encoder",
        choices=("true", "false"),
        default="true",
        help="Whether to freeze the vision encoder.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        choices=("true", "false"),
        default="true",
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--compile-model",
        choices=("true", "false"),
        default="false",
        help="Enable torch.compile in policy.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--job-name", default="pi05_earbud_global_mixed")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).resolve()
    model_path = Path(args.model_path).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    dataset_root = resolve_training_dataset_root(
        dataset_path=dataset_path,
        dataset_subdir=args.dataset_subdir,
        search_depth=max(1, args.dataset_search_depth),
    )

    print("========== Dataset Root ==========")
    print(f"dataset_root: {dataset_root}")
    print("")

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_unique_output_dir(output_root, args.job_name, run_tag)
    logs_dir = pick_logs_dir(output_root, args.job_name, run_tag)
    logs_dir.mkdir(parents=True, exist_ok=True)

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
        f"--policy.gradient_checkpointing={args.gradient_checkpointing}",
        f"--policy.compile_model={args.compile_model}",
        f"--policy.train_expert_only={args.train_expert_only}",
        f"--policy.freeze_vision_encoder={args.freeze_vision_encoder}",
        '--policy.normalization_mapping={"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}',
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

    print("\n========== Training Command ==========")
    print(" \\\n  ".join(shlex.quote(x) for x in cmd))
    print("")
    print("NOTE: This is one single training run, so --steps is GLOBAL steps.")
    print(f"global_steps: {args.steps}")
    print(f"dataset_root: {dataset_root}")
    print(f"output_dir: {output_dir}")
    print("")

    raw_lines = []
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
    for line in process.stdout:
        print(line, end="")
        raw_lines.append(
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "line": line.rstrip("\n"),
            }
        )
    return_code = process.wait()

    raw_log_txt = logs_dir / "train_stdout.log"
    with raw_log_txt.open("w", encoding="utf-8") as f:
        for item in raw_lines:
            f.write(f"[{item['ts']}] {item['line']}\n")

    summary_json = logs_dir / "training_summary.json"
    summary = {
        "return_code": return_code,
        "run_tag": run_tag,
        "dataset_root": str(dataset_root),
        "global_steps": args.steps,
        "save_freq": args.save_freq,
        "output_dir": str(output_dir),
        "logs_dir": str(logs_dir),
        "raw_log_txt": str(raw_log_txt),
        "command": cmd,
        "start_config": vars(args),
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n========== Run Summary ==========")
    print(f"return_code: {return_code}")
    print(f"summary_json: {summary_json}")
    print(f"raw_log_txt: {raw_log_txt}")

    maybe_copy_logs_into_output(logs_dir, output_dir)
    sys.exit(return_code)


if __name__ == "__main__":
    main()
