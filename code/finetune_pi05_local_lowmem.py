#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


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


def find_lerobot_dataset_roots(search_root: Path, max_depth: int = 2) -> list[Path]:
    roots = []

    def walk(base: Path, depth: int):
        if depth > max_depth:
            return
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            if is_lerobot_dataset_root(child):
                roots.append(child)
                continue
            if depth < max_depth:
                walk(child, depth + 1)

    walk(search_root, depth=1)
    return roots


def resolve_dataset_root(dataset_path: Path, dataset_subdir: str, search_depth: int) -> Path:
    def pick_latest(candidates: list[Path], search_base: Path, label: str) -> Path:
        chosen = max(candidates, key=lambda p: p.stat().st_mtime)
        display = chosen
        if chosen.is_relative_to(search_base):
            display = chosen.relative_to(search_base)
        print(
            f"[dataset] {label} has {len(candidates)} sub-datasets "
            f"(search_depth={search_depth}); auto picked latest modified: {display}"
        )
        return chosen

    if dataset_subdir:
        candidate = (dataset_path / dataset_subdir).resolve()
        if is_lerobot_dataset_root(candidate):
            return candidate
        if not candidate.exists() or not candidate.is_dir():
            raise FileNotFoundError(
                f"Dataset subdir not found: {candidate}"
            )
        candidates = find_lerobot_dataset_roots(candidate, max_depth=search_depth)
        if not candidates:
            raise FileNotFoundError(
                "Dataset subdir is not a valid LeRobot dataset root, and no nested "
                f"LeRobot roots were found: {candidate}"
            )
        return pick_latest(candidates, dataset_path, f"subdir '{dataset_subdir}'")

    if is_lerobot_dataset_root(dataset_path):
        return dataset_path

    candidates = find_lerobot_dataset_roots(dataset_path, max_depth=search_depth)

    if not candidates:
        raise FileNotFoundError(
            "No valid LeRobot dataset root found. "
            f"Checked: {dataset_path} and nested subdirectories up to depth={search_depth}."
        )

    if len(candidates) == 1:
        return candidates[0]
    return pick_latest(candidates, dataset_path, "dataset-path")


def path_display(path: Path, base: Path) -> str:
    try:
        if path.is_relative_to(base):
            return str(path.relative_to(base))
    except Exception:
        pass
    return str(path)


def resolve_dataset_roots(
    dataset_path: Path,
    dataset_subdir: str,
    search_depth: int,
    dataset_mode: str,
) -> list[Path]:
    if dataset_mode == "single":
        return [resolve_dataset_root(dataset_path, dataset_subdir, search_depth)]

    if dataset_subdir:
        search_base = (dataset_path / dataset_subdir).resolve()
        if not search_base.exists() or not search_base.is_dir():
            raise FileNotFoundError(f"Dataset subdir not found: {search_base}")
    else:
        search_base = dataset_path

    if is_lerobot_dataset_root(search_base):
        roots = [search_base]
    else:
        roots = find_lerobot_dataset_roots(search_base, max_depth=search_depth)

    if not roots:
        raise FileNotFoundError(
            "No valid LeRobot dataset roots found for full-dataset mode. "
            f"Checked: {search_base} and nested subdirectories up to depth={search_depth}."
        )

    # Keep deterministic ordering before shuffling into training schedule.
    roots = sorted({p.resolve() for p in roots}, key=lambda p: str(p))
    print(
        f"[dataset] discovered {len(roots)} dataset roots under "
        f"{path_display(search_base, dataset_path)}"
    )
    return roots


def build_training_schedule(dataset_roots: list[Path], seed: int, rounds: int) -> list[Path]:
    if not dataset_roots:
        return []

    rng = random.Random(seed)
    schedule = []
    for _ in range(max(1, rounds)):
        one_round = list(dataset_roots)
        rng.shuffle(one_round)
        schedule.extend(one_round)
    return schedule


def is_policy_artifact_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    marker_files = (
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "adapter_config.json",
    )
    if any((path / name).exists() for name in marker_files):
        return True
    if list(path.glob("*.safetensors")):
        return True
    return False


def resolve_next_policy_path(stage_output_dir: Path) -> Path:
    preferred_candidates = [
        stage_output_dir / "checkpoints" / "last" / "pretrained_model",
        stage_output_dir / "checkpoints" / "last",
        stage_output_dir / "pretrained_model",
        stage_output_dir,
    ]
    for candidate in preferred_candidates:
        if is_policy_artifact_dir(candidate):
            return candidate

    # Fallback: search recursively for likely policy directories.
    for candidate in sorted(stage_output_dir.rglob("pretrained_model")):
        if is_policy_artifact_dir(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not locate a valid policy directory in stage output. "
        f"Checked under: {stage_output_dir}"
    )


def try_float(val: str):
    try:
        if val.lower() in ("true", "false"):
            return val.lower() == "true"
        if re.match(r"^[+-]?\d+$", val):
            return int(val)
        if re.match(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$", val):
            return float(val)
        return val
    except Exception:
        return val


def parse_line_to_kv(line: str):
    parsed = {}

    # key=value
    for k, v in re.findall(r"([A-Za-z0-9_.]+)=([^\s,]+)", line):
        parsed[k] = try_float(v.strip())

    # key: value
    for k, v in re.findall(r"([A-Za-z0-9_.]+)\s*:\s*([^\s,]+)", line):
        if k not in parsed:
            parsed[k] = try_float(v.strip())

    # Effective batch size: 2 x 1 = 2
    m = re.search(r"Effective batch size:\s*(\d+)\s*x\s*(\d+)\s*=\s*(\d+)", line)
    if m:
        parsed["batch_size"] = int(m.group(1))
        parsed["grad_accum_steps"] = int(m.group(2))
        parsed["effective_batch_size"] = int(m.group(3))

    return parsed


def extract_metric_point(kv: dict, step_fallback: int):
    numeric = {}
    for k, v in kv.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            numeric[k] = v
    if not numeric:
        return None

    step = None
    for key in ("step", "steps", "global_step", "train_step"):
        if key in numeric:
            step = int(numeric[key])
            break
    if step is None:
        step = step_fallback

    numeric["_step"] = step
    return numeric


def pick_plot_metric(metric_points, preferred="loss"):
    if not metric_points:
        return None
    keys = set()
    for p in metric_points:
        keys.update(p.keys())
    keys.discard("_step")
    if preferred in keys:
        return preferred
    for name in ("loss", "train_loss", "total_loss", "policy_loss", "action_loss"):
        if name in keys:
            return name
    return next(iter(keys), None)


def save_line_plot(metric_points, out_png: Path, metric_name: str):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        return {"ok": False, "reason": f"matplotlib unavailable: {e}"}

    xs = []
    ys = []
    for p in metric_points:
        if metric_name in p:
            xs.append(p["_step"])
            ys.append(float(p[metric_name]))

    if not xs:
        return {"ok": False, "reason": f"metric '{metric_name}' not found in parsed points"}

    plt.figure(figsize=(9, 5))
    plt.plot(xs, ys, linewidth=1.8)
    plt.title(f"Training Curve - {metric_name}")
    plt.xlabel("Step")
    plt.ylabel(metric_name)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    return {"ok": True, "path": str(out_png)}


def ensure_unique_output_dir(output_root: Path, job_name: str, run_tag: str) -> Path:
    # LeRobot requires output_dir to not exist when resume=false.
    base = output_root / f"{job_name}_{run_tag}"
    return unique_path(base)


def pick_logs_dir(output_root: Path, job_name: str, run_tag: str) -> Path:
    # Keep logs separate so we don't pre-create trainer output_dir by mistake.
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
        default="/root/autodl-tmp/openpi_earbud_proto/lerobot_ready/earbud_insert_batch_v3_split",
    )
    parser.add_argument(
        "--dataset-subdir",
        default="",
        help=(
            "Optional: pick a subdirectory under dataset-path. If this directory is not a "
            "LeRobot root, nested LeRobot roots will be auto-resolved inside it."
        ),
    )
    parser.add_argument(
        "--dataset-search-depth",
        type=int,
        default=2,
        help="Max depth for auto-resolving nested LeRobot dataset roots.",
    )
    parser.add_argument(
        "--dataset-mode",
        choices=("all", "single"),
        default="all",
        help=(
            "all: discover all dataset roots and train them in random order; "
            "single: keep old behavior and train only one resolved dataset root."
        ),
    )
    parser.add_argument(
        "--dataset-rounds",
        type=int,
        default=1,
        help=(
            "How many random full passes to run when dataset-mode=all. "
            "Each round includes every discovered dataset root exactly once."
        ),
    )
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
        default="local/earbud_insert_batch_v3_split",
    )
    parser.add_argument("--job-name", default="pi05_earbud_lowmem")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-metric", default="loss")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).resolve()
    model_path = Path(args.model_path).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    dataset_roots = resolve_dataset_roots(
        dataset_path,
        args.dataset_subdir,
        search_depth=max(1, args.dataset_search_depth),
        dataset_mode=args.dataset_mode,
    )
    if args.dataset_mode == "single":
        schedule = [dataset_roots[0]]
    else:
        schedule = build_training_schedule(
            dataset_roots,
            seed=args.seed,
            rounds=max(1, args.dataset_rounds),
        )
    if not schedule:
        raise RuntimeError("Training schedule is empty.")

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    run_root = ensure_unique_output_dir(output_root, args.job_name, run_tag)
    run_root.mkdir(parents=True, exist_ok=False)
    stages_root = run_root / "stages"
    stages_root.mkdir(parents=True, exist_ok=True)

    launcher = (
        ["lerobot-train"]
        if shutil.which("lerobot-train")
        else [sys.executable, "-m", "lerobot.scripts.lerobot_train"]
    )

    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    print("\n========== Dataset Schedule ==========")
    print(f"mode: {args.dataset_mode}")
    print(f"search_depth: {max(1, args.dataset_search_depth)}")
    print(f"rounds: {max(1, args.dataset_rounds) if args.dataset_mode == 'all' else 1}")
    print(f"unique_dataset_roots: {len(dataset_roots)}")
    print(f"scheduled_stages: {len(schedule)}")
    for idx, ds in enumerate(schedule, start=1):
        print(f"  [{idx:03d}] {path_display(ds, dataset_path)}")
    print("")

    logs_dir = pick_logs_dir(output_root, args.job_name, run_tag)
    logs_dir.mkdir(parents=True, exist_ok=True)

    raw_lines = []
    parsed_entries = []
    metric_points = []
    step_fallback = 0
    stage_summaries = []
    current_policy_path = model_path
    return_code = 0

    for stage_idx, dataset_root in enumerate(schedule, start=1):
        dataset_display = path_display(dataset_root, dataset_path)
        stage_output_dir = unique_path(stages_root / f"{stage_idx:03d}_{dataset_root.name}")
        stage_job_name = f"{args.job_name}_{run_tag}_s{stage_idx:03d}"

        cmd = launcher + [
            f"--dataset.repo_id={args.dataset_repo_id}",
            f"--dataset.root={dataset_root}",
            "--dataset.revision=v3.0",
            f"--policy.path={current_policy_path}",
            "--policy.device=cuda",
            "--policy.dtype=bfloat16",
            "--policy.gradient_checkpointing=true",
            "--policy.compile_model=false",
            "--policy.train_expert_only=true",
            "--policy.freeze_vision_encoder=true",
            '--policy.normalization_mapping={"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}',
            f"--output_dir={stage_output_dir}",
            f"--job_name={stage_job_name}",
            "--policy.push_to_hub=false",
            "--wandb.enable=false",
            "--eval_freq=0",
            f"--batch_size={args.batch_size}",
            f"--num_workers={args.num_workers}",
            f"--steps={args.steps}",
            "--log_freq=10",
            "--save_freq=200",
            f"--seed={args.seed}",
            "--resume=false",
        ]

        print("\n========== Training Stage ==========")
        print(f"stage: {stage_idx}/{len(schedule)}")
        print(f"dataset: {dataset_display}")
        print(f"policy_in: {current_policy_path}")
        print(f"stage_output_dir: {stage_output_dir}")
        print("command:")
        print(" \\\n  ".join(shlex.quote(x) for x in cmd))
        print("")

        stage_raw_count = 0
        stage_parsed_count = 0
        stage_metric_count = 0
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
            clean = line.rstrip("\n")
            ts = datetime.now().isoformat(timespec="seconds")
            raw_lines.append(
                {
                    "ts": ts,
                    "stage": stage_idx,
                    "dataset_root": str(dataset_root),
                    "dataset_display": dataset_display,
                    "line": clean,
                }
            )
            stage_raw_count += 1

            kv = parse_line_to_kv(clean)
            if kv:
                parsed_entries.append(
                    {
                        "ts": ts,
                        "stage": stage_idx,
                        "dataset_root": str(dataset_root),
                        "dataset_display": dataset_display,
                        "line": clean,
                        "kv": kv,
                    }
                )
                stage_parsed_count += 1
                metric = extract_metric_point(kv, step_fallback=step_fallback)
                if metric is not None:
                    metric["stage"] = stage_idx
                    metric_points.append(metric)
                    stage_metric_count += 1
                    step_fallback += 1

        stage_rc = process.wait()
        stage_summary = {
            "stage": stage_idx,
            "dataset_root": str(dataset_root),
            "dataset_display": dataset_display,
            "policy_in": str(current_policy_path),
            "stage_output_dir": str(stage_output_dir),
            "return_code": stage_rc,
            "raw_line_count": stage_raw_count,
            "parsed_entry_count": stage_parsed_count,
            "metric_point_count": stage_metric_count,
        }

        if stage_rc != 0:
            return_code = stage_rc
            stage_summary["error"] = "trainer returned non-zero exit code"
            stage_summaries.append(stage_summary)
            print(f"[ERROR] stage {stage_idx} failed with return code {stage_rc}")
            break

        try:
            next_policy_path = resolve_next_policy_path(stage_output_dir)
            stage_summary["policy_out"] = str(next_policy_path)
            current_policy_path = next_policy_path
        except Exception as e:
            return_code = 1
            stage_summary["error"] = f"failed to locate next policy path: {e}"
            stage_summaries.append(stage_summary)
            print(f"[ERROR] stage {stage_idx} finished but next policy path was not found: {e}")
            break

        stage_summaries.append(stage_summary)

    raw_log_txt = logs_dir / "train_stdout.log"
    with raw_log_txt.open("w", encoding="utf-8") as f:
        for item in raw_lines:
            f.write(
                f"[{item['ts']}][stage={item['stage']}][dataset={item['dataset_display']}] "
                f"{item['line']}\n"
            )

    run_json = logs_dir / "training_log_dump.json"
    run_payload = {
        "return_code": return_code,
        "start_config": vars(args),
        "run_root": str(run_root),
        "resolved_dataset_roots": [str(p) for p in dataset_roots],
        "training_schedule": [str(p) for p in schedule],
        "stage_summaries": stage_summaries,
        "final_policy_path": str(current_policy_path),
        "launcher": launcher,
        "raw_line_count": len(raw_lines),
        "parsed_entry_count": len(parsed_entries),
        "metric_point_count": len(metric_points),
        "raw_lines": raw_lines,
        "parsed_entries": parsed_entries,
    }
    with run_json.open("w", encoding="utf-8") as f:
        json.dump(run_payload, f, ensure_ascii=False, indent=2)

    metric_name = pick_plot_metric(metric_points, preferred=args.plot_metric)
    plot_info = {"ok": False, "reason": "no metric points parsed"}
    if metric_name is not None:
        plot_info = save_line_plot(
            metric_points,
            out_png=logs_dir / "training_curve.png",
            metric_name=metric_name,
        )

    summary_json = logs_dir / "training_summary.json"
    summary_payload = {
        "return_code": return_code,
        "dataset_mode": args.dataset_mode,
        "dataset_root_count": len(dataset_roots),
        "scheduled_stage_count": len(schedule),
        "completed_stage_count": sum(1 for s in stage_summaries if s.get("return_code") == 0),
        "run_root": str(run_root),
        "final_policy_path": str(current_policy_path),
        "stage_summaries": stage_summaries,
        "logs_dir": str(logs_dir),
        "log_json": str(run_json),
        "raw_log_txt": str(raw_log_txt),
        "plot_metric": metric_name,
        "plot_info": plot_info,
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    print("\n========== Run Summary ==========")
    print(f"return_code: {return_code}")
    print(f"run_root: {run_root}")
    print(f"final_policy_path: {current_policy_path}")
    print(f"log_json: {run_json}")
    print(f"summary_json: {summary_json}")
    print(f"logs_dir: {logs_dir}")
    if plot_info.get("ok"):
        print(f"training_curve: {plot_info.get('path')}")
    else:
        print(f"training_curve not generated: {plot_info.get('reason')}")

    # Copy run logs into run_root for convenience.
    maybe_copy_logs_into_output(logs_dir, run_root)

    sys.exit(return_code)


if __name__ == "__main__":
    main()
