#!/usr/bin/env python3
"""
Tiny PI0.5 canary overfit test.

This script runs two short, offline training checks on the same 8 LeRobot
training frames:

  1. real labels: use the dataset action chunks.
  2. random labels: replace those chunks with random actions sampled from the
     dataset action range.

Interpretation:
  - If neither real nor random loss drops, the trainable path / optimizer /
    LoRA target / processor path is broken or too weak.
  - If random drops but real does not, the real labels are likely conflicting,
    mis-normalized, or time-shifted.
  - If both drop, the model can memorize and the larger run failure is likely
    data coverage, augmentation, or longer-run configuration.

The script intentionally does not start LIBERO or a policy server.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_TASK = "Pick up the red rectangular peg, keep it vertical, and insert it into the rectangular slot."
DEFAULT_IMAGE_KEY = "observation.images.image"
DEFAULT_IMAGE2_KEY = "observation.images.image2"
DEFAULT_STATE_KEY = "observation.state"
DEFAULT_ACTION_KEY = "action"


try:
    from finetune_pi05_lora_red_peg_overfit import DEFAULT_LORA_TARGET_MODULES
except Exception:
    DEFAULT_LORA_TARGET_MODULES = (
        r"(.*\.paligemma\.model\.language_model\.layers\.\d+\."
        r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)|"
        r".*\.paligemma\.model\.multi_modal_projector\.linear|"
        r".*\.gemma_expert\..*\."
        r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj|"
        r"input_layernorm\.dense|post_attention_layernorm\.dense|norm\.dense)|"
        r"model\.(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out))"
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def image_to_uint8_hwc(value: Any) -> np.ndarray:
    arr = to_numpy(value)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if float(np.nanmax(arr)) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def sample_task(sample: dict[str, Any], default_task: str) -> str:
    task = sample.get("task", default_task)
    if isinstance(task, (list, tuple)) and task:
        task = task[0]
    if not isinstance(task, str):
        task = default_task
    return task.strip() or default_task


def read_actions_and_episodes_fast(dataset_root: Path, action_key: str) -> tuple[np.ndarray, np.ndarray]:
    """Read action and episode_index from LeRobot parquet files without decoding videos."""
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("pyarrow is required for the fast parquet path") from exc

    files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")

    actions: list[Any] = []
    episodes: list[Any] = []
    for file in files:
        table = pq.read_table(file, columns=[action_key, "episode_index"])
        actions.extend(table[action_key].to_pylist())
        episodes.extend(table["episode_index"].to_pylist())

    action_np = np.asarray(actions, dtype=np.float32)
    episode_np = np.asarray(episodes, dtype=np.int64).reshape(-1)
    if action_np.ndim != 2:
        raise RuntimeError(f"Expected action array shape (N,D), got {action_np.shape}")
    return action_np, episode_np


def read_actions_and_episodes_slow(ds: Any, action_key: str) -> tuple[np.ndarray, np.ndarray]:
    actions = []
    episodes = []
    for i in range(len(ds)):
        sample = ds[i]
        actions.append(to_numpy(sample[action_key]).astype(np.float32).reshape(-1))
        if "episode_index" in sample:
            episodes.append(int(to_numpy(sample["episode_index"]).reshape(-1)[0]))
        else:
            episodes.append(0)
    return np.asarray(actions, dtype=np.float32), np.asarray(episodes, dtype=np.int64)


def select_indices(
    actions: np.ndarray,
    episodes: np.ndarray,
    n_frames: int,
    selection: str,
    start_index: int,
    explicit_indices: str,
    seed: int,
) -> list[int]:
    n = len(actions)
    if explicit_indices.strip():
        out = [int(x) for x in explicit_indices.replace(" ", "").split(",") if x != ""]
        if len(out) != n_frames:
            raise ValueError(f"--indices provided {len(out)} ids, but --n-frames={n_frames}")
        return out

    if selection == "sequential":
        start = max(0, min(int(start_index), max(0, n - n_frames)))
        return list(range(start, min(n, start + n_frames)))

    rng = np.random.default_rng(seed)
    picks: list[int] = []

    def add(candidates: Any, limit: int | None = None) -> None:
        added = 0
        for idx in candidates:
            idx = int(idx)
            if idx < 0 or idx >= n or idx in picks:
                continue
            # Leave enough room for a chunk inside the same episode when possible.
            if idx + 1 < n and episodes[idx + 1] != episodes[idx]:
                continue
            picks.append(idx)
            added += 1
            if limit is not None and added >= limit:
                break

    action_dim = actions.shape[1]
    grip = actions[:, action_dim - 1]
    z = actions[:, 2] if action_dim >= 3 else np.zeros(n, dtype=np.float32)
    xy = np.linalg.norm(actions[:, :2], axis=1) if action_dim >= 2 else np.zeros(n, dtype=np.float32)

    add(np.argsort(-np.abs(np.diff(grip, prepend=grip[0]))), limit=max(1, n_frames // 4))
    add(np.argsort(z), limit=max(1, n_frames // 4))
    add(np.argsort(-z), limit=max(1, n_frames // 4))
    add(np.argsort(-xy), limit=max(1, n_frames // 4))

    if len(picks) < n_frames:
        remaining = np.setdiff1d(np.arange(n), np.asarray(picks, dtype=np.int64), assume_unique=False)
        if len(remaining) > 0:
            add(rng.choice(remaining, size=min(n_frames - len(picks), len(remaining)), replace=False))

    if len(picks) < n_frames:
        add(range(n))

    return picks[:n_frames]


def make_real_chunk(actions: np.ndarray, episodes: np.ndarray, idx: int, horizon: int) -> np.ndarray:
    action_dim = actions.shape[1]
    chunk = np.zeros((horizon, action_dim), dtype=np.float32)
    ep = episodes[idx]
    end = idx
    while end + 1 < len(actions) and episodes[end + 1] == ep and end - idx + 1 < horizon:
        end += 1
    window = actions[idx : end + 1].astype(np.float32)
    chunk[: len(window)] = window
    if len(window) < horizon:
        chunk[len(window) :] = window[-1]
    return chunk


def make_random_chunks(
    n_frames: int,
    horizon: int,
    action_min: np.ndarray,
    action_max: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lo = action_min.astype(np.float32)
    hi = action_max.astype(np.float32)
    span = np.maximum(hi - lo, 1e-3)
    chunks = rng.uniform(lo - 0.05 * span, hi + 0.05 * span, size=(n_frames, horizon, len(lo))).astype(np.float32)
    # Keep gripper in a normal action range if present.
    if chunks.shape[-1] >= 7:
        chunks[..., 6] = rng.uniform(-1.0, 1.0, size=chunks[..., 6].shape).astype(np.float32)
    return chunks


def normalize_state_dim(state: np.ndarray, target_dim: int | None) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if target_dim is None or target_dim <= 0:
        return state
    if len(state) == target_dim:
        return state
    if len(state) > target_dim:
        return state[:target_dim]
    out = np.zeros(target_dim, dtype=np.float32)
    out[: len(state)] = state
    return out


def infer_state_dim(policy: Any) -> int | None:
    cfg = getattr(policy, "config", None)
    if cfg is None:
        return None
    for attr in ("input_features", "observation_features"):
        feats = getattr(cfg, attr, None)
        if feats is None:
            continue
        if "observation.state" in feats:
            shape = getattr(feats["observation.state"], "shape", None)
            if shape is not None and len(shape) > 0:
                return int(shape[0])
    max_state_dim = getattr(cfg, "max_state_dim", None)
    return int(max_state_dim) if max_state_dim is not None else None


def collate_processed(items: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = items[0].keys()
    for key in keys:
        vals = [item[key] for item in items]
        first = vals[0]
        if isinstance(first, torch.Tensor):
            if key.startswith("observation.images."):
                out[key] = torch.cat(vals, dim=0) if first.ndim == 4 and first.shape[0] == 1 else torch.stack(vals, dim=0)
            elif key == "observation.state":
                out[key] = torch.cat(vals, dim=0) if first.ndim == 2 and first.shape[0] == 1 else torch.stack(vals, dim=0)
            elif key == "action":
                out[key] = torch.cat(vals, dim=0) if first.ndim == 3 and first.shape[0] == 1 else torch.stack(vals, dim=0)
            elif "language" in key or "attention_mask" in key:
                out[key] = torch.cat(vals, dim=0) if first.ndim >= 2 and first.shape[0] == 1 else torch.stack(vals, dim=0)
            else:
                try:
                    out[key] = torch.cat(vals, dim=0) if first.ndim >= 1 and first.shape[0] == 1 else torch.stack(vals, dim=0)
                except Exception:
                    out[key] = vals
        elif isinstance(first, np.ndarray):
            if key == "action" and first.ndim == 2:
                out[key] = torch.from_numpy(np.stack(vals, axis=0))
            elif key == "observation.state" and first.ndim == 1:
                out[key] = torch.from_numpy(np.stack(vals, axis=0))
            elif key.startswith("observation.images.") and first.ndim == 3:
                out[key] = torch.from_numpy(np.stack(vals, axis=0))
            else:
                out[key] = torch.from_numpy(np.stack(vals, axis=0))
        elif isinstance(first, str):
            out[key] = vals
        else:
            out[key] = vals
    return out


def prepare_batch_for_policy(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """A local copy of the server batching helper, kept here to avoid server startup."""
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, str):
            out[key] = value
            continue
        if isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], str):
            out[key] = list(value)
            continue

        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value)
        elif isinstance(value, torch.Tensor):
            tensor = value
        elif isinstance(value, (list, tuple)):
            try:
                arr = np.asarray(value)
                if arr.dtype.kind in ("U", "S", "O"):
                    out[key] = value
                    continue
                tensor = torch.from_numpy(arr)
            except Exception:
                out[key] = value
                continue
        else:
            out[key] = value
            continue

        if key.startswith("observation.images."):
            if tensor.ndim == 3:
                if tensor.shape[-1] in (1, 3):
                    tensor = tensor.permute(2, 0, 1).contiguous()
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim == 4 and tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(0, 3, 1, 2).contiguous()
        elif key == "observation.state" and tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif key == "action" and tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif ("language" in key or "attention_mask" in key) and tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        out[key] = tensor.to(device)
    return out


def postprocess_action_array(action_tensor: torch.Tensor, postprocess: Any) -> np.ndarray:
    original_shape = tuple(action_tensor.shape)
    if action_tensor.ndim == 3:
        batch, horizon, action_dim = action_tensor.shape
        flat = action_tensor.reshape(batch * horizon, action_dim)
        processed = postprocess(flat)
        arr = tensor_or_action_dict_to_numpy(processed)
        return np.asarray(arr, dtype=np.float32).reshape(batch, horizon, -1)

    processed = postprocess(action_tensor)
    arr = tensor_or_action_dict_to_numpy(processed)
    arr = np.asarray(arr, dtype=np.float32)
    if len(original_shape) == 2 and arr.ndim == 1:
        arr = arr.reshape(original_shape[0], -1)
    return arr


def tensor_or_action_dict_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict) and "action" in value:
        action = value["action"]
        if isinstance(action, torch.Tensor):
            return action.detach().cpu().numpy()
        return np.asarray(action)
    return np.asarray(value)


def build_raw_items(
    ds: Any,
    indices: list[int],
    chunks: np.ndarray,
    state_dim: int | None,
    image_key: str,
    image2_key: str,
    state_key: str,
    action_key: str,
    default_task: str,
    task_override: str,
    state_batched: bool,
    action_batched: bool,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for local_i, idx in enumerate(indices):
        sample = ds[idx]
        image = image_to_uint8_hwc(sample[image_key])
        image2 = image_to_uint8_hwc(sample[image2_key])
        state = normalize_state_dim(to_numpy(sample[state_key]), state_dim)
        action = chunks[local_i].astype(np.float32)

        raw_state = state[None, :] if state_batched else state
        raw_action = action[None, :, :] if action_batched else action
        task = task_override.strip() or sample_task(sample, default_task)
        items.append(
            {
                image_key: image,
                image2_key: image2,
                "observation.images.empty_camera_0": np.zeros_like(image, dtype=np.uint8),
                state_key: raw_state.astype(np.float32),
                action_key: raw_action.astype(np.float32),
                "task": task,
                "task_description": task,
                "language_instruction": task,
            }
        )
    return items


def make_policy_batch(
    policy: Any,
    preprocess: Any,
    ds: Any,
    indices: list[int],
    chunks: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    state_dim = infer_state_dim(policy)
    last_error: Exception | None = None
    for state_batched in (False, True):
        for action_batched in (False, True):
            try:
                raw_items = build_raw_items(
                    ds,
                    indices,
                    chunks,
                    state_dim,
                    args.image_key,
                    args.image2_key,
                    args.state_key,
                    args.action_key,
                    args.default_task,
                    args.task_override,
                    state_batched=state_batched,
                    action_batched=action_batched,
                )
                processed = [preprocess(item) for item in raw_items]
                batch = prepare_batch_for_policy(collate_processed(processed), device)
                with torch.no_grad():
                    loss, _ = policy(batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite dry-run loss: {float(loss.detach().cpu())}")
                print(
                    f"[batch] preprocess ok: state_batched={state_batched} "
                    f"action_batched={action_batched} dry_loss={float(loss.detach().cpu()):.6f}"
                )
                print("[batch] keys/shapes:")
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: shape={tuple(value.shape)} dtype={value.dtype}")
                    elif isinstance(value, list):
                        print(f"  {key}: list_len={len(value)}")
                    else:
                        print(f"  {key}: {type(value).__name__}")
                return batch
            except Exception as exc:
                last_error = exc
                if args.verbose:
                    print(f"[batch] failed state_batched={state_batched} action_batched={action_batched}: {exc}")
                continue
    raise RuntimeError("Could not build a valid PI0.5 training batch") from last_error


def import_pi05_policy():
    try:
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    except Exception:
        try:
            from lerobot.policies.pi05 import PI05Policy
        except Exception:
            from lerobot.policies.pi05 import Pi05Policy as PI05Policy
    return PI05Policy


def build_runtime_model_dir(policy_path: str, tokenizer_path: str) -> str:
    try:
        from policy_server_pi05 import build_runtime_model_dir as server_build_runtime_model_dir

        return server_build_runtime_model_dir(policy_path, tokenizer_path, overlay_path=None)
    except Exception as exc:
        print(f"[warn] could not reuse policy_server_pi05.build_runtime_model_dir: {exc}")
        return policy_path


def make_processors(policy: Any, runtime_model_dir: str, tokenizer_path: str, force_build_processors: bool):
    from lerobot.policies.factory import make_pre_post_processors

    overrides = {
        "device_processor": {"device": str(next(policy.parameters()).device)},
        "tokenizer_processor": {"tokenizer_name": tokenizer_path},
    }

    if force_build_processors:
        # Some local PI0.5 mirrors have processor JSON files whose state entries
        # are not available locally. LeRobot then calls hf_hub_download with the
        # absolute runtime path as repo_id. For this canary we only need a valid
        # training batch, so rebuilding processors is the safest default.
        print("[processors] force_build_processors=true: building processors from policy config")
        try:
            return make_pre_post_processors(
                policy.config,
                None,
                preprocessor_overrides=overrides,
            )
        except TypeError:
            # Keep compatibility with LeRobot variants that prefer a keyword.
            return make_pre_post_processors(
                policy.config,
                pretrained_path=None,
                preprocessor_overrides=overrides,
            )

    try:
        return make_pre_post_processors(
            policy.config,
            runtime_model_dir,
            preprocessor_overrides=overrides,
        )
    except Exception as exc:
        print(f"[processors] loading checkpoint processors failed: {repr(exc)}")
        print("[processors] retrying with pretrained_path=None")
        return make_pre_post_processors(
            policy.config,
            None,
            preprocessor_overrides=overrides,
        )


def load_base_policy(args: argparse.Namespace, device: torch.device):
    runtime_model_dir = build_runtime_model_dir(args.base_policy_path, args.tokenizer_path)
    PI05Policy = import_pi05_policy()
    print(f"[load] runtime_model_dir={runtime_model_dir}")
    policy = PI05Policy.from_pretrained(runtime_model_dir).to(device)
    if args.dtype == "bfloat16" and device.type == "cuda":
        policy = policy.to(dtype=torch.bfloat16)
    elif args.dtype == "float16" and device.type == "cuda":
        policy = policy.to(dtype=torch.float16)
    policy.train()
    preprocess, postprocess = make_processors(
        policy,
        runtime_model_dir,
        args.tokenizer_path,
        force_build_processors=bool(args.force_build_processors),
    )
    return policy, preprocess, postprocess, runtime_model_dir


def named_trainable_parameters(model: torch.nn.Module):
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def setup_trainable(policy: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    if args.train_mode == "lora":
        from peft import LoraConfig, get_peft_model

        matches = [
            name
            for name, module in policy.named_modules()
            if re.fullmatch(args.lora_target_modules, name) and isinstance(module, torch.nn.Linear)
        ]
        print(f"[lora] regex matched Linear modules: {len(matches)}")
        for name in matches[:20]:
            print(f"  [lora_match] {name}")
        if len(matches) > 20:
            print(f"  ... {len(matches) - 20} more")

        lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=args.lora_target_modules,
        )
        policy = get_peft_model(policy, lora_cfg)
        if hasattr(policy, "print_trainable_parameters"):
            policy.print_trainable_parameters()
        return policy

    for _, param in policy.named_parameters():
        param.requires_grad_(False)

    if args.train_mode == "heads":
        pattern = re.compile(args.heads_regex)
    elif args.train_mode == "expert_heads":
        pattern = re.compile(args.expert_heads_regex)
    elif args.train_mode == "all":
        pattern = None
    else:
        raise ValueError(f"Unknown train_mode: {args.train_mode}")

    for name, param in policy.named_parameters():
        if pattern is None or pattern.search(name):
            param.requires_grad_(True)

    trainable = named_trainable_parameters(policy)
    print(f"[trainable] mode={args.train_mode} tensors={len(trainable)} params={sum(p.numel() for _, p in trainable):,}")
    for name, param in trainable[:30]:
        print(f"  [trainable] {name} shape={tuple(param.shape)}")
    if len(trainable) > 30:
        print(f"  ... {len(trainable) - 30} more")
    return policy


def grad_stats(policy: torch.nn.Module) -> dict[str, Any]:
    total_sq = 0.0
    max_abs = 0.0
    nonzero_tensors = 0
    none_tensors = 0
    examples = []
    for name, param in named_trainable_parameters(policy):
        if param.grad is None:
            none_tensors += 1
            continue
        grad = param.grad.detach()
        norm = float(grad.float().norm().cpu())
        absmax = float(grad.float().abs().max().cpu()) if grad.numel() else 0.0
        total_sq += norm * norm
        max_abs = max(max_abs, absmax)
        if norm > 0:
            nonzero_tensors += 1
            if len(examples) < 10:
                examples.append({"name": name, "grad_norm": norm, "grad_absmax": absmax})
    return {
        "grad_norm": math.sqrt(total_sq),
        "grad_absmax": max_abs,
        "nonzero_grad_tensors": nonzero_tensors,
        "none_grad_tensors": none_tensors,
        "examples": examples,
    }


def snapshot_trainable(policy: torch.nn.Module, max_tensors: int = 8) -> dict[str, torch.Tensor]:
    snap = {}
    for name, param in named_trainable_parameters(policy):
        if len(snap) >= max_tensors:
            break
        snap[name] = param.detach().float().cpu().clone()
    return snap


def weight_delta(policy: torch.nn.Module, snap: dict[str, torch.Tensor]) -> dict[str, float]:
    deltas = []
    for name, old in snap.items():
        param = dict(policy.named_parameters()).get(name)
        if param is None:
            continue
        new = param.detach().float().cpu()
        deltas.append(float((new - old).abs().mean()))
    if not deltas:
        return {"mean_abs_delta": 0.0, "max_abs_delta": 0.0}
    return {"mean_abs_delta": float(np.mean(deltas)), "max_abs_delta": float(np.max(deltas))}


@torch.no_grad()
def eval_forward_loss(policy: torch.nn.Module, batch: dict[str, Any], use_autocast: bool) -> tuple[float, list[float] | None]:
    policy.eval()
    device = next(policy.parameters()).device
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast and device.type == "cuda"):
        loss, info = policy(batch)
    policy.train()
    per_dim = info.get("loss_per_dim") if isinstance(info, dict) else None
    return float(loss.detach().float().cpu()), per_dim


@torch.no_grad()
def eval_prediction_mae(
    policy: torch.nn.Module,
    batch: dict[str, Any],
    raw_chunks: np.ndarray,
    postprocess: Any,
    use_autocast: bool,
) -> dict[str, float]:
    if not hasattr(policy, "predict_action_chunk"):
        return {}
    policy.eval()
    device = next(policy.parameters()).device
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast and device.type == "cuda"):
        pred = policy.predict_action_chunk(batch)
    pred_raw = postprocess_action_array(pred, postprocess)
    label = raw_chunks[:, : pred_raw.shape[1], : pred_raw.shape[2]]
    out = {
        "pred_first_mae": float(np.mean(np.abs(pred_raw[:, 0, :7] - label[:, 0, :7]))),
        "pred_chunk_mae": float(np.mean(np.abs(pred_raw[..., :7] - label[..., :7]))),
    }
    if pred_raw.shape[-1] >= 7:
        out["pred_first_gripper_mean"] = float(np.mean(pred_raw[:, 0, 6]))
    policy.train()
    return out


@dataclass
class RunResult:
    name: str
    selected_indices: list[int]
    initial_loss: float
    final_loss: float
    best_loss: float
    first_grad_stats: dict[str, Any]
    first_weight_delta: dict[str, float]
    final_weight_delta: dict[str, float]
    initial_pred: dict[str, float]
    final_pred: dict[str, float]
    loss_history: list[dict[str, float]]


def train_one_case(
    case_name: str,
    raw_chunks: np.ndarray,
    ds: Any,
    indices: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> RunResult:
    print("\n" + "=" * 80)
    print(f"[case] {case_name}")
    print("=" * 80)

    policy, preprocess, postprocess, _runtime_dir = load_base_policy(args, device)
    policy = setup_trainable(policy, args).to(device).train()
    trainable = named_trainable_parameters(policy)
    if not trainable:
        raise RuntimeError("No trainable parameters. Check --train-mode / target regex.")

    batch = make_policy_batch(policy, preprocess, ds, indices, raw_chunks, args, device)

    optimizer = torch.optim.AdamW(
        [param for _, param in trainable],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    use_autocast = bool(args.autocast)
    initial_loss, initial_per_dim = eval_forward_loss(policy, batch, use_autocast)
    initial_pred = {} if args.skip_predict else eval_prediction_mae(policy, batch, raw_chunks, postprocess, use_autocast)
    print(f"[{case_name}] initial_loss={initial_loss:.6f}")
    if initial_per_dim is not None:
        print(f"[{case_name}] initial_loss_per_dim={np.asarray(initial_per_dim)[:7].round(6).tolist()}")
    if initial_pred:
        print(f"[{case_name}] initial_pred={initial_pred}")

    snap_before = snapshot_trainable(policy)
    first_grad: dict[str, Any] = {}
    first_delta: dict[str, float] = {}
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    t0 = time.time()

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast and device.type == "cuda"):
            loss, info = policy(batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss}")
        loss.backward()

        if step == 1:
            first_grad = grad_stats(policy)
            print(f"[{case_name}] first_grad={json.dumps(first_grad, ensure_ascii=False, indent=2)}")

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([param for _, param in trainable], args.grad_clip)
        optimizer.step()

        if step == 1:
            first_delta = weight_delta(policy, snap_before)
            print(f"[{case_name}] first_weight_delta={first_delta}")

        loss_value = float(loss.detach().float().cpu())
        best_loss = min(best_loss, loss_value)
        if step == 1 or step % args.log_freq == 0 or step == args.steps:
            elapsed = time.time() - t0
            row = {"step": float(step), "loss": loss_value, "elapsed_s": elapsed}
            history.append(row)
            print(f"[{case_name}] step={step:04d}/{args.steps} loss={loss_value:.6f} elapsed={elapsed:.1f}s")

    final_loss, final_per_dim = eval_forward_loss(policy, batch, use_autocast)
    final_delta = weight_delta(policy, snap_before)
    final_pred = {} if args.skip_predict else eval_prediction_mae(policy, batch, raw_chunks, postprocess, use_autocast)
    print(f"[{case_name}] final_loss={final_loss:.6f} best_train_loss={best_loss:.6f}")
    if final_per_dim is not None:
        print(f"[{case_name}] final_loss_per_dim={np.asarray(final_per_dim)[:7].round(6).tolist()}")
    if final_pred:
        print(f"[{case_name}] final_pred={final_pred}")
    print(f"[{case_name}] final_weight_delta={final_delta}")

    result = RunResult(
        name=case_name,
        selected_indices=indices,
        initial_loss=initial_loss,
        final_loss=final_loss,
        best_loss=best_loss,
        first_grad_stats=first_grad,
        first_weight_delta=first_delta,
        final_weight_delta=final_delta,
        initial_pred=initial_pred,
        final_pred=final_pred,
        loss_history=history,
    )

    del policy, preprocess, postprocess, optimizer, batch
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--base-policy-path", required=True)
    parser.add_argument("--tokenizer-path", default="/root/autodl-tmp/cache/huggingface/google/paligemma-3b-pt-224")
    parser.add_argument("--output-json", default="pi05_canary_overfit_8frames_summary.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--autocast", action="store_true", help="Use CUDA bfloat16 autocast around forward/backward.")
    parser.add_argument("--offline", action="store_true", help="Set HF/Transformers offline env flags.")
    parser.add_argument(
        "--force-build-processors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build processors from policy config instead of loading checkpoint processor state files.",
    )

    parser.add_argument("--n-frames", type=int, default=8)
    parser.add_argument("--selection", choices=["diverse", "sequential"], default="diverse")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--indices", default="", help="Comma-separated explicit frame indices, e.g. 100,120,...")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-freq", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train-mode", choices=["lora", "heads", "expert_heads", "all"], default="lora")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target-modules", default=DEFAULT_LORA_TARGET_MODULES)
    parser.add_argument(
        "--heads-regex",
        default=r"(action_in_proj|action_out_proj|time_mlp|action_time_mlp|state_proj)",
    )
    parser.add_argument(
        "--expert-heads-regex",
        default=r"(gemma_expert|action_in_proj|action_out_proj|time_mlp|action_time_mlp|state_proj)",
    )

    parser.add_argument("--image-key", default=DEFAULT_IMAGE_KEY)
    parser.add_argument("--image2-key", default=DEFAULT_IMAGE2_KEY)
    parser.add_argument("--state-key", default=DEFAULT_STATE_KEY)
    parser.add_argument("--action-key", default=DEFAULT_ACTION_KEY)
    parser.add_argument("--default-task", default=DEFAULT_TASK)
    parser.add_argument("--task-override", default="")
    parser.add_argument("--skip-predict", action="store_true", help="Skip predict_action_chunk MAE checks.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.tokenizer_path:
        os.environ["PI0_LOCAL_TOKENIZER_PATH"] = args.tokenizer_path

    set_seed(args.seed)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    try:
        ds = LeRobotDataset(repo_id=args.repo_id, root=str(dataset_root), return_uint8=True)
    except TypeError:
        ds = LeRobotDataset(repo_id=args.repo_id, root=str(dataset_root))

    try:
        actions, episodes = read_actions_and_episodes_fast(dataset_root, args.action_key)
    except Exception as exc:
        print(f"[warn] fast parquet action read failed, falling back to LeRobotDataset iteration: {exc}")
        actions, episodes = read_actions_and_episodes_slow(ds, args.action_key)

    if len(actions) != len(ds):
        print(f"[warn] parquet action count {len(actions)} != dataset len {len(ds)}; continuing with min length.")
        n = min(len(actions), len(ds))
        actions = actions[:n]
        episodes = episodes[:n]

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    indices = select_indices(
        actions,
        episodes,
        n_frames=args.n_frames,
        selection=args.selection,
        start_index=args.start_index,
        explicit_indices=args.indices,
        seed=args.seed,
    )

    real_chunks = np.stack([make_real_chunk(actions, episodes, idx, args.chunk_size) for idx in indices], axis=0)
    random_chunks = make_random_chunks(
        n_frames=len(indices),
        horizon=args.chunk_size,
        action_min=np.min(actions, axis=0),
        action_max=np.max(actions, axis=0),
        seed=args.seed + 123,
    )

    print("[dataset]", dataset_root, "len=", len(ds), "actions_shape=", actions.shape)
    print("[selected_indices]", indices)
    for local_i, idx in enumerate(indices):
        a0 = real_chunks[local_i, 0, :7]
        ar = random_chunks[local_i, 0, :7]
        print(
            f"  idx={idx:06d} real0=({a0[0]:+.4f},{a0[1]:+.4f},{a0[2]:+.4f},"
            f"{a0[3]:+.4f},{a0[4]:+.4f},{a0[5]:+.4f},{a0[6]:+.4f}) "
            f"rand0=({ar[0]:+.4f},{ar[1]:+.4f},{ar[2]:+.4f},"
            f"{ar[3]:+.4f},{ar[4]:+.4f},{ar[5]:+.4f},{ar[6]:+.4f})"
        )

    results = []
    results.append(train_one_case("real_labels", real_chunks, ds, indices, args, device))
    results.append(train_one_case("random_labels", random_chunks, ds, indices, args, device))

    payload = {
        "args": vars(args),
        "dataset_root": str(dataset_root),
        "actions_shape": list(actions.shape),
        "selected_indices": indices,
        "results": [asdict(result) for result in results],
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("[summary]")
    for result in results:
        ratio = result.final_loss / max(result.initial_loss, 1e-12)
        print(
            f"{result.name}: initial={result.initial_loss:.6f} final={result.final_loss:.6f} "
            f"best={result.best_loss:.6f} final/initial={ratio:.4f} "
            f"grad_norm={result.first_grad_stats.get('grad_norm', float('nan')):.6f} "
            f"delta1={result.first_weight_delta.get('mean_abs_delta', 0.0):.3e}"
        )
    print(f"[saved] {output_json}")


if __name__ == "__main__":
    main()
