#!/usr/bin/env python3
"""
Train a small behavior-cloning policy for the red-peg LIBERO task.

This is intentionally independent of PI0/PI0.5. It reads the same LeRobot
dataset, predicts an action chunk, and can be served by
policy_server_small_bc_red_peg.py behind the existing benchmark client.

Two model kinds are supported:
  - state_mlp: fastest sanity check for state/action/chunk plumbing
  - image_state_cnn: small visual policy using both cameras plus robot state
  - image_state_text_cnn: VLA-style variant with an extra language branch
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DEFAULT_IMAGE_KEY = "observation.images.image"
DEFAULT_IMAGE2_KEY = "observation.images.image2"
DEFAULT_STATE_KEY = "observation.state"
DEFAULT_ACTION_KEY = "action"
DEFAULT_TASK = "Pick up the red rectangular peg, keep it vertical, and insert it into the rectangular slot."


@dataclass
class ModelConfig:
    model_kind: str
    state_dim: int
    action_dim: int
    horizon: int
    image_size: int
    cnn_channels: int
    hidden_dim: int
    state_mean: list[float]
    state_std: list[float]
    action_clip: list[float]
    image_key: str
    image2_key: str
    state_key: str
    action_key: str
    task_text: str = DEFAULT_TASK
    text_max_len: int = 160
    text_embed_dim: int = 128
    pos_label_scale: float = 1.0
    rot_label_scale: float = 1.0
    index_start: int = 0
    index_end: int = -1
    use_progress: bool = False
    progress_denominator: float = 500.0


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


def image_to_chw_float(value: Any, image_size: int) -> torch.Tensor:
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
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    if image_size > 0 and (t.shape[-2] != image_size or t.shape[-1] != image_size):
        t = F.interpolate(t.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False)[0]
    # Fixed normalization keeps the checkpoint self-contained and server-simple.
    return (t - 0.5) / 0.5


def scalar_int(value: Any) -> int:
    arr = to_numpy(value)
    return int(arr.reshape(-1)[0])


def text_to_byte_tokens(text: str, max_len: int) -> torch.Tensor:
    """Tiny dependency-free tokenizer: UTF-8 bytes + 1, with 0 as padding."""
    max_len = int(max_len)
    tokens = np.zeros(max_len, dtype=np.int64)
    if max_len <= 0:
        return torch.from_numpy(tokens)
    raw = (text or "").encode("utf-8")[:max_len]
    if raw:
        tokens[: len(raw)] = np.frombuffer(raw, dtype=np.uint8).astype(np.int64) + 1
    return torch.from_numpy(tokens)


def read_arrays_from_parquet(dataset_root: Path, state_key: str, action_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast path for labels and episode ids without decoding videos."""
    import pyarrow.parquet as pq

    files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")

    states: list[Any] = []
    actions: list[Any] = []
    episodes: list[Any] = []
    for file in files:
        table = pq.read_table(file, columns=[state_key, action_key, "episode_index"])
        states.extend(table[state_key].to_pylist())
        actions.extend(table[action_key].to_pylist())
        episodes.extend(table["episode_index"].to_pylist())

    state_np = np.asarray(states, dtype=np.float32)
    action_np = np.asarray(actions, dtype=np.float32)
    episode_np = np.asarray(episodes, dtype=np.int64).reshape(-1)
    if state_np.ndim != 2:
        raise RuntimeError(f"Expected states shape (N,D), got {state_np.shape}")
    if action_np.ndim != 2:
        raise RuntimeError(f"Expected actions shape (N,D), got {action_np.shape}")
    return state_np, action_np, episode_np


def build_action_chunks(actions: np.ndarray, episodes: np.ndarray, horizon: int) -> np.ndarray:
    n, action_dim = actions.shape
    chunks = np.zeros((n, horizon, action_dim), dtype=np.float32)
    for i in range(n):
        ep = episodes[i]
        end = i
        while end + 1 < n and episodes[end + 1] == ep and end - i + 1 < horizon:
            end += 1
        window = actions[i : end + 1]
        chunks[i, : len(window)] = window
        if len(window) < horizon:
            chunks[i, len(window) :] = window[-1]
    return chunks


def select_training_indices(actions: np.ndarray, max_samples: int | None) -> np.ndarray:
    n = len(actions)
    if max_samples is None or max_samples <= 0 or max_samples >= n:
        return np.arange(n, dtype=np.int64)

    rng = np.random.default_rng(0)
    picks: list[int] = []

    def add(indices: np.ndarray | list[int]) -> None:
        for idx in indices:
            idx = int(idx)
            if 0 <= idx < n and idx not in picks:
                picks.append(idx)

    g = actions[:, 6]
    z = actions[:, 2]
    xy = np.linalg.norm(actions[:, :2], axis=1)
    add(np.where(g < -0.5)[0])
    add(np.where(z > 0.01)[0])
    add(np.where(xy > 0.05)[0])
    add(np.argsort(-np.abs(np.diff(g, prepend=g[0])))[: max_samples // 4])
    remaining = np.setdiff1d(np.arange(n), np.asarray(picks, dtype=np.int64), assume_unique=False)
    if len(picks) < max_samples and len(remaining) > 0:
        need = max_samples - len(picks)
        add(rng.choice(remaining, size=min(need, len(remaining)), replace=False))
    return np.asarray(picks[:max_samples], dtype=np.int64)


class SmallBCDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        repo_id: str,
        model_kind: str,
        image_size: int,
        image_key: str,
        image2_key: str,
        state_key: str,
        action_key: str,
        horizon: int,
        max_samples: int | None = None,
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.dataset_root = Path(dataset_root)
        self.model_kind = model_kind
        self.image_size = int(image_size)
        self.image_key = image_key
        self.image2_key = image2_key
        self.state_key = state_key
        self.action_key = action_key
        self.task_text = DEFAULT_TASK
        self.text_max_len = 160
        self.text_tokens = text_to_byte_tokens(self.task_text, self.text_max_len)

        try:
            self.ds = LeRobotDataset(repo_id=repo_id, root=str(dataset_root), return_uint8=True)
        except TypeError:
            self.ds = LeRobotDataset(repo_id=repo_id, root=str(dataset_root))

        states, actions, episodes = read_arrays_from_parquet(self.dataset_root, state_key, action_key)
        if len(self.ds) != len(actions):
            print(f"[warn] LeRobotDataset len={len(self.ds)} parquet rows={len(actions)}; using min length.")
            n = min(len(self.ds), len(actions))
            states, actions, episodes = states[:n], actions[:n], episodes[:n]

        self.states = states.astype(np.float32)
        self.actions = actions.astype(np.float32)
        self.episodes = episodes.astype(np.int64)
        self.action_chunks = build_action_chunks(self.actions, self.episodes, horizon=int(horizon))
        self.indices = select_training_indices(self.actions, max_samples=max_samples)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[item])
        out = {
            "state": torch.from_numpy(self.states[idx]),
            "action_chunk": torch.from_numpy(self.action_chunks[idx]),
            "action_first": torch.from_numpy(self.actions[idx]),
            "index": torch.tensor(idx, dtype=torch.long),
        }
        if self.model_kind != "state_mlp":
            sample = self.ds[idx]
            out["image"] = image_to_chw_float(sample[self.image_key], self.image_size)
            out["image2"] = image_to_chw_float(sample[self.image2_key], self.image_size)
        if self.model_kind == "image_state_text_cnn":
            out["text_tokens"] = self.text_tokens.clone()
        return out


class StateMLPPolicy(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, horizon: int, action_dim: int):
        super().__init__()
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, horizon * action_dim),
        )

    def forward(
        self,
        state: torch.Tensor,
        image: torch.Tensor | None = None,
        image2: torch.Tensor | None = None,
        text_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del image, image2, text_tokens
        out = self.net(state)
        return out.view(state.shape[0], self.horizon, self.action_dim)


class TinyImageEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int):
        super().__init__()
        c = int(base_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, c),
            nn.SiLU(),
            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, c * 2),
            nn.SiLU(),
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, c * 4),
            nn.SiLU(),
            nn.Conv2d(c * 4, c * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, c * 4),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.out_dim = c * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImageStateCNNPolicy(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, horizon: int, action_dim: int, cnn_channels: int):
        super().__init__()
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.image_encoder = TinyImageEncoder(in_channels=6, base_channels=cnn_channels)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.SiLU(),
        )
        fused_dim = self.image_encoder.out_dim + hidden_dim // 2
        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, horizon * action_dim),
        )

    def forward(
        self,
        state: torch.Tensor,
        image: torch.Tensor | None = None,
        image2: torch.Tensor | None = None,
        text_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del text_tokens
        if image is None or image2 is None:
            raise RuntimeError("image_state_cnn requires both image tensors")
        vis = self.image_encoder(torch.cat([image, image2], dim=1))
        st = self.state_encoder(state)
        out = self.head(torch.cat([vis, st], dim=1))
        return out.view(state.shape[0], self.horizon, self.action_dim)


class TinyTextEncoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(257, embed_dim, padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
        )
        self.out_dim = int(embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        emb = self.embedding(tokens.clamp(min=0, max=256))
        mask = (tokens != 0).float().unsqueeze(-1)
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled)


class ImageStateTextCNNPolicy(nn.Module):
    """Small VLA-style policy: two cameras + proprioceptive state + task text."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        horizon: int,
        action_dim: int,
        cnn_channels: int,
        text_embed_dim: int,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.image_encoder = TinyImageEncoder(in_channels=6, base_channels=cnn_channels)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.SiLU(),
        )
        self.text_encoder = TinyTextEncoder(text_embed_dim)
        fused_dim = self.image_encoder.out_dim + hidden_dim // 2 + self.text_encoder.out_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, horizon * action_dim),
        )

    def forward(
        self,
        state: torch.Tensor,
        image: torch.Tensor | None = None,
        image2: torch.Tensor | None = None,
        text_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if image is None or image2 is None:
            raise RuntimeError("image_state_text_cnn requires both image tensors")
        if text_tokens is None:
            raise RuntimeError("image_state_text_cnn requires text_tokens")
        vis = self.image_encoder(torch.cat([image, image2], dim=1))
        st = self.state_encoder(state)
        txt = self.text_encoder(text_tokens)
        out = self.head(torch.cat([vis, st, txt], dim=1))
        return out.view(state.shape[0], self.horizon, self.action_dim)


def make_model(cfg: ModelConfig) -> nn.Module:
    if cfg.model_kind == "state_mlp":
        return StateMLPPolicy(cfg.state_dim, cfg.hidden_dim, cfg.horizon, cfg.action_dim)
    if cfg.model_kind == "image_state_cnn":
        return ImageStateCNNPolicy(cfg.state_dim, cfg.hidden_dim, cfg.horizon, cfg.action_dim, cfg.cnn_channels)
    if cfg.model_kind == "image_state_text_cnn":
        return ImageStateTextCNNPolicy(
            cfg.state_dim,
            cfg.hidden_dim,
            cfg.horizon,
            cfg.action_dim,
            cfg.cnn_channels,
            cfg.text_embed_dim,
        )
    raise ValueError(f"Unknown model_kind: {cfg.model_kind}")


def weighted_bc_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    z_pos_weight: float,
    large_xy_weight: float,
    gripper_mse_weight: float,
    gripper_bce_weight: float,
    xy_threshold: float,
    z_pos_threshold: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    weight = torch.ones_like(target)
    z_pos = target[..., 2] > float(z_pos_threshold)
    large_xy = torch.linalg.norm(target[..., :2], dim=-1) > float(xy_threshold)

    weight[..., 2] = torch.where(z_pos, torch.full_like(weight[..., 2], z_pos_weight), weight[..., 2])
    xy_w = torch.where(large_xy, torch.full_like(target[..., 0], large_xy_weight), torch.ones_like(target[..., 0]))
    weight[..., 0] = weight[..., 0] * xy_w
    weight[..., 1] = weight[..., 1] * xy_w
    weight[..., 6] = weight[..., 6] * float(gripper_mse_weight)

    mse = ((pred - target) ** 2 * weight).mean()

    gripper_target_pos = (target[..., 6] > 0).float()
    gripper_logits = pred[..., 6] * 3.0
    grip_bce_raw = F.binary_cross_entropy_with_logits(gripper_logits, gripper_target_pos, reduction="none")
    grip_event_weight = torch.where(
        target[..., 6] < -0.5,
        torch.full_like(target[..., 6], 2.0),
        torch.ones_like(target[..., 6]),
    )
    grip_bce = (grip_bce_raw * grip_event_weight).mean()
    loss = mse + float(gripper_bce_weight) * grip_bce

    with torch.no_grad():
        pred_first = pred[:, 0]
        target_first = target[:, 0]
        metrics = {
            "loss": float(loss.detach().cpu()),
            "mse": float(mse.detach().cpu()),
            "grip_bce": float(grip_bce.detach().cpu()),
            "mae_xyz": float((pred_first[:, :3] - target_first[:, :3]).abs().mean().detach().cpu()),
            "mae_action": float((pred_first[:, :7] - target_first[:, :7]).abs().mean().detach().cpu()),
            "z_sign": float((torch.sign(pred_first[:, 2]) == torch.sign(target_first[:, 2])).float().mean().detach().cpu()),
            "g_sign": float((torch.sign(pred_first[:, 6]) == torch.sign(target_first[:, 6])).float().mean().detach().cpu()),
            "pred_g_mean": float(pred_first[:, 6].mean().detach().cpu()),
            "pred_z_mean": float(pred_first[:, 2].mean().detach().cpu()),
        }
    return loss, metrics


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, cfg: ModelConfig, device: torch.device, args) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    state_mean = torch.tensor(cfg.state_mean, dtype=torch.float32, device=device)
    state_std = torch.tensor(cfg.state_std, dtype=torch.float32, device=device)
    for batch_i, batch in enumerate(loader):
        if batch_i >= args.eval_batches:
            break
        state = batch["state"].to(device)
        target = batch["action_chunk"].to(device)
        image = batch.get("image")
        image2 = batch.get("image2")
        text_tokens = batch.get("text_tokens")
        if image is not None:
            image = image.to(device)
            image2 = image2.to(device)
        if text_tokens is not None:
            text_tokens = text_tokens.to(device)
        state_norm = (state - state_mean) / state_std
        pred = model(state_norm, image, image2, text_tokens)
        _, metrics = weighted_bc_loss(
            pred,
            target,
            args.z_pos_weight,
            args.large_xy_weight,
            args.gripper_mse_weight,
            args.gripper_bce_weight,
            args.xy_threshold,
            args.z_pos_threshold,
        )
        bs = int(state.shape[0])
        count += bs
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + float(v) * bs
    model.train()
    return {k: v / max(1, count) for k, v in totals.items()}


def save_checkpoint(path: Path, model: nn.Module, cfg: ModelConfig, step: int, optimizer: torch.optim.Optimizer | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "step": int(step),
        "config": asdict(cfg),
        "model": model.state_dict(),
    }
    if optimizer is not None:
        obj["optimizer"] = optimizer.state_dict()
    torch.save(obj, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--model-kind",
        choices=["state_mlp", "image_state_cnn", "image_state_text_cnn"],
        default="image_state_cnn",
    )
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--cnn-channels", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument("--eval-freq", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-key", default=DEFAULT_IMAGE_KEY)
    parser.add_argument("--image2-key", default=DEFAULT_IMAGE2_KEY)
    parser.add_argument("--state-key", default=DEFAULT_STATE_KEY)
    parser.add_argument("--action-key", default=DEFAULT_ACTION_KEY)
    parser.add_argument("--task-text", default=DEFAULT_TASK)
    parser.add_argument("--text-max-len", type=int, default=160)
    parser.add_argument("--text-embed-dim", type=int, default=128)
    parser.add_argument("--index-start", type=int, default=0, help="First dataset row to train on, useful for single-episode overfit.")
    parser.add_argument("--index-end", type=int, default=-1, help="Exclusive end row. Use 360-ish to drop unstable release tail.")
    parser.add_argument("--pos-label-scale", type=float, default=1.0, help="Scale action xyz labels before training.")
    parser.add_argument("--rot-label-scale", type=float, default=1.0, help="Scale action rotation labels before training.")
    parser.add_argument(
        "--use-progress",
        action="store_true",
        help="Append normalized dataset row index to state so one-demo BC can learn the task phase.",
    )
    parser.add_argument(
        "--progress-denominator",
        type=float,
        default=500.0,
        help="Normalize progress as row_index / denominator. Use 500 to match the LIBERO eval max_steps.",
    )
    parser.add_argument("--z-pos-weight", type=float, default=5.0)
    parser.add_argument("--large-xy-weight", type=float, default=3.0)
    parser.add_argument("--gripper-mse-weight", type=float, default=2.0)
    parser.add_argument("--gripper-bce-weight", type=float, default=0.2)
    parser.add_argument("--xy-threshold", type=float, default=0.05)
    parser.add_argument("--z-pos-threshold", type=float, default=0.01)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SmallBCDataset(
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        model_kind=args.model_kind,
        image_size=args.image_size,
        image_key=args.image_key,
        image2_key=args.image2_key,
        state_key=args.state_key,
        action_key=args.action_key,
        horizon=args.horizon,
        max_samples=args.max_samples if args.max_samples > 0 else None,
    )
    if args.pos_label_scale != 1.0:
        train_ds.actions[:, :3] *= float(args.pos_label_scale)
        train_ds.action_chunks[..., :3] *= float(args.pos_label_scale)
    if args.rot_label_scale != 1.0:
        train_ds.actions[:, 3:6] *= float(args.rot_label_scale)
        train_ds.action_chunks[..., 3:6] *= float(args.rot_label_scale)
    if args.use_progress:
        denom = max(1.0, float(args.progress_denominator))
        progress = (np.arange(len(train_ds.states), dtype=np.float32) / denom).reshape(-1, 1)
        train_ds.states = np.concatenate([train_ds.states, progress], axis=1).astype(np.float32)
    if args.index_start > 0 or args.index_end > 0:
        start = max(0, int(args.index_start))
        end = len(train_ds.actions) if args.index_end <= 0 else min(len(train_ds.actions), int(args.index_end))
        train_ds.indices = train_ds.indices[(train_ds.indices >= start) & (train_ds.indices < end)]
        if len(train_ds.indices) == 0:
            raise RuntimeError(f"No training rows left after --index-start {start} --index-end {end}")
    train_ds.task_text = args.task_text
    train_ds.text_max_len = int(args.text_max_len)
    train_ds.text_tokens = text_to_byte_tokens(args.task_text, args.text_max_len)
    selected_states = train_ds.states[train_ds.indices]
    selected_actions = train_ds.actions[train_ds.indices]
    state_mean = selected_states.mean(axis=0).astype(np.float32)
    state_std = selected_states.std(axis=0).astype(np.float32)
    state_std = np.maximum(state_std, 1e-6)
    action_abs = np.percentile(np.abs(selected_actions), 99.5, axis=0).astype(np.float32)
    action_clip = np.maximum(action_abs, np.asarray([0.08, 0.08, 0.08, 0.10, 0.10, 0.08, 1.0], dtype=np.float32))

    cfg = ModelConfig(
        model_kind=args.model_kind,
        state_dim=int(train_ds.states.shape[1]),
        action_dim=int(train_ds.actions.shape[1]),
        horizon=int(args.horizon),
        image_size=int(args.image_size),
        cnn_channels=int(args.cnn_channels),
        hidden_dim=int(args.hidden_dim),
        state_mean=state_mean.tolist(),
        state_std=state_std.tolist(),
        action_clip=action_clip.tolist(),
        image_key=args.image_key,
        image2_key=args.image2_key,
        state_key=args.state_key,
        action_key=args.action_key,
        task_text=args.task_text,
        text_max_len=int(args.text_max_len),
        text_embed_dim=int(args.text_embed_dim),
        pos_label_scale=float(args.pos_label_scale),
        rot_label_scale=float(args.rot_label_scale),
        index_start=int(args.index_start),
        index_end=int(args.index_end),
        use_progress=bool(args.use_progress),
        progress_denominator=float(args.progress_denominator),
    )

    cfg_path = args.output_dir / "config.json"
    cfg_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device(args.device)
    model = make_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))
    state_mean_t = torch.tensor(cfg.state_mean, dtype=torch.float32, device=device)
    state_std_t = torch.tensor(cfg.state_std, dtype=torch.float32, device=device)

    print("[dataset]", args.dataset_root, "repo_id=", args.repo_id)
    print("[dataset] total rows:", len(train_ds.actions), "training rows:", len(train_ds))
    print("[dataset] index range:", int(train_ds.indices.min()), int(train_ds.indices.max()))
    print("[dataset] gripper neg ratio:", float((selected_actions[:, 6] < -0.5).mean()))
    print("[dataset] z pos ratio:", float((selected_actions[:, 2] > 0.01).mean()))
    print("[dataset] big xy ratio:", float((np.linalg.norm(selected_actions[:, :2], axis=1) > 0.05).mean()))
    if args.use_progress:
        print("[dataset] progress range:", float(selected_states[:, -1].min()), float(selected_states[:, -1].max()))
    print("[model]", cfg)
    print("[output]", args.output_dir)

    model.train()
    step = 0
    t0 = time.time()
    running: dict[str, float] = {}
    running_count = 0
    data_iter = iter(loader)

    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        state = batch["state"].to(device, non_blocking=True)
        target = batch["action_chunk"].to(device, non_blocking=True)
        image = batch.get("image")
        image2 = batch.get("image2")
        text_tokens = batch.get("text_tokens")
        if image is not None:
            image = image.to(device, non_blocking=True)
            image2 = image2.to(device, non_blocking=True)
        if text_tokens is not None:
            text_tokens = text_tokens.to(device, non_blocking=True)
        state_norm = (state - state_mean_t) / state_std_t

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=bool(args.amp and device.type == "cuda")):
            pred = model(state_norm, image, image2, text_tokens)
            loss, metrics = weighted_bc_loss(
                pred,
                target,
                args.z_pos_weight,
                args.large_xy_weight,
                args.gripper_mse_weight,
                args.gripper_bce_weight,
                args.xy_threshold,
                args.z_pos_threshold,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()

        step += 1
        running_count += 1
        for k, v in metrics.items():
            running[k] = running.get(k, 0.0) + float(v)

        if step % args.log_freq == 0 or step == 1:
            avg = {k: v / max(1, running_count) for k, v in running.items()}
            elapsed = time.time() - t0
            msg = " ".join(
                [
                    f"step={step:05d}/{args.steps}",
                    f"loss={avg.get('loss', 0):.5f}",
                    f"mae_xyz={avg.get('mae_xyz', 0):.5f}",
                    f"mae_action={avg.get('mae_action', 0):.5f}",
                    f"z_sign={avg.get('z_sign', 0):.3f}",
                    f"g_sign={avg.get('g_sign', 0):.3f}",
                    f"pred_z={avg.get('pred_z_mean', 0):+.4f}",
                    f"pred_g={avg.get('pred_g_mean', 0):+.4f}",
                    f"{elapsed:.1f}s",
                ]
            )
            print("[train]", msg, flush=True)
            running.clear()
            running_count = 0

        if step % args.eval_freq == 0:
            eval_metrics = evaluate(model, loader, cfg, device, args)
            print(
                "[eval] "
                + " ".join(
                    [
                        f"step={step:05d}",
                        f"loss={eval_metrics.get('loss', 0):.5f}",
                        f"mae_xyz={eval_metrics.get('mae_xyz', 0):.5f}",
                        f"mae_action={eval_metrics.get('mae_action', 0):.5f}",
                        f"z_sign={eval_metrics.get('z_sign', 0):.3f}",
                        f"g_sign={eval_metrics.get('g_sign', 0):.3f}",
                        f"pred_z={eval_metrics.get('pred_z_mean', 0):+.4f}",
                        f"pred_g={eval_metrics.get('pred_g_mean', 0):+.4f}",
                    ]
                ),
                flush=True,
            )

        if step % args.save_freq == 0:
            save_checkpoint(args.output_dir / f"checkpoint_step_{step:06d}.pt", model, cfg, step, optimizer)
            save_checkpoint(args.output_dir / "last.pt", model, cfg, step, optimizer)
            print("[saved]", args.output_dir / "last.pt", flush=True)

    save_checkpoint(args.output_dir / "last.pt", model, cfg, step, optimizer)
    print("[done] saved:", args.output_dir / "last.pt")


if __name__ == "__main__":
    main()
