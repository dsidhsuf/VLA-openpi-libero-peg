#!/usr/bin/env python3
"""
HTTP policy server for train_small_bc_red_peg.py checkpoints.

It implements the same minimal interface used by the current LIBERO eval
client:
  GET  /health
  POST /reset
  POST /infer   -> {"action": [[... 7 dims ...], ...]}
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from train_small_bc_red_peg import (
    ImageStateCNNPolicy,
    ImageStateTextCNNPolicy,
    ModelConfig,
    StateMLPPolicy,
    text_to_byte_tokens,
)


SERVER_STATE: dict[str, Any] = {
    "model": None,
    "cfg": None,
    "device": None,
    "checkpoint": None,
    "clip_actions": True,
    "pos_clip": 0.08,
    "rot_xy_clip": 0.10,
    "rot_z_clip": 0.08,
    "gripper_clip": 1.0,
    "step": 0,
}


def decode_image_b64(value: str) -> np.ndarray:
    raw = base64.b64decode(value)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def image_to_tensor(value: np.ndarray, image_size: int) -> torch.Tensor:
    arr = np.asarray(value)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    if image_size > 0 and (t.shape[-2] != image_size or t.shape[-1] != image_size):
        t = F.interpolate(t.unsqueeze(0), size=(image_size, image_size), mode="bilinear", align_corners=False)[0]
    return (t - 0.5) / 0.5


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


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, ModelConfig, int]:
    obj = torch.load(path, map_location=device)
    cfg = ModelConfig(**obj["config"])
    model = make_model(cfg).to(device)
    model.load_state_dict(obj["model"], strict=True)
    model.eval()
    return model, cfg, int(obj.get("step", -1))


def get_policy_progress(payload: dict[str, Any], cfg: ModelConfig) -> float:
    if "policy_progress" in payload:
        progress = float(payload["policy_progress"])
    elif "progress" in payload:
        progress = float(payload["progress"])
    else:
        raw_step = payload.get("policy_step", payload.get("step", payload.get("timestep", SERVER_STATE["step"])))
        progress = float(raw_step) / max(1.0, float(cfg.progress_denominator))
    return float(np.clip(progress, 0.0, 2.0))


def normalize_state(
    state: np.ndarray,
    cfg: ModelConfig,
    device: torch.device,
    progress: float | None = None,
) -> torch.Tensor:
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if cfg.use_progress and state.shape[0] == cfg.state_dim - 1:
        if progress is None:
            progress = float(SERVER_STATE["step"]) / max(1.0, float(cfg.progress_denominator))
        state = np.concatenate([state, np.asarray([progress], dtype=np.float32)], axis=0)
    if state.shape[0] > cfg.state_dim:
        state = state[: cfg.state_dim]
    elif state.shape[0] < cfg.state_dim:
        padded = np.zeros(cfg.state_dim, dtype=np.float32)
        padded[: state.shape[0]] = state
        state = padded
    mean = np.asarray(cfg.state_mean, dtype=np.float32)
    std = np.asarray(cfg.state_std, dtype=np.float32)
    state = (state - mean) / np.maximum(std, 1e-6)
    return torch.from_numpy(state).float().unsqueeze(0).to(device)


@torch.inference_mode()
def infer_action(payload: dict[str, Any]) -> list[list[float]]:
    model: nn.Module = SERVER_STATE["model"]
    cfg: ModelConfig = SERVER_STATE["cfg"]
    device: torch.device = SERVER_STATE["device"]

    progress = get_policy_progress(payload, cfg) if cfg.use_progress else None
    state = normalize_state(payload.get(cfg.state_key) or payload.get("observation.state"), cfg, device, progress)
    image = None
    image2 = None
    text_tokens = None
    if cfg.model_kind != "state_mlp":
        image_payload = payload.get(cfg.image_key) or payload.get("observation.images.image")
        image2_payload = payload.get(cfg.image2_key) or payload.get("observation.images.image2")
        if image_payload is None or image2_payload is None:
            raise RuntimeError(f"{cfg.model_kind} checkpoint requires both image payloads")
        image = image_to_tensor(decode_image_b64(image_payload), cfg.image_size).unsqueeze(0).to(device)
        image2 = image_to_tensor(decode_image_b64(image2_payload), cfg.image_size).unsqueeze(0).to(device)
    if cfg.model_kind == "image_state_text_cnn":
        task_text = payload.get("task") or payload.get("language_instruction") or cfg.task_text
        text_tokens = text_to_byte_tokens(str(task_text), cfg.text_max_len).unsqueeze(0).to(device)

    pred = model(state, image, image2, text_tokens)[0].detach().cpu().numpy().astype(np.float32)
    if SERVER_STATE["clip_actions"]:
        clip = np.asarray(cfg.action_clip, dtype=np.float32)
        if clip.shape[0] >= pred.shape[1]:
            pred = np.clip(pred, -clip[: pred.shape[1]], clip[: pred.shape[1]])
        pred[:, :3] = np.clip(pred[:, :3], -float(SERVER_STATE["pos_clip"]), float(SERVER_STATE["pos_clip"]))
        pred[:, 3:5] = np.clip(
            pred[:, 3:5],
            -float(SERVER_STATE["rot_xy_clip"]),
            float(SERVER_STATE["rot_xy_clip"]),
        )
        pred[:, 5] = np.clip(pred[:, 5], -float(SERVER_STATE["rot_z_clip"]), float(SERVER_STATE["rot_z_clip"]))
        pred[:, 6] = np.clip(
            pred[:, 6],
            -float(SERVER_STATE["gripper_clip"]),
            float(SERVER_STATE["gripper_clip"]),
        )
    return pred[:, :7].tolist()


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj: Any, code: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            cfg: ModelConfig = SERVER_STATE["cfg"]
            self._send_json(
                {
                    "status": "ok",
                    "policy_path": str(SERVER_STATE["checkpoint"]),
                    "policy_type": "small_bc",
                    "model_kind": cfg.model_kind,
                    "adapter_loaded": False,
                    "state_dim": cfg.state_dim,
                    "action_dim": cfg.action_dim,
                    "return_chunk": True,
                    "chunk_len": cfg.horizon,
                    "image_size": cfg.image_size,
                    "task_text": cfg.task_text,
                    "input_keys": [cfg.image_key, cfg.image2_key, cfg.state_key],
                    "clip_actions": bool(SERVER_STATE["clip_actions"]),
                    "pos_clip": float(SERVER_STATE["pos_clip"]),
                    "rot_xy_clip": float(SERVER_STATE["rot_xy_clip"]),
                    "rot_z_clip": float(SERVER_STATE["rot_z_clip"]),
                    "gripper_clip": float(SERVER_STATE["gripper_clip"]),
                    "use_progress": bool(cfg.use_progress),
                    "progress_denominator": float(cfg.progress_denominator),
                    "server_step": int(SERVER_STATE["step"]),
                }
            )
            return
        self._send_json({"error": f"not found: {self.path}"}, code=404)

    def do_POST(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8") or "{}")

            if self.path == "/reset":
                SERVER_STATE["step"] = 0
                self._send_json({"status": "ok"})
                return
            if self.path == "/infer":
                action = infer_action(payload)
                SERVER_STATE["step"] = int(SERVER_STATE["step"]) + 1
                self._send_json({"action": action})
                return
            self._send_json({"error": f"not found: {self.path}"}, code=404)
        except Exception as exc:
            print("[server] exception during request:")
            traceback.print_exc()
            self._send_json({"error": str(exc)}, code=500)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable-action-clip", action="store_true")
    parser.add_argument("--pos-clip", type=float, default=0.08)
    parser.add_argument("--rot-xy-clip", type=float, default=0.10)
    parser.add_argument("--rot-z-clip", type=float, default=0.08)
    parser.add_argument("--gripper-clip", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, cfg, step = load_checkpoint(args.checkpoint, device)
    SERVER_STATE["model"] = model
    SERVER_STATE["cfg"] = cfg
    SERVER_STATE["device"] = device
    SERVER_STATE["checkpoint"] = args.checkpoint.resolve()
    SERVER_STATE["clip_actions"] = not args.disable_action_clip
    SERVER_STATE["pos_clip"] = float(args.pos_clip)
    SERVER_STATE["rot_xy_clip"] = float(args.rot_xy_clip)
    SERVER_STATE["rot_z_clip"] = float(args.rot_z_clip)
    SERVER_STATE["gripper_clip"] = float(args.gripper_clip)

    print("[server] checkpoint:", args.checkpoint)
    print("[server] step:", step)
    print("[server] model_kind:", cfg.model_kind)
    print("[server] horizon:", cfg.horizon)
    print("[server] device:", device)
    print(f"[server] listening on http://{args.host}:{args.port}")

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
