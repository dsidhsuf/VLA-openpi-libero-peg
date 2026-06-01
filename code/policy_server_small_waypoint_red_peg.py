#!/usr/bin/env python3
"""
HTTP server for a small waypoint-target VLA-style policy.

The checkpoint is trained with train_small_bc_red_peg.py on labels where:
  pred[:3] = absolute target EEF xyz
  pred[3:6] = rotation action
  pred[6] = gripper command

This server converts predicted target xyz into a stable servo action:
  action[:3] = kp_pos * (pred_target_xyz - current_eef_xyz)
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
    "step": 0,
    "kp_pos": 12.0,
    "pos_clip": 0.32,
    "rot_scale": 1.0,
    "rot_xy_clip": 0.10,
    "rot_z_clip": 0.08,
    "gripper_clip": 1.0,
    "binarize_gripper": True,
    "hold_open_until_error": 0.0,
    "force_gripper_start": -1,
    "force_gripper_until": -1,
    "z_clip_after": -1,
    "z_clip_after_value": 0.025,
    "return_target_debug": True,
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


def get_current_eef(payload: dict[str, Any]) -> np.ndarray:
    state = payload.get("observation.state")
    if state is None:
        raise ValueError("Missing observation.state in request payload.")
    arr = np.asarray(state, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 3:
        raise ValueError(f"observation.state must contain eef xyz, got shape={arr.shape}")
    return arr[:3].astype(np.float32)


def get_request_step(payload: dict[str, Any]) -> int:
    raw = payload.get("policy_step", payload.get("step", payload.get("timestep", SERVER_STATE["step"])))
    return int(raw)


def get_policy_progress(payload: dict[str, Any], cfg: ModelConfig) -> float:
    if "policy_progress" in payload:
        progress = float(payload["policy_progress"])
    elif "progress" in payload:
        progress = float(payload["progress"])
    else:
        progress = float(get_request_step(payload)) / max(1.0, float(cfg.progress_denominator))
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
            progress = float(get_request_step({})) / max(1.0, float(cfg.progress_denominator))
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
def predict_waypoint(payload: dict[str, Any]) -> np.ndarray:
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
    if pred.shape[-1] < 7:
        raise RuntimeError(f"Waypoint checkpoint must output at least 7 dims, got {pred.shape}")
    return pred[:, :7]


def waypoint_to_action(row: np.ndarray, current_eef: np.ndarray, request_step: int) -> np.ndarray:
    target = np.asarray(row[:3], dtype=np.float32)
    action = np.zeros(7, dtype=np.float32)
    action[:3] = float(SERVER_STATE["kp_pos"]) * (target - current_eef)
    action[:3] = np.clip(action[:3], -float(SERVER_STATE["pos_clip"]), float(SERVER_STATE["pos_clip"]))

    z_clip_after = int(SERVER_STATE["z_clip_after"])
    if z_clip_after >= 0 and request_step >= z_clip_after:
        z_clip_value = abs(float(SERVER_STATE["z_clip_after_value"]))
        action[2] = np.clip(action[2], -z_clip_value, z_clip_value)

    action[3:6] = np.asarray(row[3:6], dtype=np.float32) * float(SERVER_STATE["rot_scale"])
    action[3:5] = np.clip(
        action[3:5],
        -float(SERVER_STATE["rot_xy_clip"]),
        float(SERVER_STATE["rot_xy_clip"]),
    )
    action[5] = np.clip(action[5], -float(SERVER_STATE["rot_z_clip"]), float(SERVER_STATE["rot_z_clip"]))

    grip = float(np.clip(row[6], -float(SERVER_STATE["gripper_clip"]), float(SERVER_STATE["gripper_clip"])))
    hold_threshold = float(SERVER_STATE["hold_open_until_error"])
    if hold_threshold > 0.0 and grip > 0.0:
        err = float(np.linalg.norm(target - current_eef))
        if err > hold_threshold:
            grip = -1.0
    if bool(SERVER_STATE["binarize_gripper"]):
        grip = 1.0 if grip >= 0.0 else -1.0

    force_start = int(SERVER_STATE["force_gripper_start"])
    force_until = int(SERVER_STATE["force_gripper_until"])
    if force_start >= 0 and force_until >= force_start and force_start <= request_step <= force_until:
        grip = 1.0
    action[6] = grip
    return action


def infer_chunk(payload: dict[str, Any]) -> dict[str, Any]:
    current_eef = get_current_eef(payload)
    request_step = get_request_step(payload)
    pred = predict_waypoint(payload)

    actions = []
    for i in range(pred.shape[0]):
        actions.append(waypoint_to_action(pred[i], current_eef, request_step + i))
    action_np = np.asarray(actions, dtype=np.float32)
    SERVER_STATE["step"] = request_step + 1

    out: dict[str, Any] = {"action": action_np.tolist()}
    if bool(SERVER_STATE["return_target_debug"]):
        out["target"] = pred[:, :3].astype(np.float32).tolist()
        out["model_output"] = pred.astype(np.float32).tolist()
    return out


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
                    "policy_type": "small_waypoint_target",
                    "model_kind": cfg.model_kind,
                    "adapter_loaded": False,
                    "state_dim": cfg.state_dim,
                    "action_dim": cfg.action_dim,
                    "return_chunk": True,
                    "chunk_len": cfg.horizon,
                    "image_size": cfg.image_size,
                    "task_text": cfg.task_text,
                    "input_keys": [cfg.image_key, cfg.image2_key, cfg.state_key],
                    "label_semantics": "model_output[:3]=absolute_target_eef_xyz; server servoes to target",
                    "kp_pos": float(SERVER_STATE["kp_pos"]),
                    "pos_clip": float(SERVER_STATE["pos_clip"]),
                    "rot_scale": float(SERVER_STATE["rot_scale"]),
                    "rot_xy_clip": float(SERVER_STATE["rot_xy_clip"]),
                    "rot_z_clip": float(SERVER_STATE["rot_z_clip"]),
                    "gripper_clip": float(SERVER_STATE["gripper_clip"]),
                    "binarize_gripper": bool(SERVER_STATE["binarize_gripper"]),
                    "hold_open_until_error": float(SERVER_STATE["hold_open_until_error"]),
                    "force_gripper_start": int(SERVER_STATE["force_gripper_start"]),
                    "force_gripper_until": int(SERVER_STATE["force_gripper_until"]),
                    "z_clip_after": int(SERVER_STATE["z_clip_after"]),
                    "z_clip_after_value": float(SERVER_STATE["z_clip_after_value"]),
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
                self._send_json({"status": "ok", "step": int(SERVER_STATE["step"])})
                return
            if self.path == "/infer":
                self._send_json(infer_chunk(payload))
                return
            self._send_json({"error": f"not found: {self.path}"}, code=404)
        except Exception as exc:
            print("[waypoint-small-server] exception during request:")
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
    parser.add_argument("--kp-pos", type=float, default=12.0)
    parser.add_argument("--pos-clip", type=float, default=0.32)
    parser.add_argument("--rot-scale", type=float, default=1.0)
    parser.add_argument("--rot-xy-clip", type=float, default=0.10)
    parser.add_argument("--rot-z-clip", type=float, default=0.08)
    parser.add_argument("--gripper-clip", type=float, default=1.0)
    parser.add_argument("--no-binarize-gripper", action="store_true")
    parser.add_argument("--hold-open-until-error", type=float, default=0.0)
    parser.add_argument("--force-gripper-start", type=int, default=-1)
    parser.add_argument("--force-gripper-until", type=int, default=-1)
    parser.add_argument("--z-clip-after", type=int, default=-1)
    parser.add_argument("--z-clip-after-value", type=float, default=0.025)
    parser.add_argument("--no-target-debug", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, cfg, step = load_checkpoint(args.checkpoint, device)
    SERVER_STATE["model"] = model
    SERVER_STATE["cfg"] = cfg
    SERVER_STATE["device"] = device
    SERVER_STATE["checkpoint"] = args.checkpoint.resolve()
    SERVER_STATE["kp_pos"] = float(args.kp_pos)
    SERVER_STATE["pos_clip"] = float(args.pos_clip)
    SERVER_STATE["rot_scale"] = float(args.rot_scale)
    SERVER_STATE["rot_xy_clip"] = float(args.rot_xy_clip)
    SERVER_STATE["rot_z_clip"] = float(args.rot_z_clip)
    SERVER_STATE["gripper_clip"] = float(args.gripper_clip)
    SERVER_STATE["binarize_gripper"] = not bool(args.no_binarize_gripper)
    SERVER_STATE["hold_open_until_error"] = float(args.hold_open_until_error)
    SERVER_STATE["force_gripper_start"] = int(args.force_gripper_start)
    SERVER_STATE["force_gripper_until"] = int(args.force_gripper_until)
    SERVER_STATE["z_clip_after"] = int(args.z_clip_after)
    SERVER_STATE["z_clip_after_value"] = float(args.z_clip_after_value)
    SERVER_STATE["return_target_debug"] = not bool(args.no_target_debug)

    print("[waypoint-small-server] checkpoint:", args.checkpoint)
    print("[waypoint-small-server] step:", step)
    print("[waypoint-small-server] model_kind:", cfg.model_kind)
    print("[waypoint-small-server] horizon:", cfg.horizon)
    print("[waypoint-small-server] use_progress:", cfg.use_progress)
    print("[waypoint-small-server] kp_pos:", args.kp_pos)
    print("[waypoint-small-server] pos_clip:", args.pos_clip)
    print("[waypoint-small-server] device:", device)
    print(f"[waypoint-small-server] listening on http://{args.host}:{args.port}")

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
