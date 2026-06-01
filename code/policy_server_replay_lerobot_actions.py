#!/usr/bin/env python3
"""
Replay actions from a LeRobot dataset through the same HTTP policy interface.

This is a diagnostic server, not a learned policy. It answers an important
question before more training:

  "Can the recorded LeRobot actions solve the benchmark when replayed through
   the current eval client and simulator?"

If this fails on the original successful demonstration, check action sign,
action scaling, image/state conversion, or benchmark initial state mismatch
before blaming the model.
"""

from __future__ import annotations

import argparse
import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np


SERVER_STATE: dict[str, Any] = {
    "actions": None,
    "step": 0,
    "start_step": 0,
    "chunk_len": 50,
    "advance": 1,
    "dataset_root": None,
    "repo_id": None,
    "episode_index": None,
    "invert_gripper": False,
    "binarize_gripper": False,
    "pos_scale": 1.0,
    "rot_scale": 1.0,
    "gripper_scale": 1.0,
    "clip_actions": True,
}


def load_actions_from_lerobot(dataset_root: Path, repo_id: str, episode_index: int, action_key: str) -> np.ndarray:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    try:
        ds = LeRobotDataset(repo_id=repo_id, root=str(dataset_root), return_uint8=True)
    except TypeError:
        ds = LeRobotDataset(repo_id=repo_id, root=str(dataset_root))

    actions = []
    for i in range(len(ds)):
        sample = ds[i]
        ep = sample.get("episode_index", None)
        if ep is not None:
            ep_arr = np.asarray(ep)
            ep_id = int(ep_arr.reshape(-1)[0])
            if ep_id != int(episode_index):
                continue
        action = np.asarray(sample[action_key], dtype=np.float32).reshape(-1)[:7]
        actions.append(action)
    if not actions:
        raise RuntimeError(f"No actions found for episode_index={episode_index} in {dataset_root}")
    return np.asarray(actions, dtype=np.float32)


def process_actions(actions: np.ndarray) -> np.ndarray:
    out = np.asarray(actions, dtype=np.float32).copy()
    out[:, :3] *= float(SERVER_STATE["pos_scale"])
    out[:, 3:6] *= float(SERVER_STATE["rot_scale"])
    out[:, 6] *= float(SERVER_STATE["gripper_scale"])
    if SERVER_STATE["invert_gripper"]:
        out[:, 6] *= -1.0
    if SERVER_STATE["binarize_gripper"]:
        out[:, 6] = np.where(out[:, 6] >= 0.0, 1.0, -1.0)
    if SERVER_STATE["clip_actions"]:
        out[:, :3] = np.clip(out[:, :3], -0.08, 0.08)
        out[:, 3:5] = np.clip(out[:, 3:5], -0.10, 0.10)
        out[:, 5] = np.clip(out[:, 5], -0.08, 0.08)
        out[:, 6] = np.clip(out[:, 6], -1.0, 1.0)
    return out


def infer_chunk() -> list[list[float]]:
    actions: np.ndarray = SERVER_STATE["actions"]
    step = int(SERVER_STATE["step"])
    chunk_len = int(SERVER_STATE["chunk_len"])
    advance = int(SERVER_STATE["advance"])

    if step >= len(actions):
        chunk = np.repeat(actions[-1][None, :], chunk_len, axis=0)
    else:
        end = min(len(actions), step + chunk_len)
        chunk = actions[step:end]
        if len(chunk) < chunk_len:
            pad = np.repeat(chunk[-1][None, :], chunk_len - len(chunk), axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)

    SERVER_STATE["step"] = step + max(1, advance)
    return process_actions(chunk).tolist()


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
            self._send_json(
                {
                    "status": "ok",
                    "policy_type": "replay_lerobot_actions",
                    "policy_path": str(SERVER_STATE["dataset_root"]),
                    "repo_id": SERVER_STATE["repo_id"],
                    "episode_index": SERVER_STATE["episode_index"],
                    "num_actions": int(len(SERVER_STATE["actions"])),
                    "step": int(SERVER_STATE["step"]),
                    "start_step": int(SERVER_STATE["start_step"]),
                    "return_chunk": True,
                    "chunk_len": int(SERVER_STATE["chunk_len"]),
                    "advance": int(SERVER_STATE["advance"]),
                    "invert_gripper": bool(SERVER_STATE["invert_gripper"]),
                    "binarize_gripper": bool(SERVER_STATE["binarize_gripper"]),
                    "pos_scale": float(SERVER_STATE["pos_scale"]),
                    "rot_scale": float(SERVER_STATE["rot_scale"]),
                    "gripper_scale": float(SERVER_STATE["gripper_scale"]),
                    "clip_actions": bool(SERVER_STATE["clip_actions"]),
                    "adapter_loaded": False,
                }
            )
            return
        self._send_json({"error": f"not found: {self.path}"}, code=404)

    def do_POST(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length > 0:
                self.rfile.read(content_length)
            if self.path == "/reset":
                SERVER_STATE["step"] = int(SERVER_STATE["start_step"])
                self._send_json({"status": "ok", "step": int(SERVER_STATE["step"])})
                return
            if self.path == "/infer":
                self._send_json({"action": infer_chunk()})
                return
            self._send_json({"error": f"not found: {self.path}"}, code=404)
        except Exception as exc:
            print("[replay-server] exception:")
            traceback.print_exc()
            self._send_json({"error": str(exc)}, code=500)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--chunk-len", type=int, default=50)
    parser.add_argument("--advance", type=int, default=5, help="How many env steps the eval client executes per /infer.")
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="Start replay from this action index. Useful for checking demo/env time alignment.",
    )
    parser.add_argument("--invert-gripper", action="store_true")
    parser.add_argument("--binarize-gripper", action="store_true")
    parser.add_argument("--pos-scale", type=float, default=1.0, help="Multiply xyz actions before returning them.")
    parser.add_argument("--rot-scale", type=float, default=1.0, help="Multiply rotation actions before returning them.")
    parser.add_argument("--gripper-scale", type=float, default=1.0, help="Multiply gripper action before returning it.")
    parser.add_argument("--no-server-clip", action="store_true", help="Do not clip actions inside the replay server.")
    args = parser.parse_args()

    actions = load_actions_from_lerobot(args.dataset_root, args.repo_id, args.episode_index, args.action_key)
    SERVER_STATE.update(
        {
            "actions": actions,
            "step": int(np.clip(args.start_step, 0, max(0, len(actions) - 1))),
            "start_step": int(np.clip(args.start_step, 0, max(0, len(actions) - 1))),
            "chunk_len": int(args.chunk_len),
            "advance": int(args.advance),
            "dataset_root": args.dataset_root.resolve(),
            "repo_id": args.repo_id,
            "episode_index": int(args.episode_index),
            "invert_gripper": bool(args.invert_gripper),
            "binarize_gripper": bool(args.binarize_gripper),
            "pos_scale": float(args.pos_scale),
            "rot_scale": float(args.rot_scale),
            "gripper_scale": float(args.gripper_scale),
            "clip_actions": not bool(args.no_server_clip),
        }
    )

    print("[replay-server] dataset:", args.dataset_root)
    print("[replay-server] repo_id:", args.repo_id)
    print("[replay-server] episode_index:", args.episode_index)
    print("[replay-server] actions:", actions.shape)
    print("[replay-server] start_step:", SERVER_STATE["start_step"])
    print("[replay-server] pos_scale:", SERVER_STATE["pos_scale"])
    print("[replay-server] rot_scale:", SERVER_STATE["rot_scale"])
    print("[replay-server] gripper_scale:", SERVER_STATE["gripper_scale"])
    print("[replay-server] clip_actions:", SERVER_STATE["clip_actions"])
    print("[replay-server] first action:", actions[0].tolist())
    print("[replay-server] last action:", actions[-1].tolist())
    print(f"[replay-server] listening on http://{args.host}:{args.port}")

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
