#!/usr/bin/env python3
"""
Replay a raw demo as end-effector waypoints through the HTTP policy interface.

This is a diagnostic controller. It does not return the recorded xyz actions.
Instead, it reads the current EEF position from observation.state and servoes
toward the recorded raw EEF position at the matching time index.

If this succeeds while raw action replay fails, the benchmark state/contact
geometry is usable and the remaining mismatch is action scale/control semantics.
"""

from __future__ import annotations

import argparse
import base64
import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np


SERVER_STATE: dict[str, Any] = {
    "raw_episode": None,
    "eef_pos": None,
    "eef_quat": None,
    "raw_actions": None,
    "step": 0,
    "start_step": 0,
    "lookahead": 1,
    "chunk_len": 50,
    "advance": 1,
    "kp_pos": 4.0,
    "pos_clip": 0.08,
    "rot_scale": 1.0,
    "rot_servo_to_raw": False,
    "kp_rot": 2.0,
    "rot_clip": 0.10,
    "invert_gripper": False,
    "binarize_gripper": True,
    "gripper_delay": 0,
    "hold_open_until_error": 0.0,
    "target_offset_xyz": np.zeros(3, dtype=np.float32),
    "target_offset_start": -1,
    "target_offset_end": -1,
    "raw_peg_pos": None,
    "track_current_peg": False,
    "peg_relative_offset_xyz": np.zeros(3, dtype=np.float32),
    "peg_relative_start": -1,
    "peg_relative_end": -1,
    "hold_target_after": -1,
    "z_clip_after": -1,
    "z_clip_after_value": 0.08,
    "force_gripper_start": -1,
    "force_gripper_until": -1,
    "release_ramp_start": -1,
    "release_ramp_end": -1,
    "zero_motion_start": -1,
    "zero_motion_end": -1,
}


def load_json_request(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    if content_length <= 0:
        return {}
    body = handler.rfile.read(content_length)
    return json.loads(body.decode("utf-8"))


def get_current_eef(payload: dict[str, Any]) -> np.ndarray:
    state = payload.get("observation.state", None)
    if state is None:
        raise ValueError("Missing observation.state in request payload.")
    arr = np.asarray(state, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 3:
        raise ValueError(f"observation.state must contain eef xyz, got shape={arr.shape}")
    return arr[:3].astype(np.float32)


def get_current_eef_quat(payload: dict[str, Any]) -> np.ndarray | None:
    state = payload.get("observation.state", None)
    if state is None:
        return None
    arr = np.asarray(state, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 7:
        return None
    q = arr[3:7].astype(np.float32)
    return q / (np.linalg.norm(q) + 1e-12)


def get_current_peg(payload: dict[str, Any]) -> np.ndarray | None:
    peg = payload.get("observation.privileged.peg_pos", None)
    if peg is None:
        return None
    arr = np.asarray(peg, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 3:
        raise ValueError(f"observation.privileged.peg_pos must contain xyz, got shape={arr.shape}")
    return arr[:3].astype(np.float32)


def parse_xyz(value: str) -> np.ndarray:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected comma-separated xyz, e.g. 0.005,-0.002,-0.002")
    try:
        return np.asarray([float(p) for p in parts], dtype=np.float32)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def apply_target_offset(target: np.ndarray, idx: int) -> np.ndarray:
    start = int(SERVER_STATE["target_offset_start"])
    end = int(SERVER_STATE["target_offset_end"])
    if start >= 0 and end >= start and start <= idx <= end:
        return target + np.asarray(SERVER_STATE["target_offset_xyz"], dtype=np.float32)
    return target


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64)
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float32) / (float(np.dot(q, q)) + 1e-12)


def quat_error_axis_angle(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    current = np.asarray(current, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    current = current / (np.linalg.norm(current) + 1e-12)
    target = target / (np.linalg.norm(target) + 1e-12)
    if float(np.dot(current, target)) < 0.0:
        target = -target
    q_err = quat_mul(target, quat_inv(current))
    q_err = q_err / (np.linalg.norm(q_err) + 1e-12)
    w = float(np.clip(q_err[0], -1.0, 1.0))
    angle = 2.0 * np.arccos(w)
    if angle > np.pi:
        angle -= 2.0 * np.pi
    s = np.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-6:
        axis = q_err[1:4].astype(np.float32)
    else:
        axis = (q_err[1:4] / s).astype(np.float32)
    return (axis * float(angle)).astype(np.float32)


def in_window(idx: int, start_key: str, end_key: str) -> bool:
    start = int(SERVER_STATE[start_key])
    end = int(SERVER_STATE[end_key])
    return start >= 0 and end >= start and start <= idx <= end


def maybe_make_peg_relative_target(current_peg: np.ndarray | None, idx: int) -> np.ndarray | None:
    if not bool(SERVER_STATE["track_current_peg"]):
        return None
    if current_peg is None:
        return None
    if not in_window(idx, "peg_relative_start", "peg_relative_end"):
        return None

    eef_pos: np.ndarray = SERVER_STATE["eef_pos"]
    raw_peg_pos: np.ndarray = SERVER_STATE["raw_peg_pos"]
    raw_rel = eef_pos[idx].astype(np.float32) - raw_peg_pos[idx].astype(np.float32)
    offset = np.asarray(SERVER_STATE["peg_relative_offset_xyz"], dtype=np.float32)
    return current_peg.astype(np.float32) + raw_rel + offset


def make_action(
    current_eef: np.ndarray,
    target_idx: int,
    current_peg: np.ndarray | None = None,
    current_quat: np.ndarray | None = None,
) -> np.ndarray:
    eef_pos: np.ndarray = SERVER_STATE["eef_pos"]
    eef_quat: np.ndarray = SERVER_STATE["eef_quat"]
    raw_actions: np.ndarray = SERVER_STATE["raw_actions"]
    request_idx = int(target_idx)
    idx = int(np.clip(request_idx, 0, len(eef_pos) - 1))
    hold_target_after = int(SERVER_STATE["hold_target_after"])
    motion_idx = idx
    if hold_target_after >= 0 and request_idx >= hold_target_after:
        motion_idx = int(np.clip(hold_target_after, 0, len(eef_pos) - 1))

    target = maybe_make_peg_relative_target(current_peg, motion_idx)
    if target is None:
        target = apply_target_offset(eef_pos[motion_idx].astype(np.float32), motion_idx)

    action = np.zeros(7, dtype=np.float32)
    action[:3] = float(SERVER_STATE["kp_pos"]) * (target - current_eef)
    action[:3] = np.clip(action[:3], -float(SERVER_STATE["pos_clip"]), float(SERVER_STATE["pos_clip"]))
    z_clip_after = int(SERVER_STATE["z_clip_after"])
    if z_clip_after >= 0 and request_idx >= z_clip_after:
        z_clip_value = abs(float(SERVER_STATE["z_clip_after_value"]))
        action[2] = np.clip(action[2], -z_clip_value, z_clip_value)
    if bool(SERVER_STATE["rot_servo_to_raw"]) and current_quat is not None:
        target_quat = eef_quat[motion_idx].astype(np.float32)
        rot_vec = quat_error_axis_angle(current_quat, target_quat)
        action[3:6] = float(SERVER_STATE["kp_rot"]) * rot_vec
        action[3:6] = np.clip(action[3:6], -float(SERVER_STATE["rot_clip"]), float(SERVER_STATE["rot_clip"]))
    else:
        action[3:6] = raw_actions[motion_idx, 3:6] * float(SERVER_STATE["rot_scale"])

    grip_idx = request_idx - int(SERVER_STATE["gripper_delay"])
    grip_idx = int(np.clip(grip_idx, 0, len(raw_actions) - 1))
    action[6] = raw_actions[grip_idx, 6]
    hold_threshold = float(SERVER_STATE["hold_open_until_error"])
    if hold_threshold > 0.0 and action[6] > 0.0:
        # In this dataset +1 closes the gripper. If the EEF is still far from
        # the raw waypoint, hold it open to avoid an empty close before contact.
        err = float(np.linalg.norm(target - current_eef))
        if err > hold_threshold:
            action[6] = -1.0
    if SERVER_STATE["invert_gripper"]:
        action[6] *= -1.0
    if SERVER_STATE["binarize_gripper"]:
        action[6] = 1.0 if action[6] >= 0.0 else -1.0
    force_gripper_start = int(SERVER_STATE["force_gripper_start"])
    force_gripper_until = int(SERVER_STATE["force_gripper_until"])
    if (
        force_gripper_start >= 0
        and force_gripper_until >= force_gripper_start
        and force_gripper_start <= request_idx <= force_gripper_until
    ):
        action[6] = 1.0
    ramp_start = int(SERVER_STATE["release_ramp_start"])
    ramp_end = int(SERVER_STATE["release_ramp_end"])
    if ramp_start >= 0 and ramp_end > ramp_start and ramp_start <= request_idx <= ramp_end:
        alpha = float(request_idx - ramp_start) / float(ramp_end - ramp_start)
        action[6] = 1.0 - 2.0 * alpha
    zero_start = int(SERVER_STATE["zero_motion_start"])
    zero_end = int(SERVER_STATE["zero_motion_end"])
    if zero_start >= 0 and request_idx >= zero_start and (zero_end < zero_start or request_idx <= zero_end):
        action[:6] = 0.0
    action[3:5] = np.clip(action[3:5], -0.10, 0.10)
    action[5] = np.clip(action[5], -0.08, 0.08)
    action[6] = np.clip(action[6], -1.0, 1.0)
    return action


def infer_chunk(payload: dict[str, Any]) -> list[list[float]]:
    current_eef = get_current_eef(payload)
    current_quat = get_current_eef_quat(payload)
    current_peg = get_current_peg(payload)
    step = int(SERVER_STATE["step"])
    lookahead = int(SERVER_STATE["lookahead"])
    chunk_len = int(SERVER_STATE["chunk_len"])
    advance = int(SERVER_STATE["advance"])

    # Only the first action is exact because future states are unknown. This is
    # fine for exec_horizon=1; later chunk entries are useful only as fallback.
    actions = []
    for i in range(chunk_len):
        actions.append(
            make_action(current_eef, step + lookahead + i, current_peg=current_peg, current_quat=current_quat)
        )
    SERVER_STATE["step"] = step + max(1, advance)
    return np.asarray(actions, dtype=np.float32).tolist()


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
                    "policy_type": "replay_raw_eef_waypoints",
                    "raw_episode": str(SERVER_STATE["raw_episode"]),
                    "num_waypoints": int(len(SERVER_STATE["eef_pos"])),
                    "step": int(SERVER_STATE["step"]),
                    "start_step": int(SERVER_STATE["start_step"]),
                    "lookahead": int(SERVER_STATE["lookahead"]),
                    "return_chunk": True,
                    "chunk_len": int(SERVER_STATE["chunk_len"]),
                    "advance": int(SERVER_STATE["advance"]),
                    "kp_pos": float(SERVER_STATE["kp_pos"]),
                    "pos_clip": float(SERVER_STATE["pos_clip"]),
                    "rot_scale": float(SERVER_STATE["rot_scale"]),
                    "rot_servo_to_raw": bool(SERVER_STATE["rot_servo_to_raw"]),
                    "kp_rot": float(SERVER_STATE["kp_rot"]),
                    "rot_clip": float(SERVER_STATE["rot_clip"]),
                    "invert_gripper": bool(SERVER_STATE["invert_gripper"]),
                    "binarize_gripper": bool(SERVER_STATE["binarize_gripper"]),
                    "gripper_delay": int(SERVER_STATE["gripper_delay"]),
                    "hold_open_until_error": float(SERVER_STATE["hold_open_until_error"]),
                    "target_offset_xyz": np.asarray(SERVER_STATE["target_offset_xyz"]).tolist(),
                    "target_offset_start": int(SERVER_STATE["target_offset_start"]),
                    "target_offset_end": int(SERVER_STATE["target_offset_end"]),
                    "track_current_peg": bool(SERVER_STATE["track_current_peg"]),
                    "peg_relative_offset_xyz": np.asarray(SERVER_STATE["peg_relative_offset_xyz"]).tolist(),
                    "peg_relative_start": int(SERVER_STATE["peg_relative_start"]),
                    "peg_relative_end": int(SERVER_STATE["peg_relative_end"]),
                    "hold_target_after": int(SERVER_STATE["hold_target_after"]),
                    "z_clip_after": int(SERVER_STATE["z_clip_after"]),
                    "z_clip_after_value": float(SERVER_STATE["z_clip_after_value"]),
                    "force_gripper_start": int(SERVER_STATE["force_gripper_start"]),
                    "force_gripper_until": int(SERVER_STATE["force_gripper_until"]),
                    "release_ramp_start": int(SERVER_STATE["release_ramp_start"]),
                    "release_ramp_end": int(SERVER_STATE["release_ramp_end"]),
                    "zero_motion_start": int(SERVER_STATE["zero_motion_start"]),
                    "zero_motion_end": int(SERVER_STATE["zero_motion_end"]),
                    "adapter_loaded": False,
                }
            )
            return
        self._send_json({"error": f"not found: {self.path}"}, code=404)

    def do_POST(self) -> None:
        try:
            payload = load_json_request(self)
            if self.path == "/reset":
                SERVER_STATE["step"] = int(SERVER_STATE["start_step"])
                self._send_json({"status": "ok", "step": int(SERVER_STATE["step"])})
                return
            if self.path == "/infer":
                self._send_json({"action": infer_chunk(payload)})
                return
            self._send_json({"error": f"not found: {self.path}"}, code=404)
        except Exception as exc:
            print("[waypoint-server] exception:")
            traceback.print_exc()
            self._send_json({"error": str(exc)}, code=500)

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-episode", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--chunk-len", type=int, default=50)
    parser.add_argument("--advance", type=int, default=1)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--lookahead", type=int, default=1)
    parser.add_argument("--kp-pos", type=float, default=4.0)
    parser.add_argument("--pos-clip", type=float, default=0.08)
    parser.add_argument("--rot-scale", type=float, default=1.0)
    parser.add_argument(
        "--rot-servo-to-raw",
        action="store_true",
        help="Servo wrist orientation toward raw robot0_eef_quat_wxyz instead of replaying raw rotation actions.",
    )
    parser.add_argument("--kp-rot", type=float, default=2.0)
    parser.add_argument("--rot-clip", type=float, default=0.10)
    parser.add_argument("--invert-gripper", action="store_true")
    parser.add_argument("--no-binarize-gripper", action="store_true")
    parser.add_argument(
        "--gripper-delay",
        type=int,
        default=0,
        help="Use raw gripper command from this many steps earlier; positive values delay closing.",
    )
    parser.add_argument(
        "--hold-open-until-error",
        type=float,
        default=0.0,
        help="If >0, keep gripper open when a close command is requested while EEF waypoint error exceeds this distance in meters.",
    )
    parser.add_argument(
        "--target-offset-xyz",
        type=parse_xyz,
        default=np.zeros(3, dtype=np.float32),
        help="Comma-separated xyz offset added to raw EEF targets only inside [target-offset-start, target-offset-end].",
    )
    parser.add_argument("--target-offset-start", type=int, default=-1)
    parser.add_argument("--target-offset-end", type=int, default=-1)
    parser.add_argument(
        "--track-current-peg",
        action="store_true",
        help="Inside the peg-relative window, target current_peg + raw(eef-peg) instead of raw world EEF.",
    )
    parser.add_argument(
        "--peg-relative-offset-xyz",
        type=parse_xyz,
        default=np.zeros(3, dtype=np.float32),
        help="Extra xyz offset added to current_peg + raw(eef-peg) inside the peg-relative window.",
    )
    parser.add_argument("--peg-relative-start", type=int, default=-1)
    parser.add_argument("--peg-relative-end", type=int, default=-1)
    parser.add_argument(
        "--hold-target-after",
        type=int,
        default=-1,
        help="If >=0, keep servoing to this raw waypoint index after the request index reaches it.",
    )
    parser.add_argument(
        "--z-clip-after",
        type=int,
        default=-1,
        help="If >=0, clamp only the z action to +/- z-clip-after-value from this raw waypoint index onward.",
    )
    parser.add_argument(
        "--z-clip-after-value",
        type=float,
        default=0.08,
        help="Late-stage z action clip used after --z-clip-after.",
    )
    parser.add_argument(
        "--force-gripper-start",
        type=int,
        default=-1,
        help="If set with --force-gripper-until, start forcing +1 gripper command at this raw waypoint index.",
    )
    parser.add_argument(
        "--force-gripper-until",
        type=int,
        default=-1,
        help="If set with --force-gripper-start, force +1 gripper command through this raw waypoint index.",
    )
    parser.add_argument(
        "--release-ramp-start",
        type=int,
        default=-1,
        help="If set with --release-ramp-end, linearly ramp gripper from +1 to -1 over this raw waypoint window.",
    )
    parser.add_argument("--release-ramp-end", type=int, default=-1)
    parser.add_argument(
        "--zero-motion-start",
        type=int,
        default=-1,
        help="If >=0, set xyz/rot actions to zero from this request index onward while preserving gripper commands.",
    )
    parser.add_argument(
        "--zero-motion-end",
        type=int,
        default=-1,
        help="Optional inclusive end index for --zero-motion-start. Default means no end.",
    )
    args = parser.parse_args()

    raw_episode = args.raw_episode.resolve()
    traj_path = raw_episode / "trajectory.npz"
    raw = np.load(traj_path, allow_pickle=True)
    eef_pos = np.asarray(raw["robot0_eef_pos"], dtype=np.float32)
    eef_quat = np.asarray(raw["robot0_eef_quat_wxyz"], dtype=np.float32)
    raw_peg_pos = np.asarray(raw["earbud_1_pos"], dtype=np.float32)
    raw_actions = np.asarray(raw["action"], dtype=np.float32)
    start_step = int(np.clip(args.start_step, 0, max(0, len(eef_pos) - 1)))

    SERVER_STATE.update(
        {
            "raw_episode": raw_episode,
            "eef_pos": eef_pos,
            "eef_quat": eef_quat,
            "raw_peg_pos": raw_peg_pos,
            "raw_actions": raw_actions,
            "step": start_step,
            "start_step": start_step,
            "lookahead": int(args.lookahead),
            "chunk_len": int(args.chunk_len),
            "advance": int(args.advance),
            "kp_pos": float(args.kp_pos),
            "pos_clip": float(args.pos_clip),
            "rot_scale": float(args.rot_scale),
            "rot_servo_to_raw": bool(args.rot_servo_to_raw),
            "kp_rot": float(args.kp_rot),
            "rot_clip": float(args.rot_clip),
            "invert_gripper": bool(args.invert_gripper),
            "binarize_gripper": not bool(args.no_binarize_gripper),
            "gripper_delay": int(args.gripper_delay),
            "hold_open_until_error": float(args.hold_open_until_error),
            "target_offset_xyz": np.asarray(args.target_offset_xyz, dtype=np.float32),
            "target_offset_start": int(args.target_offset_start),
            "target_offset_end": int(args.target_offset_end),
            "track_current_peg": bool(args.track_current_peg),
            "peg_relative_offset_xyz": np.asarray(args.peg_relative_offset_xyz, dtype=np.float32),
            "peg_relative_start": int(args.peg_relative_start),
            "peg_relative_end": int(args.peg_relative_end),
            "hold_target_after": int(args.hold_target_after),
            "z_clip_after": int(args.z_clip_after),
            "z_clip_after_value": float(args.z_clip_after_value),
            "force_gripper_start": int(args.force_gripper_start),
            "force_gripper_until": int(args.force_gripper_until),
            "release_ramp_start": int(args.release_ramp_start),
            "release_ramp_end": int(args.release_ramp_end),
            "zero_motion_start": int(args.zero_motion_start),
            "zero_motion_end": int(args.zero_motion_end),
        }
    )

    print("[waypoint-server] raw_episode:", raw_episode)
    print("[waypoint-server] waypoints:", eef_pos.shape)
    print("[waypoint-server] start_step:", start_step)
    print("[waypoint-server] lookahead:", args.lookahead)
    print("[waypoint-server] kp_pos:", args.kp_pos)
    print("[waypoint-server] pos_clip:", args.pos_clip)
    print("[waypoint-server] rot_servo_to_raw:", args.rot_servo_to_raw)
    print("[waypoint-server] kp_rot:", args.kp_rot)
    print("[waypoint-server] rot_clip:", args.rot_clip)
    print("[waypoint-server] gripper_delay:", args.gripper_delay)
    print("[waypoint-server] hold_open_until_error:", args.hold_open_until_error)
    print("[waypoint-server] target_offset_xyz:", np.asarray(args.target_offset_xyz).tolist())
    print("[waypoint-server] target_offset_window:", (args.target_offset_start, args.target_offset_end))
    print("[waypoint-server] track_current_peg:", args.track_current_peg)
    print("[waypoint-server] peg_relative_offset_xyz:", np.asarray(args.peg_relative_offset_xyz).tolist())
    print("[waypoint-server] peg_relative_window:", (args.peg_relative_start, args.peg_relative_end))
    print("[waypoint-server] hold_target_after:", args.hold_target_after)
    print("[waypoint-server] z_clip_after:", args.z_clip_after)
    print("[waypoint-server] z_clip_after_value:", args.z_clip_after_value)
    print("[waypoint-server] force_gripper_window:", (args.force_gripper_start, args.force_gripper_until))
    print("[waypoint-server] release_ramp_window:", (args.release_ramp_start, args.release_ramp_end))
    print("[waypoint-server] zero_motion_window:", (args.zero_motion_start, args.zero_motion_end))
    print(f"[waypoint-server] listening on http://{args.host}:{args.port}")
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
