#!/usr/bin/env python3
"""
Compare a closed-loop replay state trace against the raw recorded trajectory.

For direct replay with --advance 1 and --exec_horizon 1, eval step N applies raw
action N and records the post-step observation. That should match raw trajectory
index N if the action/environment semantics are identical.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def vec3(row, prefix: str) -> np.ndarray:
    return np.asarray([row[f"{prefix}_x"], row[f"{prefix}_y"], row[f"{prefix}_z"]], dtype=np.float64)


def quat_cols(frame: pd.DataFrame, prefix: str):
    cols = [f"{prefix}_qw", f"{prefix}_qx", f"{prefix}_qy", f"{prefix}_qz"]
    if not all(c in frame.columns for c in cols):
        return None
    q = frame[cols].to_numpy(dtype=np.float64)
    return q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)


def quat_angle_error(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = q1 / (np.linalg.norm(q1, axis=1, keepdims=True) + 1e-12)
    q2 = q2 / (np.linalg.norm(q2, axis=1, keepdims=True) + 1e-12)
    dot = np.abs(np.sum(q1 * q2, axis=1))
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def parse_window(value: str) -> tuple[int, int] | None:
    value = str(value).strip()
    if not value:
        return None
    if ":" not in value:
        step = int(value)
        return step, step
    start_s, end_s = value.split(":", 1)
    start = int(start_s) if start_s.strip() else 0
    end = int(end_s) if end_s.strip() else 10**9
    return start, end


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-trace-csv", required=True, type=Path)
    parser.add_argument("--raw-episode", required=True, type=Path)
    parser.add_argument("--max-rows", type=int, default=500)
    parser.add_argument("--print-steps", default="0,25,50,75,100,112,125,144,160,200,250,300,366")
    parser.add_argument(
        "--print-window",
        default="",
        help="Optional inclusive window start:end to print as a compact table, e.g. 100:180.",
    )
    parser.add_argument("--print-every", type=int, default=1, help="Stride for --print-window.")
    args = parser.parse_args()

    trace = pd.read_csv(args.state_trace_csv)
    raw_path = args.raw_episode / "trajectory.npz"
    raw = np.load(raw_path, allow_pickle=True)

    n = min(len(trace), int(args.max_rows), raw["robot0_eef_pos"].shape[0])
    trace = trace.iloc[:n].copy()
    raw_eef = np.asarray(raw["robot0_eef_pos"][:n], dtype=np.float64)
    raw_peg = np.asarray(raw["earbud_1_pos"][:n], dtype=np.float64)
    raw_slot = np.asarray(raw["charging_slot_1_pos"][:n], dtype=np.float64)
    raw_grip = np.mean(np.abs(np.asarray(raw["robot0_gripper_qpos"][:n], dtype=np.float64)), axis=1)
    raw_eef_q = np.asarray(raw["robot0_eef_quat_wxyz"][:n], dtype=np.float64) if "robot0_eef_quat_wxyz" in raw else None
    raw_peg_q = np.asarray(raw["earbud_1_quat_wxyz"][:n], dtype=np.float64) if "earbud_1_quat_wxyz" in raw else None
    raw_slot_q = (
        np.asarray(raw["charging_slot_1_quat_wxyz"][:n], dtype=np.float64)
        if "charging_slot_1_quat_wxyz" in raw
        else None
    )

    rep_eef = trace[["eef_x", "eef_y", "eef_z"]].to_numpy(dtype=np.float64)
    rep_peg = trace[["peg_x", "peg_y", "peg_z"]].to_numpy(dtype=np.float64)
    rep_slot = trace[["slot_x", "slot_y", "slot_z"]].to_numpy(dtype=np.float64)
    rep_grip = trace["gripper_qpos_mean_abs"].to_numpy(dtype=np.float64)
    rep_eef_q = quat_cols(trace, "eef")
    rep_peg_q = quat_cols(trace, "peg")
    rep_slot_q = quat_cols(trace, "slot")

    eef_err = np.linalg.norm(rep_eef - raw_eef, axis=1)
    peg_err = np.linalg.norm(rep_peg - raw_peg, axis=1)
    rel_err = np.linalg.norm((rep_peg - rep_eef) - (raw_peg - raw_eef), axis=1)
    slot_err = np.linalg.norm(rep_slot - raw_slot, axis=1)
    grip_err = np.abs(rep_grip - raw_grip)
    eef_ang = quat_angle_error(rep_eef_q, raw_eef_q) if rep_eef_q is not None and raw_eef_q is not None else None
    peg_ang = quat_angle_error(rep_peg_q, raw_peg_q) if rep_peg_q is not None and raw_peg_q is not None else None
    slot_ang = quat_angle_error(rep_slot_q, raw_slot_q) if rep_slot_q is not None and raw_slot_q is not None else None

    print("[summary]")
    print(f"rows compared: {n}")
    print(f"eef_err mean/p50/max: {eef_err.mean():.6f} {np.median(eef_err):.6f} {eef_err.max():.6f}")
    print(f"peg_err mean/p50/max: {peg_err.mean():.6f} {np.median(peg_err):.6f} {peg_err.max():.6f}")
    print(f"obj_minus_eef_err mean/p50/max: {rel_err.mean():.6f} {np.median(rel_err):.6f} {rel_err.max():.6f}")
    print(f"slot_err mean/p50/max: {slot_err.mean():.6f} {np.median(slot_err):.6f} {slot_err.max():.6f}")
    print(f"gripper_qpos_abs_err mean/p50/max: {grip_err.mean():.6f} {np.median(grip_err):.6f} {grip_err.max():.6f}")
    if eef_ang is not None:
        eef_ang_deg = np.degrees(eef_ang)
        print(f"eef_ang_deg mean/p50/max: {eef_ang_deg.mean():.4f} {np.median(eef_ang_deg):.4f} {eef_ang_deg.max():.4f}")
    if peg_ang is not None:
        peg_ang_deg = np.degrees(peg_ang)
        print(f"peg_ang_deg mean/p50/max: {peg_ang_deg.mean():.4f} {np.median(peg_ang_deg):.4f} {peg_ang_deg.max():.4f}")
    if slot_ang is not None:
        slot_ang_deg = np.degrees(slot_ang)
        print(f"slot_ang_deg mean/p50/max: {slot_ang_deg.mean():.4f} {np.median(slot_ang_deg):.4f} {slot_ang_deg.max():.4f}")

    bad = np.where(eef_err > 0.01)[0]
    if bad.size:
        print(f"first eef_err > 1cm: step {int(bad[0])} err={eef_err[bad[0]]:.6f}")
    bad_rel = np.where(rel_err > 0.01)[0]
    if bad_rel.size:
        print(f"first obj_minus_eef_err > 1cm: step {int(bad_rel[0])} err={rel_err[bad_rel[0]]:.6f}")

    wanted = [int(x.strip()) for x in args.print_steps.split(",") if x.strip()]
    print("\n[steps]")
    for step in wanted:
        if step < 0 or step >= n:
            continue
        print(
            f"step={step:04d} "
            f"eef_err={eef_err[step]:.5f} peg_err={peg_err[step]:.5f} rel_err={rel_err[step]:.5f} "
            f"rep_eef={rep_eef[step].round(5).tolist()} raw_eef={raw_eef[step].round(5).tolist()} "
            f"rep_peg={rep_peg[step].round(5).tolist()} raw_peg={raw_peg[step].round(5).tolist()} "
            f"rep_grip={rep_grip[step]:.5f} raw_grip={raw_grip[step]:.5f}"
            + (f" eef_ang_deg={np.degrees(eef_ang[step]):.3f}" if eef_ang is not None else "")
            + (f" peg_ang_deg={np.degrees(peg_ang[step]):.3f}" if peg_ang is not None else "")
            + (f" slot_ang_deg={np.degrees(slot_ang[step]):.3f}" if slot_ang is not None else "")
        )

    window = parse_window(args.print_window)
    if window is not None:
        start, end = window
        start = max(0, start)
        end = min(n - 1, end)
        stride = max(1, int(args.print_every))
        print("\n[window]")
        print(
            "step eef_err peg_err rel_err "
            "dpeg_x dpeg_y dpeg_z "
            "dslot_x dslot_y dslot_z "
            "drel_x drel_y drel_z "
            "rep_grip raw_grip rep_slot_xy rep_slot_z raw_slot_xy raw_slot_z "
            "eef_ang_deg peg_ang_deg slot_ang_deg"
        )
        for step in range(start, end + 1, stride):
            dpeg = rep_peg[step] - raw_peg[step]
            dslot = rep_slot[step] - raw_slot[step]
            drel = (rep_peg[step] - rep_eef[step]) - (raw_peg[step] - raw_eef[step])
            rep_obj_slot = rep_peg[step] - rep_slot[step]
            raw_obj_slot = raw_peg[step] - raw_slot[step]
            print(
                f"{step:04d} "
                f"{eef_err[step]:.5f} {peg_err[step]:.5f} {rel_err[step]:.5f} "
                f"{dpeg[0]:+.5f} {dpeg[1]:+.5f} {dpeg[2]:+.5f} "
                f"{dslot[0]:+.5f} {dslot[1]:+.5f} {dslot[2]:+.5f} "
                f"{drel[0]:+.5f} {drel[1]:+.5f} {drel[2]:+.5f} "
                f"{rep_grip[step]:.5f} {raw_grip[step]:.5f} "
                f"{np.linalg.norm(rep_obj_slot[:2]):.5f} {rep_obj_slot[2]:+.5f} "
                f"{np.linalg.norm(raw_obj_slot[:2]):.5f} {raw_obj_slot[2]:+.5f}"
                + (f" {np.degrees(eef_ang[step]):.3f}" if eef_ang is not None else " nan")
                + (f" {np.degrees(peg_ang[step]):.3f}" if peg_ang is not None else " nan")
                + (f" {np.degrees(slot_ang[step]):.3f}" if slot_ang is not None else " nan")
            )


if __name__ == "__main__":
    main()
