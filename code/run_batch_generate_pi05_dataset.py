import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Batch generate LIBERO dataset: easy/medium/hard each N episodes.")
    parser.add_argument("--script-path", type=str, default="./full_chain_pick_random_wrist_align_descend_release_lerobot_pi05_3cam_1min.py")
    parser.add_argument("--out-root", type=str, default="./libero_lerobot_dataset_pi05_3cam_1min")
    parser.add_argument("--python-bin", type=str, default=sys.executable)

    parser.add_argument("--episodes-per-level", type=int, default=20)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--seed-easy", type=int, default=0)
    parser.add_argument("--seed-medium", type=int, default=1000)
    parser.add_argument("--seed-hard", type=int, default=2000)

    parser.add_argument("--camera-names", type=str, default="agentview,frontview,robot0_eye_in_hand")
    parser.add_argument("--camera-size", type=int, default=768)
    parser.add_argument("--video-fps", type=int, default=0)
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument("--target-video-duration-sec", type=float, default=60.0)
    parser.add_argument("--max-video-frames", type=int, default=0)

    parser.add_argument("--random-yaw-min-deg", type=float, default=-90.0)
    parser.add_argument("--random-yaw-max-deg", type=float, default=90.0)
    parser.add_argument("--flat-rest-prob", type=float, default=0.5)

    parser.add_argument("--bddl-base-dir", type=str, default=None, help="Optional. Override BDDL base dir in cloud env.")
    args = parser.parse_args()

    script_path = Path(args.script_path).as_posix()
    out_root = Path(args.out_root).as_posix()

    levels = [
        ("easy", args.seed_easy),
        ("medium", args.seed_medium),
        ("hard", args.seed_hard),
    ]

    for level, seed in levels:
        cmd = [
            args.python_bin,
            script_path,
            "--level", level,
            "--episodes", str(args.episodes_per_level),
            "--seed", str(seed),
            "--seed-step", str(args.seed_step),
            "--out-root", out_root,
            "--camera-names", args.camera_names,
            "--camera-size", str(args.camera_size),
            "--video-fps", str(args.video_fps),
            "--playback-speed", str(args.playback_speed),
            "--target-video-duration-sec", str(args.target_video_duration_sec),
            "--max-video-frames", str(args.max_video_frames),
            "--random-yaw-min-deg", str(args.random_yaw_min_deg),
            "--random-yaw-max-deg", str(args.random_yaw_max_deg),
            "--flat-rest-prob", str(args.flat_rest_prob),
        ]
        if args.bddl_base_dir:
            cmd += ["--bddl-base-dir", args.bddl_base_dir]

        run_cmd(cmd)

    total = 3 * args.episodes_per_level
    print(f"\nDone. Generated {total} episodes (easy/medium/hard each {args.episodes_per_level}).")
    print(f"Output root: {out_root}")


if __name__ == "__main__":
    main()