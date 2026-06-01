import argparse
import json
from pathlib import Path

from full_chain_release_random_wrist_align_with_demo import rollout


def run_one(level: str, seed: int, demo_dir: Path, yaw_min: float, yaw_max: float, flat_rest_prob: float):
    stem = f"earbud_insert_{level}_ep{seed:04d}"
    npz_path = demo_dir / f"{stem}.npz"
    json_path = demo_dir / f"{stem}.json"

    # remove stale files for same seed
    if npz_path.exists():
        npz_path.unlink()
    if json_path.exists():
        json_path.unlink()

    print(f"\n===== level={level} seed={seed} =====")
    rollout(
        level=level,
        seed=seed,
        random_yaw_min_deg=yaw_min,
        random_yaw_max_deg=yaw_max,
        flat_rest_prob=flat_rest_prob,
        save_demo=True,
        demo_dir=str(demo_dir),
    )

    if not json_path.exists():
        print(f"[warn] meta file not found: {json_path}")
        return False, None

    meta = json.loads(json_path.read_text(encoding="utf-8"))
    success = bool(meta.get("success", False))
    if not success:
        print(f"[drop failed demo] {json_path.name}")
        try:
            json_path.unlink()
        except FileNotFoundError:
            pass
        try:
            npz_path.unlink()
        except FileNotFoundError:
            pass
        return False, meta

    print(f"[keep success] {json_path.name}")
    return True, meta


def count_successes(demo_dir: Path, level: str) -> int:
    n = 0
    for fp in demo_dir.glob(f"earbud_insert_{level}_ep*.json"):
        try:
            meta = json.loads(fp.read_text(encoding="utf-8"))
            n += int(bool(meta.get("success", False)))
        except Exception:
            pass
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-dir", type=str, default="/root/autodl-tmp/openpi_earbud_proto/demo_examples")
    parser.add_argument("--easy", type=int, default=5)
    parser.add_argument("--medium", type=int, default=5)
    parser.add_argument("--hard", type=int, default=5)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--max-seed", type=int, default=9999)
    parser.add_argument("--random-yaw-min-deg", type=float, default=-90.0)
    parser.add_argument("--random-yaw-max-deg", type=float, default=90.0)
    parser.add_argument("--flat-rest-prob", type=float, default=0.5)
    args = parser.parse_args()

    demo_dir = Path(args.demo_dir)
    demo_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        "easy": args.easy,
        "medium": args.medium,
        "hard": args.hard,
    }

    seed = args.start_seed
    while seed <= args.max_seed:
        done_all = True
        for level in ["easy", "medium", "hard"]:
            current = count_successes(demo_dir, level)
            target = targets[level]
            print(f"[progress] {level}: {current}/{target}")
            if current >= target:
                continue
            done_all = False
            ok, meta = run_one(
                level=level,
                seed=seed,
                demo_dir=demo_dir,
                yaw_min=args.random_yaw_min_deg,
                yaw_max=args.random_yaw_max_deg,
                flat_rest_prob=args.flat_rest_prob,
            )
            if ok:
                current = count_successes(demo_dir, level)
                print(f"[updated] {level}: {current}/{target}")

        if done_all:
            print("\nAll target demo counts reached.")
            break
        seed += 1

    print("\nFinal counts:")
    for level in ["easy", "medium", "hard"]:
        print(f"  {level}: {count_successes(demo_dir, level)}/{targets[level]}")


if __name__ == "__main__":
    main()
