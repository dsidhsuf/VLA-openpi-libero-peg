#!/usr/bin/env python3
import argparse
import json
import random
import shutil
from pathlib import Path


def is_lerobot_episode_root(path: Path) -> bool:
    return (path / "meta" / "info.json").exists() and (path / "data").exists()


def collect_episodes(src_root: Path, categories: list[str]) -> list[Path]:
    episodes = []
    for cat in categories:
        cat_dir = src_root / cat
        if not cat_dir.exists() or not cat_dir.is_dir():
            continue
        for child in sorted(cat_dir.iterdir()):
            if child.is_dir() and is_lerobot_episode_root(child):
                episodes.append(child.resolve())
    return episodes


def unique_destination_name(index: int, category: str, episode_name: str) -> str:
    return f"{index:06d}_{category}_{episode_name}"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Flatten LeRobot episodes from easy/hard/medium into a single folder "
            "with randomized order and numbered names."
        )
    )
    parser.add_argument(
        "--src-root",
        default="/root/autodl-tmp/openpi_earbud_proto/lerobot_ready/earbud_insert_batch_v3_split",
        help="Source root that contains easy/hard/medium.",
    )
    parser.add_argument(
        "--dst-root",
        required=True,
        help="Destination root. Will be created and filled with shuffled episodes.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["easy", "medium", "hard"],
        help="Category folders under src-root to scan.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffle order.",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "move", "symlink"],
        default="move",
        help="How to transfer episodes into dst-root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, remove existing dst-root before writing.",
    )
    args = parser.parse_args()

    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"Source root not found: {src_root}")

    if dst_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Destination already exists: {dst_root}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(dst_root)

    dst_root.mkdir(parents=True, exist_ok=True)

    episodes = collect_episodes(src_root, args.categories)
    if not episodes:
        raise RuntimeError(
            "No valid LeRobot episode roots found. "
            f"Checked categories under: {src_root}"
        )

    rng = random.Random(args.seed)
    shuffled = list(episodes)
    rng.shuffle(shuffled)

    manifest_items = []
    for idx, episode in enumerate(shuffled, start=1):
        category = episode.parent.name
        dst_name = unique_destination_name(idx, category, episode.name)
        dst_episode = dst_root / dst_name

        if args.mode == "copy":
            shutil.copytree(episode, dst_episode)
        elif args.mode == "move":
            shutil.move(str(episode), str(dst_episode))
        else:
            dst_episode.symlink_to(episode, target_is_directory=True)

        manifest_items.append(
            {
                "order": idx,
                "category": category,
                "source": str(episode),
                "destination": str(dst_episode),
            }
        )

        print(f"[{idx:04d}/{len(shuffled):04d}] {category} -> {dst_episode.name}")

    manifest = {
        "src_root": str(src_root),
        "dst_root": str(dst_root),
        "mode": args.mode,
        "seed": args.seed,
        "count": len(shuffled),
        "categories": args.categories,
        "items": manifest_items,
    }
    manifest_path = dst_root / "shuffle_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Total episodes: {len(shuffled)}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
