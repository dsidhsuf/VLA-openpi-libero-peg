#!/usr/bin/env python3
"""Create a one-episode raw LIBERO tree with a prompt aligned to the visual task.

The converter prefers task text already present in metadata.json over
--default-task. This helper builds a lightweight raw tree for exactly one
episode and rewrites the language fields, while symlinking large trajectory and
video files instead of copying them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


LANGUAGE_KEYS = (
    "task",
    "task_description",
    "instruction",
    "language_instruction",
    "prompt",
    "language",
)


DEFAULT_TASK = "Pick up the red rectangular peg, keep it vertical, and insert it into the rectangular slot."


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        os.symlink(src.resolve(), dst, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/autodl-tmp/openpi_earbud_proto/libero_lerobot_better_dataset_prompt_single"),
    )
    parser.add_argument("--level", type=str, default=None)
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    episode_dir = args.episode_dir.resolve()
    if not episode_dir.is_dir():
        raise FileNotFoundError(f"episode dir not found: {episode_dir}")
    meta_path = episode_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {meta_path}")

    level = args.level or episode_dir.parent.name
    dst_root = args.output_root.resolve()
    dst_episode = dst_root / level / episode_dir.name

    if dst_episode.exists() or dst_episode.is_symlink():
        if not args.force:
            raise FileExistsError(f"{dst_episode} already exists; pass --force to overwrite it.")
        if dst_episode.is_dir() and not dst_episode.is_symlink():
            shutil.rmtree(dst_episode)
        else:
            dst_episode.unlink()

    dst_episode.mkdir(parents=True, exist_ok=True)

    for child in episode_dir.iterdir():
        dst = dst_episode / child.name
        if child.name == "metadata.json":
            continue
        link_or_copy(child, dst)

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    for key in LANGUAGE_KEYS:
        meta[key] = args.task
    meta["prompt_aligned_source_episode"] = str(episode_dir)

    with (dst_episode / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    manifest = {
        "source_episode": str(episode_dir),
        "output_root": str(dst_root),
        "output_episode": str(dst_episode),
        "level": level,
        "task": args.task,
    }
    with (dst_root / "prompt_alignment_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[done] prompt-aligned raw root: {dst_root}")
    print(f"[done] episode: {dst_episode}")
    print(f"[done] task: {args.task}")


if __name__ == "__main__":
    main()
