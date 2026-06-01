import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import imageio.v2 as imageio


def copy_or_write_frames(video_dir: Path, episode_stem: str, frames: np.ndarray, fps: int = 20):
    ep_dir = video_dir / episode_stem
    ep_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        imageio.imwrite(ep_dir / f"{i:06d}.png", frame)
    return ep_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demo_dir', required=True, help='directory containing *.npz and *.json demos')
    ap.add_argument('--out_dir', required=True, help='output lerobot-style dataset dir')
    ap.add_argument('--fps', type=int, default=20)
    args = ap.parse_args()

    demo_dir = Path(args.demo_dir)
    out_dir = Path(args.out_dir)
    data_dir = out_dir / 'data'
    video_dir = out_dir / 'videos'
    meta_dir = out_dir / 'meta'
    data_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    episodes = []
    total_frames = 0
    levels = {}

    npz_files = sorted(demo_dir.glob('*.npz'))
    if not npz_files:
        raise SystemExit(f'No .npz demos found in {demo_dir}')

    for ep_idx, npz_path in enumerate(npz_files):
        arr = np.load(npz_path, allow_pickle=True)
        stem = npz_path.stem
        json_path = npz_path.with_suffix('.json')
        meta = json.loads(json_path.read_text(encoding='utf-8')) if json_path.exists() else {}

        agent = arr['agentview_image']
        wrist = arr['wrist_image']
        state = arr['state8']
        action = arr['action']
        task_arr = arr['task']
        success = bool(arr['success'][0])

        if len(agent) != len(action):
            raise ValueError(f'{npz_path}: image/action length mismatch {len(agent)} vs {len(action)}')

        ep_video_dir = copy_or_write_frames(video_dir, stem + '_agentview', agent, fps=args.fps)
        ep_wrist_dir = copy_or_write_frames(video_dir, stem + '_wrist', wrist, fps=args.fps)

        ep_payload = {
            'episode_index': ep_idx,
            'num_frames': int(len(action)),
            'task': str(task_arr[0]) if len(task_arr) else meta.get('task_text', ''),
            'level': meta.get('level', 'unknown'),
            'success': success,
            'agentview_dir': str(ep_video_dir),
            'wrist_dir': str(ep_wrist_dir),
            'state8_shape': list(state.shape),
            'action_shape': list(action.shape),
        }
        (data_dir / f'{stem}.npz').write_bytes(npz_path.read_bytes())
        (meta_dir / f'{stem}.json').write_text(json.dumps(ep_payload, indent=2, ensure_ascii=False), encoding='utf-8')

        episodes.append(ep_payload)
        total_frames += len(action)
        levels[ep_payload['level']] = levels.get(ep_payload['level'], 0) + 1

    info = {
        'format': 'custom_lerobot_like_v1',
        'source_demo_dir': str(demo_dir),
        'total_episodes': len(episodes),
        'total_frames': total_frames,
        'fps': args.fps,
        'levels': levels,
        'notes': [
            'Each episode keeps raw NPZ trajectory in data/*.npz',
            'Rendered frames are exploded to videos/*/*.png for later packing/conversion',
            'This is an intermediate format before final LeRobot dataset packing.'
        ]
    }
    (out_dir / 'manifest.json').write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
