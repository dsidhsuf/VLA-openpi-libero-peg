
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demo_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--fps', type=int, default=20)
    args = ap.parse_args()

    demo_dir = Path(args.demo_dir)
    out_dir = Path(args.out_dir)
    data_dir = out_dir / 'data' / 'chunk-000'
    videos_agent_dir = out_dir / 'videos' / 'chunk-000' / 'observation.images.image'
    videos_wrist_dir = out_dir / 'videos' / 'chunk-000' / 'observation.images.image2'
    meta_dir = out_dir / 'meta'
    for d in [data_dir, videos_agent_dir, videos_wrist_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    episodes = []
    index = 0
    npz_files = sorted(demo_dir.glob('*.npz'))
    if not npz_files:
        raise SystemExit(f'No .npz demos found in {demo_dir}')

    for episode_index, npz_path in enumerate(npz_files):
        stem = npz_path.stem
        arr = np.load(npz_path, allow_pickle=True)
        json_path = npz_path.with_suffix('.json')
        meta = json.loads(json_path.read_text(encoding='utf-8')) if json_path.exists() else {}

        agent = arr['agentview_image']
        wrist = arr['wrist_image']
        state = arr['state8']
        action = arr['action']
        task_arr = arr['task']
        success = bool(arr['success'][0]) if 'success' in arr else bool(meta.get('success', False))
        task = str(task_arr[0]) if len(task_arr) else meta.get('task_text', 'insert the earbud into the charging slot')
        level = meta.get('level', 'unknown')

        ep_agent_dir = videos_agent_dir / f'episode_{episode_index:06d}'
        ep_wrist_dir = videos_wrist_dir / f'episode_{episode_index:06d}'
        ep_agent_dir.mkdir(parents=True, exist_ok=True)
        ep_wrist_dir.mkdir(parents=True, exist_ok=True)

        for i in range(len(action)):
            # save frames
            import imageio.v2 as imageio
            img_rel = Path('videos/chunk-000/observation.images.image') / f'episode_{episode_index:06d}' / f'{i:06d}.png'
            wrist_rel = Path('videos/chunk-000/observation.images.image2') / f'episode_{episode_index:06d}' / f'{i:06d}.png'
            imageio.imwrite(out_dir / img_rel, agent[i])
            imageio.imwrite(out_dir / wrist_rel, wrist[i])

            rows.append({
                'index': index,
                'episode_index': episode_index,
                'frame_index': i,
                'timestamp': i / float(args.fps),
                'task_index': episode_index,
                'task': task,
                'level': level,
                'success': success,
                'observation.state': state[i].tolist(),
                'action': action[i].tolist(),
                'observation.images.image': str(img_rel),
                'observation.images.image2': str(wrist_rel),
            })
            index += 1

        episodes.append({
            'episode_index': episode_index,
            'num_frames': int(len(action)),
            'task': task,
            'level': level,
            'success': success,
            'source_npz': str(npz_path),
        })

    df = pd.DataFrame(rows)
    parquet_path = data_dir / 'episode_data.parquet'
    df.to_parquet(parquet_path, index=False)

    info = {
        'codebase_version': 'custom_v1',
        'robot_type': 'libero_single_arm',
        'total_episodes': len(episodes),
        'total_frames': len(rows),
        'total_tasks': len(episodes),
        'chunks_size': 1,
        'fps': args.fps,
        'splits': {'train': f'0:{len(episodes)}'},
        'data_path': 'data/chunk-000/episode_data.parquet',
        'video_path': 'videos/chunk-000',
        'features': {
            'observation.images.image': {'dtype': 'image_path'},
            'observation.images.image2': {'dtype': 'image_path'},
            'observation.state': {'dtype': 'float32', 'shape': [8]},
            'action': {'dtype': 'float32', 'shape': [7]},
            'task': {'dtype': 'string'},
            'timestamp': {'dtype': 'float32'},
            'episode_index': {'dtype': 'int64'},
            'frame_index': {'dtype': 'int64'},
        },
    }
    stats = {
        'episode_index': [e['episode_index'] for e in episodes],
        'num_frames': [e['num_frames'] for e in episodes],
        'levels': {e['level']: sum(1 for x in episodes if x['level']==e['level']) for e in episodes},
    }

    (meta_dir / 'info.json').write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    (meta_dir / 'stats.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding='utf-8')
    (meta_dir / 'episodes.json').write_text(json.dumps(episodes, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'out_dir': str(out_dir), 'episodes': len(episodes), 'frames': len(rows), 'parquet': str(parquet_path)}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
