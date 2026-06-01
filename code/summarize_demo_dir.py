import argparse, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demo_dir', required=True)
    args = ap.parse_args()
    demo_dir = Path(args.demo_dir)
    stats = {'episodes': 0, 'success': 0, 'levels': {}, 'avg_steps': 0.0}
    total_steps = 0
    for jp in sorted(demo_dir.glob('*.json')):
        meta = json.loads(jp.read_text(encoding='utf-8'))
        stats['episodes'] += 1
        stats['success'] += int(bool(meta.get('success', False)))
        total_steps += int(meta.get('num_steps', 0))
        lvl = meta.get('level', 'unknown')
        stats['levels'][lvl] = stats['levels'].get(lvl, 0) + 1
    stats['avg_steps'] = total_steps / max(stats['episodes'], 1)
    stats['success_rate'] = stats['success'] / max(stats['episodes'], 1)
    print(json.dumps(stats, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
