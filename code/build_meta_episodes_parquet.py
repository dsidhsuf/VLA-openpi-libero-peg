import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    data_path = dataset_dir / "data" / "chunk-000" / "episode_data.parquet"
    tasks_path = dataset_dir / "meta" / "tasks.parquet"
    out_dir = dataset_dir / "meta" / "episodes" / "chunk-000"
    out_path = out_dir / "file-000.parquet"

    if not data_path.exists():
        raise FileNotFoundError(data_path)

    df = pd.read_parquet(data_path)
    if len(df) == 0:
        raise ValueError("episode_data.parquet is empty")

    # 统一准备一个全局顺序索引，作为 dataset frame index 的后备
    df = df.reset_index(drop=True)

    if "episode_index" not in df.columns:
        raise KeyError("episode_data.parquet missing required column: episode_index")

    # task 映射
    task_map = {}
    if tasks_path.exists():
        tasks_df = pd.read_parquet(tasks_path)
        if "task" in tasks_df.columns and "task_index" in tasks_df.columns:
            task_map = dict(zip(tasks_df["task"].astype(str), tasks_df["task_index"].astype(int)))

    rows = []
    for ep_idx, g in df.groupby("episode_index", sort=True):
        g = g.sort_index()

        # dataset_from_index / to_index 优先用现成 index 列，否则用行号
        if "index" in g.columns:
            from_idx = int(g["index"].iloc[0])
            to_idx = int(g["index"].iloc[-1]) + 1
        else:
            from_idx = int(g.index[0])
            to_idx = int(g.index[-1]) + 1

        length = int(len(g))

        # task_index 优先用现成列，否则从 task 文本恢复
        if "task_index" in g.columns:
            task_index = int(g["task_index"].iloc[0])
        elif "task" in g.columns:
            task_name = str(g["task"].iloc[0])
            task_index = int(task_map.get(task_name, 0))
        else:
            task_index = 0

        row = {
            "episode_index": int(ep_idx),
            "task_index": task_index,
            "length": length,
            "dataset_from_index": from_idx,
            "dataset_to_index": to_idx,
            "data_chunk_index": 0,
            "data_file_index": 0,
        }

        # 尽量补一些常见字段，方便兼容不同读取逻辑
        if "task" in g.columns:
            row["task"] = str(g["task"].iloc[0])

        rows.append(row)

    episodes_df = pd.DataFrame(rows).sort_values("episode_index").reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_df.to_parquet(out_path, index=False)

    print("saved:", out_path)
    print(episodes_df)


if __name__ == "__main__":
    main()
