import json
from pathlib import Path

import pandas as pd

DATASET_DIR = Path("/root/autodl-tmp/lerobot_datasets/local/earbud_insert")
INFO_PATH = DATASET_DIR / "meta" / "info.json"
DATA_PATH = DATASET_DIR / "data" / "chunk-000" / "episode_data.parquet"
TASKS_PATH = DATASET_DIR / "meta" / "tasks.parquet"
EPISODES_DIR = DATASET_DIR / "meta" / "episodes"

def main():
    info = json.load(open(INFO_PATH, "r", encoding="utf-8"))
    features = info.get("features", {})
    if not isinstance(features, dict):
        raise ValueError("info['features'] is not a dict")

    df = pd.read_parquet(DATA_PATH)

    # 1) 先找 info.json 里声明为 string 的 feature
    string_feature_keys = [
        k for k, v in features.items()
        if isinstance(v, dict) and v.get("dtype") == "string"
    ]

    # 2) 再找 parquet 里真实存在的字符串列
    string_cols_in_df = [
        c for c in df.columns
        if str(df[c].dtype) in ("object", "string")
    ]

    print("string_feature_keys in info.json:", string_feature_keys)
    print("string_cols_in_episode_data:", string_cols_in_df)

    # 3) 如果 task_index 不存在，但 task 存在，则先从 tasks.parquet 恢复 task_index
    if "task_index" not in df.columns and "task" in df.columns and TASKS_PATH.exists():
        tasks_df = pd.read_parquet(TASKS_PATH)
        if "task" in tasks_df.columns and "task_index" in tasks_df.columns:
            mapping = dict(zip(tasks_df["task"].astype(str), tasks_df["task_index"].astype(int)))
            df["task_index"] = df["task"].astype(str).map(mapping).fillna(0).astype("int64")
            print("reconstructed task_index from tasks.parquet")

            if "task_index" not in features:
                features["task_index"] = {
                    "dtype": "int64",
                    "shape": []
                }

    # 4) 删除 episode_data.parquet 中的所有字符串列
    drop_cols = [c for c in string_cols_in_df if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print("dropped string columns from episode_data:", drop_cols)

    df.to_parquet(DATA_PATH, index=False)
    print("rewrote:", DATA_PATH)

    # 5) 删除 info.json 中所有 string feature
    for k in list(features.keys()):
        v = features[k]
        if isinstance(v, dict) and v.get("dtype") == "string":
            del features[k]

    with open(INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print("rewrote:", INFO_PATH)

    # 6) 顺手把 meta/episodes 下 parquet 里的字符串列也删掉
    if EPISODES_DIR.exists():
        for pq in EPISODES_DIR.rglob("*.parquet"):
            edf = pd.read_parquet(pq)
            str_cols = [c for c in edf.columns if str(edf[c].dtype) in ("object", "string")]
            if str_cols:
                edf = edf.drop(columns=str_cols)
                edf.to_parquet(pq, index=False)
                print("dropped string columns from", pq, "->", str_cols)

    # 7) 最终检查
    new_info = json.load(open(INFO_PATH, "r", encoding="utf-8"))
    remain = []
    for k, v in new_info.get("features", {}).items():
        if isinstance(v, dict) and v.get("dtype") == "string":
            remain.append((k, v))
    print("remaining string features:", remain)

if __name__ == "__main__":
    main()
