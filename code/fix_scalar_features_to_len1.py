import json
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_DIR = Path("/root/autodl-tmp/lerobot_datasets/local/earbud_insert")
INFO_PATH = DATASET_DIR / "meta" / "info.json"
DATA_PATH = DATASET_DIR / "data" / "chunk-000" / "episode_data.parquet"

def is_scalar_number(x):
    if isinstance(x, (np.generic, int, float, bool)):
        return True
    if x is None:
        return False
    arr = np.asarray(x)
    return arr.shape == ()

def wrap_len1(x):
    if x is None:
        return [0]
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return [x.item()]
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, np.generic):
        return [x.item()]
    return [x]

info = json.load(open(INFO_PATH, "r", encoding="utf-8"))
features = info.get("features", {})
df = pd.read_parquet(DATA_PATH)

patched_features = []
patched_columns = []

# 1) 修 info.json 里的 scalar schema
for k, ft in features.items():
    if not isinstance(ft, dict):
        continue

    dtype = ft.get("dtype")
    shape = ft.get("shape")

    if dtype in {"float32", "float64", "int32", "int64", "bool"} and (shape == [] or shape == () or shape is None):
        ft["shape"] = [1]
        ft["names"] = None
        patched_features.append(k)

# 2) 修 parquet 里的 scalar 列
for c in df.columns:
    # 跳过图像列
    if "image" in c.lower() or "rgb" in c.lower() or "camera" in c.lower():
        continue

    sample = None
    for v in df[c]:
        if v is not None:
            sample = v
            break

    if sample is None:
        continue

    if is_scalar_number(sample):
        df[c] = df[c].apply(wrap_len1)
        patched_columns.append(c)

df.to_parquet(DATA_PATH, index=False)

with open(INFO_PATH, "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)

print("patched features:")
for k in patched_features:
    print(" -", k)

print("patched columns:")
for c in patched_columns:
    print(" -", c)

print("rewrote:", INFO_PATH)
print("rewrote:", DATA_PATH)
