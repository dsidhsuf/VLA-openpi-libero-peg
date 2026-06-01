import json
import traceback
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset

from lerobot.datasets.feature_utils import get_hf_features_from_features

DATASET_DIR = Path("/root/autodl-tmp/lerobot_datasets/local/earbud_insert")
INFO_PATH = DATASET_DIR / "meta" / "info.json"
DATA_PATH = DATASET_DIR / "data" / "chunk-000" / "episode_data.parquet"

print("=" * 80)
print("1) parquet schema")
print("=" * 80)
table = pq.read_table(DATA_PATH)
print(table.schema)

print("\n" + "=" * 80)
print("2) pandas dtypes")
print("=" * 80)
df = pd.read_parquet(DATA_PATH)
print(df.dtypes)

print("\n" + "=" * 80)
print("3) first row python types")
print("=" * 80)
row = df.iloc[0].to_dict()
for k, v in row.items():
    print(k, type(v), v if isinstance(v, (int, float, str, bool)) else f"sample={str(v)[:120]}")

print("\n" + "=" * 80)
print("4) info.json features")
print("=" * 80)
info = json.load(open(INFO_PATH, "r", encoding="utf-8"))
for k, v in info["features"].items():
    print(k, v)

print("\n" + "=" * 80)
print("5) HF inferred load WITHOUT features")
print("=" * 80)
try:
    ds = Dataset.from_parquet([str(DATA_PATH)])
    print("load without features: OK")
    print(ds.features)
except Exception as e:
    print("load without features: FAIL")
    traceback.print_exc()

print("\n" + "=" * 80)
print("6) HF load WITH lerobot features")
print("=" * 80)
try:
    hf_features = get_hf_features_from_features(info["features"])
    print("converted hf features:")
    print(hf_features)
    ds = Dataset.from_parquet([str(DATA_PATH)], features=hf_features)
    print("load with features: OK")
    print(ds.features)
except Exception as e:
    print("load with features: FAIL")
    traceback.print_exc()
    if getattr(e, "__cause__", None) is not None:
        print("\nCAUSE:")
        traceback.print_exception(type(e.__cause__), e.__cause__, e.__cause__.__traceback__)
    if getattr(e, "__context__", None) is not None:
        print("\nCONTEXT:")
        traceback.print_exception(type(e.__context__), e.__context__, e.__context__.__traceback__)
