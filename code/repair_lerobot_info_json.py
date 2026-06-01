import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def infer_hw3(dataset_dir: Path):
    pngs = sorted((dataset_dir / "videos").rglob("*.png"))
    if not pngs:
        return None
    img = Image.open(pngs[0]).convert("RGB")
    w, h = img.size
    return [h, w, 3]


def to_shape_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    if hasattr(x, "shape"):
        return list(x.shape)
    arr = np.asarray(x)
    if arr.shape == ():
        return []
    return list(arr.shape)


def infer_from_first_row(parquet_path: Path):
    df = pd.read_parquet(parquet_path)
    if len(df) == 0:
        return {}
    row = df.iloc[0].to_dict()
    out = {}
    for k, v in row.items():
        out[k] = to_shape_list(v)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    info_path = dataset_dir / "meta" / "info.json"
    parquet_path = dataset_dir / "data" / "chunk-000" / "episode_data.parquet"

    if not info_path.exists():
        raise FileNotFoundError(info_path)
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    features = info.get("features", {})
    if not isinstance(features, dict):
        raise ValueError("Expected info['features'] to be a dict")

    inferred = infer_from_first_row(parquet_path)
    hw3 = infer_hw3(dataset_dir)

    updated = 0

    for name, ft in features.items():
        if not isinstance(ft, dict):
            continue

        if "shape" in ft and ft["shape"] is not None:
            continue

        shape = None

        # 1) 先用 parquet 第一行推断
        if name in inferred:
            shape = inferred[name]

        # 2) 图像字段兜底
        if shape is None and (
            "image" in name.lower()
            or "rgb" in name.lower()
            or "camera" in name.lower()
        ):
            shape = hw3 if hw3 is not None else []

        # 3) 常见标量字段兜底
        if shape is None and name in {
            "episode_index",
            "frame_index",
            "index",
            "task_index",
            "timestamp",
            "next.reward",
            "next.done",
            "next.truncated",
        }:
            shape = []

        # 4) 常见文本字段兜底
        if shape is None and name in {
            "task",
            "task_description",
            "language_instruction",
        }:
            shape = []

        # 5) 最后兜底
        if shape is None:
            shape = []

        ft["shape"] = shape
        updated += 1

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "info_path": str(info_path),
        "updated_features": updated,
        "sample_features": info.get("features", {}),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
