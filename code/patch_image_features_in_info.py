import json
from pathlib import Path
from PIL import Image

DATASET_DIR = Path("/root/autodl-tmp/lerobot_datasets/local/earbud_insert")
INFO_PATH = DATASET_DIR / "meta" / "info.json"

def find_any_image_shape(dataset_dir: Path):
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        files = sorted(dataset_dir.rglob(ext))
        if files:
            img = Image.open(files[0]).convert("RGB")
            w, h = img.size
            return [h, w, 3], str(files[0])
    return [224, 224, 3], None

with open(INFO_PATH, "r", encoding="utf-8") as f:
    info = json.load(f)

features = info.get("features", {})
if not isinstance(features, dict):
    raise ValueError("info.json['features'] is not a dict")

img_shape, sample_path = find_any_image_shape(DATASET_DIR)

patched = []
for k, ft in features.items():
    if not isinstance(ft, dict):
        continue

    dtype = ft.get("dtype")
    is_image_key = (
        "image" in k.lower()
        or "rgb" in k.lower()
        or "camera" in k.lower()
        or dtype == "image_path"
    )

    if dtype == "image_path":
        ft["dtype"] = "image"

    if is_image_key and ft.get("dtype") == "image":
        ft["shape"] = img_shape
        ft["names"] = ["height", "width", "channel"]
        patched.append(k)

with open(INFO_PATH, "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)

print("patched keys:")
for k in patched:
    print(" -", k)
print("sample image:", sample_path)
print("image shape:", img_shape)
print("info path:", INFO_PATH)
