import os
from datasets import load_dataset

ds = load_dataset("Alabi-Ayobami/rubiks-cube-obb")

print("Splits:", {k: len(v) for k, v in ds.items()})

feats = ds["train"].features["objects_category_id"]
print("Category feature:", feats)
if hasattr(feats, "names"):
    print("Class names:", feats.names)

out = "data/real"
os.makedirs(out, exist_ok=True)

for i, ex in enumerate(ds["train"]):
    if i >= 20:
        break
    img = ex["image"]
    n = len(ex["objects_bbox"])
    img.save(os.path.join(out, f"{i:03d}_objs{n}.jpg"))
    print(f"{i:03d}: {img.width}x{img.height}, {n} objects")