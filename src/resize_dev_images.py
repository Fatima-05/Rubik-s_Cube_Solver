import os
from pathlib import Path
import cv2

source = Path("data/real")
out = Path("data/dev")
out.mkdir(exist_ok=True)

for f in source.glob("*.jpg"):
    if "annotated" in f.name:
        continue
    img = cv2.imread(str(f))
    h, w = img.shape[:2]
    scale = 1200 / max(h, w)
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out / f.name), small)
    print(f.name, "->", small.shape[1], "x", small.shape[0])