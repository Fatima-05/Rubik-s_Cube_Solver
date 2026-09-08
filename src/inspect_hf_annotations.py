from datasets import load_dataset
from PIL import Image, ImageDraw

ds = load_dataset("Alabi-Ayobami/rubiks-cube-obb")
idx = 18
ex = ds["train"][idx]
img = ex["image"]
w, h = img.size

names = ['Blue', 'Green', 'Orange', 'Red', 'White', 'Yellow', 'cube_face', 'side_face']
colors = [(0,51,255),(0,155,72),(255,88,0),(183,18,52),(255,255,255),(255,204,0),(128,128,128),(255,0,255)]

for i, (cat, b) in enumerate(zip(ex["objects_category_id"], ex["objects_bbox"])):
    print(f"obj {i}: {names[cat]:10s} quad={[round(v,3) for v in b]}")

scale = 900 / max(w, h)
small = img.resize((int(w * scale), int(h * scale)))
draw = ImageDraw.Draw(small)
for cat, b in zip(ex["objects_category_id"], ex["objects_bbox"]):
    pts = [(round(b[k] * small.width), round(b[k + 1] * small.height)) for k in (0, 2, 4, 6)]
    draw.polygon(pts, outline=colors[cat], width=4)
small.save(f"data/real/{idx:03d}_annotated.jpg")
print("saved data/real/018_annotated.jpg")