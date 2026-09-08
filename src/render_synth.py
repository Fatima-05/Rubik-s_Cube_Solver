import numpy as np, cv2, json, os

W, H = 900, 700

V0 = np.array([450.0, 600.0])
U = np.array([0.0, -200.0])
R = np.array([200.0, -100.0])
L = np.array([-200.0, -100.0])

FACES = {
    "top":   [V0 + U, V0 + U + R, V0 + U + R + L, V0 + U + L],
    "right": [V0, V0 + U, V0 + U + R, V0 + R],
    "left":  [V0, V0 + U, V0 + U + L, V0 + L],
}
COLOR = {"top": (200, 200, 200), "right": (0, 0, 255), "left": (255, 0, 0)}

def sticker_cells(corners):
    c0 = np.array(corners[0]); d01 = np.array(corners[1]) - c0; d03 = np.array(corners[3]) - c0
    cells = []
    for i in range(3):
        for j in range(3):
            ll = c0 + d01 * (i / 3.0) + d03 * (j / 3.0)
            cells.append([ll, ll + d01 / 3, ll + d01 / 3 + d03 / 3, ll + d03 / 3])
    return cells

img = np.zeros((H, W, 3), np.uint8)
truth = {}
for name, corners in FACES.items():
    truth[name] = []
    for ci, cell in enumerate(sticker_cells(corners)):
        pair = np.array(cell).astype(np.int32)
        cv2.fillConvexPoly(img, pair.reshape(1, 4, 2), COLOR[name])
        cv2.polylines(img, [pair], True, (0, 0, 0), 2)
        truth[name].append({"color": list(COLOR[name]), "quad": pair.tolist(), "cell": ci})

for name, corners in FACES.items():
    print(name, "face corners:", [[int(p[0]), int(p[1])] for p in corners])
cv2.imwrite("data/syn/cube.png", img)
with open("data/syn/truth.json", "w") as f:
    json.dump(truth, f)
print("saved data/syn/cube.png + truth.json (3 faces, 9 sticker cells each)")