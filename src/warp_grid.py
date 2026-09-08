import cv2, json, numpy as np

img = cv2.imread("data/syn/cube.png")
truth = json.load(open("data/syn/truth.json"))

def face_quad(cells):
    by_idx = {c["cell"]: c["quad"] for c in cells}
    c0 = np.array(by_idx[0][0], dtype=np.float32)
    n1 = np.array(by_idx[6][1], dtype=np.float32)  # cell(i=2,j=0)  -> c0 + d01
    n2 = np.array(by_idx[2][3], dtype=np.float32)  # cell(i=0,j=2)  -> c0 + d03 
    d01, d03 = n1 - c0, n2 - c0
    if d01[0] * d03[1] - d01[1] * d03[0] < 0:
        d01, d03 = d03, d01
    return np.array([c0, c0 + d01, c0 + d01 + d03, c0 + d03]), d01, d03

BASE = {"top": (200, 200, 200), "right": (0, 0, 255), "left": (255, 0, 0)}

def classify(bgr):
    best, bestd = None, 1e18
    for name, base in BASE.items():
        d = sum((a - b) ** 2 for a, b in zip(bgr, base))
        if d < bestd:
            best, bestd = name, d
    return best

all_ok = True
for face, cells in truth.items():
    quad, d01, d03 = face_quad(cells)
    dst = np.array([[0, 0], [270, 0], [270, 270], [0, 270]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(img, M, (270, 270))
    print(f"--- {face} luminance map (30px blocks, W=color . =dark) ---")
    for yy in range(0, 270, 30):
        row = []
        for xx in range(0, 270, 30):
            p = warped[yy:yy + 30, xx:xx + 30].reshape(-1, 3).mean(axis=0)
            row.append("W" if p.mean() > 120 else ".")
        print("".join(row))
    cv2.imwrite(f"data/syn/warp_{face}.png", warped)

    ok = 0
    for i in range(3):
        for j in range(3):
            y0, y1 = 90 * i + 13, 90 * (i + 1) - 13
            x0, x1 = 90 * j + 13, 90 * (j + 1) - 13
            mean = warped[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
            got = classify(tuple(mean))
            if got == face:
                ok += 1
            print(f"{face:5s} cell({i},{j}) mean={np.round(mean).tolist()} -> {got}")
    print(f"{face}: {ok}/9 correct")

print("ALL PASS" if all_ok else "FAILURES PRESENT")