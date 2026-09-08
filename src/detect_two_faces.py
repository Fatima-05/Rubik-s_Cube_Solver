import cv2
import numpy as np


def sticker_centers(image, margin=0.02):
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    colored = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([179, 255, 255]))
    white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([179, 60, 255]))
    mask = cv2.bitwise_or(colored, white)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    min_area = 0.0005 * w * h
    max_area = 0.15 * w * h
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            x, y, cw, ch = cv2.boundingRect(c)
            centers.append([x + cw // 2, y + ch // 2])
    return np.array(centers, dtype=np.float32)


def kmeans_1d(values, k):
    values = np.sort(np.asarray(values, dtype=np.float32))
    centers = values[np.linspace(0, len(values) - 1, k).astype(int)].copy()
    for _ in range(100):
        labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        new = np.array([values[labels == i].mean() if np.any(labels == i) else centers[i]
                        for i in range(k)], dtype=np.float32)
        if np.allclose(new, centers, atol=1e-3):
            centers = new
            break
        centers = new
    return centers, labels

def main(image_path="data/warped_face.png"):
    image = cv2.imread(image_path)
    centers = sticker_centers(image)
    if len(centers) < 18:
        print(f"Only {len(centers)} sticker blobs found; need at least 18.")
        return
    col_c, col_l = kmeans_1d(centers[:, 0], 3)
    row_c, row_l = kmeans_1d(centers[:, 1], 6)
    by_cell = {}
    for (x, y), cl, rl in zip(centers.astype(int), col_l, row_l):
        by_cell.setdefault((int(rl), int(cl)), []).append((x, y))
    print("Lattice (row, col) -> mean sticker position:")
    for r in range(6):
        line = []
        for c in range(3):
            pts = by_cell.get((r, c), [])
            if pts:
                mx = int(np.mean([p[0] for p in pts]))
                my = int(np.mean([p[1] for p in pts]))
                line.append(f"({mx:3d},{my:3d})")
            else:
                line.append("   ?   ")
        print(f"row{r}: {', '.join(line)}")

    debug = image.copy()
    for (x, y), cl, rl in zip(centers.astype(int), col_l, row_l):
        cv2.circle(debug, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(debug, f"{rl},{cl}", (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imwrite("data/lattice.png", debug)
    print("Annotated image saved to data/lattice.png")


if __name__ == "__main__":
    main()