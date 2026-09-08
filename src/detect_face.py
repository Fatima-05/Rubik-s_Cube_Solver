import argparse

import cv2
import numpy as np
from pathlib import Path


def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def best_face_quad(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape[:2]
    min_area = 0.01 * h * w
    max_area = 0.6 * h * w
    best = None
    best_area = 0
    for thresh_val in (220, 128, 60):
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            quad = approx.reshape(4, 2).astype(np.float32)
            if area > best_area:
                best_area = area
                best = quad
    return best


def order_points(pts):
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(pts.sum(axis=1))]
    ordered[2] = pts[np.argmax(pts.sum(axis=1))]
    diff = np.diff(pts, axis=1)[:, 0]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def warp_face(image, corners, size=600):
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(image, matrix, (size, size))


def grid_hsv_stats(warped, grid=3, margin=0.15):
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    cell_h, cell_w = h // grid, w // grid
    means = np.zeros((grid * grid, 3), dtype=np.float32)
    for r in range(grid):
        for c in range(grid):
            cell = hsv[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
            mh, mw = cell.shape[:2]
            y0, y1 = int(mh * margin), int(mh * (1 - margin))
            x0, x1 = int(mw * margin), int(mw * (1 - margin))
            means[r * grid + c] = cell[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
    return means


def main(image_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    image = load_image(image_path)
    corners = best_face_quad(image)
    if corners is None:
        print("No contours found; could not detect a cube face.")
        return
    corners = order_points(corners)
    overlay = image.copy()
    cv2.polylines(overlay, [corners.astype(np.int32)], True, (0, 255, 0), 3)
    for point in corners.astype(np.int32):
        cv2.circle(overlay, tuple(point), 6, (0, 0, 255), -1)
    cv2.imwrite(str(out / "detected_face.png"), overlay)
    warped = warp_face(image, corners)
    grid_overlay = warped.copy()
    for i in range(1, 3):
        cv2.line(grid_overlay, (i * 200, 0), (i * 200, warped.shape[0]), (0, 0, 255), 3)
        cv2.line(grid_overlay, (0, i * 200), (warped.shape[1], i * 200), (0, 0, 255), 3)
    cv2.imwrite(str(out / "warped_grid.png"), grid_overlay)
    cv2.imwrite(str(out / "warped_face.png"), warped)
    means = grid_hsv_stats(warped)
    print("Per-sticker mean HSV in reading order (top-left to bottom-right):")
    for i, mean in enumerate(means):
        print(f"  sticker {i}: H={mean[0]:6.1f} S={mean[1]:6.1f} V={mean[2]:6.1f}")
    print(f"Visualizations saved to: {out / 'detected_face.png'} and {out / 'warped_face.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect the 3x3 sticker grid of a Rubik's cube face.")
    parser.add_argument("--image", default="data/sample_cube.png", help="Path to the face photo.")
    parser.add_argument("--out", default="data", help="Directory for output images.")
    args = parser.parse_args()
    main(args.image, args.out)