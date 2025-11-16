########################## Imports ##########################
import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")  # robust OpenCV display on Linux
import re
import glob
import sys
from pathlib import PurePosixPath, Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.camera_distortion import undistort_image

import cv2
import numpy as np


########################## File naming helpers ##########################
# Accept names like: x_0p325__y_m0p050.png
FNAME_RE = re.compile(r"x_([A-Za-z0-9.\-]+)__y_([A-Za-z0-9.\-]+)\.png$", re.IGNORECASE)

def decode_coord(token: str) -> float:
    """ Decode the coordinate from the filename. """
    return float(token.replace("m", "-").replace("p", "."))

########################## Drawing & UI ##########################

def draw_cross(img: np.ndarray, center, size: int = 20, thickness: int = 2) -> None:
    u, v = int(round(center[0])), int(round(center[1]))
    cv2.line(img, (u - size, v), (u + size, v), (0, 255, 0), thickness, cv2.LINE_AA)
    cv2.line(img, (u, v - size), (u, v + size), (0, 255, 0), thickness, cv2.LINE_AA)
    cv2.circle(img, (u, v), 3, (0, 0, 255), -1, cv2.LINE_AA)

def put_text_box(img: np.ndarray, lines: list[str], org: tuple[int, int] = (10, 25)) -> None:
    x, y = org
    for line in lines:
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        y += 24

########################## Detection params ##########################

HSV_LOWER = np.array([85, 120, 40], dtype=np.uint8)
HSV_UPPER = np.array([115, 255, 160], dtype=np.uint8)
MORPH_KERNEL = np.ones((30, 30), np.uint8)

########################## Core logic ##########################

def detect_ring_center_uv(bgr: np.ndarray) -> tuple[float, float] | None:
    """Return (u, v) pixel center of the ring if found; otherwise None."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    mask = cv2.dilate(mask, MORPH_KERNEL, iterations=1)

    masked = cv2.bitwise_and(bgr, bgr, mask=mask)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=200,
        param2=100,
        minRadius=100,
        maxRadius=0,
    )
    if circles is None:
        return None

    c = np.uint16(np.around(circles))[0, 0]  # best candidate
    return float(c[0]), float(c[1])


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    images_dir = script_dir / "images"
    out_path = script_dir / "homography.npy"

    paths = sorted(glob.glob(str(images_dir / "x_*__y_*.png")))
    if not paths:
        raise SystemExit(f"No images found in {images_dir}")

    kept_pixel_uv: list[tuple[float, float]] = []
    kept_robot_xy: list[tuple[float, float]] = []

    for path in paths:
        basename = os.path.basename(path)
        m = FNAME_RE.search(basename)
        if not m:
            print(f"[skip] unexpected filename: {basename}")
            continue

        X, Y = decode_coord(m.group(1)), decode_coord(m.group(2))
        X = X + 0.135
        
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] could not read {path}", file=sys.stderr)
            continue

        # undistort image
        img = undistort_image(img)

        uv = detect_ring_center_uv(img)

        preview = img.copy()
        if uv is not None:
            draw_cross(preview, uv)
            lines = [
                f"File: {basename}",
                f"Robot XY from filename: ({X:.3f}, {Y:.3f})",
                f"Detected pixel center: ({uv[0]:.1f}, {uv[1]:.1f})",
                "Press 'k' to KEEP, 'i' to IGNORE",
            ]
        else:
            lines = [
                f"File: {basename}",
                f"Robot XY from filename: ({X:.3f}, {Y:.3f})",
                "No circle detected.",
                "Press 'i' to IGNORE",
            ]
        put_text_box(preview, lines, org=(10, 30))

        while True:
            key = ord('k')
            if key in (ord('k'), ord('K')):
                if uv is None:
                    print(f"[info] no detection in {basename}; cannot keep, ignoring.", file=sys.stderr)
                    break
                kept_pixel_uv.append(uv)
                kept_robot_xy.append((X, Y))
                print(f"[keep] {basename}  ->  pixel=({uv[0]:.1f},{uv[1]:.1f})  robot=({X:.3f},{Y:.3f})")
                break
            elif key in (ord('i'), ord('I')):
                print(f"[ignore] {basename}")
                break
            elif key == 27:  # ESC
                raise SystemExit("Aborted by user (ESC).")

    if len(kept_pixel_uv) < 4:
        raise SystemExit(f"Need at least 4 kept detections; got {len(kept_pixel_uv)}")

    src = np.array(kept_pixel_uv, dtype=np.float32)
    dst = np.array(kept_robot_xy, dtype=np.float32)

    H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if H is None:
        raise SystemExit("cv2.findHomography failed (returned None).")

    np.save(out_path, H)
    inliers = int(mask.sum()) if mask is not None else len(src)
    print(f"\nSaved homography to {out_path}  |  inliers: {inliers}/{len(src)}")
    print("Matrix:\n", H)


if __name__ == "__main__":
    main()



