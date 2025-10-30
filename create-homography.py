# HOW TO LOAD LATER:
# import numpy as np
# H = np.load("homography.npy")   # 3x3 matrix mapping pixel (u,v,1) -> robot (X,Y,w)
# Example:
#   import numpy as np
#   H = np.load("homography.npy")
#   u, v = 960, 540
#   Xw, Yw, w = H @ np.array([u, v, 1.0], dtype=float)
#   X, Y = Xw / w, Yw / w

import os, re, glob, sys
import cv2
import numpy as np

# Filenames must look like: x_<X>__y_<Y>.png  (e.g., x_100.0__y_250.0.png)
FNAME_RE = re.compile(r"x_(-?\d+(?:\.\d+)?)__y_(-?\d+(?:\.\d+)?)\.png$", re.IGNORECASE)

def draw_cross(img, center, size=20, thickness=2):
    u, v = int(round(center[0])), int(round(center[1]))
    cv2.line(img, (u - size, v), (u + size, v), (0, 255, 0), thickness, cv2.LINE_AA)
    cv2.line(img, (u, v - size), (u, v + size), (0, 255, 0), thickness, cv2.LINE_AA)
    cv2.circle(img, (u, v), 3, (0, 0, 255), -1, cv2.LINE_AA)

def put_text_box(img, lines, org=(10, 25)):
    x, y = org
    for line in lines:
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        y += 24

def main():
    images_dir = "./images"
    out_path = "homography.npy"

    paths = sorted(glob.glob(os.path.join(images_dir, "x_*__y_*.png")))
    if not paths:
        raise SystemExit(f"No images found in {images_dir}")

    kept_pixel_uv = []
    kept_robot_xy = []

    # Use the exact detection approach you provided:
    lower_color = np.array([85, 120, 40], dtype=np.uint8)
    upper_color = np.array([115, 255, 160], dtype=np.uint8)
    kernel_size = 30
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)

    for path in paths:
        m = FNAME_RE.search(os.path.basename(path))
        if not m:
            continue
        X, Y = float(m.group(1)), float(m.group(2))

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] could not read {path}", file=sys.stderr)
            continue

        # ---- Your pipeline ----
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower_color, upper_color)
        mask = cv2.dilate(mask, kernel, iterations=1)

        masked_img = cv2.bitwise_and(img, img, mask=mask)
        gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=200,
            param2=100,
            minRadius=100,
            maxRadius=0
        )
        # -----------------------

        # Build a preview:
        preview = img.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles))
            i = circles[0, 0]  # take best/first
            u, v = float(i[0]), float(i[1])
            draw_cross(preview, (u, v))
            put_text_box(
                preview,
                [
                    f"File: {os.path.basename(path)}",
                    f"Robot XY from filename: ({X:.3f}, {Y:.3f})",
                    f"Detected pixel center: ({u:.1f}, {v:.1f})",
                    "Press 'k' to KEEP, 'i' to IGNORE"
                ],
                org=(10, 30)
            )
        else:
            put_text_box(
                preview,
                [
                    f"File: {os.path.basename(path)}",
                    f"Robot XY from filename: ({X:.3f}, {Y:.3f})",
                    "No circle detected.",
                    "Press 'i' to IGNORE"
                ],
                org=(10, 30)
            )

        cv2.imshow("preview", preview)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord('k'), ord('K')):
                if circles is None:
                    # nothing to keep, force ignore
                    print(f"[info] no detection in {path}; cannot keep, ignoring.", file=sys.stderr)
                    break
                kept_pixel_uv.append((u, v))
                kept_robot_xy.append((X, Y))
                print(f"[keep] {os.path.basename(path)}  ->  pixel=({u:.1f},{v:.1f})  robot=({X:.3f},{Y:.3f})")
                break
            elif key in (ord('i'), ord('I')):
                print(f"[ignore] {os.path.basename(path)}")
                break
            # (Optional: allow ESC to abort)
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                raise SystemExit("Aborted by user (ESC).")

    cv2.destroyAllWindows()

    if len(kept_pixel_uv) < 4:
        raise SystemExit(f"Need at least 4 kept detections; got {len(kept_pixel_uv)}")

    src = np.array(kept_pixel_uv, dtype=np.float32)  # pixels (u,v)
    dst = np.array(kept_robot_xy, dtype=np.float32)  # robot (X,Y)
    H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if H is None:
        raise SystemExit("cv2.findHomography failed (returned None).")

    np.save(out_path, H)
    inliers = int(mask.sum()) if mask is not None else len(src)
    print(f"\nSaved homography to {out_path}  |  inliers: {inliers}/{len(src)}")
    print("Matrix:")
    print(H)

if __name__ == "__main__":
    main()
