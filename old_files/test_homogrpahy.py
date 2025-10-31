#!/usr/bin/env python3
"""
test_homogrpahy.py
- Loads a 3x3 homography (homography.npy)
- Moves the robot to a RANDOM (x, y, z) within the capture area limits using the
  SAME orientation/IK approach as capture_images.py
- Captures an image, detects the marker center in pixels, maps (u, v) -> (X, Y) via homography
- Prints the real (commanded) XY vs the homography-estimated XY (and the error)
- After each run, asks for input: press 'r' to repeat, anything else to exit.
"""

from pathlib import Path
import sys
import time
import cv2
import numpy as np

# --- Use your robot driver (same as capture_images.py) ---
from ctu_crs import CRS97  # or CRS93

# ------------------ CONFIG ------------------
# Area limits (same as capture_images.get_photo_positions corners):
LEFT_BOTTOM  = np.array([0.40, -0.20, 0.054])
LEFT_TOP     = np.array([0.250, -0.20, 0.054])
RIGHT_TOP    = np.array([0.250,  0.15, 0.054])
RIGHT_BOTTOM = np.array([0.40,   0.15, 0.054])

# For this rectangle, X in [0.250, 0.400], Y in [-0.200, 0.150], Z fixed to 0.054
X_MIN, X_MAX = min(LEFT_TOP[0], LEFT_BOTTOM[0], RIGHT_TOP[0], RIGHT_BOTTOM[0]), max(LEFT_TOP[0], LEFT_BOTTOM[0], RIGHT_TOP[0], RIGHT_BOTTOM[0])
Y_MIN, Y_MAX = min(LEFT_TOP[1], LEFT_BOTTOM[1], RIGHT_TOP[1], RIGHT_BOTTOM[1]), max(LEFT_TOP[1], LEFT_BOTTOM[1], RIGHT_TOP[1], RIGHT_BOTTOM[1])
Z_FIXED = float(LEFT_BOTTOM[2])

H_PATH = Path("homography.npy")   # produced by create-homography.py
SHOW_WINDOW = True                # set False if you don't want OpenCV windows

# Detection parameters (same pipeline as create-homography.py)
LOWER_HSV = np.array([85, 120, 40], dtype=np.uint8)
UPPER_HSV = np.array([115, 255, 160], dtype=np.uint8)
KERNEL_SIZE = 30
HOUGH_DP = 1
HOUGH_MIN_DIST = 100
HOUGH_PARAM1 = 200
HOUGH_PARAM2 = 100
HOUGH_MIN_RADIUS = 100
HOUGH_MAX_RADIUS = 0   # 0 = auto upper bound

# ------------- Helpers / Utilities -------------
def move_robot_xy(robot, position: np.ndarray):
    """
    Copy of the orientation & IK logic from capture_images.py:move_robot_xy.
    Moves the robot using the closest IK solution to the current joint state.
    """
    rotation_euler = np.array([-0.06, np.pi + 0.05, 0.0])
    target_pose = np.eye(4)

    R_x = np.array([[1, 0, 0],
                    [0, np.cos(rotation_euler[0]), -np.sin(rotation_euler[0])],
                    [0, np.sin(rotation_euler[0]),  np.cos(rotation_euler[0])]])
    R_y = np.array([[ np.cos(rotation_euler[1]), 0, np.sin(rotation_euler[1])],
                    [0, 1, 0],
                    [-np.sin(rotation_euler[1]), 0, np.cos(rotation_euler[1])]])
    R_z = np.array([[ np.cos(rotation_euler[2]), -np.sin(rotation_euler[2]), 0],
                    [ np.sin(rotation_euler[2]),  np.cos(rotation_euler[2]), 0],
                    [0, 0, 1]])
    R = R_z @ R_y @ R_x
    target_pose[:3, :3] = R
    target_pose[:3, 3] = position

    ik_sols = robot.ik(target_pose)
    if len(ik_sols) == 0:
        raise RuntimeError("No IK solutions found for the target position.")
    closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - robot.get_q()))
    robot.move_to_q(closest_solution)
    robot.wait_for_motion_stop()

def random_position() -> np.ndarray:
    x = np.random.uniform(X_MIN, X_MAX)
    y = np.random.uniform(Y_MIN, Y_MAX)
    z = Z_FIXED
    return np.array([x, y, z], dtype=float)

def detect_uv_center(img: np.ndarray):
    """
    Apply the same HSV mask + dilation + Hough circle approach as in create-homography.py.
    Returns (u, v) as floats or None if not found.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=HOUGH_MIN_RADIUS,
        maxRadius=HOUGH_MAX_RADIUS
    )
    if circles is None:
        return None
    circles = np.uint16(np.around(circles))
    best = circles[0, 0]
    u, v = float(best[0]), float(best[1])
    return (u, v)

def pixel_to_robot_xy(H: np.ndarray, u: float, v: float):
    vec = np.array([u, v, 1.0], dtype=float)
    Xw, Yw, w = H @ vec
    if abs(w) < 1e-12:
        raise ZeroDivisionError("Homography mapping produced w ~ 0.")
    return float(Xw / w), float(Yw / w)

def draw_preview(img: np.ndarray, real_xy, est_xy):
    u_text = [
        f"Real XY: ({real_xy[0]:.3f}, {real_xy[1]:.3f})",
        f"Est. XY: ({est_xy[0]:.3f}, {est_xy[1]:.3f})",
        f"Error dX={est_xy[0]-real_xy[0]:+.3f}  dY={est_xy[1]-real_xy[1]:+.3f}  |err|={np.hypot(est_xy[0]-real_xy[0], est_xy[1]-real_xy[1]):.3f}"
    ]
    y = 30
    for line in u_text:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        y += 24
    return img

# ------------------ Main ------------------
def main():
    if not H_PATH.exists():
        raise SystemExit(f"Missing homography file: {H_PATH.resolve()}")
    H = np.load(H_PATH)

    robot = CRS97()
    try:
        print("Initializing robot...")
        robot.initialize()  # full init by default (matches capture_images default path)
        while True:
            target = random_position()
            print(f"Moving robot to RANDOM position: x={target[0]:.3f}, y={target[1]:.3f}, z={target[2]:.3f}")
            try:
                move_robot_xy(robot, target)
            except Exception as e:
                print(f"[Move error] {e}")
                # try another random point or exit
                ch = input("Press 'r' to try another random point, anything else to exit: ").strip().lower()
                if ch == 'r':
                    continue
                else:
                    break

            # Grab image
            print("Capturing image...")
            img = None
            for attempt in range(5):
                img = robot.grab_image()
                if isinstance(img, np.ndarray):
                    break
                print("grab_image() returned invalid data, retrying...")
                time.sleep(0.2)
            if not isinstance(img, np.ndarray):
                print("Failed to grab a valid image. Aborting this round.")
                ch = input("Press 'r' to retry, anything else to exit: ").strip().lower()
                if ch == 'r':
                    continue
                else:
                    break

            # Detect pixel center
            uv = detect_uv_center(img)
            if uv is None:
                print("No circle detected in image.")
                ch = input("Press 'r' to retry, anything else to exit: ").strip().lower()
                if ch == 'r':
                    continue
                else:
                    break

            u, v = uv
            # Map to robot XY via homography
            try:
                est_x, est_y = pixel_to_robot_xy(H, u, v)
            except Exception as e:
                print(f"Homography mapping failed: {e}")
                ch = input("Press 'r' to retry, anything else to exit: ").strip().lower()
                if ch == 'r':
                    continue
                else:
                    break

            real_xy = (float(target[0]), float(target[1]))
            est_xy = (float(est_x), float(est_y))
            dx, dy = est_xy[0] - real_xy[0], est_xy[1] - real_xy[1]
            err = float(np.hypot(dx, dy))

            print("\n=== RESULT ===")
            print(f"Real XY (commanded): ({real_xy[0]:.3f}, {real_xy[1]:.3f})")
            print(f"Estimated XY (homog): ({est_xy[0]:.3f}, {est_xy[1]:.3f})")
            print(f"Error: dX={dx:+.3f}, dY={dy:+.3f}, |err|={err:.3f}\n")

            if SHOW_WINDOW:
                preview = img.copy()
                preview = draw_preview(preview, real_xy, est_xy)
                cv2.imshow("test_homogrpahy preview", preview)
                key = cv2.waitKey(1) & 0xFF
                # (window will remain until next loop or exit)

            ch = input("Press 'r' to repeat, anything else to exit: ").strip().lower()
            if ch != 'r':
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        try:
            robot.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("Terminated.")

if __name__ == "__main__":
    main()
