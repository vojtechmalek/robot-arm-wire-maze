########################## Imports ##########################
from pathlib import Path

import cv2
import numpy as np

from ctu_crs import CRS97

########################## Constants ##########################
# Camera height above table for imaging. Tune as needed.
Z_DEFAULT = 0.054

# Default orientation: slight roll, camera looking down (Y flipped by π), zero yaw.
# Matches the style used in utils.py, but explicit for readability here.
RING_VIEW_RPY = np.array([-0.06, np.pi + 0.05, 0.0])  # [roll, pitch, yaw] (radians)

LEFT_BOTTOM  = np.array([0.400, -0.200, Z_DEFAULT])
LEFT_TOP     = np.array([0.250, -0.200, Z_DEFAULT])
RIGHT_TOP    = np.array([0.250,  0.150, Z_DEFAULT])
RIGHT_BOTTOM = np.array([0.400,  0.150, Z_DEFAULT])

########################## Rotation matrix ##########################
rotation_euler = np.array([-0.06, np.pi + 0.05, 0.0])

R_x = np.array([[1, 0, 0],
                [0, np.cos(rotation_euler[0]), -np.sin(rotation_euler[0])],
                [0, np.sin(rotation_euler[0]), np.cos(rotation_euler[0])]])
R_y = np.array([[np.cos(rotation_euler[1]), 0, np.sin(rotation_euler[1])],
                [0, 1, 0],
                [-np.sin(rotation_euler[1]), 0, np.cos(rotation_euler[1])]])
R_z = np.array([[np.cos(rotation_euler[2]), -np.sin(rotation_euler[2]), 0],
                [np.sin(rotation_euler[2]), np.cos(rotation_euler[2]), 0],
                [0, 0, 1]])
R = R_z @ R_y @ R_x


########################## Small utilities ##########################
def safe_coord_str(val: float) -> str:
    """Encode a float using 'm' for minus and 'p' for decimal point."""
    s = f"{val:.3f}"
    return s.replace("-", "m").replace(".", "p")


def show_image_and_get_action(img: np.ndarray, auto_keep: bool = False) -> str:
    """Display an image and return 'k' (keep) or 'esc'.
    If auto_keep=True, shows for ~0.5 s and returns 'k' automatically.
    """
    window = "Preview (k=keep, Esc=quit)"
    cv2.imshow(window, img)

    if auto_keep:
        cv2.waitKey(500)
        cv2.destroyWindow(window)
        return 'k'

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord('k'), ord('K')):
            cv2.destroyWindow(window)
            return 'k'
        elif key == 27:  # ESC
            cv2.destroyWindow(window)
            return 'esc'


########################## Robot motion ##########################
def move_robot_xyz(robot: CRS97, position_xyz: np.ndarray) -> None:
    """Move robot flange to a pose whose origin is at position_xyz with
    a fixed orientation (RING_VIEW_RPY). Chooses the IK solution closest
    to the current joint state for smooth motion.
    """
    target_pose = np.eye(4)

    target_pose[:3, :3] = R
    target_pose[:3, 3] = position_xyz

    print("Target pose (T_base_flange):\n", target_pose)

    ik_sols = robot.ik(target_pose)
    if len(ik_sols) == 0:
        raise RuntimeError("No IK solutions found for the target pose.")

    q_now = robot.get_q()
    closest = min(ik_sols, key=lambda q: np.linalg.norm(q - q_now))
    robot.move_to_q(closest)
    robot.wait_for_motion_stop()


########################## Grid generation ##########################
def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + t * (b - a)


def photo_positions(points_horizontal: int, points_vertical: int) -> list[np.ndarray]:
    """Return a list of (x, y, z) positions that cover the convex quad
    defined by the four corners, sampled on an H×V grid in row-major order.
    """
    if points_horizontal < 2 or points_vertical < 2:
        raise ValueError("Use at least a 2×2 grid.")

    positions = []
    for i in range(points_horizontal):
        s = i / (points_horizontal - 1)
        bottom = lerp(LEFT_BOTTOM, RIGHT_BOTTOM, s)
        top    = lerp(LEFT_TOP,    RIGHT_TOP,    s)
        for j in range(points_vertical):
            t = j / (points_vertical - 1)
            positions.append(lerp(bottom, top, t))
    return positions


########################## Main ##########################
def main() -> None:
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    robot = CRS97()
    try:
        # Full initialize (homes robot). Press 's' to skip homing but still connect.
        print("Initializing robot...")
        do_skip = input("Press 's' + Enter to skip homing, or Enter to fully initialize: ").strip().lower() == 's'
        if do_skip:
            print("Skipping full initialization (home=False)…")
            robot.initialize(home=False)
            robot.move_to_q(robot.q_home)
            robot.wait_for_motion_stop()
        else:
            robot.initialize()

        # Iterate grid and capture
        for pos in photo_positions(points_horizontal=5, points_vertical=5):
            print(f"\nMoving to: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
            try:
                move_robot_xyz(robot, pos)
            except Exception as e:
                print(f"[move-error] {e}")
                continue

            # Acquire frame; auto-keep for unattended capture
            print("Capturing…")
            while True:
                img = robot.grab_image()
                if img is None or not isinstance(img, np.ndarray):
                    print("[warn] grab_image() failed; retrying…")
                    continue

                action = show_image_and_get_action(img, auto_keep=True)
                if action == 'esc':
                    print("Exiting early at user request.")
                    return
                if action == 'k':
                    break

            # Save with encoded coordinates (X/Y only; Z is fixed for homography)
            fname = f"x_{safe_coord_str(float(pos[0]))}__y_{safe_coord_str(float(pos[1]))}.png"
            fpath = out_dir / fname
            ok = cv2.imwrite(str(fpath), img)
            print(f"Saved: {fpath.resolve()}" if ok else "[error] Failed to save image.")

    finally:
        # Always close robot & windows
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

