from pathlib import Path

import cv2
import numpy as np

# --- Use your robot driver ---
from ctu_crs import CRS97  # or CRS93

z = 0.054  # default Z to use when moving

# -------- cross-platform single-keypress helpers --------
def read_single_keypress():
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch.lower()

def wait_for_key(prompt, valid=("q",)):
    print(prompt, flush=True)
    while True:
        ch = read_single_keypress()
        if ch in valid or ord(ch) == 27:  # Esc
            return ch

def safe_coord_str(val: float) -> str:
    s = f"{val:.3f}"           # 3 decimal places
    s = s.replace("-", "m")    # avoid leading '-' weirdness
    s = s.replace(".", "p")    # dots are fine but make it consistent
    return s

def show_image_and_get_action(img: np.ndarray) -> str:
    """
    Show image and wait for a decision:
      - 'k' keep
      - Esc to quit
    Returns 'k' or 'esc'
    """
    window = "Preview (k=keep, Esc=quit)"
    cv2.imshow(window, img)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord('k'), ord('K')):
            cv2.destroyWindow(window)
            return 'k'
        elif key == 27:
            cv2.destroyWindow(window)
            return 'esc'

def prompt_float(prompt_text: str) -> float:
    while True:
        try:
            s = input(prompt_text).strip()
            return float(s)
        except ValueError:
            print("Invalid number. Try again (example: 12.34).")

def move_robot_xy(robot, position: np.ndarray):
    """
    Move the robot to a position using closest IK solution, as per user-provided example.
    """

    rotation_euler = np.array([-0.06, np.pi + 0.05, 0.0])
    target_pose = np.eye(4)


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
    target_pose[:3, :3] = R
    target_pose[:3, 3] = position

    print(target_pose)

    ik_sols = robot.ik(target_pose)
    if len(ik_sols) == 0:
        raise RuntimeError("No IK solutions found for the target position.")
    
    closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - robot.get_q()))
    robot.move_to_q(closest_solution)
    robot.wait_for_motion_stop()


def get_photo_positions(points_horizontal: int, points_vertical: int):
    """
    Interpolates the corner positions to get the photo positions.
    """

    left_bottom_corner = np.array([0.40, -0.2, 0.054])
    left_top_corner = np.array([0.250, -0.2, 0.054])
    right_top_corner = np.array([0.250, 0.2, 0.054])
    right_bottom_corner = np.array([0.40, 0.2, 0.054])

    diff_horiz_bottom = (right_bottom_corner - left_bottom_corner) / (points_horizontal - 1)
    diff_horiz_top = (right_top_corner - left_top_corner) / (points_horizontal - 1)

    photo_positions = []
    for i in range(points_horizontal):
        for j in range(points_vertical):
            pos = left_bottom_corner + diff_horiz_bottom * i + (left_top_corner + diff_horiz_top * i - left_bottom_corner - diff_horiz_bottom * i) / (points_vertical - 1) * j
            photo_positions.append(pos)

    return photo_positions

def main():
    out_dir = Path("images")
    out_dir.mkdir(parents=True, exist_ok=True)

    robot = CRS97()
    try:
        print("Initializing robot...")
        if (wait_for_key("Press 's' to skip full initialization, or any other key to continue.", valid=("s",)) == 's'):
            print("Skipping full initialization.")
            robot.initialize(home=False)
            robot.move_to_q(robot.q_home)
            robot.wait_for_motion_stop()
        else:
            robot.initialize()

        for position in get_photo_positions(5, 5):
            print(f"Moving robot to position: x={position[0]:.3f}, y={position[1]:.3f}, z={position[2]:.3f}")
            try:
                move_robot_xy(robot, position)
            except KeyboardInterrupt:
                print("\nInterrupted during move. Exiting.")
                break
            except Exception as e:
                print(f"Error while moving robot: {e}")
                continue

            print("Capturing...")
            while True:
                img = robot.grab_image()

                if img is None:
                    print("Error: grab_image() returned None. Trying again...")
                    continue
                if not isinstance(img, np.ndarray):
                    print("Error: Unexpected image type. Trying again...")
                    continue

                action = show_image_and_get_action(img)
                if action == 'esc':
                    print("Exiting.")
                    return
                elif action == 'r':
                    print("Retaking...")
                    continue
                elif action == 'k':
                    break

            fname = f"x_{safe_coord_str(position[0])}__y_{safe_coord_str(position[1])}.png"
            fpath = out_dir / fname
            ok = cv2.imwrite(str(fpath), img)
            if ok:
                print(f"Saved: {fpath.resolve()}")
            else:
                print("Error: Failed to save image.")


    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # Always close the robot connection
        try:
            robot.close()
        except Exception:
            pass

        # Make sure all OpenCV windows are closed
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("Terminated.")

if __name__ == "__main__":
    main()
