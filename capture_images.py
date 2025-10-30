from pathlib import Path

import cv2
import numpy as np

# --- Use your robot driver ---
from ctu_crs import CRS97  # or CRS93

z = 0.2  # default Z to use when moving

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

def move_robot_xy(robot, x: float, y: float, z_value: float):
    """
    Move the robot to a position using closest IK solution, as per user-provided example.
    """
    #TODO

def main():
    out_dir = Path("images")
    out_dir.mkdir(parents=True, exist_ok=True)

    robot = CRS97()
    try:
        print("Initializing robot...")
        robot.initialize()

        while True:
            # 1) Get target X/Y first
            print("\nEnter target coordinates (Esc at any prompt to quit).")
            x_str = input("Enter X coordinate (decimal): ").strip()
            if x_str == "":
                print("Empty input detected. Exiting.")
                break
            try:
                x_val = float(x_str)
            except ValueError:
                print("Invalid X number. Please start over.")
                continue

            y_str = input("Enter Y coordinate (decimal): ").strip()
            if y_str == "":
                print("Empty input detected. Exiting.")
                break
            try:
                y_val = float(y_str)
            except ValueError:
                print("Invalid Y number. Please start over.")
                continue

            print(f"Moving robot to X={x_val:.3f}, Y={y_val:.3f}, Z={z:.3f} ...")
            try:
                move_robot_xy(robot, x_val, y_val, z)
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

            fname = f"x_{safe_coord_str(x_val)}__y_{safe_coord_str(y_val)}.png"
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
