from pathlib import Path

import cv2
import numpy as np

# --- Use your robot driver ---
from ctu_crs import CRS97  # or CRS93

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
      - 'r' retake
      - Esc to quit
    Returns 'k', 'r', or 'esc'
    """
    window = "Preview (k=keep, r=retake, Esc=quit)"
    cv2.imshow(window, img)
    # Wait indefinitely for a key
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord('k'), ord('K')):
            cv2.destroyWindow(window)
            return 'k'
        elif key == 27:
            cv2.destroyWindow(window)
            return 'esc'

def main():
    out_dir = Path("images")
    out_dir.mkdir(parents=True, exist_ok=True)

    robot = CRS97()
    try:
        print("Initializing robot...")
        robot.initialize()

        while True:
            ch = wait_for_key("Press 'q' to TAKE a picture (Esc to quit).", valid=("q",))
            if ord(ch) == 27:
                print("Exiting.")
                break

            print("Capturing...")
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
                break

            elif action == 'k':
                while True:
                    try:
                        x_str = input("Enter X coordinate (decimal): ").strip()
                        x_val = float(x_str)
                        break
                    except ValueError:
                        print("Invalid number. Try again (example: 12.34).")

                while True:
                    try:
                        y_str = input("Enter Y coordinate (decimal): ").strip()
                        y_val = float(y_str)
                        break
                    except ValueError:
                        print("Invalid number. Try again (example: -5.75).")

            # Build filename from X,Y
            fname = f"x_{safe_coord_str(x_val)}__y_{safe_coord_str(y_val)}.png"
            fpath = out_dir / fname

            # Save PNG (lossless). If you prefer JPEG, switch to .jpg and add quality params.
            ok = cv2.imwrite(str(fpath), img)
            if ok:
                print(f"Saved: {fpath.resolve()}")
            else:
                print("Error: Failed to save image.")

            # After saving, ask if user wants to take another
            ch2 = wait_for_key("Press 'q' to take ANOTHER photo, or Esc to quit.", valid=("q",))
            if ord(ch2) == 27:
                print("Done.")
                break

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

if __name__ == "__main__":
    main()
