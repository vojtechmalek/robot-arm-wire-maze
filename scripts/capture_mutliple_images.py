import cv2
import numpy as np
from ctu_crs import CRS97  # or CRS93

robot = CRS97()

try:
    print("Initializing robot...")
    robot.initialize(home=False)
    robot.move_to_q(robot.q_home)
    robot.wait_for_motion_stop()

    counter = 0

    while True:
        input("Press Enter to capture an image (or Ctrl+C to exit)...")
        counter += 1
        filename = f"maze_detection_images/E/capture_{counter:03d}.png"
    
        print("Capturing image...")
        img = robot.grab_image()

        if img is None or not isinstance(img, np.ndarray):
            raise RuntimeError("Error: grab_image() returned invalid image.")

        if cv2.imwrite(filename, img):
            print(f"Image saved as {filename}")
        else:
            print("Failed to save image.")

except Exception as e:
    print(f"Error: {e}")

finally:
    try:
        robot.close()
    except Exception:
        pass
    cv2.destroyAllWindows()
