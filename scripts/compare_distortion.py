import cv2
from matplotlib import pyplot as plt
import numpy as np
from ctu_crs import CRS97  # or CRS93
from camera_distortion import undistort_image, show_image

robot = CRS97()

try:
    print("Initializing robot...")
    robot.initialize(home=False)
    robot.move_to_q(robot.q_home)
    robot.wait_for_motion_stop()

    print("Capturing image...")
    img = robot.grab_image()

    if img is None or not isinstance(img, np.ndarray):
        raise RuntimeError("Error: grab_image() returned invalid image.")

    show_image("Captured Image", img, wait_time=10)
    undistorted_img = undistort_image(img)
    show_image("Undistorted Image", undistorted_img, wait_time=10)
    input("Press Enter to continue...")


except Exception as e:
    print(f"Error: {e}")

finally:
    try:
        robot.close()
    except Exception:
        pass
