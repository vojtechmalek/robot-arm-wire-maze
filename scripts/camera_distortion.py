import cv2
import numpy as np
from ctu_crs import CRS97  # or CRS93

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
    
    # Calculate and display distortion from checkeredboard pattern
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Captured Image', gray)
    cv2.waitKey(1000)
 
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (6,6), None)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print("found corners:", ret)
    # If found, add object points, image points (after refining them)
    if ret == True:
 
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
 
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7,6), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(5000)

except Exception as e:
    print(f"Error: {e}")

finally:
    try:
        robot.close()
    except Exception:
        pass
    cv2.destroyAllWindows()
