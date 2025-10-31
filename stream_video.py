import cv2
import numpy as np
from ctu_crs import CRS97  # or CRS93

robot = CRS97()

try:
    print("Initializing robot...")
    robot.initialize(home=False)
    # robot.move_to_q(robot.q_home)
    # robot.wait_for_motion_stop()

    while True:
        # print("Capturing image...")
        img = robot.grab_image()
        # scale down for faster display
        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

        if img is None or not isinstance(img, np.ndarray):
            raise RuntimeError("Error: grab_image() returned invalid image.")
        
        cv2.imshow('Live Video Stream', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


except Exception as e:
    print(f"Error: {e}")

finally:
    try:
        robot.close()
    except Exception:
        pass
    cv2.destroyAllWindows()
