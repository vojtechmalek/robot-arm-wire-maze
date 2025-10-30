import cv2
import numpy as np
from ctu_crs import CRS97  # or CRS93

robot = CRS97()

try:
    print("Initializing robot...")
    robot.initialize()

    print("Capturing image...")
    img = robot.grab_image()

    if img is None or not isinstance(img, np.ndarray):
        raise RuntimeError("Error: grab_image() returned invalid image.")

    filename = "capture.png"
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
