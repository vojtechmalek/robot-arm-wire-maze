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

    # detect aruco markers
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    corners, ids, rejectedImgPoints = detector.detectMarkers(img)

    if ids is not None:
        print(f"Detected {len(ids)} markers: {ids.flatten()}")
        img_markers = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)

        if len(ids) == 2:
            print("Two markers detected, calculating distance...")
            # Calculate the center points of the two markers
            center1 = np.mean(corners[0][0], axis=0)
            center2 = np.mean(corners[1][0], axis=0)

            # Calculate center of the centers
            center_of_centers = (center1 + center2) / 2
            cv2.circle(img_markers, tuple(center_of_centers.astype(int)), 5, (0, 255, 0), -1)

        for i, corner in enumerate(corners):
            # calculate angle of the marker
            vector = corner[0][1] - corner[0][0]
            angle = np.arctan2(vector[1], vector[0]) * 180 / np.pi
            cv2.putText(img_markers, f"Angle: {angle:.1f}", tuple(corner[0][0].astype(int) - np.array([0, 10])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("Detected Aruco Markers", img_markers)
        cv2.waitKey(0)
    else:
        print("No markers detected.")

except Exception as e:
    print(f"Error: {e}")

finally:
    try:
        robot.close()
    except Exception:
        pass
    cv2.destroyAllWindows()
