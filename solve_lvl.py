import cv2
import numpy as np
from ctu_crs import CRS97  # or CRS93
import PybulletRobot
import utils
import json
import time

robot = PybulletRobot.PybulletRobot()


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

    center_pos = np.array([0,0])
    center_rot = 0.0

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
            center_pos = center_of_centers

        for i, corner in enumerate(corners):
            # calculate angle of the marker
            vector = corner[0][1] - corner[0][0]
            angle = np.arctan2(vector[1], vector[0]) * 180 / np.pi
            cv2.putText(img_markers, f"Angle: {angle:.1f}", tuple(corner[0][0].astype(int) - np.array([0, 10])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            center_rot = angle

        # scale down for display
        img_markers = cv2.resize(img_markers, (0,0), fx=0.5, fy=0.5)
        cv2.imshow("Detected Aruco Markers", img_markers)
        cv2.waitKey(0)
    else:
        print("No markers detected.")
        exit(-1)

    H = np.load("homography.npy")

    uv1 = np.array([center_pos[0], center_pos[1], 1.0])
    xy1 = H @ uv1
    xy1 /= xy1[2]

    print(f"Marker position in robot frame: x={xy1[0]:.3f}, y={xy1[1]:.3f}, rotation={center_rot:.2f} deg")
    
    base_center_robot = np.array([xy1[0], xy1[1], 0.054])

    print("Moving robot above the marker...")
    # Move robot above the marker

    # load json file from path
    waypoints_raw = json.load(open("waypoints/waypoints_A.json", "r"))

    utils.move_robot_by_ring_pose(robot, base_center_robot + np.array([0.0, 0.0, 0.3]), np.array([0., np.pi, np.pi/2]))

    used_waypoints_indices = []
    for i in range(0, len(waypoints_raw), 10):
        wp = waypoints_raw[i]
        pos = np.array(wp["pos"])
        rpy = np.array(wp["rpy"])
        print(f"Moving to waypoint {i}: pos={pos}, rpy={rpy}")

        position = base_center_robot + pos
        rotation_euler = np.array([0., np.pi, np.pi/2])

        if position[2] < 0.06:
            break

        used_waypoints_indices.append(i)

        utils.move_robot_by_ring_pose(robot, position, rotation_euler, False)
        # time.sleep(0.01)

    time.sleep(2.0)

    print("Moving robot back up...")


    # Move robot back up
    reversed_indices = used_waypoints_indices[::-1]
    for i in reversed_indices:
        wp = waypoints_raw[i]
        pos = np.array(wp["pos"])
        rpy = np.array(wp["rpy"])
        print(f"Moving back to waypoint {i}: pos={pos}, rpy={rpy}")

        position = base_center_robot + pos
        rotation_euler = np.array([0., np.pi, np.pi/2])

        utils.move_robot_by_ring_pose(robot, position, rotation_euler, False)
        # time.sleep(0.01)

    # home the robot
    print("Homing the robot...")
    robot.move_to_q(robot.q_home)
    robot.wait_for_motion_stop()

    print("LET'S GO DONE!")


except Exception as e:
    print(f"Error: {e}")

finally:
    try:
        robot.close()
    except Exception:
        pass
    cv2.destroyAllWindows()
