import cv2
import numpy as np
from ctu_crs import CRS97  # or CRS93

robot = CRS97()


# TODO: move to packages
def move_robot_xyz(robot, position: np.ndarray):
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
    
    print(f"Found {len(ik_sols)} IK solutions.")

    # sort solutions by distance to current joint state
    ik_sols.sort(key=lambda q: np.linalg.norm(q - robot.get_q()))

    for q in ik_sols:
        try:
            robot.move_to_q(q)
            robot.wait_for_motion_stop()
            print("Moved to IK solution:", q)
            return
        except Exception as e:
            print(f"Failed to move to IK solution {q}: {e}")
        
    raise RuntimeError("Failed to move to any IK solution.")

    # closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - robot.get_q()))
    # robot.move_to_q(closest_solution)
    # robot.wait_for_motion_stop()



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
    
    base_center_robot = np.array([xy1[0], xy1[1], 0.054])  # set Z to 15 cm above the marker

    print("Moving robot above the marker...")
    # Move robot above the marker
    
    move_robot_xyz(robot, base_center_robot + np.array([0.0, 0.0, 0.3]))

    print("Moving robot down to the marker...")
    # Move robot down to the marker
    # move in smaller steps to avoid collisions
    for z in np.linspace(0.3, 0.02, num=10):
        move_robot_xyz(robot, base_center_robot + np.array([0.0, 0.0, z]))
    # move_robot_xyz(robot, base_center_robot + np.array([0.0, 0.0, 0.02]))
    
    # wait for a moment
    import time
    time.sleep(1.0)

    print("Moving robot back up...")
    # Move robot back up
    for z in np.linspace(0.02, 0.3, num=10):
        move_robot_xyz(robot, base_center_robot + np.array([0.0, 0.0, z]))
    # move_robot_xyz(robot, base_center_robot + np.array([0.0, 0.0, 0.3]))

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
