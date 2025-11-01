import cv2
import numpy as np
from ctu_crs import CRS97  # or CRS93
from PybulletRobot import PybulletRobot
import base_detection
import utils
import json
import time

robot = PybulletRobot(maze_rotation_deg=-17.2, maze_urdf="robot/mazes/B.urdf")


try:
    print("Initializing robot...")
    robot.initialize(home=False)
    robot.move_to_q(robot.q_home)
    robot.wait_for_motion_stop()

    print("Capturing image...")
    img = robot.grab_image()

    xy1, center_rot = base_detection.detect_base_position(img, True)
    center_rot = np.rad2deg(center_rot)

    print(f"Marker position in robot frame: x={xy1[0]:.3f}, y={xy1[1]:.3f}, rotation={center_rot:.2f} deg")
    
    base_center_robot = np.array([xy1[0], xy1[1], 0.054])
    base_center_robot_se3 = utils.trans_se3(xy1[0], xy1[1], 0.054)

    print("Moving robot above the marker...")
    # Move robot above the marker

    # load json file from path
    waypoints_raw = json.load(open("waypoints/waypoints_B.json", "r"))

    # LVL B needs a 180 deg rotation around Z
    utils.move_robot_by_ring_pose(robot, base_center_robot + np.array([0.0, 0.0, 0.3]), np.array([0., np.pi, -np.pi/2]))

    used_waypoints_indices = []
    used_waypoints_poses = []
    for i in range(0, len(waypoints_raw), 10):
        wp = waypoints_raw[i]
        pos = np.array(wp["pos"])
        rpy = np.array(wp["rpy"])
        print(f"Moving to waypoint {i}: pos={pos}, rpy={rpy}")

        pose_matrix = np.eye(4)
        # pose_matrix = pose_matrix @ utils.rotxyz_se3(rpy[0], rpy[1], rpy[2])
        pose_matrix = pose_matrix @ utils.trans_se3_vec(base_center_robot)

        pose_matrix = pose_matrix @ utils.rotzyx_se3(0, 0, (center_rot + 90) * np.pi / 180)

        pose_matrix = pose_matrix @ utils.trans_se3_vec(pos)
        # pose_matrix = pose_matrix @ utils.trans_se3(0, 0, 0.054)

        pose_matrix = pose_matrix @ utils.rotzyx_se3(0, 0, -(center_rot + 90) * np.pi / 180)

        # LVL B needs a 180 deg rotation around Z
        pose_matrix = pose_matrix @ utils.rotzyx_se3(0, np.pi, -np.pi/2)


        # if pose_matrix[2, 3] < 0.06:
        #     break

        position = base_center_robot + pos
        rotation_euler = np.array([0., np.pi, np.pi/2])

        position_from_matrix = pose_matrix[:3, 3]
        position_difference = position_from_matrix - position
        print(f"difference: {position_difference}")
        
        if position[2] < 0.06:
            break


        used_waypoints_indices.append(i)
        used_waypoints_poses.append(pose_matrix)

        # utils.move_robot_by_ring_pose(robot, position, rotation_euler, False)
        if not utils.move_robot_by_ring_pose_matrix(robot, pose_matrix, False):
            print(f"Failed to move to waypoint {i}")
            robot.move_to_q(robot.q_home)
            robot.wait_for_motion_stop()
            exit(-1)


        # time.sleep(0.01)

    time.sleep(2.0)

    print("Moving robot back up...")


    # Move robot back up
    reversed_poses = used_waypoints_poses[::-1]
    for pose_matrix in reversed_poses:
        utils.move_robot_by_ring_pose_matrix(robot, pose_matrix, False)
        # time.sleep(0.01)

    # home the robot
    print("Homing the robot...")
    robot.move_to_q(robot.q_home)
    robot.wait_for_motion_stop()

    print("LET'S GO DONE!")


except Exception as e:
    print(f"Error: {e}")
    print(f"Line number:", e.__traceback__.tb_lineno)
    print(f"Traceback:", e.__traceback__)


finally:
    try:
        robot.close()
    except Exception:
        pass
    cv2.destroyAllWindows()
