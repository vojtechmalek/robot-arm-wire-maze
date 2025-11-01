import cv2
import numpy as np
from ctu_crs import CRS97  # or CRS93
from PybulletRobot import PybulletRobot
import base_detection
import utils
import json
import time

robot = PybulletRobot(maze_rotation_deg=-16.288773)

def get_waypoint_pose(waypoint, base_center_robot, center_rot, angle_offset = 0.0) -> np.ndarray:
    pos = np.array(waypoint["pos"])
    rpy = np.array(waypoint["rpy"])
    # print(f"Moving to waypoint: pos={pos}, rpy={rpy}")

    pose_matrix = np.eye(4)
    # pose_matrix = pose_matrix @ utils.rotxyz_se3(rpy[0], rpy[1], rpy[2])
    pose_matrix = pose_matrix @ utils.trans_se3_vec(base_center_robot)

    pose_matrix = pose_matrix @ utils.rotzyx_se3(0, 0, (center_rot + 90) * np.pi / 180)

    pose_matrix = pose_matrix @ utils.trans_se3_vec(pos + np.array([-0.005, -0.005, 0.0]))
    # pose_matrix = pose_matrix @ utils.trans_se3_vec(pos)

    # pose_matrix = pose_matrix @ utils.rotzyx_se3(0, 0, -(-center_rot + 90) * np.pi / 180)

    # LVL B needs a 180 deg rotation around Z
    # pose_matrix = pose_matrix @ utils.rotzyx_se3(0, np.pi, -np.pi/2)
    pose_matrix = pose_matrix @ utils.rotzyx_se3(0, np.pi, np.pi/2 + angle_offset)

    return pose_matrix

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
    
    base_center_robot = np.array([xy1[0], xy1[1], 0.032])
    base_center_robot_se3 = utils.trans_se3(xy1[0], xy1[1], 0.032)

    print("Moving robot above the marker...")
    # Move robot above the marker

    # load json file from path
    waypoints_raw = json.load(open("waypoints/waypoints_C.json", "r"))

    # LVL B needs a 180 deg rotation around Z
    utils.move_robot_by_ring_pose(robot, base_center_robot + np.array([0.0, 0.0, 0.3]), np.array([0., np.pi, -np.pi/2]))


    used_waypoints_indices = []
    used_waypoints_poses = []

    # first point rotation correction
    # wp_first = waypoints_raw[0]
    # wp_first["pos"] = np.array([0.0, 0.0, 0.3])
    # pose_matrix_correction = get_waypoint_pose(wp_first, base_center_robot, center_rot)
    # utils.move_robot_by_ring_pose_matrix(robot, pose_matrix_correction, False)

    prev_angle_offset = 0.0
    for i in range(0, len(waypoints_raw), 5):
        wp = waypoints_raw[i]
        print(f"Moving to waypoint {i}:")

        percentage_of_path = i / len(waypoints_raw)

        quit_loop = False
        moved = False

        pose_matrix = np.eye(4)

        for angle_offset in np.linspace(0.0, 2 * np.pi, num=16):
            angle_offset = angle_offset + prev_angle_offset
            # angle_offset = percentage_of_path * np.pi
            pose_matrix = get_waypoint_pose(wp, base_center_robot, center_rot, angle_offset)

            if pose_matrix[2, 3] < 0.06:
                quit_loop = True
                break

            # utils.move_robot_by_ring_pose(robot, position, rotation_euler, False)
            if not utils.move_robot_by_ring_pose_matrix(robot, pose_matrix, False):
                print(f"  - Failed to move with angle offset {angle_offset:.2f} rad.")
                continue
            
            used_waypoints_indices.append(i)
            used_waypoints_poses.append(pose_matrix)

            moved = True
            prev_angle_offset = angle_offset
            break

        if quit_loop:
            break

        if not moved:
            print(f"Failed to move to waypoint {i} with any angle offset.")
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
