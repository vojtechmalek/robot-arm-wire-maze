import cv2
from ctu_crs import CRS97
import numpy as np
from core import PybulletRobot, base_detection, utils
from time import sleep
from core.pose import Pose
    
# Configuration
maze = "D"
local_mode = False
use_gui = True

NUM_ANGLE_SAMPLES = 32
STEP = 1

if local_mode:
    aruco_img = cv2.imread("aruco_images/aruco_5.png")
else:
    robot = CRS97()
    robot.initialize(home=False)  # initialize connection to the robot, perform hard and soft home
    # robot.move_to_q(robot.q_home)
    # robot.wait_for_motion_stop()
    # aruco_img = robot.grab_image()
    aruco_img = cv2.imread("aruco_images/aruco_5.png")

print(robot.fk(robot.get_q()))

# detect maze position from image
print("[MAIN] Detecting maze position from image...")
maze_pose = base_detection.detect_base_position(aruco_img)

if not local_mode:

    # test_pose = maze_pose @ pose.Pose(p=[0, 0, 0.03])
    # test_pose = test_pose @ pose.Pose(rpy=[0, 0, 4* np.pi/2])

    test_pose = np.eye(4)
    # set position
    # test_pose[0:3, 3] = [0.450, 0.0, 0.037]
    test_pose[0:3, 3] = [0.450, 0.0, 0.036 + 0.026]
    pose = Pose(T=test_pose)

    pose = pose @ Pose(rpy=np.array([0, 0, np.pi/2]))
    pose = pose @ Pose(rpy=np.array([2 * np.pi/2, 0, 0]))
    pose = pose @ Pose(rpy=np.array([0, np.pi/2, 0]))
    # pose = pose @ Pose(rpy=np.array([-np.pi/2, 0, 0]))
    # pose = pose @ Pose(rpy=np.array([0, 0, np.pi/2]))

    utils.move_robot_by_ring_pose(robot, pose)
    robot.wait_for_motion_stop()

    # save image
    img = robot.grab_image()
    cv2.imwrite("test_angle2.png", img)

    # angle_test_samples = 16
    # for i in range(angle_test_samples):
    #     angle = i * (2 * np.pi) / angle_test_samples
    #     print(f"[MAIN] Moving to ring pose at angle {angle:.2f} rad...")
    #     ring_pose = pose @ Pose(rpy=np.array([0, 0, angle]))
    #     utils.move_robot_by_ring_pose(robot, ring_pose)
    #     robot.wait_for_motion_stop()


# Clean up
print("[MAIN] Done!")
