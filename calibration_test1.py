import time
from ctu_crs import CRS97 # or CRS97
import numpy as np

from calibration.PybulletRobotDouble import PybulletRobotDouble
from core.pose import Pose

robot = PybulletRobotDouble()  # set argument tty_dev=None if you are not connected to robot,
# it will allow you to compute FK and IK offline
robot.initialize(home=False)  # initialize connection to the robot, perform hard and soft home

step_count = 1000
# make array 0 to step_count-1 and back to 0
angles = np.concatenate([np.linspace(0, np.pi/2, step_count), np.linspace(np.pi/2, 0, step_count)])

for i in range(angles.shape[0]):
    angle = angles[i]

    position = np.array([0.550, 0.0, 0.254])
    # position -= np.array([0.135, 0.0, 0.0])
    # rotation_euler = np.array([-0.06, np.pi + 0.15, 0.0])
    rotation_euler = np.array([0, np.pi + angle, np.pi/2 + 0])


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

    target_pose = np.eye(4)
    target_pose[:3, :3] = R
    target_pose[:3, 3] = position



    # offset by flange length along tool X axis
    flange_length = 0.135  # meters, distance to hoop center
    # flange_length = 0.135 + 0.0696/2  # meters, distance to end of hoop

    flange_z_offset = -0.013 # meters

    movement_offset_matrix = np.eye(4)
    movement_offset_matrix[:3, 3] = np.array([flange_length, 0.0, flange_z_offset])
    target_pose = target_pose @ movement_offset_matrix

    # print(target_pose)


    ik_sols = robot.ik(target_pose)
    assert len(ik_sols) > 0
    # print(f"Found {len(ik_sols)} IK solutions.")

    q0 = robot.get_q()
    closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - q0))

    robot._set_q(closest_solution)
    # robot.wait_for_motion_stop()
    
    time.sleep(0.005)

time.sleep(2)
