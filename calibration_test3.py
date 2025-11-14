import time
from ctu_crs import CRS97 # or CRS97
import numpy as np

from calibration.PybulletRobotDouble import PybulletRobotDouble
from core.pose import Pose

robot = CRS97() # set argument tty_dev=None if you are not connected to robot,
# it will allow you to compute FK and IK offline
robot.initialize(home=False)  # initialize connection to the robot, perform hard and soft home


Z_OFFSET = 0.01  # meters, vertical offset based on hoop angle

step_count = 100
# make array 0 to step_count-1 and back to 0
angles = np.concatenate([np.linspace(0, np.pi/2, step_count), np.linspace(np.pi/2, 0, step_count)])

for i in range(angles.shape[0]):
    angle = angles[i]

    position = np.array([0.650, -0.10, 0.195])
    # position -= np.array([0.135, 0.0, 0.0])
    # rotation_euler = np.array([-0.06, np.pi + 0.15, 0.0])
    rotation_euler = np.array([angle, np.pi + angle, angle])


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

    flange_z_offset = -(0.01 + 0.00325) # meters

    movement_offset_matrix = np.eye(4)
    movement_offset_matrix[:3, 3] = np.array([flange_length, 0.0, flange_z_offset])
    target_pose = target_pose @ movement_offset_matrix

    #! TEMP: adjust height based on angle
    # get hoop rotation z vector
    hoop_z = target_pose[:3, 0]
    world_z = np.array([0, 0, 1])
    z_dot_product = np.dot(hoop_z, world_z)

    print("A")
    print(target_pose)
    correction_term = Z_OFFSET * np.abs(z_dot_product)
    print("correction term:", correction_term)
    target_pose[2, 3] -= correction_term
    # pose._p[2] -= 0.05

    print(target_pose)


    ik_sols = robot.ik(target_pose)
    assert len(ik_sols) > 0
    # print(f"Found {len(ik_sols)} IK solutions.")

    q0 = robot.get_q()
    closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - q0))

    robot.move_to_q(closest_solution)
    # robot.wait_for_motion_stop()
    
    time.sleep(0.00001)

time.sleep(2)
