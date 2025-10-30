from ctu_crs import CRS93 # or CRS97
import numpy as np

robot = CRS93()  # set argument tty_dev=None if you are not connected to robot,
# it will allow you to compute FK and IK offline
robot.initialize(home=False)  # initialize connection to the robot, perform hard and soft home

q0 = robot.get_q()
print(q0)

current_pose = robot.fk(robot.get_q())
print(current_pose)

current_pose[:3, 3] -= np.array([0.0, 0.0, 0.1])
print(current_pose)


rotation_euler_left_bottom_maybe = np.array([-0.06, np.pi + 0.15, 0.0])
left_bottom_corner = np.array([0.40, -0.2, 0.054])
left_top_corner = np.array([0.250, -0.2, 0.054])
right_top_corner = np.array([0.250, 0.2, 0.054])
right_bottom_corner = np.array([0.40, 0.2, 0.054])



position = np.array([0.40, -0.2, 0.054])
rotation_euler = np.array([-0.06, np.pi + 0.15, 0.0])


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
current_pose[:3, :3] = R
current_pose[:3, 3] = position


# current_pose = np.array([[0, 0, 1, 0.4],
#                          [0, 1, 0, 0],
#                          [-1, 0, 0, 0.64],
#                          [0, 0, 0, 1]])
print(current_pose)


ik_sols = robot.ik(current_pose)
assert len(ik_sols) > 0
closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - q0))
robot.move_to_q(closest_solution)
robot.wait_for_motion_stop()

