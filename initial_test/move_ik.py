from ctu_crs import CRS93 # or CRS97
import numpy as np

robot = CRS93()  # set argument tty_dev=None if you are not connected to robot,
# it will allow you to compute FK and IK offline
robot.initialize(home=False)  # initialize connection to the robot, perform hard and soft home

q0 = robot.q_home
print(q0)

current_pose = robot.fk(robot.get_q())
print(current_pose)

current_pose[:3, 3] -= np.array([0.0, 0.0, 0.1])
print(current_pose)

# current_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
print(current_pose)


ik_sols = robot.ik(current_pose)
assert len(ik_sols) > 0
closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - q0))
robot.move_to_q(closest_solution)
robot.wait_for_motion_stop()

