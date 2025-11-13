from ctu_crs import CRS97 # or CRS97
import numpy as np

robot = CRS97()  # set argument tty_dev=None if you are not connected to robot,
# it will allow you to compute FK and IK offline
robot.initialize(home=False)  # initialize connection to the robot, perform hard and soft home

q0 = robot.get_q()

position = np.array([0.450, -0.30, 0.100])
rotation_euler = np.array([0, np.pi + 0, np.pi/2 + 0.0])

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
current_pose = np.eye(4)
current_pose[:3, :3] = R
current_pose[:3, 3] = position



ik_sols = robot.ik(current_pose)
filtered_ik_sols = [q for q in ik_sols if robot.in_limits(q)]
print(f"Found {len(filtered_ik_sols)} IK solutions.")
assert len(filtered_ik_sols) > 0

for i, q in enumerate(filtered_ik_sols):
    print(f"IK solution {i}: {q}")
    input("Press Enter to move to this solution...")
    # closest_solution = min(ik_sols, key=lambda q: np.linalg.norm(q - q0))
    if (not robot.in_limits(q)):
        continue

    robot.move_to_q(q)
    robot.wait_for_motion_stop()

