from ctu_crs import CRS93 # or CRS97

robot = CRS93()  # set argument tty_dev=None if you are not connected to robot,
# it will allow you to compute FK and IK offline
robot.initialize(home=False)  # initialize connection to the robot, perform hard and soft home

print(robot.get_q())

# [-4.71238898e-05 -1.57079633e-05 -7.85413871e-01  3.11048778e-05 -7.85366747e-01  0.00000000e+00]
home_q = robot.q_home
print(home_q)

robot.move_to_q(home_q)
robot.wait_for_motion_stop()