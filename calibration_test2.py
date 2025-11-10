import time
from ctu_crs import CRS97 # or CRS97
import numpy as np

from calibration.PybulletRobotDouble import PybulletRobotDouble
from core.pose import Pose

robot = CRS97() # set argument tty_dev=None if you are not connected to robot,
# it will allow you to compute FK and IK offline
robot.initialize(home=False)  # initialize connection to the robot, perform hard and soft home

q = robot.q_home

robot.move_to_q(q)
robot.wait_for_motion_stop()

q[5] = -np.pi
robot.move_to_q(q)
robot.wait_for_motion_stop()

q[5] = np.pi
robot.move_to_q(q)
robot.wait_for_motion_stop()

time.sleep(2)
