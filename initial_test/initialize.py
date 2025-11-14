from ctu_crs import CRS97 # or CRS97

robot = CRS97()  # set argument tty_dev=None if you are not connected to robot,
# it will allow you to compute FK and IK offline
robot.initialize()  # initialize connection to the robot, perform hard and soft home
