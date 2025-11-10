import cv2
from ctu_crs import CRS97
from core import PybulletRobot, base_detection
from time import sleep
    
# Configuration
maze = "E"
local_mode = True
use_gui = True

NUM_ANGLE_SAMPLES = 32
STEP = 1
COLLISION_PADDING = 0.0  # Minimum distance to consider as collision (meters)
MAX_TILT_ANGLE = 45  # Maximum tilt angle to try (degrees)
NUM_TILT_DIRECTIONS = 8  # Number of tilt directions (4 = cardinal, 8 = cardinal + diagonal)

if local_mode:
    aruco_img = cv2.imread("aruco_images/aruco_5.png")
else:
    robot = CRS97()
    robot.initialize(home=False)  # initialize connection to the robot, perform hard and soft home
    robot.move_to_q(robot.q_home)
    robot.wait_for_motion_stop()
    aruco_img = robot.grab_image()

# detect maze position from image
print("[MAIN] Detecting maze position from image...")
maze_pose = base_detection.detect_base_position(aruco_img)

# Initialize robot simulation (with GUI for visualization)
print("[MAIN] Initializing robot simulation...")
pybullet_robot = PybulletRobot(maze=maze, maze_pose=maze_pose, use_gui=use_gui, collision_padding=COLLISION_PADDING)

# Find path (calculates state space on-the-fly)
print("[MAIN] Finding path")
q_path = pybullet_robot.find_path(
    num_angle_samples=NUM_ANGLE_SAMPLES, 
    step=STEP,
    max_tilt_angle=MAX_TILT_ANGLE,
    num_tilt_directions=NUM_TILT_DIRECTIONS
)

if use_gui:
    # Visualize state space (interactive plot)
    print("[MAIN] Opening interactive state space visualization...")
    print("[MAIN] Click or drag on the plot to move the robot to different configurations.")
    pybullet_robot.visualize_state_space()


if not local_mode and q_path is not None and len(q_path) > 0:
        
    # move to start of path
    robot.move_to_q(q_path[0])
    robot.wait_for_motion_stop()


    print("[MAIN] Moving robot according to path...")
    robot.initialize(home=False)
    for q in q_path:
        robot.move_to_q(q)
        robot.wait_for_motion_stop()

    sleep(10)  # small delay between movements

    for q in reversed(q_path):
        robot.move_to_q(q)
        robot.wait_for_motion_stop()

    robot.move_to_q(robot.q_home)
    robot.wait_for_motion_stop()

# Clean up
pybullet_robot.close()
print("[MAIN] Done!")
