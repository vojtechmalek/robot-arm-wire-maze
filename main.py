import cv2
from ctu_crs import CRS97
from core import PybulletRobot, base_detection
from time import sleep
from datetime import datetime

from detection.detect_maze import get_maze_type
from scripts.camera_distortion import undistort_image, show_image

    
# Configuration
local_mode = True # If you want to run simulation on your own computer, without the actual robot (uses pre-captured images)
use_gui = True # If you want to see the simulation to open a gui window
run_robot_calibration = False # If you want to run the initialization sequence of the robot

NUM_ANGLE_SAMPLES = 30 # Number of angle samples to try for each waypoint (360 degrees / NUM_ANGLE_SAMPLES = angle step)
STEP = 1 # Step size for the pathfinding algorithm (1 = every waypoint, 2 = every other waypoint, etc.)
COLLISION_PADDING = 0.02  # Minimum distance to consider as collision (meters)
MAX_TILT_ANGLE = 40  # Maximum tilt angle to try (degrees)
NUM_TILT_DIRECTIONS = 8  # Number of tilt directions (4 = cardinal, 8 = cardinal + diagonal)

if local_mode: # What aruco image to use in local mode
    aruco_img = cv2.imread("aruco_images/aruco_20.png")
    # 1-4 A
    # 5-8 B
    # 9-12 C
    # 13-16 D # no solution for: 13,14,15
    # 17-22 E # no solution for: 17, 21, 22
else:
    # If not local mode, connect to the actual robot, initialize and grab an image
    robot = CRS97()
    robot.initialize(home=False)
    robot.move_to_q(robot.q_home)
    robot.wait_for_motion_stop()
    aruco_img = robot.grab_image()

# undistort image using saved calibration
aruco_img = undistort_image(aruco_img)

# Detect the maze type from the image
maze = get_maze_type(aruco_img)
print(f"[MAIN] Detected maze type: {maze}")

# Confirm maze type with user
user_input = input(f"Detected maze type is '{maze}'. Is this correct? Press enter to confirm or type correct maze type (A, B, C, D, E): ").strip().upper()
if user_input in ["A", "B", "C", "D", "E"]:
    maze = user_input
    print(f"[MAIN] Using maze type: {maze}")
else:
    print(f"[MAIN] Confirmed maze type: {maze}")

# Detect maze position from image
print("[MAIN] Detecting maze position from image...")
maze_pose = base_detection.detect_base_position(aruco_img, display=False)

# Initialize robot simulation
print("[MAIN] Initializing robot simulation...")
pybullet_robot = PybulletRobot(maze=maze, maze_pose=maze_pose, use_gui=use_gui, collision_padding=COLLISION_PADDING)

if not local_mode and run_robot_calibration:
    print("[MAIN] Running robot calibration...")
    robot.initialize()

# Find path
print("[MAIN] Finding path...")
start_time = datetime.now()
q_path = pybullet_robot.find_path(
    num_angle_samples=NUM_ANGLE_SAMPLES, 
    step=STEP,
    max_tilt_angle=MAX_TILT_ANGLE,
    num_tilt_directions=NUM_TILT_DIRECTIONS
)
end_time = datetime.now()

if not local_mode and run_robot_calibration:
    print("[MAIN] Moving robot to home position...")
    robot.move_to_q(robot.q_home)
    robot.wait_for_motion_stop()

# Calculate and print total time
elapsed_time = (end_time - start_time).total_seconds()
print(f"[MAIN] Pathfinding completed in {elapsed_time:.3f} seconds")

if use_gui:
    # Visualize state space (interactive plot)
    print("[MAIN] Opening interactive state space visualization...")
    print("[MAIN] Click or drag on the plot to move the robot to different configurations.")
    pybullet_robot.visualize_state_space()


# If we are not in local mode and a path was found, move according to path
if not local_mode and q_path is not None and len(q_path) > 0:
        
    # move to start of path
    robot.move_to_q(q_path[0])
    robot.wait_for_motion_stop()


    print("[MAIN] Moving robot according to path...")
    robot.initialize(home=False)
    for q in q_path:
        robot.move_to_q(q)
        robot.wait_for_motion_stop()

    sleep(2)  # small delay between movements

    for q in reversed(q_path):
        robot.move_to_q(q)
        robot.wait_for_motion_stop()

    robot.move_to_q(robot.q_home)
    robot.wait_for_motion_stop()

# Clean up
pybullet_robot.close()
print("[MAIN] Done!")
