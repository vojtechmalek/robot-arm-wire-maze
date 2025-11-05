import cv2
from ctu_crs import CRS97
from core import PybulletRobot, base_detection
from time import sleep
    
# Configuration
maze = "C"
local_mode = True
use_gui = True

NUM_ANGLE_SAMPLES = 8
STEP = 1

if local_mode:
    aruco_img = cv2.imread("aruco_images/aruco_7.png")
else:
    robot = CRS97()
    aruco_img = robot.grab_image()

# detect maze position from image
print("[MAIN] Detecting maze position from image...")
maze_pose = base_detection.detect_base_position(aruco_img)

# Initialize robot simulation (with GUI for visualization)
print("[MAIN] Initializing robot simulation...")
pybullet_robot = PybulletRobot(maze=maze, maze_pose=maze_pose, use_gui=use_gui)

# Calculate state space
print("[MAIN] Calculating state space (this may take a while)...")
pybullet_robot.calculate_state_space(num_angle_samples=NUM_ANGLE_SAMPLES, step=STEP)

print("[MAIN] Finding path...")
q_path = pybullet_robot.get_q_path()

if use_gui:
    # Visualize state space (interactive plot)
    print("[MAIN] Opening interactive state space visualization...")
    print("[MAIN] Click or drag on the plot to move the robot to different configurations.")
    pybullet_robot.visualize_state_space()

if not local_mode:
    print("[MAIN] Moving robot according to path...")
    for q in q_path:
        robot.move_to_q(q)
        robot.wait_for_motion_stop()

# Clean up
pybullet_robot.close()
print("[MAIN] Done!")
