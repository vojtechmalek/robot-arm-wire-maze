import time
import numpy as np
from ctu_crs import CRS97

# Import the new movement function and its helpers
from utils import move_robot_by_ring_pose

# ----------------------------------------------
# CONFIGURATION
# ----------------------------------------------
RING_CENTER = np.array([0.5, 0.0, 0.20])  # target ring center in meters

# Several test orientations [roll, pitch, yaw] in radians
# Small angles to avoid collisions; yaw sweeps illustrate rotation around Z
TEST_RPY_LIST = [
    np.array([ 0.00,  np.pi+0.00,  0.00]),
    np.array([ 0.00,  np.pi+0.00,  0.40]),
    np.array([ 0.00,  np.pi+0.00,  0.80]), 
    np.array([ 0.00,  np.pi+0.00,  1.20]),
    np.array([ 0.00,  np.pi+0.00,  1.60]),
    np.array([ 0.00,  np.pi+0.00,  2.00]),
    np.array([ 0.00,  np.pi+0.00,  2.40]),
]

# ----------------------------------------------
# MAIN
# ----------------------------------------------
def main():
    robot = CRS97()
    try:
        print("Initializing robot...")
        robot.initialize(home=False)
        print("Moving to home position...")
        robot.move_to_q(robot.q_home)
        robot.wait_for_motion_stop()

        print("\n--- Starting ring-center pose tests ---")
        for rpy in TEST_RPY_LIST:
            print(f"Moving ring to {RING_CENTER} with RPY={np.round(rpy,3)}")
            try:
                print("Starting new move ...")
                move_robot_by_ring_pose(robot, RING_CENTER, rpy, True)
                print("Move complete.")
            except Exception as e:
                print(f"❌ Error at orientation {rpy}: {e}")
            time.sleep(2.0)  # short pause between moves so you can observe

        print("\n✅ Test sequence finished successfully.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        try:
            robot.close()
        except Exception:
            pass
        print("Robot connection closed.")

if __name__ == "__main__":
    main()
