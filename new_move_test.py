import time
import numpy as np
from ctu_crs import CRS97

# Import the new movement function and its helpers
from utils import move_robot_by_ring_pose

# ----------------------------------------------
# CONFIGURATION
# ----------------------------------------------
RING_CENTER = np.array([0.325, 0.0, 0.300])  # target ring center in meters

# Several test orientations [roll, pitch, yaw] in radians
# Small angles to avoid collisions; yaw sweeps illustrate rotation around Z
TEST_RPY_LIST = [
    np.array([-0.06,  0.05,  0.0]),      # neutral / default orientation
    np.array([-0.06,  0.05,  0.5]),      # yaw +0.5 rad (~28.6°)
    np.array([-0.06,  0.05, -0.5]),      # yaw -0.5 rad
    np.array([ 0.20,  0.05,  0.0]),      # roll +0.2 rad (~11°)
    np.array([-0.20,  0.05,  0.0]),      # roll -0.2 rad
    np.array([-0.06,  0.25,  0.0]),      # pitch +0.25 rad (~14°)
    np.array([-0.06, -0.10,  0.0]),      # pitch -0.10 rad (~-6°)
]

# ----------------------------------------------
# MAIN
# ----------------------------------------------
def main():
    robot = CRS97()
    try:
        print("Initializing robot...")
        robot.initialize()
        print("Moving to home position...")
        robot.move_to_q(robot.q_home)
        robot.wait_for_motion_stop()

        print("\n--- Starting ring-center pose tests ---")
        for rpy in TEST_RPY_LIST:
            print(f"Moving ring to {RING_CENTER} with RPY={np.round(rpy,3)}")
            try:
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
