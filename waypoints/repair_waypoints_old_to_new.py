import json
import numpy as np

INPUT_FILE = "waypoints_E.json"   # reads and overwrites this file


def rpy_zyx_to_rotm(roll, pitch, yaw):
    """
    Convert roll/pitch/yaw (radians) -> 3x3 rotation matrix using ZYX convention:
    R = Rz(yaw) * Ry(pitch) * Rx(roll)

    This reconstructs the same matrix that was originally saved in rpy,
    i.e. TOOL->WORLD in your old data.
    """
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)

    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]])
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0],
                   [0, cr,-sr],
                   [0, sr, cr]])

    return Rz @ Ry @ Rx  # 3x3


def rotm_to_rpy_zyx(R):
    """
    Convert rotation matrix -> [roll, pitch, yaw] using ZYX convention.
    Returns roll, pitch, yaw in radians.
    """
    r11, r12, r13 = R[0,0], R[0,1], R[0,2]
    r21, r22, r23 = R[1,0], R[1,1], R[1,2]
    r31, r32, r33 = R[2,0], R[2,1], R[2,2]

    yaw   = np.arctan2(r21, r11)
    pitch = np.arcsin(-r31)
    roll  = np.arctan2(r32, r33)

    return [float(roll), float(pitch), float(yaw)]


def convert_file(path):
    # load
    with open(path, "r") as f:
        waypoints = json.load(f)

    for wp in waypoints:
        roll, pitch, yaw = wp["rpy"]

        # 1) rebuild TOOL->WORLD rotation from old rpy
        R_tool_to_world = rpy_zyx_to_rotm(roll, pitch, yaw)

        # 2) convert to WORLD->TOOL by transpose
        R_world_to_tool = R_tool_to_world.T

        # 3) get new rpy in world-coords convention
        new_rpy = rotm_to_rpy_zyx(R_world_to_tool)

        wp["rpy"] = new_rpy

    # overwrite same file
    with open(path, "w") as f:
        json.dump(waypoints, f, indent=2)

    print(f"Updated RPY in {path}")


if __name__ == "__main__":
    convert_file(INPUT_FILE)
