import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D plots)

WAYPOINT_FILE = "waypoints_M.json"  # path 

# ------------ CONFIG ------------
ARROW_STRIDE = 50       # draw orientation triads every Nth waypoint
DRAW_STRIDE  = 10     # draw a ring every Nth waypoint (can be same as ARROW_STRIDE)
AXIS_LEN     = 0.015    # length of axis arrows (m)
TOOL_AXIS    = 'X'      # which tool axis is aligned with path tangent: 'Z' or 'X'

RING_RADIUS  = 0.03     # ring radius in meters (distance from center to wire)
RING_RES     = 64       # how smooth the drawn circle is

# ------------ HELPERS ------------

def load_waypoints(path):
    with open(path, "r") as f:
        return json.load(f)

def rpy_zyx_to_rotm(roll, pitch, yaw):
    """
    Convert roll/pitch/yaw (radians) to rotation matrix using ZYX convention:
    R = Rz(yaw) * Ry(pitch) * Rx(roll)

    This must match how rpy was produced (rotm_to_rpy_zyx in generator).
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

def make_ring_points(center, R, tool_axis, radius, resolution):
    """
    Create 3D points of a ring (a circle) centered at `center`
    and oriented by rotation matrix R.

    If tool_axis == 'Z':
        - ring's plane is tool X-Y plane
        - circle is parameterized in (tool X, tool Y)
        - tool Z is ring normal
    If tool_axis == 'X':
        - ring's plane is tool Y-Z plane
        - tool X is ring normal
    """
    thetas = np.linspace(0, 2*np.pi, resolution)

    if tool_axis.upper() == 'Z':
        # local circle in XY plane of tool frame
        local_pts = np.stack([
            radius * np.cos(thetas),  # along tool X
            radius * np.sin(thetas),  # along tool Y
            np.zeros_like(thetas)     # no tool Z
        ], axis=1)  # shape (res, 3)

        # Map tool frame -> world: p_world = center + R * p_local
        world_pts = center + local_pts @ R.T  # (res,3)

    elif tool_axis.upper() == 'X':
        # local circle in YZ plane of tool frame
        local_pts = np.stack([
            np.zeros_like(thetas),    # no tool X
            radius * np.cos(thetas),  # along tool Y
            radius * np.sin(thetas),  # along tool Z
        ], axis=1)

        world_pts = center + local_pts @ R.T

    else:
        raise ValueError("Unsupported TOOL_AXIS, use 'Z' or 'X'.")

    return world_pts  # (res,3)

# ------------ MAIN ------------

def main():
    waypoints = load_waypoints(WAYPOINT_FILE)

    # Extract XYZ path for continuous line plot
    xs = [wp["pos"][0] for wp in waypoints]
    ys = [wp["pos"][1] for wp in waypoints]
    zs = [wp["pos"][2] for wp in waypoints]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the full path
    ax.plot(xs, ys, zs, label="path")

    # Mark start and end
    ax.scatter(xs[0],  ys[0],  zs[0],  marker='o', s=50, label="start")
    ax.scatter(xs[-1], ys[-1], zs[-1], marker='^', s=50, label="end")

    # Draw orientations + rings at stride
    for k in range(0, len(waypoints), ARROW_STRIDE):
        wp = waypoints[k]

        pos = np.array(wp["pos"])       # [x,y,z] (m)
        roll, pitch, yaw = wp["rpy"]    # radians

        # rotation matrix of tool frame in world frame
        R = rpy_zyx_to_rotm(roll, pitch, yaw)

        # Tool frame basis vectors in world coords
        x_axis = R[:,0]
        y_axis = R[:,1]
        z_axis = R[:,2]

        # Pick which axis is the ring normal (the direction of travel)
        if TOOL_AXIS.upper() == 'Z':
            main_axis = z_axis
        elif TOOL_AXIS.upper() == 'X':
            main_axis = x_axis
        else:
            main_axis = z_axis  # fallback

        # 1. Draw a main axis arrow (shows direction the ring is "heading")
        ax.quiver(
            pos[0], pos[1], pos[2],
            main_axis[0]*AXIS_LEN,
            main_axis[1]*AXIS_LEN,
            main_axis[2]*AXIS_LEN,
            length=1.0,
            normalize=False,
            linewidth=1.5
        )

        # 2. Draw the tool triad (X=red, Y=green, Z=blue)
        triad_scale = AXIS_LEN * 0.6

        ax.plot(
            [pos[0], pos[0] + x_axis[0]*triad_scale],
            [pos[1], pos[1] + x_axis[1]*triad_scale],
            [pos[2], pos[2] + x_axis[2]*triad_scale],
            color='tab:red',
            linewidth=1,
        )
        ax.plot(
            [pos[0], pos[0] + y_axis[0]*triad_scale],
            [pos[1], pos[1] + y_axis[1]*triad_scale],
            [pos[2], pos[2] + y_axis[2]*triad_scale],
            color='tab:green',
            linewidth=1,
        )
        ax.plot(
            [pos[0], pos[0] + z_axis[0]*triad_scale],
            [pos[1], pos[1] + z_axis[1]*triad_scale],
            [pos[2], pos[2] + z_axis[2]*triad_scale],
            color='tab:blue',
            linewidth=1,
        )

    # Draw the ring itself (circle in the tool plane) every DRAW_STRIDE points
    for k in range(0, len(waypoints), DRAW_STRIDE):
        wp = waypoints[k]

        pos = np.array(wp["pos"])
        roll, pitch, yaw = wp["rpy"]
        R = rpy_zyx_to_rotm(roll, pitch, yaw)

        ring_pts_world = make_ring_points(
            center=pos,
            R=R,
            tool_axis=TOOL_AXIS,
            radius=RING_RADIUS,
            resolution=RING_RES,
        )

        ax.plot(
            ring_pts_world[:,0],
            ring_pts_world[:,1],
            ring_pts_world[:,2],
            color='orange',
            linewidth=1.0,
        )

    # Labels, formatting
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()

    # roughly equal aspect ratio so it doesn't look squished
    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    zs_arr = np.array(zs)

    max_range = np.array([
        xs_arr.max()-xs_arr.min(),
        ys_arr.max()-ys_arr.min(),
        zs_arr.max()-zs_arr.min()
    ]).max() / 2.0

    mid_x = (xs_arr.max()+xs_arr.min()) * 0.5
    mid_y = (ys_arr.max()+ys_arr.min()) * 0.5
    mid_z = (zs_arr.max()+zs_arr.min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

if __name__ == "__main__":
    main()
