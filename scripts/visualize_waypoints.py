import sys
from pathlib import Path
# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json

from core import utils


waypoints_raw = json.load(open("waypoints/opt_waypoint_paths/waypoints_E.json", "r"))
# use only every nth waypoint for clarity


# use matplotlib to visualize waypoints in 3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# draw waypoints
for wp in waypoints_raw:
    ax.scatter(*wp["pos"], c='k', s=8)

# positions
pts = np.array([wp["pos"] for wp in waypoints_raw])
xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]

# approximate frame scale based on scene size
scene_scale = np.max(np.ptp(pts, axis=0)) if pts.shape[0] > 1 else 1.0
frame_scale = scene_scale * 0.1

# draw points
ax.scatter(xs, ys, zs, c='k', s=8)

# draw frames (local axes) at each waypoint
for wp in waypoints_raw:
    p = np.array(wp["pos"], dtype=float)

    R = np.eye(3)
    # wp has "mat"
    if "mat" in wp:
        R = np.array(wp["mat"]).reshape((3, 3))

    # wp has "rpy"
    elif "rpy" in wp:
        rpy = wp["rpy"]
        roll, pitch, yaw = rpy
        # compute rotation matrix from rpy
        R = R @ utils.rotzyx(roll, 0, 0)
        R = R @ utils.rotzyx(0, pitch, 0)
        R = R @ utils.rotzyx(0, 0, yaw)


    # local axes in world coordinates (columns of R)
    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]

    ax.quiver(p[0], p[1], p[2], x_axis[0], x_axis[1], x_axis[2],
              length=frame_scale, color='r', normalize=True)
    ax.quiver(p[0], p[1], p[2], y_axis[0], y_axis[1], y_axis[2],
              length=frame_scale, color='g', normalize=True)
    ax.quiver(p[0], p[1], p[2], z_axis[0], z_axis[1], z_axis[2],
              length=frame_scale, color='b', normalize=True)

# labels and equal aspect
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
try:
    ax.set_box_aspect([1, 1, 1])
except Exception:
    # fallback: set equal scaling by setting limits
    max_range = scene_scale / 2.0 if scene_scale > 0 else 1.0
    mid_x, mid_y, mid_z = xs.mean(), ys.mean(), zs.mean()
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

# make all axes have the same scale
plt.axis('equal')  # does not work for 3D

plt.show()