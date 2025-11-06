import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d)

WAYPOINT_FILE = "waypoints_M.json"  # path 

def load_waypoints(path):
    with open(path, "r") as f:
        return json.load(f)

def main():
    waypoints = load_waypoints(WAYPOINT_FILE)

    # extract XYZ from the waypoints
    xs = [wp["pos"][0] for wp in waypoints]
    ys = [wp["pos"][1] for wp in waypoints]
    zs = [wp["pos"][2] for wp in waypoints]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # draw the full path as a connected line
    ax.plot(xs, ys, zs)

    # highlight start and end
    ax.scatter(xs[0], ys[0], zs[0], marker='o', s=50)   # start
    ax.scatter(xs[-1], ys[-1], zs[-1], marker='^', s=50)  # end


    # draw little orientation arrows every N points
    N = 50  # every 50th point to keep it readable
    scale = 0.02  # arrow length in meters

    for i in range(0, len(waypoints)-1, N):
        p = np.array(waypoints[i]["pos"])
        p_next = np.array(waypoints[i+1]["pos"])
        t = p_next - p
        t = t / np.linalg.norm(t)
        ax.quiver(
            p[0], p[1], p[2],
            t[0]*scale, t[1]*scale, t[2]*scale,
            length=1.0, normalize=False
        )


    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Waypoint Path (Maze Frame M)")

    # make the aspect ratio less distorted
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
