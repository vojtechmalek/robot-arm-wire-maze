import json
import matplotlib.pyplot as plt
import numpy as np
from ctu_crs import CRS97
from PybulletRobot import PybulletRobot

import utils  # or CRS93

def get_waypoint_pose(waypoint, base_center_robot, center_rot, angle_offset = 0.0) -> np.ndarray:
    pos = np.array(waypoint["pos"])
    rpy = np.array(waypoint["rpy"])
    # print(f"Moving to waypoint: pos={pos}, rpy={rpy}")

    pose_matrix = np.eye(4)
    # pose_matrix = pose_matrix @ utils.rotxyz_se3(rpy[0], rpy[1], rpy[2])
    pose_matrix = pose_matrix @ utils.trans_se3_vec(base_center_robot)

    pose_matrix = pose_matrix @ utils.rotzyx_se3(0, 0, (-center_rot + 90) * np.pi / 180)

    pose_matrix = pose_matrix @ utils.trans_se3_vec(pos + np.array([-0.005, -0.005, 0.0]))
    # pose_matrix = pose_matrix @ utils.trans_se3_vec(pos)

    # pose_matrix = pose_matrix @ utils.rotzyx_se3(0, 0, -(-center_rot + 90) * np.pi / 180)

    # LVL B needs a 180 deg rotation around Z
    # pose_matrix = pose_matrix @ utils.rotzyx_se3(0, np.pi, -np.pi/2)
    pose_matrix = pose_matrix @ utils.rotzyx_se3(0, np.pi, np.pi/2 + angle_offset)

    return pose_matrix


robot = PybulletRobot(tty_dev=None)

maze_base_center_robot = np.array([0.409188095097534, 0.07017433, 0.032])
maze_center_rot = 16.288773  # degrees

print(f"Marker position in robot frame: x={maze_base_center_robot[0]:.3f}, y={maze_base_center_robot[1]:.3f}, rotation={maze_center_rot:.2f} deg")

base_center_robot_se3 = utils.trans_se3(maze_base_center_robot[0], maze_base_center_robot[1], 0.032)


waypoints_raw = json.load(open("waypoints/waypoints_C.json", "r"))



grid_count_x = len(waypoints_raw)
grid_count_y = 32
grid = np.zeros((grid_count_y, grid_count_x))


for ix in range(grid_count_x):
    wp = waypoints_raw[ix]
    print(f"Processing waypoint {ix}:")

    found_solution = False

    for iy in range(grid_count_y):
        angle_offset = (iy / (grid_count_y - 1)) * 2 * np.pi

        pose_matrix = get_waypoint_pose(wp, maze_base_center_robot, maze_center_rot, angle_offset)

        movement_offset_matrix = np.eye(4)
        movement_offset_matrix[:3, 3] = np.array([utils.HOOP_CENTER_DISTANCE, 0.0, 0.0])
        pose_matrix = pose_matrix @ movement_offset_matrix

        ik_sols = robot.ik(pose_matrix)
        if len(ik_sols) == 0:
            # print(f"  - No IK solution found for angle offset {angle_offset:.2f} rad.")
            continue

        filtered_sols = lambda sols: [q for q in sols if robot.in_limits(q)]

        found_solution = True
        grid[iy, ix] = len(filtered_sols(ik_sols))
        # print(f"  - IK solution found for angle offset {angle_offset:.2f} rad.")
        # break

    if not found_solution:
        grid[:, ix] = 0












#####################


fig, ax = plt.subplots()
extent = [0, 1, 0, 360]  # x_min, x_max, y_min, y_max
img = ax.imshow(grid, cmap="viridis", interpolation="nearest", extent=extent, origin="lower", aspect="auto")
ax.set_xlim(0, 1)
ax.set_ylim(0, 360)

def on_click(event):
    if event.inaxes != ax:
        return
    xdata, ydata = event.xdata, event.ydata
    if xdata is None or ydata is None:
        return

    # Convert axis coordinates → grid indices
    ix = int(np.floor((xdata - extent[0]) / (extent[1] - extent[0]) * grid_count_x))
    iy = int(np.floor((ydata - extent[2]) / (extent[3] - extent[2]) * grid_count_y))
    ix = np.clip(ix, 0, grid_count_x - 1)
    iy = np.clip(iy, 0, grid_count_y - 1)

    print(f"Clicked data coords: x={xdata:.3f}, y={ydata:.3f}")
    print(f"Clicked cell: ({iy}, {ix}), value={grid[iy, ix]}")

    # show first IK solution for the clicked cell
    wp = waypoints_raw[ix]
    angle_offset = (iy / (grid_count_y - 1)) * 2 * np.pi
    pose_matrix = get_waypoint_pose(wp, maze_base_center_robot, maze_center_rot, angle_offset)
    movement_offset_matrix = np.eye(4)
    movement_offset_matrix[:3, 3] = np.array([utils.HOOP_CENTER_DISTANCE, 0.0, 0.0])
    pose_matrix = pose_matrix @ movement_offset_matrix
    ik_sols = robot.ik(pose_matrix)
    filtered_sols = [q for q in ik_sols if robot.in_limits(q)]
    robot._set_q(filtered_sols[0])

cid = fig.canvas.mpl_connect('button_press_event', on_click)
# also handle when holding button and moving mouse
def on_move(event):
    if event.inaxes != ax:
        return
    if not event.button:
        return
    xdata, ydata = event.xdata, event.ydata
    if xdata is None or ydata is None:
        return

    # Convert axis coordinates → grid indices
    ix = int(np.floor((xdata - extent[0]) / (extent[1] - extent[0]) * grid_count_x))
    iy = int(np.floor((ydata - extent[2]) / (extent[3] - extent[2]) * grid_count_y))
    ix = np.clip(ix, 0, grid_count_x - 1)
    iy = np.clip(iy, 0, grid_count_y - 1)

    print(f"Moved to data coords: x={xdata:.3f}, y={ydata:.3f}")
    print(f"Moved to cell: ({iy}, {ix}), value={grid[iy, ix]}")

    # show first IK solution for the clicked cell
    wp = waypoints_raw[ix]
    angle_offset = (iy / (grid_count_y - 1)) * 2 * np.pi
    pose_matrix = get_waypoint_pose(wp, maze_base_center_robot, maze_center_rot, angle_offset)
    movement_offset_matrix = np.eye(4)
    movement_offset_matrix[:3, 3] = np.array([utils.HOOP_CENTER_DISTANCE, 0.0, 0.0])
    pose_matrix = pose_matrix @ movement_offset_matrix
    ik_sols = robot.ik(pose_matrix)
    filtered_sols = [q for q in ik_sols if robot.in_limits(q)]
    robot._set_q(filtered_sols[0])


cid2 = fig.canvas.mpl_connect('motion_notify_event', on_move)
plt.show()
