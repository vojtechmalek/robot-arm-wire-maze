import json
import matplotlib.pyplot as plt
import numpy as np
from ctu_crs import CRS97
from PybulletRobot import PybulletRobot

from PybulletRobotSimulation import PybulletRobotSimulation
import utils  # or CRS93

def get_waypoint_pose(waypoint, base_center_robot, center_rot, angle_offset = 0.0) -> np.ndarray:
    pos = np.array(waypoint["pos"])
    rpy = np.array(waypoint["rpy"])
    # print(f"Moving to waypoint: pos={pos}, rpy={rpy}")

    # start with identity matrix
    pose_matrix = np.eye(4)

    # move to maze base center in robot frame
    pose_matrix = pose_matrix @ utils.trans_se3_vec(base_center_robot)

    # rotate to align with maze orientation
    pose_matrix = pose_matrix @ utils.rotzyx_se3(0, 0, (-center_rot + 90) * np.pi / 180)

    # move to waypoint position relative to maze center
    pose_matrix = pose_matrix @ utils.trans_se3_vec(pos)

    # apply waypoint orientation
    pose_matrix = pose_matrix @ utils.rotzyx_se3(rpy[0], 0, 0)
    pose_matrix = pose_matrix @ utils.rotzyx_se3(0, rpy[1], 0)
    pose_matrix = pose_matrix @ utils.rotzyx_se3(0, 0, rpy[2])

    # adjust for robot end effector orientation
    pose_matrix = pose_matrix @ utils.rotzyx_se3(0, 0, np.pi/2)
    pose_matrix = pose_matrix @ utils.rotzyx_se3(np.pi/2, 0, 0)
    
    # apply additional angle offset around Z
    pose_matrix = pose_matrix @ utils.rotzyx_se3(0, 0, angle_offset)

    return pose_matrix


robot = PybulletRobot(tty_dev=None)

maze_base_center_robot = np.array([0.409188095097534, 0.07017433, 0.032])
maze_center_rot = 16.288773  # degrees

print(f"Marker position in robot frame: x={maze_base_center_robot[0]:.3f}, y={maze_base_center_robot[1]:.3f}, rotation={maze_center_rot:.2f} deg")

base_center_robot_se3 = utils.trans_se3(maze_base_center_robot[0], maze_base_center_robot[1], 0.032)


waypoints_raw = json.load(open("waypoints/waypoints_C.json", "r"))



step = 1
grid_count_x = len(waypoints_raw)
# grid_count_x = 50

grid_count_y = 32

grid = np.zeros((grid_count_y, grid_count_x))

simulation_robot = PybulletRobotSimulation()


for ix in range(0, grid_count_x, step):
    wp = waypoints_raw[ix]
    print(f"Processing waypoint {ix}:")

    found_solution = False

    for iy in range(grid_count_y):
        angle_offset = (iy / (grid_count_y - 1)) * 2 * np.pi

        pose_matrix = get_waypoint_pose(wp, maze_base_center_robot, maze_center_rot, angle_offset)

        movement_offset_matrix = np.eye(4)
        movement_offset_matrix[:3, 3] = np.array([utils.HOOP_CENTER_DISTANCE, 0.0, 0.0])
        pose_matrix = pose_matrix @ movement_offset_matrix

        if (pose_matrix[2, 3] < 0.04):
            continue

        ik_sols = robot.ik(pose_matrix)
        if len(ik_sols) == 0:
            # print(f"  - No IK solution found for angle offset {angle_offset:.2f} rad.")
            continue

        ik_filtered_sols = [q for q in ik_sols if robot.in_limits(q)]

        if len(ik_filtered_sols) == 0:
            # print(f"  - No valid IK solutions found for angle offset {angle_offset:.2f} rad.")
            continue

        # grid[iy, ix] = len(ik_filtered_sols)


        # test for collisions in simulation
        collision_found = False
        valid_solutions = 0
        for ik_sol in ik_filtered_sols:
            simulation_robot.move_to_q(ik_sol)
            # print("robot moved to: ", simulation_robot.get_q())
            simulation_robot.step_simulation()
            if simulation_robot.is_colliding_with_maze():
                collision_found = True
            else:
                valid_solutions += 1

        grid[iy, ix] = valid_solutions
        if (valid_solutions == 0):
            grid[iy, ix] = -2








#####################

def move_robot_to_waypoint(ix, iy):
    wp = waypoints_raw[ix]
    angle_offset = (iy / (grid_count_y - 1)) * 2 * np.pi
    pose_matrix = get_waypoint_pose(wp, maze_base_center_robot, maze_center_rot, angle_offset)
    movement_offset_matrix = np.eye(4)
    movement_offset_matrix[:3, 3] = np.array([utils.HOOP_CENTER_DISTANCE, 0.0, 0.0])
    pose_matrix = pose_matrix @ movement_offset_matrix
    ik_sols = robot.ik(pose_matrix)
    
    # filtered_sols = ik_sols
    filtered_sols = [q for q in ik_sols if robot.in_limits(q)]

    if len(filtered_sols) == 0:
        print("No valid IK solutions found.")
        return
    robot._set_q(filtered_sols[0])


def draw_robot_frame_at_waypoint(ix, iy):
    wp = waypoints_raw[ix]
    angle_offset = (iy / (grid_count_y - 1)) * 2 * np.pi
    pose_matrix = get_waypoint_pose(wp, maze_base_center_robot, maze_center_rot, angle_offset)
    robot._draw_frame(pose_matrix, length=0.1, width=0.01)


# visualize grid using matplotlib
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

    # draw_robot_frame_at_waypoint(ix, iy)
    move_robot_to_waypoint(ix, iy)

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

    # draw_robot_frame_at_waypoint(ix, iy)
    move_robot_to_waypoint(ix, iy)


cid2 = fig.canvas.mpl_connect('motion_notify_event', on_move)
plt.show()
