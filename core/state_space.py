import json
from typing import Optional, TYPE_CHECKING, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import heapq

from .pose import Pose
from . import utils

if TYPE_CHECKING:
    from .pybullet_robot import PybulletRobot

GROUND_LIMIT = 0.05  # meters


class StateSpace:
    """
    Represents and calculates the state space for robot path planning.
    
    The state space is a 2D grid where:
    - X-axis: waypoint indices along the path
    - Y-axis: angle offsets (0 to 2π)
    - Values: robot-to-maze distance for the first IK solution (furthest from maze), or -0.05 if no valid solutions
    
    IK solutions are sorted by distance from the maze (furthest first) to make path finding easier
    by preferring safer configurations.
    """
    
    def __init__(self, robot: 'PybulletRobot', maze: str, maze_pose: Pose):
        ################################################################################################################
        # INITIALIZATION
        ################################################################################################################

        self._robot = robot
        self._maze = maze
        self._maze_pose = maze_pose
        
        # State space data (calculated on demand)
        self._grid: Optional[np.ndarray] = None
        self._ik_solutions: Optional[list] = None  # List of lists: viable IK solutions per grid cell, sorted by distance from maze (furthest first)
        self._waypoints: Optional[list] = None
        self._num_angle_samples: Optional[int] = None
        self._step: Optional[int] = None
        self._path_states: Optional[List[Tuple[int, int]]] = None  # Path states (waypoint_idx, angle_idx) for visualization
        self._search_tree = None  # For pathfinding optimization
    
    
    def calculate(self, waypoints_path, num_angle_samples, step, use_separate_simulation):
        """
        Calculate the state space matrix
        """
        
        waypoints = json.load(open(waypoints_path, "r"))
        
        # remove last waypoint that is in the base
        waypoints = waypoints[:-1]

        self._waypoints = waypoints
        
        # Extract maze position and rotation
        base_center_robot = self._maze_pose.p
        center_rot = np.rad2deg(self._maze_pose.rpy[2])
        
        grid_count_x = len(waypoints)
        grid_count_y = num_angle_samples
        
        # Initialize IK solutions storage: list of lists, one per grid cell
        ik_solutions = [[None] * grid_count_y for _ in range(grid_count_x)]
        
        # Create separate simulation for collision testing if requested
        test_robot = self._robot
        created_separate_sim = False
        
        if use_separate_simulation and self._robot._bullet_client.isConnected():
            # Import here to avoid circular dependency
            from .pybullet_robot import PybulletRobot
            # Create a headless simulation for faster collision testing
            test_robot = PybulletRobot(maze=self._maze, maze_pose=self._maze_pose, use_gui=False)
            created_separate_sim = True
        
        try:
            for ix in tqdm(range(0, grid_count_x, step), desc="Calculating state space"):
                wp = waypoints[ix]
                
                for iy in range(grid_count_y):
                    angle_offset = (iy / (grid_count_y - 1)) * 2 * np.pi
                    
                    pose = utils.get_waypoint_pose(
                        wp, base_center_robot, center_rot, angle_offset
                    )
                    
                    # Apply movement offset (hoop center distance)
                    movement_offset = Pose(p=np.array([utils.HOOP_CENTER_DISTANCE, 0.0, -utils.HOOP_OFFSET_Z]))
                    pose = pose @ movement_offset
                    
                    # Skip if Z position is too low
                    if pose.p[2] < GROUND_LIMIT:
                        continue
                    
                    # Solve IK (robot.ik expects SE3 matrix)
                    ik_sols = self._robot.ik(pose.T)
                    if len(ik_sols) == 0:
                        continue
                    
                    # Filter by joint limits
                    ik_filtered_sols = [q for q in ik_sols if self._robot.in_limits(q)]
                    if len(ik_filtered_sols) == 0:
                        continue
                    
                    # Test for collisions in simulation and save viable solutions
                    viable_solutions = []
                    for ik_sol in ik_filtered_sols:
                        test_robot._set_q(ik_sol)
                        test_robot.step_simulation()
                        if not test_robot.is_colliding_with_maze():
                            viable_solutions.append(ik_sol)
                    
                    # Sort viable solutions by distance from maze (furthest first)
                    # This makes path finding easier by preferring safer configurations
                    # if len(viable_solutions) > 0:
                    #     solution_distances = []
                    #     for ik_sol in viable_solutions:
                    #         test_robot._set_q(ik_sol)
                    #         test_robot.step_simulation()
                    #         distance = test_robot.get_robot_to_maze_distance()
                    #         solution_distances.append((distance, ik_sol))
                        
                    #     # Sort by distance (furthest first = highest distance first)
                    #     solution_distances.sort(key=lambda x: x[0], reverse=True)
                    #     viable_solutions = [sol for _, sol in solution_distances]
                    
                    # Store viable IK solutions (empty list if none)
                    ik_solutions[ix][iy] = viable_solutions
        
        finally:
            # Clean up separate simulation if we created one
            if created_separate_sim:
                test_robot.close()
        
        # Store calculated IK solutions and parameters
        self._ik_solutions = ik_solutions
        self._num_angle_samples = num_angle_samples
        self._step = step
        # Clear any previous path when recalculating state space
        self._path_states = None

    def find_path(self) -> Optional[List[np.ndarray]]:
        """
        Find a joint-space path through the maze from start to goal using A* search.
        Returns a list of joint configurations (q) if a path is found, otherwise None.
        """
        if self._ik_solutions is None:
            raise ValueError("State space not calculated. Call calculate() first.")
        
        # ik_solutions is a 3D array [waypoint_idx][angle_idx][ik_solution_idx]
        ik_solutions = self._ik_solutions
        grid_count_y = self._num_angle_samples
        grid_count_x = len(self._waypoints)

        MAX_ANGLE_CHANGE = np.deg2rad(45)  # maximum allowed joint change between steps

        q_path = []
        path_states = []
        # previous_q = None
        # for i in range(grid_count_x):
        #     q_path.append(ik_solutions[i][0][0] if ik_solutions[i][0] else None)
        #     path_states.append((i, 0, 0))

        # BFS

        state_queue = []
        visited_states = set()

        # append all start angles
        for angle_idx in range(grid_count_y):
            state_queue.append( (0, angle_idx, 0, 0, -1, -1, -1) )
        # state_queue.append( (0, 0, 0, 0, -1, -1, -1) )  # waypoint_idx, angle_idx, ik_solution_idx, cost_so_far, prev_waypoint_idx, prev_angle_idx, prev_ik_solution_idx
        


        reached_goal = False

        while state_queue:
            waypoint_idx, angle_idx, ik_solution_idx, cost_so_far, prev_waypoint_idx, prev_angle_idx, prev_ik_solution_idx = state_queue.pop(0)

            if (waypoint_idx, angle_idx, ik_solution_idx) in visited_states:
                continue
            visited_states.add( (waypoint_idx, angle_idx, ik_solution_idx) )

            viable_sols = ik_solutions[waypoint_idx][angle_idx]
            if viable_sols is None or len(viable_sols) == 0:
                continue

            this_q = viable_sols[ik_solution_idx]
            q_path.append(this_q)
            path_states.append((waypoint_idx, angle_idx, ik_solution_idx, cost_so_far, prev_waypoint_idx, prev_angle_idx, prev_ik_solution_idx))

            if waypoint_idx == grid_count_x - 1:
                # Reached goal waypoint
                reached_goal = True
                break

            # enqueue next and previous angle offsets
            for delta_angle in [-1, 1]:
                next_angle_idx = (angle_idx + delta_angle) % grid_count_y
                next_viable_sols = ik_solutions[waypoint_idx][next_angle_idx]
                if next_viable_sols is None or len(next_viable_sols) < ik_solution_idx + 1:
                    continue
                ik_solution = next_viable_sols[ik_solution_idx]
                if ik_solution is not None:
                    q_diff = ik_solution - this_q
                    # max joint change must be respected
                    if np.any(np.abs(q_diff) > MAX_ANGLE_CHANGE):
                        continue
                    state_queue.append( (waypoint_idx, next_angle_idx, ik_solution_idx, cost_so_far + 1,
                                     waypoint_idx, angle_idx, ik_solution_idx) )
            
            # enqueue next and previous waypoints
            for delta_wp in [-1, 1]:
                next_waypoint_idx = waypoint_idx + delta_wp
                if next_waypoint_idx < 0 or next_waypoint_idx >= grid_count_x:
                    continue

                next_viable_sols = ik_solutions[next_waypoint_idx][angle_idx]
                if next_viable_sols is None or len(next_viable_sols) < ik_solution_idx + 1:
                    continue

                ik_solution = next_viable_sols[ik_solution_idx]
                if ik_solution is not None:
                    q_diff = ik_solution - this_q
                    # max joint change must be respected
                    if np.any(np.abs(q_diff) > MAX_ANGLE_CHANGE):
                        continue
                    state_queue.append( (next_waypoint_idx, angle_idx, ik_solution_idx, cost_so_far + 1,
                                         waypoint_idx, angle_idx, ik_solution_idx) )

            # enqueue all ik solutions for current waypoint and angle
            for next_ik_solution_idx in range(len(viable_sols)):
                if next_ik_solution_idx == ik_solution_idx:
                    continue
                next_ik_solution = viable_sols[next_ik_solution_idx]
                q_diff = next_ik_solution - this_q
                # max joint change must be respected
                if np.any(np.abs(q_diff) > MAX_ANGLE_CHANGE):
                    continue

                state_queue.append( (waypoint_idx, angle_idx, next_ik_solution_idx, cost_so_far + 1,
                                     waypoint_idx, angle_idx, ik_solution_idx) )


            # # Enqueue next states
            # next_waypoint_idx = waypoint_idx + 1

            # for next_angle_idx in range(grid_count_y):
            #     next_viable_sols = ik_solutions[next_waypoint_idx][next_angle_idx]
            #     if next_viable_sols is None or len(next_viable_sols) == 0:
            #         continue
            #     for next_ik_solution_idx in range(len(next_viable_sols)):
            #         # get q difference to previous q
            #         next_q = next_viable_sols[next_ik_solution_idx]
            #         q_diff = next_q - this_q
            #         # max joint change must be respected
            #         if np.any(np.abs(q_diff) > MAX_ANGLE_CHANGE):
            #             continue

            #         state_queue.append( (next_waypoint_idx, next_angle_idx, next_ik_solution_idx, cost_so_far + 1,
            #                              waypoint_idx, angle_idx, ik_solution_idx) )

        self._search_tree = path_states

        if len(path_states) == 0:
            return None
        # if we have something we at least want to visualize the search tree

        # get last path state
        current_state = path_states[-1]

        # reconstruct path
        reconstructed_path_states = []
        while True:
            waypoint_idx, angle_idx, ik_solution_idx, cost_so_far, prev_waypoint_idx, prev_angle_idx, prev_ik_solution_idx = current_state
            reconstructed_path_states.append((waypoint_idx, angle_idx, ik_solution_idx, cost_so_far))

            if current_state[4] == -1:
                break

            current_state = next((s for s in path_states if s[0] == prev_waypoint_idx and s[1] == prev_angle_idx and s[2] == prev_ik_solution_idx), None)

        self._path_states = reconstructed_path_states[::-1]  # reverse to get from start to goal
        
        # cosntruct q_path
        q_path = []
        for waypoint_idx, angle_idx, ik_solution_idx, _ in self._path_states:
            viable_sols = ik_solutions[waypoint_idx][angle_idx]
            q_path.append(viable_sols[ik_solution_idx])
        
        return q_path

    def move_to_point(self, waypoint_idx: int, angle_idx: int) -> bool:
        """
        Move robot to a specific point in the state space.
        """
        if self._ik_solutions is None:
            raise ValueError("State space not calculated. Call calculate() first.")
        
        # Check if we have a stored IK solution for this point
        stored_solutions = self._ik_solutions[waypoint_idx][angle_idx]
        if stored_solutions is not None and len(stored_solutions) > 0:
            # Use the first stored IK solution (furthest from maze after sorting)
            self._robot._set_q(stored_solutions[0])
            if self._robot._bullet_client.isConnected():
                self._robot.step_simulation()
            return True

        return False
    
    def visualize(self) -> None:
        """
        Display an interactive visualization of the state space.
        Clicking or dragging on the plot will move the robot to that configuration.
        """
        if self._ik_solutions is None:
            raise ValueError("State space not calculated. Call calculate() first.")
        
        ik_solutions = self._ik_solutions
        grid_count_x = len(self._waypoints)
        grid_count_y = self._num_angle_samples
        
        # Compute distances from saved IK solutions
        # Use the first solution (index 0) which is the furthest from the maze after sorting
        grid = np.zeros((grid_count_y, grid_count_x))
        for iy in range(grid_count_y): 
            for ix in range(grid_count_x) : 
                viable_sols = ik_solutions[ix][iy]
                if viable_sols is None or len(viable_sols) == 0:
                    grid[iy, ix] = -0.05
                else:
                    # Use the first solution (furthest from maze)
                    self._robot._set_q(viable_sols[0])
                    self._robot.step_simulation()
                    distance = self._robot.get_robot_to_maze_distance()
                    grid[iy, ix] = viable_sols[0][5]
        
        # Create visualization
        fig, ax = plt.subplots()
        extent = [0, 1, 0, 360]  # x_min, x_max, y_min, y_max
        img = ax.imshow(grid, cmap="viridis", interpolation="nearest", extent=extent, origin="lower", aspect="auto")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 360)
        ax.set_xlabel("Path Progress (normalized)")
        ax.set_ylabel("Angle Offset (degrees)")
        
        # Plot path if available
        if self._path_states is not None and len(self._path_states) > 0:
            path_x = []
            path_y = []
            point_colors = []
            max_cost = max(cs for _, _, _, cs in self._path_states)
            for i, (waypoint_idx, angle_idx, ik_solution_idx, cost_so_far) in enumerate(self._path_states):
                # Convert grid indices to plot coordinates
                x_coord = (waypoint_idx / grid_count_x) * (extent[1] - extent[0]) + extent[0]
                angle_offset_deg = (angle_idx / (grid_count_y - 1)) * 360
                path_x.append(x_coord)
                path_y.append(angle_offset_deg)

                # Map cost to color (e.g., using a colormap)
                point_colors.append(plt.cm.viridis(ik_solution_idx / 8.0))

            # Plot path as colored points
            ax.scatter(path_x, path_y, c=point_colors, s=50, marker='o', edgecolors='white', linewidths=1, zorder=10, label='Path')
            # ax.plot(path_x, path_y, c='red', linewidth=2, alpha=0.5, zorder=9)

            search_x = []
            search_y = []
            search_colors = []
            for waypoint_idx, angle_idx, ik_solution_idx, cost_so_far, prev_waypoint_idx, prev_angle_idx, prev_ik_solution_idx in self._search_tree:
                # Convert grid indices to plot coordinates
                x_coord = (waypoint_idx / grid_count_x) * (extent[1] - extent[0]) + extent[0]
                angle_offset_deg = (angle_idx / (grid_count_y - 1)) * 360
                search_x.append(x_coord)
                search_y.append(angle_offset_deg)

                point_cost = cost_so_far
                # Normalize cost for color mapping
                norm_cost = point_cost / max_cost if max_cost > 0 else 0
                # Map cost to color (e.g., using a colormap)
                search_colors.append(plt.cm.plasma(norm_cost))

            # Plot search tree as colored points
            ax.scatter(search_x, search_y, c=search_colors, s=10, marker='x', alpha=0.3, zorder=5, label='Search Tree')

            ax.legend()
        
        # Add colorbar
        plt.colorbar(img, ax=ax, label="Max Robot-Maze Distance (m)")
        
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
            
            # Get IK solutions for this point
            viable_sols = ik_solutions[ix][iy]
            if viable_sols is None or len(viable_sols) == 0:
                print(f"Clicked: waypoint={ix}, No valid configuration")
            else:
                distance = grid[iy, ix]
                num_solutions = len(viable_sols)
                print(f"Clicked: waypoint={ix}, IK solutions={num_solutions}, distance={distance:.4f}m")
            
            self.move_to_point(ix, iy)
        
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
            
            # Get IK solutions for this point
            viable_sols = ik_solutions[ix][iy]
            if viable_sols is None or len(viable_sols) == 0:
                print(f"Moved to: waypoint={ix}, - No valid configuration (inaccessible pose)")
            else:
                distance = grid[iy, ix]
                num_solutions = len(viable_sols)
                print(f"Moved to: waypoint={ix}, IK solutions={num_solutions}, distance={distance:.4f}m")
            
            self.move_to_point(ix, iy)
        
        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('motion_notify_event', on_move)
        
        self._visualization_fig = fig
        plt.show()
    
    @property
    def grid(self) -> Optional[np.ndarray]:
        """Get the calculated state space grid (None if not yet calculated)."""
        return self._grid
    
    @property
    def waypoints(self) -> Optional[list]:
        """Get the waypoints used for state space calculation."""
        return self._waypoints
    
