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

GROUND_LIMIT = 0.03  # meters


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
        self._visited_costs = None  # Visited costs from pathfinding: (waypoint_idx, angle_idx, ik_solution_idx) -> cost
        self._best_ik_at_position = None  # Best IK solution index at each (waypoint_idx, angle_idx) with minimum cost
    
    
    def calculate(self, waypoints_path, num_angle_samples, step, use_separate_simulation):
        """
        Calculate the state space matrix
        """
        
        waypoints = json.load(open(waypoints_path, "r"))
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
        # Clear any previous path and pathfinding data when recalculating state space
        self._path_states = None
        self._visited_costs = None
        self._best_ik_at_position = None

    def find_path(self) -> Optional[List[np.ndarray]]:
        """
        Find a joint-space path through the maze from start to goal using Dijkstra's algorithm.
        Returns a list of joint configurations (q) if a path is found, otherwise None.
        """
        if self._ik_solutions is None:
            raise ValueError("State space not calculated. Call calculate() first.")
        
        ik_solutions = self._ik_solutions
        grid_count_y = self._num_angle_samples
        grid_count_x = len(self._waypoints)
        
        # Configurable parameters
        MAX_JOINT_CHANGE = np.deg2rad(45)  # Maximum allowed joint change per step (per joint)
        DIAGONAL_STEP = 1  # Step size for diagonal moves (same for waypoint and angle)
        
        # Create separate simulation for collision testing if GUI is enabled
        test_robot = self._robot
        created_separate_sim = False
        
        if self._robot._use_gui and self._robot._bullet_client.isConnected():
            # Import here to avoid circular dependency
            from .pybullet_robot import PybulletRobot
            # Create a headless simulation for faster collision testing
            test_robot = PybulletRobot(maze=self._maze, maze_pose=self._maze_pose, use_gui=False)
            created_separate_sim = True
        
        # Distance function: squared joint angle differences
        def calculate_distance(last_q: np.ndarray, current_q: np.ndarray) -> float:
            """Calculate squared joint angle difference between two configurations."""
            q_diff = current_q - last_q
            return np.sum(q_diff ** 2)
        
        # Priority queue: (cost, waypoint_idx, angle_idx, ik_solution_idx, prev_waypoint_idx, prev_angle_idx, prev_ik_solution_idx)
        priority_queue = []
        heapq.heapify(priority_queue)
        
        # Visited states: (waypoint_idx, angle_idx, ik_solution_idx) -> best_cost
        visited_costs = {}
        
        # Parent tracking for path reconstruction: (waypoint_idx, angle_idx, ik_solution_idx) -> (prev_waypoint_idx, prev_angle_idx, prev_ik_solution_idx)
        parents = {}
        
        # Search tree for visualization
        search_tree = []
        
        # Initialize: start from all valid states in first column (waypoint_idx=0)
        for angle_idx in range(grid_count_y):
            viable_sols = ik_solutions[0][angle_idx]
            if viable_sols is None or len(viable_sols) == 0:
                continue
            for ik_solution_idx in range(len(viable_sols)):
                state = (0, angle_idx, ik_solution_idx)
                visited_costs[state] = 0.0
                heapq.heappush(priority_queue, (0.0, 0, angle_idx, ik_solution_idx, -1, -1, -1))
                search_tree.append((0, angle_idx, ik_solution_idx, 0.0, -1, -1, -1))
        
        # Track the best goal state (lowest cost to reach final waypoint)
        best_goal_state = None
        
        try:
            while priority_queue:
                cost_so_far, waypoint_idx, angle_idx, ik_solution_idx, prev_wp, prev_ang, prev_ik = heapq.heappop(priority_queue)
                
                # Skip if we've found a better path to this state
                state = (waypoint_idx, angle_idx, ik_solution_idx)
                if state in visited_costs and cost_so_far > visited_costs[state]:
                    continue
                
                # Check if we reached the goal (last waypoint)
                # With Dijkstra's algorithm, the first goal state we pop is guaranteed to be optimal
                if waypoint_idx == grid_count_x - 1:
                    best_goal_state = state
                    break  # Terminate immediately - we've found the optimal path
                
                # Get current joint configuration
                viable_sols = ik_solutions[waypoint_idx][angle_idx]
                if viable_sols is None or len(viable_sols) == 0:
                    continue
                if ik_solution_idx >= len(viable_sols):
                    continue
                current_q = viable_sols[ik_solution_idx]
                
                # Try all movement directions
                movements = [
                    # Right: next waypoint, same angle
                    (1, 0),
                    # Up: same waypoint, next angle
                    (0, 1),
                    # Down: same waypoint, previous angle
                    (0, -1),
                    # Right+Up: next waypoint, next angle (diagonal)
                    (DIAGONAL_STEP, DIAGONAL_STEP),
                    # Right+Down: next waypoint, previous angle (diagonal)
                    (DIAGONAL_STEP, -DIAGONAL_STEP),
                ]
                
                for delta_wp, delta_ang in movements:
                    next_waypoint_idx = waypoint_idx + delta_wp
                    next_angle_idx = (angle_idx + delta_ang) % grid_count_y
                    
                    # Check bounds
                    if next_waypoint_idx < 0 or next_waypoint_idx >= grid_count_x:
                        continue
                    
                    # Get all IK solutions at the target position
                    next_viable_sols = ik_solutions[next_waypoint_idx][next_angle_idx]
                    if next_viable_sols is None or len(next_viable_sols) == 0:
                        continue
                    
                    # Try all IK solutions at the target position
                    for next_ik_solution_idx in range(len(next_viable_sols)):
                        next_q = next_viable_sols[next_ik_solution_idx]
                        
                        # Check joint limit: each joint must change by <= MAX_JOINT_CHANGE
                        q_diff = next_q - current_q
                        if np.any(np.abs(q_diff) > MAX_JOINT_CHANGE):
                            continue
                        
                        # Check collision along path (use test_robot for collision checking)
                        if test_robot.is_path_colliding_with_maze(current_q, next_q):
                            continue
                        
                        # Calculate new cost
                        step_cost = calculate_distance(current_q, next_q)
                        new_cost = cost_so_far + step_cost
                        
                        # Check if this is a better path to the next state
                        next_state = (next_waypoint_idx, next_angle_idx, next_ik_solution_idx)
                        if next_state in visited_costs and new_cost >= visited_costs[next_state]:
                            continue
                        
                        # Update visited costs and add to queue
                        visited_costs[next_state] = new_cost
                        parents[next_state] = state
                        heapq.heappush(priority_queue, (new_cost, next_waypoint_idx, next_angle_idx, next_ik_solution_idx,
                                                         waypoint_idx, angle_idx, ik_solution_idx))
                        search_tree.append((next_waypoint_idx, next_angle_idx, next_ik_solution_idx, new_cost,
                                           waypoint_idx, angle_idx, ik_solution_idx))
        
        finally:
            # Clean up separate simulation if we created one
            if created_separate_sim:
                test_robot.close()
        
        self._search_tree = search_tree
        self._visited_costs = visited_costs
        
        # Compute best IK solution at each position (minimum cost)
        best_ik_at_position = {}
        for (waypoint_idx, angle_idx, ik_solution_idx), cost in visited_costs.items():
            key = (waypoint_idx, angle_idx)
            if key not in best_ik_at_position or cost < best_ik_at_position[key][1]:
                best_ik_at_position[key] = (ik_solution_idx, cost)
        self._best_ik_at_position = best_ik_at_position
        
        if best_goal_state is None:
            return None
        
        # Reconstruct path from best goal state to start
        path_states = []
        current_state = best_goal_state
        
        while current_state in parents:
            waypoint_idx, angle_idx, ik_solution_idx = current_state
            path_states.append((waypoint_idx, angle_idx, ik_solution_idx, visited_costs[current_state]))
            current_state = parents[current_state]
        
        # Add the starting state
        if current_state in visited_costs:
            waypoint_idx, angle_idx, ik_solution_idx = current_state
            path_states.append((waypoint_idx, angle_idx, ik_solution_idx, visited_costs[current_state]))
        
        # Reverse to get path from start to goal
        path_states = path_states[::-1]
        self._path_states = path_states
        
        # Construct q_path
        q_path = []
        for waypoint_idx, angle_idx, ik_solution_idx, _ in path_states:
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
        Display an interactive visualization of the state space in two windows:
        1. Distance-based visualization (original)
        2. Cost-based visualization (from pathfinding)
        Clicking or dragging on either plot will move the robot to that configuration.
        """
        if self._ik_solutions is None:
            raise ValueError("State space not calculated. Call calculate() first.")
        
        ik_solutions = self._ik_solutions
        grid_count_x = len(self._waypoints)
        grid_count_y = self._num_angle_samples
        extent = [0, 1, 0, 360]  # x_min, x_max, y_min, y_max
        
        # Compute distance grid (always)
        distance_grid = np.zeros((grid_count_y, grid_count_x))
        for iy in range(grid_count_y): 
            for ix in range(grid_count_x) : 
                viable_sols = ik_solutions[ix][iy]
                if viable_sols is None or len(viable_sols) == 0:
                    distance_grid[iy, ix] = -0.05
                else:
                    # Use the smallest distance across all IK solutions
                    smallest_distance = float('inf')
                    for ik_sol in viable_sols:
                        self._robot._set_q(ik_sol)
                        self._robot.step_simulation()
                        distance = self._robot.get_robot_to_maze_distance()
                        if distance < smallest_distance:
                            smallest_distance = distance
                    distance_grid[iy, ix] = smallest_distance
        
        # Compute cost grid from pathfinding (if available)
        cost_grid = np.full((grid_count_y, grid_count_x), np.nan)
        has_cost_data = self._visited_costs is not None and self._best_ik_at_position is not None
        if has_cost_data:
            for iy in range(grid_count_y):
                for ix in range(grid_count_x):
                    key = (ix, iy)
                    if key in self._best_ik_at_position:
                        ik_solution_idx, cost = self._best_ik_at_position[key]
                        cost_grid[iy, ix] = cost
        
        # Helper function to create click/move handlers
        def create_handlers(ax, use_cost_based_ik):
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
                    if use_cost_based_ik and self._best_ik_at_position is not None:
                        key = (ix, iy)
                        if key in self._best_ik_at_position:
                            ik_solution_idx, cost = self._best_ik_at_position[key]
                            print(f"Clicked: waypoint={ix}, angle={iy}, cost={cost:.4f}, using IK solution {ik_solution_idx}")
                            self._robot._set_q(viable_sols[ik_solution_idx])
                            if self._robot._bullet_client.isConnected():
                                self._robot.step_simulation()
                        else:
                            distance = distance_grid[iy, ix]
                            num_solutions = len(viable_sols)
                            print(f"Clicked: waypoint={ix}, IK solutions={num_solutions}, distance={distance:.4f}m (not reached by pathfinding)")
                            self.move_to_point(ix, iy)
                    else:
                        distance = distance_grid[iy, ix]
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
                    if use_cost_based_ik and self._best_ik_at_position is not None:
                        key = (ix, iy)
                        if key in self._best_ik_at_position:
                            ik_solution_idx, cost = self._best_ik_at_position[key]
                            print(f"Moved to: waypoint={ix}, angle={iy}, cost={cost:.4f}, using IK solution {ik_solution_idx}")
                            self._robot._set_q(viable_sols[ik_solution_idx])
                            if self._robot._bullet_client.isConnected():
                                self._robot.step_simulation()
                        else:
                            distance = distance_grid[iy, ix]
                            num_solutions = len(viable_sols)
                            print(f"Moved to: waypoint={ix}, IK solutions={num_solutions}, distance={distance:.4f}m (not reached by pathfinding)")
                            self.move_to_point(ix, iy)
                    else:
                        distance = distance_grid[iy, ix]
                        num_solutions = len(viable_sols)
                        print(f"Moved to: waypoint={ix}, IK solutions={num_solutions}, distance={distance:.4f}m")
                        self.move_to_point(ix, iy)
            
            return on_click, on_move
        
        # Window 1: Distance-based visualization
        fig1, ax1 = plt.subplots(num="Distance Visualization")
        img1 = ax1.imshow(distance_grid, cmap="viridis", interpolation="nearest", extent=extent, origin="lower", aspect="auto")
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 360)
        ax1.set_xlabel("Path Progress (normalized)")
        ax1.set_ylabel("Angle Offset (degrees)")
        ax1.set_title("Robot-Maze Distance")
        plt.colorbar(img1, ax=ax1, label="Robot-Maze Distance (m)")
        
        on_click1, on_move1 = create_handlers(ax1, use_cost_based_ik=False)
        fig1.canvas.mpl_connect('button_press_event', on_click1)
        fig1.canvas.mpl_connect('motion_notify_event', on_move1)
        
        # Window 2: Cost-based visualization (if pathfinding has been run)
        if has_cost_data and not np.isnan(cost_grid).all():
            fig2, ax2 = plt.subplots(num="Path Cost Visualization")
            img2 = ax2.imshow(cost_grid, cmap="plasma", interpolation="nearest", extent=extent, origin="lower", aspect="auto")
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 360)
            ax2.set_xlabel("Path Progress (normalized)")
            ax2.set_ylabel("Angle Offset (degrees)")
            ax2.set_title("Path Cost (squared joint angle difference)")
            plt.colorbar(img2, ax=ax2, label="Path Cost (squared joint angle difference)")
            
            on_click2, on_move2 = create_handlers(ax2, use_cost_based_ik=True)
            fig2.canvas.mpl_connect('button_press_event', on_click2)
            fig2.canvas.mpl_connect('motion_notify_event', on_move2)
        else:
            fig2, ax2 = plt.subplots(num="Path Cost Visualization")
            ax2.text(0.5, 0.5, "Run find_path() first to see cost visualization", 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title("Path Cost (squared joint angle difference)")
        
        # Plot path and search tree on both windows if available
        if self._path_states is not None and len(self._path_states) > 0:
            path_x = []
            path_y = []
            point_colors = []
            max_cost = max(cs for _, _, _, cs in self._path_states)
            for i, (waypoint_idx, angle_idx, ik_solution_idx, cost_so_far) in enumerate(self._path_states):
                x_coord = (waypoint_idx / grid_count_x) * (extent[1] - extent[0]) + extent[0]
                angle_offset_deg = (angle_idx / (grid_count_y - 1)) * 360
                path_x.append(x_coord)
                path_y.append(angle_offset_deg)
                point_colors.append(plt.cm.viridis(ik_solution_idx / 8.0))
            
            # Add path to both windows
            ax1.scatter(path_x, path_y, c=point_colors, s=50, marker='o', edgecolors='white', linewidths=1, zorder=10, label='Path')
            if has_cost_data:
                ax2.scatter(path_x, path_y, c=point_colors, s=50, marker='o', edgecolors='white', linewidths=1, zorder=10, label='Path')
            
            # Add search tree to both windows
            search_x = []
            search_y = []
            search_colors = []
            for waypoint_idx, angle_idx, ik_solution_idx, cost_so_far, prev_waypoint_idx, prev_angle_idx, prev_ik_solution_idx in self._search_tree:
                x_coord = (waypoint_idx / grid_count_x) * (extent[1] - extent[0]) + extent[0]
                angle_offset_deg = (angle_idx / (grid_count_y - 1)) * 360
                search_x.append(x_coord)
                search_y.append(angle_offset_deg)
                norm_cost = cost_so_far / max_cost if max_cost > 0 else 0
                search_colors.append(plt.cm.plasma(norm_cost))
            
            ax1.scatter(search_x, search_y, c=search_colors, s=10, marker='x', alpha=0.3, zorder=5, label='Search Tree')
            if has_cost_data:
                ax2.scatter(search_x, search_y, c=search_colors, s=10, marker='x', alpha=0.3, zorder=5, label='Search Tree')
            
            ax1.legend()
            if has_cost_data:
                ax2.legend()
        
        self._visualization_fig = [fig1, fig2]
        plt.show()
    
    @property
    def grid(self) -> Optional[np.ndarray]:
        """Get the calculated state space grid (None if not yet calculated)."""
        return self._grid
    
    @property
    def waypoints(self) -> Optional[list]:
        """Get the waypoints used for state space calculation."""
        return self._waypoints
    
