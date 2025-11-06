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
        ik_solutions = [[None] * grid_count_x for _ in range(grid_count_y)]
        
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
                    movement_offset = Pose(p=np.array([utils.HOOP_CENTER_DISTANCE, 0.0, 0.0]))
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
                    if len(viable_solutions) > 0:
                        solution_distances = []
                        for ik_sol in viable_solutions:
                            test_robot._set_q(ik_sol)
                            test_robot.step_simulation()
                            distance = test_robot.get_robot_to_maze_distance()
                            solution_distances.append((distance, ik_sol))
                        
                        # Sort by distance (furthest first = highest distance first)
                        solution_distances.sort(key=lambda x: x[0], reverse=True)
                        viable_solutions = [sol for _, sol in solution_distances]
                    
                    # Store viable IK solutions (empty list if none)
                    ik_solutions[iy][ix] = viable_solutions
        
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
    
    def move_to_point(self, waypoint_idx: int, angle_idx: int) -> bool:
        """
        Move robot to a specific point in the state space.
        """
        if self._ik_solutions is None:
            raise ValueError("State space not calculated. Call calculate() first.")
        
        # Check if we have a stored IK solution for this point
        stored_solutions = self._ik_solutions[angle_idx][waypoint_idx]
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
        grid_count_y = self._num_angle_samples
        grid_count_x = len(self._waypoints)
        
        # Compute distances from saved IK solutions
        # Use the first solution (index 0) which is the furthest from the maze after sorting
        grid = np.zeros((grid_count_y, grid_count_x))
        for iy in range(grid_count_y):
            for ix in range(grid_count_x):
                viable_sols = ik_solutions[iy][ix]
                if viable_sols is None or len(viable_sols) == 0:
                    grid[iy, ix] = -0.05
                else:
                    # Use the first solution (furthest from maze)
                    self._robot._set_q(viable_sols[0])
                    self._robot.step_simulation()
                    distance = self._robot.get_robot_to_maze_distance()
                    grid[iy, ix] = distance
        
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
            for waypoint_idx, angle_idx in self._path_states:
                # Convert grid indices to plot coordinates
                x_coord = (waypoint_idx / grid_count_x) * (extent[1] - extent[0]) + extent[0]
                angle_offset_deg = (angle_idx / (grid_count_y - 1)) * 360
                path_x.append(x_coord)
                path_y.append(angle_offset_deg)
            
            # Plot path as colored points
            ax.scatter(path_x, path_y, c='red', s=50, marker='o', edgecolors='white', linewidths=1, zorder=10, label='Path')
            ax.plot(path_x, path_y, c='red', linewidth=2, alpha=0.5, zorder=9)
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
            viable_sols = ik_solutions[iy][ix]
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
            viable_sols = ik_solutions[iy][ix]
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
    
    ################################################################################################################
    # PATH FINDING
    ################################################################################################################
    
    def get_q_for_state(self, waypoint_idx: int, angle_idx: int) -> Optional[np.ndarray]:
        """Get the first (safest) IK solution for a given state, or None if invalid."""
        if self._ik_solutions is None:
            return None
        if waypoint_idx < 0 or waypoint_idx >= len(self._waypoints):
            return None
        if angle_idx < 0 or angle_idx >= self._num_angle_samples:
            return None
        
        viable_sols = self._ik_solutions[angle_idx][waypoint_idx]
        if viable_sols is None or len(viable_sols) == 0:
            return None
        return viable_sols[0]  # First solution is furthest from maze (safest)
    
    def _joint_distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """Calculate L2 distance in joint space."""
        return np.linalg.norm(q1 - q2)
    
    def _get_neighbors(self, waypoint_idx: int, angle_idx: int, step_size: int) -> List[Tuple[int, int]]:
        """
        Get valid neighboring states within step_size distance.
        Returns list of (waypoint_idx, angle_idx) tuples.
        
        Note: The angle dimension is circular (0 and num_angle_samples-1 are adjacent).
        """
        if self._ik_solutions is None:
            return []
        
        grid_count_x = len(self._waypoints)
        grid_count_y = self._num_angle_samples
        neighbors = []
        
        # Generate neighbors within step_size in both dimensions
        for dx in range(-step_size, step_size + 1):
            for dy in range(-step_size, step_size + 1):
                if dx == 0 and dy == 0:
                    continue  # Skip self
                
                new_x = waypoint_idx + dx
                
                # Handle waypoint bounds (not circular)
                if new_x < 0 or new_x >= grid_count_x:
                    continue
                
                # Handle angle bounds (circular - wrap around)
                new_y = angle_idx + dy
                # Wrap around if out of bounds
                if new_y < 0:
                    new_y = grid_count_y + new_y  # Wrap to end
                elif new_y >= grid_count_y:
                    new_y = new_y - grid_count_y  # Wrap to start
                
                # Check if state has valid IK solutions
                if self.get_q_for_state(new_x, new_y) is not None:
                    neighbors.append((new_x, new_y))
        
        return neighbors
    
    def _calculate_edge_cost(self, 
                             state1: Tuple[int, int], 
                             state2: Tuple[int, int],
                             alpha: float,
                             beta: float,
                             test_robot: 'PybulletRobot',
                             num_path_steps: int = 10) -> float:
        """
        Calculate cost of moving from state1 to state2.
        Cost = alpha * joint_distance - beta * min_distance_from_maze
        
        To ensure non-negative costs, we add a base offset. The relative ordering
        is preserved: shorter paths with larger maze distances have lower cost.
        
        Args:
            test_robot: Robot instance to use for collision/distance checking (can be separate simulation)
        
        Returns float cost, or float('inf') if path is invalid/colliding.
        """
        q1 = self.get_q_for_state(state1[0], state1[1])
        q2 = self.get_q_for_state(state2[0], state2[1])
        
        if q1 is None or q2 is None:
            return float('inf')
        
        # Check for collision along path (use fewer steps for speed)
        if test_robot.is_path_colliding_with_maze(q1, q2, num_path_steps):
            return float('inf')
        
        # Calculate joint space distance
        joint_dist = self._joint_distance(q1, q2)
        
        # For cost calculation, use a simplified distance check (only check endpoints and middle)
        # This is much faster than checking all num_path_steps
        # Check start, middle, and end points
        min_maze_dist = float('inf')
        for t in [0.0, 0.5, 1.0]:
            q = q1 * (1 - t) + q2 * t
            test_robot._set_q(q)
            test_robot.step_simulation()
            dist = test_robot.get_robot_to_maze_distance()
            if dist < min_maze_dist:
                min_maze_dist = dist
        
        # Cost function: penalize joint movement, reward distance from maze
        # Also add a small penalty for going backwards in waypoint index
        # But the main cost is based on actual joint distance (which already accounts for both dimensions)
        waypoint_progress = state2[0] - state1[0]  # Positive = forward, negative = backward
        angle_change = abs(state2[1] - state1[1])   # Angle dimension change
        
        backward_penalty = 0.0
        if waypoint_progress < 0:
            # Penalize going backwards - make it more expensive
            # Scale penalty based on how far backward we go
            backward_penalty = alpha * abs(waypoint_progress) * 0.1  # Extra cost for backward movement
        
        # The joint_dist already accounts for the full 2D movement in joint space,
        # so we don't need to add separate waypoint/angle costs here.
        # The backward_penalty just discourages unnecessary backward movement.
        
        # Add a base offset to ensure costs are non-negative (required for A*)
        # Using max expected maze distance as offset (typically 0.1-0.2m)
        # Increase base offset to make maze distance rewards more significant
        base_offset = beta * 0.3  # Increased from 0.2 to give more weight to maze distance
        cost = alpha * joint_dist - beta * min_maze_dist + backward_penalty + base_offset
        
        # Ensure cost is non-negative (safety check)
        return max(0.0, cost)
    
    def _heuristic(self, 
                   state: Tuple[int, int], 
                   goal: Tuple[int, int],
                   alpha: float,
                   beta: float) -> float:
        """
        Heuristic function for A*.
        Estimates cost from state to goal using grid distance.
        Must be admissible (never overestimate) for A* to find optimal paths.
        
        We use a simple grid-based heuristic that doesn't modify robot state.
        Properly accounts for both waypoint (x) and angle (y) dimensions.
        """
        # Grid distance in both dimensions
        dx = abs(state[0] - goal[0])  # Waypoint difference (not circular)
        
        # Angle difference (circular - find shortest path around the circle)
        dy_direct = abs(state[1] - goal[1])
        grid_count_y = self._num_angle_samples
        dy_wrap = grid_count_y - dy_direct  # Distance going the other way around
        dy = min(dy_direct, dy_wrap)  # Take the shorter path
        
        # Estimate joint distance based on both dimensions
        # Waypoint changes typically require more joint movement than angle changes
        # But both should be considered
        waypoint_cost = dx * 0.15  # Cost per waypoint step
        angle_cost = dy * 0.08     # Cost per angle step (smaller but still significant)
        
        # Penalize going backwards in waypoint index (but still count the distance)
        if state[0] > goal[0]:
            # We're past the goal, so we need to go backward
            # Add extra penalty but still count the distance
            waypoint_cost *= 1.5  # Make backward movement more expensive
        
        estimated_joint_dist = waypoint_cost + angle_cost
        
        # For heuristic, we use a conservative estimate of maze distance
        # We assume worst case (minimum distance = 0) to ensure admissibility
        estimated_maze_dist = 0.0  # Conservative: assume worst case
        
        # Apply same base offset as in edge cost for consistency
        base_offset = beta * 0.2
        heuristic = alpha * estimated_joint_dist - beta * estimated_maze_dist + base_offset
        
        return max(0.0, heuristic)
    
    def find_path(self,
                  start_waypoint_idx: int,
                  start_angle_idx: int,
                  goal_waypoint_idx: int,
                  goal_angle_idx: int,
                  step_size: int = 1,
                  alpha: float = 1.0,
                  beta: float = 0.5,
                  test_robot: Optional['PybulletRobot'] = None,
                  num_path_steps: int = 10,
                  early_stop_waypoint_threshold: int = 3) -> Optional[List[np.ndarray]]:
        """
        Find a path through the state space using A* algorithm.
        
        Args:
            start_waypoint_idx: Starting waypoint index
            start_angle_idx: Starting angle index
            goal_waypoint_idx: Goal waypoint index
            goal_angle_idx: Goal angle index
            step_size: Maximum step size in grid (allows skipping adjacent states)
            alpha: Weight for joint distance (path length) - higher = prioritize shorter paths
            beta: Weight for distance from maze (safety) - higher = prioritize staying far from maze
            test_robot: Robot instance to use for collision/distance checking (if None, uses self._robot)
            num_path_steps: Number of interpolation steps for collision checking between states
            early_stop_waypoint_threshold: Stop early if we get within this many waypoints of goal (None to disable)
        
        Returns:
            List of joint configurations (q) representing the path, or None if no path found
        """
        if self._ik_solutions is None:
            raise ValueError("State space not calculated. Call calculate() first.")
        
        # Use provided test_robot or fall back to self._robot
        if test_robot is None:
            test_robot = self._robot
        
        start_state = (start_waypoint_idx, start_angle_idx)
        goal_state = (goal_waypoint_idx, goal_angle_idx)
        
        # Check if start and goal are valid
        if self.get_q_for_state(*start_state) is None:
            return None
        if self.get_q_for_state(*goal_state) is None:
            return None
        
        # A* algorithm
        open_set = []  # Priority queue: (f_score, g_score, state, parent)
        heapq.heappush(open_set, (0, 0, start_state, None))
        
        came_from = {}  # Maps state -> parent state
        g_score = {start_state: 0}  # Actual cost from start to state
        f_score = {start_state: self._heuristic(start_state, goal_state, alpha, beta)}  # Estimated total cost
        
        closed_set = set()
        nodes_explored = 0
        max_nodes_to_explore = 10000  # Safety limit to prevent infinite loops
        
        while open_set and nodes_explored < max_nodes_to_explore:
            current_f, current_g, current_state, parent = heapq.heappop(open_set)
            
            if current_state in closed_set:
                continue
            
            closed_set.add(current_state)
            came_from[current_state] = parent
            nodes_explored += 1
            
            # Progress reporting every 100 nodes
            if nodes_explored % 100 == 0:
                waypoint_dist = abs(current_state[0] - goal_state[0])
                # Calculate circular angle distance
                angle_dist_direct = abs(current_state[1] - goal_state[1])
                grid_count_y = self._num_angle_samples
                angle_dist_wrap = grid_count_y - angle_dist_direct
                angle_dist = min(angle_dist_direct, angle_dist_wrap)
                print(f"[PATHFINDING] Explored {nodes_explored} nodes, open set: {len(open_set)}, current: waypoint={current_state[0]}, angle={current_state[1]}, dist_to_goal: wp={waypoint_dist}, ang={angle_dist}")
            
            # Check if we reached the goal (exact match)
            if current_state == goal_state:
                # Reconstruct path
                path_states = []
                state = current_state
                while state is not None:
                    path_states.append(state)
                    state = came_from.get(state)
                path_states.reverse()
                
                # Store path states for visualization
                self._path_states = path_states
                
                # Convert states to joint configurations
                q_path = []
                for state in path_states:
                    q = self.get_q_for_state(state[0], state[1])
                    if q is not None:
                        q_path.append(q)
                
                return q_path if len(q_path) > 0 else None
            
            # Early stopping: if we're close enough to the goal, accept it
            if early_stop_waypoint_threshold is not None:
                waypoint_dist = abs(current_state[0] - goal_state[0])
                if waypoint_dist <= early_stop_waypoint_threshold:
                    # We're close enough - try to construct a path to the actual goal
                    # First, try to find a direct path from current to goal
                    q_current = self.get_q_for_state(current_state[0], current_state[1])
                    q_goal = self.get_q_for_state(goal_state[0], goal_state[1])
                    
                    if q_current is not None and q_goal is not None:
                        # Check if we can go directly from current to goal
                        if not test_robot.is_path_colliding_with_maze(q_current, q_goal, num_path_steps):
                            # Direct path exists - reconstruct path to current, then add goal
                            path_states = []
                            state = current_state
                            while state is not None:
                                path_states.append(state)
                                state = came_from.get(state)
                            path_states.reverse()
                            path_states.append(goal_state)  # Add goal
                            
                            # Store path states for visualization
                            self._path_states = path_states
                            
                            # Convert states to joint configurations
                            q_path = []
                            for state in path_states:
                                q = self.get_q_for_state(state[0], state[1])
                                if q is not None:
                                    q_path.append(q)
                            
                            print(f"[PATHFINDING] Early stopping: found path within {waypoint_dist} waypoints of goal")
                            return q_path if len(q_path) > 0 else None
                        # If direct path doesn't work, continue searching
            
            # Explore neighbors
            neighbors = self._get_neighbors(current_state[0], current_state[1], step_size)
            for neighbor in neighbors:
                if neighbor in closed_set:
                    continue
                
                # Calculate edge cost
                edge_cost = self._calculate_edge_cost(current_state, neighbor, alpha, beta, test_robot, num_path_steps)
                if edge_cost == float('inf'):
                    continue  # Invalid edge (collision)
                
                tentative_g = g_score[current_state] + edge_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # This path to neighbor is better
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_state, alpha, beta)
                    heapq.heappush(open_set, (f_score[neighbor], tentative_g, neighbor, current_state))
        
        # No path found - clear any previous path
        self._path_states = None
        if nodes_explored >= max_nodes_to_explore:
            print(f"[PATHFINDING] Search limit reached ({max_nodes_to_explore} nodes). No path found.")
        else:
            print(f"[PATHFINDING] Explored {nodes_explored} nodes. No path found.")
        return None
