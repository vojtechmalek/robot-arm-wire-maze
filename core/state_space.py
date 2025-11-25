import json
from typing import Optional, TYPE_CHECKING, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import heapq
import time

from .pose import Pose
from . import utils

# To avoid circular import
if TYPE_CHECKING:
    from .pybullet_robot import PybulletRobot

#TODO clean up visualization

GROUND_LIMIT = 0.05  # meters

TILT_INCREMENT = 5.0  # degrees
MAX_JOINT_CHANGE = np.deg2rad(90)  # Maximum allowed joint change per step (per joint)
        

# Tilt penalty configuration
TILT_BASE_PENALTY = 5.0  # Base penalty for any non-zero tilt
TILT_INCREMENTAL_PENALTY = 5.0  # Additional penalty per 5° increment beyond base

Z_OFFSET = 0.005  # meters, vertical offset based on hoop angle

################################################################################################################
# COST FUNCTIONS
################################################################################################################

def tilt_penalty(tilt_magnitude_deg: int) -> float:
    """
    Calculate the penalty for being in a tilted state.
    """
    if tilt_magnitude_deg > 0:
        increments = int(np.ceil(tilt_magnitude_deg / TILT_INCREMENT))
        return TILT_BASE_PENALTY + increments * TILT_INCREMENTAL_PENALTY
    return 0.0

def state_transition_cost(old_q: np.ndarray, new_q: np.ndarray, 
                          new_tilt_magnitude_deg: int) -> float:
    """
    Calculate the total cost of transitioning from one state to another.
    """
    # Joint transition cost
    step_cost = np.sum((old_q - new_q) ** 2) # sum of squared joint differences
    
    # Transition penalty: penalize any step that ends in a tilted state
    # This encourages staying at lower tilts
    tilt_penalty_value = tilt_penalty(new_tilt_magnitude_deg)
    
    return step_cost + tilt_penalty_value

################################################################################################################
# STATE SPACE
################################################################################################################

class StateSpace:

    ################################################################################################################
    # INITIALIZATION
    ################################################################################################################

    def __init__(self, robot: 'PybulletRobot', maze: str, maze_pose: Pose):
        self._robot = robot
        self._maze = maze
        self._maze_pose = maze_pose
        
        # IK solutions: dict keyed by (waypoint_idx, angle_idx, tilt_roll_delta, tilt_pitch_delta) -> list of IK solutions
        self._ik_solutions: Optional[dict] = None
        self._waypoints: Optional[list] = None
        self._num_angle_samples: Optional[int] = None
        self._path_states: Optional[List[Tuple[int, int, float, float]]] = None  # Path states (waypoint_idx, angle_idx, tilt_roll, tilt_pitch) for visualization
        self._visited_costs = None  # Visited costs: (waypoint_idx, angle_idx, tilt_roll, tilt_pitch, q_tuple) -> cost, where q_tuple is tuple(q)
        self._best_ik_at_position = None  # Best IK solution at each (waypoint_idx, angle_idx, tilt_roll, tilt_pitch) with minimum cost
        self._q_path: Optional[List[np.ndarray]] = None  # Joint-space path (list of q configurations)
        self._tested_positions = None  # All tested positions (wp_idx, ang_idx) for visualization - includes all tilts projected to 2D

    ################################################################################################################
    # IK SOLUTIONS
    ################################################################################################################

    def _get_ik_solutions(self, waypoint_idx: int, angle_idx: int, test_robot, 
                         tilt_magnitude_deg: int = 0, tilt_direction_idx: int = 0) -> List[np.ndarray]:
        """
        Get IK solutions for a specific waypoint, angle offset, and tilt.
        """
        # Create cache key using discrete (magnitude, direction_idx) - avoids float precision issues
        cache_key = (waypoint_idx, angle_idx, tilt_magnitude_deg, tilt_direction_idx)
        
        # Check if already computed
        if cache_key in self._ik_solutions:
            return self._ik_solutions[cache_key]
        
        # Convert (magnitude_deg, direction_idx) to (roll, pitch) using lookup table
        tilt_roll_delta, tilt_pitch_delta = self._tilt_key_to_roll_pitch(tilt_magnitude_deg, tilt_direction_idx)
        
        # Compute IK solutions for this position
        wp = self._waypoints[waypoint_idx]
        angle_offset = (angle_idx / (self._num_angle_samples - 1)) * 2 * np.pi
        
        # Extract maze position and rotation
        base_center_robot = self._maze_pose.p
        center_rot = np.rad2deg(self._maze_pose.rpy[2])
        
        pose = utils.get_waypoint_pose(
            wp, base_center_robot, center_rot, angle_offset,
            tilt_roll_delta=tilt_roll_delta, tilt_pitch_delta=tilt_pitch_delta
        )
        
        # Apply movement offset (hoop center distance)
        movement_offset = Pose(p=np.array([utils.HOOP_CENTER_DISTANCE, 0.0, -utils.HOOP_OFFSET_Z]))
        pose = pose @ movement_offset
        
        #! TEMP: adjust height based on angle
        # get hoop rotation z vector
        hoop_z = pose.R[:, 0]
        world_z = np.array([0, 0, 1])
        z_dot_product = np.dot(hoop_z, world_z)

        pose._p[2] -= Z_OFFSET * np.abs(z_dot_product)
        # pose._p[2] -= 0.05
        
        # Skip if Z position is too low
        if pose.p[2] < GROUND_LIMIT:
            self._ik_solutions[cache_key] = []
            return []
        
        # Solve IK (robot.ik expects SE3 matrix)
        ik_sols = self._robot.ik(pose.T)
        if len(ik_sols) == 0:
            self._ik_solutions[cache_key] = []
            return []
        
        # Filter by joint limits
        ik_filtered_sols = [q for q in ik_sols if self._robot.in_limits(q)]
        if len(ik_filtered_sols) == 0:
            self._ik_solutions[cache_key] = []
            return []
        
        # Test for collisions in simulation and save viable solutions
        viable_solutions = []
        for ik_sol in ik_filtered_sols:
            test_robot._set_q(ik_sol)
            test_robot.step_simulation()
            if not test_robot.is_colliding_with_maze():
                viable_solutions.append(ik_sol)
        
        self._ik_solutions[cache_key] = viable_solutions
        return viable_solutions

    ################################################################################################################
    # FIND PATH
    ################################################################################################################
 
    def find_path(self, waypoints_path: str, num_angle_samples: int, step: int = 1, collision_padding: float = 0.0,
                  max_tilt_angle: float = 15.0, num_tilt_directions: int = 4) -> Optional[List[np.ndarray]]:
        """
        Find a joint-space path through the maze from start to goal using Dijkstra's algorithm.
        """
        # Load waypoints
        waypoints = json.load(open(waypoints_path, "r"))
        
        # remove last n waypoints that is in the base
        waypoints = waypoints[:-2] 
        
        self._waypoints = waypoints
        self._num_angle_samples = num_angle_samples
     
        # Generate tilt angles in increments of TILT_INCREMENT
        tilt_angles_deg = []
        angle = TILT_INCREMENT
        while angle <= max_tilt_angle:
            tilt_angles_deg.append(angle)
            angle += TILT_INCREMENT
        tilt_directions = utils.get_tilt_directions(num_tilt_directions)
        
        # Precompute tilt directions at each increment (in radians) for efficiency
        # Key: tilt_angle_deg -> list of (tilt_roll_delta, tilt_pitch_delta) tuples
        precomputed_tilt_directions = {}
        for tilt_angle_deg in tilt_angles_deg:
            tilt_angle_rad = np.deg2rad(tilt_angle_deg)
            precomputed_tilt_directions[tilt_angle_deg] = [
                (roll_dir * tilt_angle_rad, pitch_dir * tilt_angle_rad)
                for roll_dir, pitch_dir in tilt_directions
            ]
        
        # Precompute all tilt vectors: (magnitude_deg, direction_idx) -> (roll, pitch)
        # This allows fast lookup instead of recalculating by scaling
        # Also enables using discrete (magnitude, direction) in cache keys instead of floats
        precomputed_all_tilt_vectors = {}
        # Add 0° tilt (magnitude=0, direction_idx=0)
        precomputed_all_tilt_vectors[(0, 0)] = (0.0, 0.0)
        # Add all other tilts
        for tilt_angle_deg in tilt_angles_deg:
            for direction_idx, (roll, pitch) in enumerate(precomputed_tilt_directions[tilt_angle_deg]):
                precomputed_all_tilt_vectors[(tilt_angle_deg, direction_idx)] = (roll, pitch)
        
        # Store for use in helper functions and lookups
        self._precomputed_all_tilt_vectors = precomputed_all_tilt_vectors
        self._precomputed_tilt_directions = precomputed_tilt_directions
        
        # Helper function to convert (magnitude_deg, direction_idx) -> (roll, pitch)
        # Simple lookup in precomputed vectors
        def tilt_key_to_roll_pitch(tilt_magnitude_deg: int, tilt_direction_idx: int) -> tuple:
            """Convert (magnitude_deg, direction_idx) to (roll, pitch) using precomputed lookup."""
            return precomputed_all_tilt_vectors.get((tilt_magnitude_deg, tilt_direction_idx), (0.0, 0.0))
        
        # Helper function to convert (roll, pitch) -> (magnitude_deg, direction_idx)
        # Used when we have roll/pitch from old states or need to convert for cache keys
        def roll_pitch_to_tilt_key(tilt_roll: float, tilt_pitch: float) -> tuple:
            """Convert (roll, pitch) to (magnitude_deg, direction_idx)."""
            if abs(tilt_roll) < 1e-6 and abs(tilt_pitch) < 1e-6:
                return (0, 0)
            
            # Calculate magnitude and round to nearest TILT_INCREMENT
            magnitude_rad = np.sqrt(tilt_roll**2 + tilt_pitch**2)
            magnitude_deg = round(np.rad2deg(magnitude_rad))
            magnitude_deg = round(magnitude_deg / TILT_INCREMENT) * TILT_INCREMENT
            
            # Fast path: check if magnitude matches a precomputed tilt angle
            if magnitude_deg in tilt_angles_deg or magnitude_deg == 0:
                # Find matching direction by comparing to precomputed vectors for this magnitude
                epsilon = 1e-5
                for direction_idx, (precomputed_roll, precomputed_pitch) in enumerate(precomputed_tilt_directions.get(magnitude_deg, [])):
                    if abs(tilt_roll - precomputed_roll) < epsilon and abs(tilt_pitch - precomputed_pitch) < epsilon:
                        return (int(magnitude_deg), direction_idx)
            
            # Slow path: no exact match (intermediate/scaled tilt), find closest direction
            best_match = (int(magnitude_deg), 0)
            min_diff = float('inf')
            epsilon = 1e-5
            for test_magnitude_deg in [0] + tilt_angles_deg:
                for direction_idx, (precomputed_roll, precomputed_pitch) in enumerate(precomputed_tilt_directions.get(test_magnitude_deg, [])):
                    diff = abs(tilt_roll - precomputed_roll) + abs(tilt_pitch - precomputed_pitch)
                    if diff < min_diff:
                        min_diff = diff
                        best_match = (int(test_magnitude_deg), direction_idx)
                        if diff < epsilon:
                            return best_match
            
            return best_match
        
        # Store helper functions as instance methods
        self._tilt_key_to_roll_pitch = tilt_key_to_roll_pitch
        self._roll_pitch_to_tilt_key = roll_pitch_to_tilt_key
        
        # Cache for tilts_to_try_deg lists (keyed by current_tilt_deg to avoid rebuilding)
        self._cached_tilts_to_try = {}

        # Try all movement directions (prefer 0° tilt when possible)
        movements = [
            # Right: next waypoint, same angle
            (step, 0),
            # Right+Up: next waypoint, next angle (diagonal)
            (step, step),
            # Right+Down: next waypoint, previous angle (diagonal)
            (step, -step),
        ]
        
        # Clear any previous path and pathfinding data
        self._path_states = None
        self._visited_costs = None
        self._best_ik_at_position = None
        
        grid_count_x = len(waypoints) # number of x-axis positions
        grid_count_y = num_angle_samples # number of y-axis positions
        
        # Initialize IK solutions storage as dict - IK solutions are computed on demand
        # Key: (waypoint_idx, angle_idx, tilt_roll_delta, tilt_pitch_delta) -> list of IK solutions
        self._ik_solutions = {}
        # Only track tested_positions if visualization will be used (saves memory and set operations)
        # Visualization is typically only used when GUI is enabled
        self._tested_positions = set() if self._robot._use_gui else None
        

        # --- Simulation setup ---
        # if we are in GUI mode, create a headless simulation for faster collision testing
        if self._robot._use_gui and self._robot._bullet_client.isConnected():
            # Import here to avoid circular dependency
            from .pybullet_robot import PybulletRobot
            # Create a headless simulation for faster collision testing
            test_robot = PybulletRobot(maze=self._maze, maze_pose=self._maze_pose, use_gui=False, collision_padding=collision_padding)
            created_separate_sim = True
        # else just use the main simulation
        else:
            test_robot = self._robot
            created_separate_sim = False

        # Priority queue: (cost, waypoint_idx, angle_idx, tilt_magnitude_deg, tilt_direction_idx, q_tuple, ...)
        # q_tuple is tuple(q) for hashing - convert back to np.array when needed
        # States now use (magnitude_deg, direction_idx) instead of (roll, pitch) to avoid float precision issues
        priority_queue = []
        heapq.heapify(priority_queue)
        
        # Visited states: (waypoint_idx, angle_idx, tilt_magnitude_deg, tilt_direction_idx, q_tuple) -> best_cost
        # q_tuple is tuple(q) for hashing
        visited_costs = {}
        
        # Parent tracking for path reconstruction: q_tuple -> (parent_q_tuple, waypoint_idx, angle_idx, tilt_magnitude_deg, tilt_direction_idx)
        # Using q_tuple as key since it uniquely identifies the robot state
        parents = {}
        
        # Function thats called when we are stuck (cant progress forwards from this state) and
        # it continues in the same direction by scaling current values
        def try_tilts_at_position(wp_idx: int, ang_idx: int, current_tilt_magnitude_deg: int, current_tilt_direction_idx: int,
                                   current_q: np.ndarray, current_cost: float, current_state: tuple):
            """
            Try tilts when stuck at a position. Progressively tries the next tilt increment.
            If at 0° tilt, tries all directions. Otherwise, continues in the same direction.
            """
            # Find the next tilt increment to try (current + TILT_INCREMENT)
            next_tilt_deg = current_tilt_magnitude_deg + TILT_INCREMENT
            
            # Only try tilts up to max_tilt_angle
            if next_tilt_deg > max_tilt_angle:
                return  # No more tilts to try
            
            # If at 0° tilt, try all directions. Otherwise, continue in the same direction.
            if current_tilt_magnitude_deg == 0:
                # At 0° tilt: try all precomputed directions at this increment
                # directions_to_try is list of (magnitude_deg, direction_idx) tuples
                directions_to_try = [(next_tilt_deg, dir_idx) for dir_idx in range(len(precomputed_tilt_directions[next_tilt_deg]))]
            else:
                # Not at 0°: continue in the same direction
                directions_to_try = [(next_tilt_deg, current_tilt_direction_idx)]
            
            # Process each direction
            for next_tilt_magnitude_deg, next_tilt_direction_idx in directions_to_try:
                # Track tested position (projected to 2D grid, so tilt doesnt matter)
                # Track it even if we dont have any IK solutions, because we need to know that we tested this position
                if self._tested_positions is not None:
                    self._tested_positions.add((wp_idx, ang_idx))
                
                # Get IK solutions with this tilt
                viable_sols = self._get_ik_solutions(wp_idx, ang_idx, test_robot, 
                                                     next_tilt_magnitude_deg, next_tilt_direction_idx)
                
                if len(viable_sols) == 0:
                    continue
                
                # Try all IK solutions
                for tilted_q in viable_sols:
                    
                    # Check joint limit transition from current state
                    q_diff = tilted_q - current_q
                    if np.any(np.abs(q_diff) > MAX_JOINT_CHANGE):
                        continue
                    
                    # Create new state with q tuple for hashing (needed for visited_costs check)
                    q_tuple = tuple(tilted_q)
                    new_state = (wp_idx, ang_idx, next_tilt_magnitude_deg, next_tilt_direction_idx, q_tuple)
                    
                    # Calculate transition cost (needed for visited_costs comparison)
                    transition_cost = state_transition_cost(current_q, tilted_q, next_tilt_magnitude_deg)
                    new_cost = current_cost + transition_cost
                    
                    # Check if this is a better path (FAST check - do this BEFORE expensive collision check)
                    existing_cost = visited_costs.get(new_state)
                    if existing_cost is not None and new_cost >= existing_cost:
                        continue
                    
                    # Check collision along path (SLOW check - only do if state is new/better)
                    if test_robot.is_path_colliding_with_maze(current_q, tilted_q):
                        continue
                    
                    # Update visited costs and add to queue
                    visited_costs[new_state] = new_cost
                    # Extract parent q_tuple from current_state (q_tuple is at index 4)
                    current_q_tuple = current_state[4]
                    parents[q_tuple] = (current_q_tuple, wp_idx, ang_idx, next_tilt_magnitude_deg, next_tilt_direction_idx)
                    # Queue format: (cost, wp, ang, tilt_magnitude_deg, tilt_direction_idx, q_tuple)
                    heapq.heappush(priority_queue, (new_cost, wp_idx, ang_idx, next_tilt_magnitude_deg, next_tilt_direction_idx, q_tuple))

        # Initialize: start from all valid states in first column (waypoint_idx=0)
        zero_tilt_total_solutions = 0
        for angle_idx in range(grid_count_y):
            viable_sols = self._get_ik_solutions(0, angle_idx, test_robot, 0, 0)
            # Track tested position (all positions tested, regardless of viable IK)
            if self._tested_positions is not None:
                self._tested_positions.add((0, angle_idx))
            if len(viable_sols) == 0:
                continue
            zero_tilt_total_solutions += len(viable_sols)
            for q in viable_sols:
                q_tuple = tuple(q)
                # Use (magnitude_deg=0, direction_idx=0) for 0° tilt
                state = (0, angle_idx, 0, 0, q_tuple)
                visited_costs[state] = 0.0
                # Start states have no parent
                parents[q_tuple] = (None, 0, angle_idx, 0, 0)
                heapq.heappush(priority_queue, (0.0, 0, angle_idx, 0, 0, q_tuple))
        
        if True: #zero_tilt_total_solutions == 0:
            # Try each tilt angle progressively
            for idx, tilt_angle_deg in enumerate(tilt_angles_deg):
                
                tilt_angle_valid_states = 0
                for angle_idx in range(grid_count_y):
                    # Use precomputed tilt directions - iterate by direction index
                    for direction_idx, (tilt_roll_delta, tilt_pitch_delta) in enumerate(precomputed_tilt_directions[tilt_angle_deg]):
                        
                        # Scale tilt by index to get progressively larger tilts
                        # Calculate actual magnitude after scaling
                        scaled_magnitude_deg = tilt_angle_deg * (idx + 1)
                        if scaled_magnitude_deg > max_tilt_angle:
                            continue
                        
                        # Get IK solutions with this tilt (use scaled magnitude and same direction)
                        # For scaled tilts, we need to find the closest precomputed vector
                        # For now, use the base tilt_angle_deg and direction_idx
                        viable_sols = self._get_ik_solutions(0, angle_idx, test_robot, 
                                                             tilt_angle_deg, direction_idx)
                        
                        if len(viable_sols) == 0:
                            continue
                        
                        tilt_angle_valid_states += len(viable_sols)
                        
                        # Start states: penalize starting in a tilted state (no joint transition cost)
                        start_cost = tilt_penalty(scaled_magnitude_deg)
                        
                        # Try all IK solutions
                        for tilted_q in viable_sols:
                            
                            # Create new state with q tuple for hashing
                            # Use (magnitude_deg, direction_idx) instead of (roll, pitch)
                            q_tuple = tuple(tilted_q)
                            new_state = (0, angle_idx, scaled_magnitude_deg, direction_idx, q_tuple)
                            
                            # Update visited costs and add to queue
                            visited_costs[new_state] = start_cost
                            # Start states have no parent
                            parents[q_tuple] = (None, 0, angle_idx, scaled_magnitude_deg, direction_idx)
                            heapq.heappush(priority_queue, (start_cost, 0, angle_idx, scaled_magnitude_deg, direction_idx, q_tuple))
                            tilt_angle_valid_states += 1
            
        # Track the best goal state (lowest cost to reach final waypoint)
        best_goal_state = None
        
        print(f"[STATE_SPACE] Starting pathfinding")

        try:
            while priority_queue:               
                cost_so_far, waypoint_idx, angle_idx, tilt_magnitude_deg, tilt_direction_idx, q_tuple = heapq.heappop(priority_queue)
                
                # Convert q_tuple back to numpy array
                current_q = np.array(q_tuple)
                
                # Skip if we've found a better path to this state
                state = (waypoint_idx, angle_idx, tilt_magnitude_deg, tilt_direction_idx, q_tuple)
                existing_cost = visited_costs.get(state)
                if existing_cost is not None and cost_so_far > existing_cost:
                    continue
                
                # Check if we reached the goal (last waypoint)
                # With Dijkstra's algorithm, the first goal state we pop is guaranteed to be optimal
                if waypoint_idx == grid_count_x - 1:
                    best_goal_state = state
                    print(f"[STATE_SPACE] Found path to goal with cost {cost_so_far:.4f}")
                    break  # Terminate immediately - we've found the optimal path
                
                # Track if we found any valid moves
                valid_moves_found = False
                
                # Convert (magnitude, direction_idx) to (roll, pitch) for calculations
                tilt_roll, tilt_pitch = self._tilt_key_to_roll_pitch(tilt_magnitude_deg, tilt_direction_idx)
                current_tilt_deg = float(tilt_magnitude_deg)
                
                # Try tilts incrementally from 0° up to current_tilt
                # This is more systematic than trying just 0° and current_tilt
                # Cache the list to avoid rebuilding for same tilt magnitude
                if tilt_magnitude_deg not in self._cached_tilts_to_try:
                    tilts_to_try_deg = []
                    tilt_deg = 0
                    while tilt_deg <= tilt_magnitude_deg:
                        tilts_to_try_deg.append(tilt_deg)
                        if tilt_deg >= tilt_magnitude_deg:
                            break
                        tilt_deg += int(TILT_INCREMENT)
                    self._cached_tilts_to_try[tilt_magnitude_deg] = tilts_to_try_deg
                else:
                    tilts_to_try_deg = self._cached_tilts_to_try[tilt_magnitude_deg]
                
                # Try each tilt increment
                for test_tilt_magnitude_deg in tilts_to_try_deg:
                    if valid_moves_found:
                        break  # Found a valid move, no need to try higher tilts
                    
                    # Determine tilt (magnitude, direction_idx) to try
                    if test_tilt_magnitude_deg == 0:
                        # 0° tilt: use (0, 0)
                        next_tilt_magnitude_deg, next_tilt_direction_idx = 0, 0
                    elif test_tilt_magnitude_deg == tilt_magnitude_deg:
                        # Current tilt: use current direction
                        next_tilt_magnitude_deg, next_tilt_direction_idx = tilt_magnitude_deg, tilt_direction_idx
                    elif test_tilt_magnitude_deg in tilt_angles_deg:
                        # Exact tilt increment: use same direction as current
                        next_tilt_magnitude_deg, next_tilt_direction_idx = test_tilt_magnitude_deg, tilt_direction_idx
                    else:
                        # Intermediate tilt: use current direction (will be scaled in IK computation)
                        # For intermediate tilts, we still use the same direction_idx
                        next_tilt_magnitude_deg, next_tilt_direction_idx = test_tilt_magnitude_deg, tilt_direction_idx
                    
                    # Convert (magnitude, direction_idx) to (roll, pitch) for IK computation
                    next_tilt_roll, next_tilt_pitch = self._tilt_key_to_roll_pitch(next_tilt_magnitude_deg, next_tilt_direction_idx)
                    
                    # For intermediate tilts, we need to scale the vector
                    if next_tilt_magnitude_deg not in [0] + tilt_angles_deg:
                        # Scale the roll/pitch proportionally
                        if tilt_magnitude_deg > 0:
                            scale_factor = next_tilt_magnitude_deg / tilt_magnitude_deg
                            next_tilt_roll = tilt_roll * scale_factor
                            next_tilt_pitch = tilt_pitch * scale_factor
                    
                    # Try each movement direction with this tilt
                    for delta_wp, delta_ang in movements:
                        next_waypoint_idx = waypoint_idx + delta_wp
                        next_angle_idx = (angle_idx + delta_ang) % grid_count_y
                        
                        # Check bounds
                        if next_waypoint_idx < 0 or next_waypoint_idx >= grid_count_x:
                            continue
                        
                        # For intermediate tilts, we need to find closest precomputed or use scaled version
                        # For now, use the base magnitude and direction_idx
                        next_viable_sols = self._get_ik_solutions(next_waypoint_idx, next_angle_idx, test_robot, 
                                                                 next_tilt_magnitude_deg, next_tilt_direction_idx)
                        
                        # Track tested position (all tilts, projected to 2D grid)
                        if self._tested_positions is not None:
                            self._tested_positions.add((next_waypoint_idx, next_angle_idx))
                        
                        if len(next_viable_sols) == 0:
                            continue
                        
                        # Try all IK solutions
                        for next_q in next_viable_sols:
                            
                            # Check joint limit: each joint must change by <= MAX_JOINT_CHANGE
                            q_diff = next_q - current_q
                            if np.any(np.abs(q_diff) > MAX_JOINT_CHANGE):
                                continue
                            
                            # Create next state with q tuple for hashing (needed for visited_costs check)
                            # Use (magnitude, direction_idx) instead of (roll, pitch)
                            next_q_tuple = tuple(next_q)
                            next_state = (next_waypoint_idx, next_angle_idx, next_tilt_magnitude_deg, next_tilt_direction_idx, next_q_tuple)
                            
                            # Calculate transition cost (needed for visited_costs comparison)
                            transition_cost = state_transition_cost(current_q, next_q, next_tilt_magnitude_deg)
                            new_cost = cost_so_far + transition_cost
                            
                            # Check if this is a better path (FAST check - do this BEFORE expensive collision check)
                            existing_cost = visited_costs.get(next_state)
                            if existing_cost is not None and new_cost >= existing_cost:
                                continue
                            
                            # Check collision along path (SLOW check - only do if state is new/better)
                            if test_robot.is_path_colliding_with_maze(current_q, next_q):
                                continue
                            
                            # Update visited costs and add to queue
                            visited_costs[next_state] = new_cost
                            # Extract parent q_tuple from current state (q_tuple is at index 4)
                            current_q_tuple = state[4]
                            parents[next_q_tuple] = (current_q_tuple, next_waypoint_idx, next_angle_idx, next_tilt_magnitude_deg, next_tilt_direction_idx)
                            heapq.heappush(priority_queue, (new_cost, next_waypoint_idx, next_angle_idx, 
                                                             next_tilt_magnitude_deg, next_tilt_direction_idx, next_q_tuple))
                            valid_moves_found = True
                
                # If still no valid moves, try increasing tilt at this position (fallback)
                if not valid_moves_found:
                    try_tilts_at_position(waypoint_idx, angle_idx, tilt_magnitude_deg, tilt_direction_idx,
                                         current_q, cost_so_far, state)
        
        finally:
            print("")  # Newline after progress output
            # Clean up separate simulation if we created one
            if created_separate_sim:
                test_robot.close()
        
        self._visited_costs = visited_costs
        
        # Compute best IK solution at each position (minimum cost) - only for 0° tilt for visualization
        # Store q_tuple instead of index for robustness
        best_ik_at_position = {}
        for (waypoint_idx, angle_idx, tilt_magnitude_deg, tilt_direction_idx, q_tuple), cost in visited_costs.items():
            # Only consider 0° tilt for grid visualization
            if tilt_magnitude_deg == 0:
                key = (waypoint_idx, angle_idx)
                if key not in best_ik_at_position or cost < best_ik_at_position[key][1]:
                    best_ik_at_position[key] = (q_tuple, cost)
        self._best_ik_at_position = best_ik_at_position
        
        if best_goal_state is None:
            print("[STATE_SPACE] No path found to goal")
            return None
        
        # Reconstruct path from best goal state to start
        # parents[q_tuple] contains (parent_q_tuple, waypoint_idx, angle_idx, tilt_magnitude_deg, tilt_direction_idx)
        path_states = []
        current_q_tuple = best_goal_state[4]
        
        # Add goal state - convert (magnitude, direction_idx) to (roll, pitch) for path_states
        waypoint_idx, angle_idx, tilt_magnitude_deg, tilt_direction_idx, q_tuple = best_goal_state
        tilt_roll, tilt_pitch = self._tilt_key_to_roll_pitch(tilt_magnitude_deg, tilt_direction_idx)
        path_states.append((waypoint_idx, angle_idx, tilt_roll, tilt_pitch, q_tuple, visited_costs[best_goal_state]))
        
        # Follow parent chain backwards
        while current_q_tuple is not None and current_q_tuple in parents:
            parent_q_tuple, waypoint_idx, angle_idx, tilt_magnitude_deg, tilt_direction_idx = parents[current_q_tuple]
            current_state = (waypoint_idx, angle_idx, tilt_magnitude_deg, tilt_direction_idx, current_q_tuple)
            tilt_roll, tilt_pitch = self._tilt_key_to_roll_pitch(tilt_magnitude_deg, tilt_direction_idx)
            path_states.append((waypoint_idx, angle_idx, tilt_roll, tilt_pitch, current_q_tuple, visited_costs[current_state]))
            current_q_tuple = parent_q_tuple
        
        # If we stopped because current_q_tuple is not in parents, it might be a start state
        # Check if it's a valid start state (waypoint_idx=0) and add it
        if current_q_tuple is not None:
            # Try to find the state in visited_costs
            for (wp_idx, ang_idx, tilt_mag_deg, tilt_dir_idx, q_t), cost in visited_costs.items():
                if q_t == current_q_tuple and wp_idx == 0:
                    tr, tp = self._tilt_key_to_roll_pitch(tilt_mag_deg, tilt_dir_idx)
                    path_states.append((wp_idx, ang_idx, tr, tp, q_t, cost))
                    break
        
        path_states = path_states[::-1]  # Reverse to get start -> goal
        self._path_states = path_states
        
        # Print path details
        print(f"\n[PATH] Path details ({len(path_states)} steps):")
        print(f"{'Step':<6} {'Waypoint':<10} {'Angle':<8} {'Tilt Roll':<12} {'Tilt Pitch':<13} {'Tilt Mag':<10} {'Cost':<10}")
        print("-" * 75)
        for i, (waypoint_idx, angle_idx, tilt_roll, tilt_pitch, q_tuple, cost_so_far) in enumerate(path_states):
            tilt_magnitude = np.sqrt(tilt_roll**2 + tilt_pitch**2)
            tilt_magnitude_deg = np.rad2deg(tilt_magnitude)
            angle_offset_deg = (angle_idx / (self._num_angle_samples - 1)) * 360
            print(f"{i:<6} {waypoint_idx:<10} {angle_offset_deg:<8.1f} {np.rad2deg(tilt_roll):<12.2f} {np.rad2deg(tilt_pitch):<13.2f} {tilt_magnitude_deg:<10.2f} {cost_so_far:<10.4f}")
        print("-" * 75)
        
        # Construct q_path directly from path_states (q_tuple is already in state)
        q_path = []
        for waypoint_idx, angle_idx, tilt_roll, tilt_pitch, q_tuple, _ in path_states:
            q_path.append(np.array(q_tuple))
        
        # Store q_path for visualization playback
        self._q_path = q_path
        
        # Debug: print path summary
        num_tilted_steps = sum(1 for _, _, tr, tp, _, _ in path_states if abs(tr) > 1e-6 or abs(tp) > 1e-6)
        print(f"[STATE_SPACE] Path found with {len(path_states)} steps, {num_tilted_steps} steps with tilt")
        
        return q_path

    ################################################################################################################
    # VISUALIZATION
    ################################################################################################################
    
    def _calculate_distance_grid_0_tilt(self) -> np.ndarray:
        """
        Calculate distance from maze for all possible x and y combinations at 0 tilt.
        Returns a grid where each cell contains the minimum distance across all IK solutions.
        """
        if self._ik_solutions is None or self._waypoints is None:
            raise ValueError("State space not calculated. Run find_path() first.")
        
        grid_count_x = len(self._waypoints)
        grid_count_y = self._num_angle_samples
        distance_grid = np.zeros((grid_count_y, grid_count_x))
        
        # Create a test robot for distance calculations if we're in GUI mode
        if self._robot._use_gui and self._robot._bullet_client.isConnected():
            from .pybullet_robot import PybulletRobot
            test_robot = PybulletRobot(maze=self._maze, maze_pose=self._maze_pose, use_gui=False, collision_padding=0.0)
            created_separate_sim = True
        else:
            test_robot = self._robot
            created_separate_sim = False
        
        try:
            print("Calculating distance grid for 0° tilt...")
            for iy in range(grid_count_y):
                for ix in range(grid_count_x):
                    # Get IK solutions for this position (computes on the fly if not cached)
                    viable_sols = self._get_ik_solutions(ix, iy, test_robot, 0, 0)
                    
                    if viable_sols is None or len(viable_sols) == 0:
                        distance_grid[iy, ix] = -0.05
                    else:
                        # Use the smallest distance across all IK solutions
                        smallest_distance = float('inf')
                        for ik_sol in viable_sols:
                            test_robot._set_q(ik_sol)
                            test_robot.step_simulation()
                            distance = test_robot.get_robot_to_maze_distance()
                            if distance < smallest_distance:
                                smallest_distance = distance
                        distance_grid[iy, ix] = smallest_distance
        finally:
            if created_separate_sim:
                test_robot.close()
        
        print("Distance grid calculation complete.")
        return distance_grid
    
    def play_path(self, num_steps_per_segment: int = 10, delay: float = 0.05) -> None:
        """
        Play the path by smoothly interpolating between waypoints.
        """
        if self._q_path is None or len(self._q_path) < 2:
            print("No path available to play. Run find_path() first.")
            return
        
        print(f"Playing path with {len(self._q_path)} waypoints, {num_steps_per_segment} steps per segment...")
        
        for i in range(len(self._q_path) - 1):
            q_start = self._q_path[i]
            q_end = self._q_path[i + 1]
            
            # Interpolate between waypoints
            interpolated_qs = self._robot.interpolate_between_qs(q_start, q_end, num_steps_per_segment)
            
            # Move through interpolated configurations
            for q in interpolated_qs:
                self._robot._set_q(q)
                if self._robot._bullet_client.isConnected():
                    self._robot.step_simulation()
                time.sleep(delay)
        
        print("Path playback complete.")
    
    def visualize(self, viz_0_tilt: bool = False) -> None:
        """
        Display an interactive visualization of the pathfinding cost space.
        Clicking or dragging on the plot will move the robot to that configuration.
        
        Args:
            viz_0_tilt: If True, visualize distance grid for 0° tilt instead of cost grid.
        """
        if self._ik_solutions is None or self._waypoints is None:
            raise ValueError("State space not calculated. Run find_path() first.")
        
        ik_solutions = self._ik_solutions
        grid_count_x = len(self._waypoints)
        grid_count_y = self._num_angle_samples
        extent = [0, 1, 0, 360]  # x_min, x_max, y_min, y_max
        
        # Compute visualization grids
        if viz_0_tilt:
            # Calculate distance grid for 0° tilt (background)
            distance_grid = self._calculate_distance_grid_0_tilt()
            
            # Compute cost grid from pathfinding (overlay)
            cost_grid = np.full((grid_count_y, grid_count_x), np.nan)
            has_cost_data = self._visited_costs is not None and self._best_ik_at_position is not None
            if has_cost_data:
                for iy in range(grid_count_y):
                    for ix in range(grid_count_x):
                        key = (ix, iy)
                        if key in self._best_ik_at_position:
                            q_tuple, cost = self._best_ik_at_position[key]
                            cost_grid[iy, ix] = cost
            
        else:
            # Compute cost grid from pathfinding only
            distance_grid = None
            cost_grid = np.full((grid_count_y, grid_count_x), np.nan)
            has_cost_data = self._visited_costs is not None and self._best_ik_at_position is not None
            if has_cost_data:
                for iy in range(grid_count_y):
                    for ix in range(grid_count_x):
                        key = (ix, iy)
                        if key in self._best_ik_at_position:
                            q_tuple, cost = self._best_ik_at_position[key]
                            cost_grid[iy, ix] = cost
        
        # Keep original title
        grid_title = "Visited Valid States (at 0° tilt)"
        
        # Create visualization
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        plt.subplots_adjust(bottom=0.15)  # Make room for button
        
        # Track colorbar positions to avoid overlap
        # This will be used later for positioning the tilt colorbar
        colorbar_pad = 0.02
        
        # Show distance grid as background if available (without colorbar)
        if viz_0_tilt and distance_grid is not None:
            # Mask invalid distances (where value is -0.05) so they appear white
            distance_grid_masked = np.ma.masked_where(distance_grid < 0, distance_grid)
            img_bg = ax.imshow(distance_grid_masked, cmap="viridis", interpolation="nearest", 
                              extent=extent, origin="lower", aspect="auto", alpha=0.6)
            # No colorbar for distance grid
        
        # Show cost grid as overlay (takes priority)
        if np.any(~np.isnan(cost_grid)):
            img_overlay = ax.imshow(cost_grid, cmap="plasma", interpolation="nearest", 
                                   extent=extent, origin="lower", aspect="auto", alpha=0.8)
            # Always show cost colorbar
            cbar_overlay = plt.colorbar(img_overlay, ax=ax, label="Path Cost")
            # When showing cost, tilt colorbar should be after it
            colorbar_pad = 0.15
        elif not viz_0_tilt:
            # Fallback: show empty grid if no cost data and not showing distance
            empty_grid = np.full((grid_count_y, grid_count_x), np.nan)
            ax.imshow(empty_grid, cmap="plasma", interpolation="nearest", 
                     extent=extent, origin="lower", aspect="auto")
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 360)
        ax.set_xlabel("Path Progress (normalized)")
        ax.set_ylabel("Angle Offset (degrees)")
        ax.set_title(grid_title)
        
        # Add Play Path button
        ax_button = plt.axes([0.4, 0.02, 0.2, 0.05])
        play_button = Button(ax_button, 'Play Path')
        
        def on_play_path_clicked(event):
            """Callback for Play Path button."""
            if self._q_path is None or len(self._q_path) < 2:
                print("No path available to play. Run find_path() first.")
                return
            
            # Default playback parameters
            num_steps_per_segment = 10
            delay = 0.05
            
            print(f"Playing path with {len(self._q_path)} waypoints, {num_steps_per_segment} steps per segment...")
            
            for i in range(len(self._q_path) - 1):
                q_start = self._q_path[i]
                q_end = self._q_path[i + 1]
                
                # Interpolate between waypoints
                interpolated_qs = self._robot.interpolate_between_qs(q_start, q_end, num_steps_per_segment)
                
                # Move through interpolated configurations
                for q in interpolated_qs:
                    self._robot._set_q(q)
                    if self._robot._bullet_client.isConnected():
                        self._robot.step_simulation()
                    time.sleep(delay)
            
            print("Path playback complete.")
        
        play_button.on_clicked(on_play_path_clicked)
        
        def on_interact(event):
            """
            Unified handler for both click and drag events.
            """
            if event.inaxes != ax:
                return
            # For motion events, only proceed if a button is pressed
            if event.name == 'motion_notify_event' and not event.button:
                return
            xdata, ydata = event.xdata, event.ydata
            if xdata is None or ydata is None:
                return
            
            # Convert axis coordinates → grid indices
            ix = int(np.floor((xdata - extent[0]) / (extent[1] - extent[0]) * grid_count_x))
            iy = int(np.floor((ydata - extent[2]) / (extent[3] - extent[2]) * grid_count_y))
            ix = np.clip(ix, 0, grid_count_x - 1)
            iy = np.clip(iy, 0, grid_count_y - 1)
            
            # Get IK solutions for this point (0° tilt only for visualization)
            cache_key = (ix, iy, 0, 0)
            viable_sols = ik_solutions.get(cache_key, [])
            
            if len(viable_sols) == 0:
                    print(f"Waypoint={ix}, angle={iy}: No valid configuration / not reached by pathfinding")
            else:
                if self._best_ik_at_position is not None:
                    key = (ix, iy)
                    if key in self._best_ik_at_position:
                        q_tuple, cost = self._best_ik_at_position[key]
                        print(f"Waypoint={ix}, angle={iy}, cost={cost:.4f}, using stored joint configuration")
                        self._robot._set_q(np.array(q_tuple))
                        if self._robot._bullet_client.isConnected():
                            self._robot.step_simulation()
                    else:
                        if len(viable_sols) > 0:
                            self._robot._set_q(viable_sols[0])
                            if self._robot._bullet_client.isConnected():
                                self._robot.step_simulation()
                else:
                    if len(viable_sols) > 0:
                        self._robot._set_q(viable_sols[0])
                        if self._robot._bullet_client.isConnected():
                            self._robot.step_simulation()
        
        fig.canvas.mpl_connect('button_press_event', on_interact)
        fig.canvas.mpl_connect('motion_notify_event', on_interact)
        
        # Plot tested positions as small crosses (all tilts projected to 2D grid)
        if self._tested_positions is not None and len(self._tested_positions) > 0:
            # tested_positions is already a set, no need to deduplicate
            tested_x = []
            tested_y = []
            for wp_idx, ang_idx in self._tested_positions:
                # Center the cross in the grid rectangle
                # x: rectangles span from wp_idx/grid_count_x to (wp_idx+1)/grid_count_x, center is at (wp_idx+0.5)/grid_count_x
                x_coord = (wp_idx + 0.5) / grid_count_x
                # y: rectangles span from ang_idx/grid_count_y*360 to (ang_idx+1)/grid_count_y*360, center is at (ang_idx+0.5)/grid_count_y*360
                angle_offset_deg = (ang_idx + 0.5) / grid_count_y * 360
                tested_x.append(x_coord)
                tested_y.append(angle_offset_deg)
            
            if len(tested_x) > 0:
                ax.scatter(tested_x, tested_y, s=20, marker='x', c='gray', alpha=1.0, linewidths=1.0, zorder=5, label='Tested (all tilts)')
        
        # Plot path if available (show all tilts with gradient)
        if self._path_states is not None and len(self._path_states) > 0:
            path_x = []
            path_y = []
            tilt_magnitudes = []
            for i, (waypoint_idx, angle_idx, tilt_roll, tilt_pitch, q_tuple, cost_so_far) in enumerate(self._path_states):
                # Center the dot in the grid rectangle
                # x: rectangles span from wp_idx/grid_count_x to (wp_idx+1)/grid_count_x, center is at (wp_idx+0.5)/grid_count_x
                x_coord = (waypoint_idx + 0.5) / grid_count_x
                # y: rectangles span from ang_idx/grid_count_y*360 to (ang_idx+1)/grid_count_y*360, center is at (ang_idx+0.5)/grid_count_y*360
                angle_offset_deg = (angle_idx + 0.5) / grid_count_y * 360
                path_x.append(x_coord)
                path_y.append(angle_offset_deg)
                # Calculate tilt magnitude in degrees
                tilt_magnitude = np.sqrt(tilt_roll**2 + tilt_pitch**2)
                tilt_magnitude_deg = np.rad2deg(tilt_magnitude)
                tilt_magnitudes.append(tilt_magnitude_deg)
            
            # Use a colormap for tilt magnitude (viridis: dark blue = 0°, bright yellow = high tilt)
            tilt_magnitudes = np.array(tilt_magnitudes)
            max_tilt = max(tilt_magnitudes) if len(tilt_magnitudes) > 0 else 1.0
            if max_tilt < 1e-6:
                max_tilt = 1.0  # Avoid division by zero
            
            # Create scatter plot with gradient colors based on tilt magnitude
            scatter = ax.scatter(path_x, path_y, c=tilt_magnitudes, s=50, marker='o', 
                               edgecolors='white', linewidths=1, zorder=10, 
                               cmap='viridis', vmin=0, vmax=max_tilt, label='Path')
            
            # Add colorbar for tilt magnitude
            # Position it after other colorbars if viz_0_tilt is enabled
            if viz_0_tilt:
                cbar_tilt = plt.colorbar(scatter, ax=ax, pad=colorbar_pad)
            else:
                cbar_tilt = plt.colorbar(scatter, ax=ax)
            cbar_tilt.set_label('Tilt Magnitude (degrees)', rotation=90, labelpad=0)
            
            ax.legend()
        self._visualization_fig = fig
        plt.show()
    
