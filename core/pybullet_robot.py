
from pathlib import Path
import yaml
from ctu_crs.crs_robot import CRSRobot

import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc
import time
import numpy as np
import threading
import inspect
from .pose import Pose
from typing import Optional
from .state_space import StateSpace


ROBOT_BASE_POSITION = [0, 0, 0.1 - 0.0250 + 0.0004]


class PybulletRobot(CRSRobot):
    ####################################################################################################################
    # INITIALIZATION
    ####################################################################################################################
    def __init__(self, maze: str,
                 maze_pose: Pose,
                 use_gui: bool = False):
        """
        Initialize PyBullet robot simulation.
        """

        # Store configuration parameters for later use
        self._maze = maze
        self._maze_pose = maze_pose
        self._use_gui = use_gui

        # Initialize parent CRSRobot class
        # CRSRobot expects a serial device (tty_dev) and baud rate, but we pass None since this is simulation-only
        # Set parameters for CRS97 robot from provided YAML file
        yaml_path = Path(inspect.getfile(CRSRobot)).parent / "params_crs97.yaml"
        with open(yaml_path, "r") as f:
            crs_params = yaml.safe_load(f)
        super().__init__(None, 19200, **crs_params)

        # Set up PyBullet connection based on use_gui parameter
        # GUI mode shows a window with 3D visualization, DIRECT mode runs headless (faster)
        connection_mode = p.GUI if use_gui else p.DIRECT
        bullet_client = bc.BulletClient(connection_mode=connection_mode)
        self._bullet_client = bullet_client
        
        # Configure PyBullet search paths
        # This allows PyBullet to find standard URDF files (like plane.urdf)
        bullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load ground plane (provides a floor for the simulation)
        bullet_client.loadURDF("plane.urdf")

        # Load robot model from URDF file and set base position and orientation
        # (offset it as in real life the floor is above the z=0 plane)
        self._robot_urdf = bullet_client.loadURDF("robot/crs97.urdf", useFixedBase=True)
        bullet_client.resetBasePositionAndOrientation(self._robot_urdf, ROBOT_BASE_POSITION, bullet_client.getQuaternionFromEuler([0, 0, 0]))

        # Load maze model from URDF file
        # The maze is positioned based on the detected pose from ArUco markers
        maze_urdf = f"robot/mazes/maze_{maze}.urdf"
        object_pos = self._maze_pose.p.tolist()  # Convert numpy array to Python list (PyBullet requirement)
        maze_yaw = self._maze_pose.rpy[2] + np.pi/2  # Add 90° offset to match URDF coordinate frame orientation
        object_rot = bullet_client.getQuaternionFromEuler([0, 0, maze_yaw])  # Convert Euler angles to quaternion
        self._maze_urdf = bullet_client.loadURDF(maze_urdf, object_pos, object_rot, flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)

        # Configure physics simulation
        # Set gravity to zero (robot and maze are static, no need for gravity)
        bullet_client.setGravity(0, 0, 0)

        # Find the first 6 revolute (rotating) joints in the robot
        self._joints = []
        for i in range(bullet_client.getNumJoints(self._robot_urdf)):
            info = bullet_client.getJointInfo(self._robot_urdf, i)
            joint_type = info[2]  # Joint type is at index 2 in joint info
            # JOINT_REVOLUTE = 0 in PyBullet (revolving/rotating joints)
            if joint_type == p.JOINT_REVOLUTE or joint_type == 0:
                self._joints.append(i)
            if len(self._joints) >= 6:
                break
        if len(self._joints) < 6:
            print(f"Warning: found only {len(self._joints)} revolute joints")

        # Disable default motor control for all joints
        # PyBullet tries to control joints by default, but we want manual control
        # Setting force=0 disables automatic motor behavior
        for i in range(bullet_client.getNumJoints(self._robot_urdf)):
            bullet_client.setJointMotorControl2(self._robot_urdf, i, p.VELOCITY_CONTROL, force=0)

        # Configure physics engine parameters
        # numSolverIterations=0 and contactSlop=0 reduce computation (we don't need precise physics)
        # This makes the simulation faster since we're mainly using it for collision detection
        bullet_client.setPhysicsEngineParameter(numSolverIterations=0, contactSlop=0)

        # Set initial robot joint positions (home position)
        # Joint values: [0, 0, -45°, 0, -45°, 0] in degrees, converted to radians
        joint_positions = [0, 0, np.deg2rad(-45), 0, np.deg2rad(-45), 0]
        self._set_q(np.array(joint_positions))

        # Start simulation stepping thread if GUI is enabled
        # PyBullet needs stepSimulation() called regularly to update the physics
        # In GUI mode, we run this in a background thread at 240 Hz (standard simulation rate)
        self._sim_thread = None
        self._stop_event = None
        if self._use_gui:
            def simulation_loop(stop_event):
                # Continuously step the simulation until stopped or disconnected
                while True:
                    if not bullet_client.isConnected() or stop_event.is_set():
                        break
                    bullet_client.stepSimulation()
                    time.sleep(1/240)

            self._stop_event = threading.Event()  # Event to signal thread to stop
            # Create daemon thread (dies when main program exits)
            self._sim_thread = threading.Thread(target=simulation_loop, args=(self._stop_event,), daemon=True)
            self._sim_thread.start()
        
        # State space will be created on demand
        self._state_space = StateSpace(robot=self, maze=self._maze, maze_pose=self._maze_pose)

    ####################################################################################################################
    # JOINT POSITIONING
    ####################################################################################################################
    
    def _set_q(self, q_target: np.ndarray):
        """
        Set the robot joint positions to the target positions.
        """
        # Adjust for joint direction differences
        q_target = q_target * np.array([1, -1, -1, 1, -1, 1])

        for jidx, pos in zip(self._joints, q_target):
            self._bullet_client.resetJointState(self._robot_urdf, jidx, pos)
    
    ####################################################################################################################
    # SIMULATION CONTROL
    ####################################################################################################################
    
    def step_simulation(self):
        """Manually step the simulation (only needed if auto_step=False)."""
        self._bullet_client.stepSimulation()

    ####################################################################################################################
    # PHYSICS CONTROL
    ####################################################################################################################
    
    def is_colliding_with_maze(self) -> bool:
        """Check if robot is colliding with maze."""
        contacts = self._bullet_client.getContactPoints(bodyA=self._robot_urdf, bodyB=self._maze_urdf)
        return len(contacts) > 0
    
    def is_path_colliding_with_maze(self, start_q: np.ndarray, end_q: np.ndarray, num_steps: int = 10) -> bool:
        """Check if collision occurs along a path."""
        for i in range(num_steps):
            t = i / (num_steps - 1)
            q = start_q * (1 - t) + end_q * t
            self._set_q(q)
            self.step_simulation()
            if self.is_colliding_with_maze():
                return True
        return False   
    
    def get_robot_to_maze_distance(self) -> float:
        """Get the minimum distance between robot and maze."""
        points = self._bullet_client.getClosestPoints(bodyA=self._robot_urdf, bodyB=self._maze_urdf, distance=100.0)
        if len(points) == 0:
            return float('inf')
        min_distance = min([p[8] for p in points])  # contactDistance is at index 8
        return min_distance

    def get_robot_to_maze_distance_along_path(self, start_q: np.ndarray, end_q: np.ndarray, num_steps: int = 10) -> float:
        """Get the minimum distance between robot and maze along a path."""
        min_distance = float('inf')
        for i in range(num_steps):
            t = i / (num_steps - 1)
            q = start_q * (1 - t) + end_q * t
            self._set_q(q)
            self.step_simulation()
            distance = self.get_robot_to_maze_distance()
            if distance < min_distance:
                min_distance = distance
        return min_distance
    
    ####################################################################################################################
    # STATE SPACE
    ####################################################################################################################
    
    @property
    def state_space(self):
        """
        Get the StateSpace instance for this robot.
        """
        return self._state_space
    
    def calculate_state_space(self,
                              num_angle_samples: int = 8,
                              step: int = 1,
                              ):
        """
        Calculate the state space matrix
        """ 
        self.state_space.calculate(
            waypoints_path=f"waypoints/waypoints_{self._maze}.json",
            num_angle_samples=num_angle_samples,
            step=step,
            use_separate_simulation=self._use_gui
        )
    
    def visualize_state_space(self) -> None:
        """se_separate_simulation: bool = True of the state space.
        Clicking or dragging on the plot will move the robot to that configuration.
        """
        self.state_space.visualize()

    ####################################################################################################################
    # PATH PLANNING
    ####################################################################################################################

    def get_q_path(self):
        return None
        
    ####################################################################################################################
    # CLEANUP
    ####################################################################################################################

    def close(self):
        # Stop the simulation thread if it exists
        if self._sim_thread is not None and self._stop_event is not None:
            self._stop_event.set()
            self._sim_thread.join()
        if self._bullet_client is not None:
            self._bullet_client.disconnect()

