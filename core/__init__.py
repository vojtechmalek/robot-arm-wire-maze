"""
Core module containing essential robot functionality and utilities.

This module provides:
- base_detection: ArUco marker detection and base position estimation
- utils: Mathematical utilities for transformations (SE3, RPY, quaternions)
- pose: Pose class for representing position and rotation
- robot_movement: High-level robot movement functions using SE3 internally
- path_planning: Path planning utilities for waypoint navigation
- PybulletRobot: PyBullet-based robot simulation with GUI and path planning
"""

from . import base_detection
from . import utils
from . import pose
from .pybullet_robot import PybulletRobot
from .pose import Pose

__all__ = ['base_detection', 'utils', 'pose', 'Pose', 'PybulletRobot']

