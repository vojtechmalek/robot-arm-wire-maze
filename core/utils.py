import numpy as np
from typing import Dict, Any
from .pose import Pose

########################## Constants ##########################

#rpy_offset = np.array([-0.06, 0.05, 0.0])

# Distance from ring center to flange
HOOP_CENTER_DISTANCE = 0.135  # 135 mm
HOOP_OFFSET_Z = 0.013        # 13 mm above ring center

########################## Rotation matrices around xyz axes ##########################

def rotx(a):
    """ roll around x axis """
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def roty(a):
    """ pitch around y axis """
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def rotz(a):
    """ yaw around z axis """
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def rotzyx(roll, pitch, yaw):
    """ combined rotation matrix from roll, pitch, yaw angles """
    R_x = rotx(roll)
    R_y = roty(pitch)
    R_z = rotz(yaw)
    return R_z @ R_y @ R_x


########################## Coordinate transformations ##########################

def get_waypoint_pose(waypoint: Dict[str, Any], 
                     base_center_robot: np.ndarray, 
                     center_rot: float, 
                     angle_offset: float = 0.0,
                     tilt_roll_delta: float = 0.0,
                     tilt_pitch_delta: float = 0.0) -> Pose:
    """
    Calculate the pose for a waypoint given maze position and rotation, with optional tilt.
    """
    pos = np.array(waypoint["pos"])
    rpy = np.array(waypoint["rpy"])
    
    # Start at robot base origin
    pose = Pose()
    
    # Move to maze base center in robot frame
    pose = pose @ Pose(p=base_center_robot)
    
    # Rotate to align with maze orientation
    pose = pose @ Pose(rpy=np.array([0, 0, (center_rot + 90) * np.pi / 180]))
    
    # Move to waypoint position relative to maze center
    pose = pose @ Pose(p=pos)
    
    # Apply waypoint orientation with tilt modifications
    if "mat" in waypoint:
        R_wp = np.array(waypoint["mat"]).reshape((3, 3))
        pose = pose @ Pose(R=R_wp)
        # Apply tilt as additional rotations (in waypoint frame)
        if abs(tilt_roll_delta) > 1e-6:
            pose = pose @ Pose(rpy=np.array([tilt_roll_delta, 0, 0]))
        if abs(tilt_pitch_delta) > 1e-6:
            pose = pose @ Pose(rpy=np.array([0, tilt_pitch_delta, 0]))
    elif "rpy" in waypoint:
        # Apply waypoint orientation (roll, pitch, yaw separately for explicit control)
        # Add tilt deltas to roll and pitch
        pose = pose @ Pose(rpy=np.array([rpy[0] + tilt_roll_delta, 0, 0]))
        pose = pose @ Pose(rpy=np.array([0, rpy[1] + tilt_pitch_delta, 0]))
        pose = pose @ Pose(rpy=np.array([0, 0, rpy[2]]))
    
    # Adjust for robot end effector orientation
    pose = pose @ Pose(rpy=np.array([0, 0, np.pi/2]))
    pose = pose @ Pose(rpy=np.array([np.pi/2, 0, 0]))
    
    # Apply additional angle offset around Z
    pose = pose @ Pose(rpy=np.array([0, 0, angle_offset]))
    
    return pose

########################## Tilt directions ##########################

def get_tilt_directions(num_directions: int = 4) -> list[tuple[float, float]]:
    """
    Get tilt direction vectors (roll_delta, pitch_delta) evenly distributed in a circle.
    """
    directions = []
    # Generate directions evenly spaced around the circle
    # Start at angle 0 (pointing right: roll=1, pitch=0) and go counter-clockwise
    for i in range(num_directions):
        angle = 2 * np.pi * i / num_directions
        roll_dir = np.cos(angle)
        pitch_dir = np.sin(angle)
        directions.append((roll_dir, pitch_dir))
    
    return directions

########################## Robot movement ##########################

#TODO remove - not used anymore
def move_robot_by_ring_pose(robot,
                            ring_pose: Pose,
                            wait_for_motion_stop: bool = True) -> bool:
    """
    Move robot to ring-center pose.

    """

    # Convert ring-center pose to flange pose
    # Flange is offset from ring center along ring's x-axis
    flange_offset = Pose(p=np.array([HOOP_CENTER_DISTANCE, 0.0, -HOOP_OFFSET_Z]))
    flange_pose = ring_pose @ flange_offset
    
    # Solve inverse kinematics (robot.ik expects SE3 matrix)
    ik_sols = robot.ik(flange_pose.T)
    if len(ik_sols) == 0:
        print("No IK solutions found for the target position.")
        return False
    
    print(f"Found {len(ik_sols)} IK solutions.")
    
    # Sort solutions by distance to current joint state
    current_q = robot.get_q()
    ik_sols.sort(key=lambda q: np.linalg.norm(q - current_q))
    
    # Try each IK solution until one succeeds
    for q in ik_sols:
        try:
            robot.move_to_q(q)
            if wait_for_motion_stop:
                print("Waiting for motion to stop...")
                robot.wait_for_motion_stop()
            print("Moved to IK solution:", q)
            return True
        except Exception as e:
            print(f"Failed to move to IK solution {q}: {e}")
            continue
    
    print("All IK solutions failed.")
    return False