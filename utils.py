import numpy as np

################################################################################
# Move functions
################################################################################

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


########################## Matrix operations ##########################
def rpy_to_R_zyx(roll, pitch, yaw):
    """ convert roll, pitch, yaw to rotation matrix """
    return rotz(yaw) @ roty(pitch) @ rotx(roll)

def make_T(R, p):
    """ create transformation matrix from rotation matrix and translation vector """
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = p; return T

def inv_T(T):
    R = T[:3,:3]; p = T[:3,3]
    Ti = np.eye(4)
    Ti[:3,:3] = R.T
    Ti[:3,3] = -R.T @ p
    return Ti

########################## Constants ##########################

rpy_offset = np.array([-0.06, 0.05, 0.0])
HOOP_CENTER_DISTANCE = 0.135    # 135 mm

# Convert the offset to a rotation matrix
R_fr = rpy_to_R_zyx(rpy_offset[0], rpy_offset[1], rpy_offset[2])

# Create a translation vector from the flange origin to the hoop center
p_fr = R_fr @ np.array([HOOP_CENTER_DISTANCE, 0.0, 0.0])

# Combine rotation (R_fr) and translation (p_fr) into a single 4x4 homogeneous matrix
T_flange_to_ring = make_T(R_fr, p_fr)


########################## Move function ##########################

def move_robot_by_ring_pose(robot,
                            ring_position_xyz: np.ndarray,
                            ring_rpy_zyx: np.ndarray,
                            wait_for_motion_stop = True
                            ) -> bool:
    """
    Command the robot by the RING-CENTER pose (XYZ + RPY),
    but internally convert to the FLANGE pose for IK/motion.
    - ring_position_xyz: [x, y, z] of ring center in base frame (meters)
    - ring_rpy_zyx: [roll, pitch, yaw] of ring center (radians)
    """

    # Build desired ring pose in base frame
    R_base_ring = rpy_to_R_zyx(ring_rpy_zyx[0], ring_rpy_zyx[1], ring_rpy_zyx[2])
    T_base_ring = make_T(R_base_ring, np.asarray(ring_position_xyz).reshape(3,))

    # Convert to flange pose expected by robot
    T_base_flange = T_base_ring @ inv_T(T_flange_to_ring)

    # Solve IK & move (reuses your logic)
    ik_sols = robot.ik(T_base_flange)
    if len(ik_sols) == 0:
        raise RuntimeError("No IK solutions found for the target position.")
    
    print(f"Found {len(ik_sols)} IK solutions.")

    # sort solutions by distance to current joint state
    ik_sols.sort(key=lambda q: np.linalg.norm(q - robot.get_q()))

    for q in ik_sols:
        try:
            robot.move_to_q(q)
            robot.wait_for_motion_stop()
            print("Moved to IK solution:", q)
            return True
        except Exception as e:
            print(f"Failed to move to IK solution {q}: {e}")
            return False
        
    print("No IK solutions found for the target position.")
    return False

################################################################################
# Find aruco marker center functions
################################################################################