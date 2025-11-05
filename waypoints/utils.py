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

def rotzyx(roll, pitch, yaw):
    """ combined rotation matrix from roll, pitch, yaw angles """
    R_x = rotx(roll)
    R_y = roty(pitch)
    R_z = rotz(yaw)
    return R_z @ R_y @ R_x

def rotzyx_se3(roll, pitch, yaw):
    """ combined rotation matrix in 4x4 SE3 form from roll, pitch, yaw angles """
    R = rotzyx(roll, pitch, yaw)
    T = np.eye(4)
    T[:3, :3] = R
    return T


########################## translation matrix ##########################

def trans_se3(x, y, z):
    return trans_se3_vec(np.array([x, y, z]))

def trans_se3_vec(pos: np.ndarray):
    T = np.eye(4)
    T[0, 3] = pos[0]
    T[1, 3] = pos[1]
    T[2, 3] = pos[2]
    return T

########################## Matrix operations ##########################

def inv_T(T):
    R = T[:3,:3]; p = T[:3,3]
    Ti = np.eye(4)
    Ti[:3,:3] = R.T
    Ti[:3,3] = -R.T @ p
    return Ti

########################## Transformations ##########################

def rotm_to_rpy_zyx(R):
    """
    Convert rotation matrix to ZYX Euler angles:
    yaw (Z), pitch (Y), roll (X)
    returns [roll, pitch, yaw] in radians
    """
    r11, r12, r13 = R[0,0], R[0,1], R[0,2]
    r21, r22, r23 = R[1,0], R[1,1], R[1,2]
    r31, r32, r33 = R[2,0], R[2,1], R[2,2]

    yaw   = np.arctan2(r21, r11)
    pitch = np.arcsin(-r31)
    roll  = np.arctan2(r32, r33)

    return [float(roll), float(pitch), float(yaw)]


########################## Vector ##########################

def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v * 0.0
    return v / n

def skew(v):
    x, y, z = v
    return np.array([[0, -z,  y],
                     [z,  0, -x],
                     [-y, x,  0]], dtype=float)

def rodrigues(axis, theta):
    axis = normalize(axis)
    K = skew(axis)
    return np.eye(3) + np.sin(theta)*K + (1.0 - np.cos(theta))*(K @ K)

def mats_equal(a, b, atol=1e-9, rtol=1e-9):
    # Compare two flattened rotation matrices (lists) robustly
    return np.allclose(np.asarray(a, dtype=float),
                       np.asarray(b, dtype=float),
                       atol=atol, rtol=rtol)

########################## Constants ##########################

#rpy_offset = np.array([-0.06, 0.05, 0.0])
HOOP_CENTER_DISTANCE = 0.135    # 135 mm

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

    position = ring_position_xyz
    rotation_euler = ring_rpy_zyx


    R_x = rotx(rotation_euler[0])
    R_y = roty(rotation_euler[1])
    R_z = rotz(rotation_euler[2])
    R = R_z @ R_y @ R_x

    target_pose = np.eye(4)
    target_pose[:3, :3] = R
    target_pose[:3, 3] = position

    return move_robot_by_ring_pose_matrix(robot, target_pose, wait_for_motion_stop)
    


def move_robot_by_ring_pose_matrix(robot, se3_matrix, wait_for_motion_stop = True) -> bool:
    """
    Command the robot by the RING-CENTER pose as a 4x4 SE3 matrix,
    but internally convert to the FLANGE pose for IK/motion.
    - se3_matrix: 4x4 SE3 matrix of ring center in base frame
    """

    target_pose = se3_matrix

    movement_offset_matrix = np.eye(4)
    movement_offset_matrix[:3, 3] = np.array([HOOP_CENTER_DISTANCE, 0.0, 0.0])
    target_pose = target_pose @ movement_offset_matrix


    # Solve IK & move (reuses your logic)
    ik_sols = robot.ik(target_pose)
    if len(ik_sols) == 0:
        # raise RuntimeError("No IK solutions found for the target position.")
        return False
    
    print(f"Found {len(ik_sols)} IK solutions.")

    # sort solutions by distance to current joint state
    ik_sols.sort(key=lambda q: np.linalg.norm(q - robot.get_q()))

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
 
    print("No IK solutions found for the target position.")
    return False

################################################################################
# Find aruco marker center functions
################################################################################