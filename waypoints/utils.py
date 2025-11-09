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

def move_back_along_x(pos, rotm, distance=0.03):
    R = np.array(rotm).reshape(3, 3)
    x_axis_world = R[:, 0]           # local X direction in world coords
    new_pos = np.array(pos) - x_axis_world * distance
    return new_pos

def move_up_along_z(pos, rotm, distance=0.1):
    R = np.array(rotm).reshape(3, 3)
    z_axis_world = R[:, 1]           # local X direction in world coords
    new_pos = np.array(pos) + z_axis_world * distance
    return new_pos


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

def points_dist_threshold_check(p1, p2, threshold=0.02):
    """
    Return True if two 3D points are closer than 'threshold' (in meters).
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    dist = np.linalg.norm(p1 - p2)
    return dist < threshold
