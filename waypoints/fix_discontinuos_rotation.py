
import json
import numpy as np
import matplotlib.pyplot as plt


waypoints_json = f"waypoints/waypoints_E.json"
waypoints = json.load(open(waypoints_json, "r"))


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

previous_x_vec = None
previous_y_vec = None
previous_z_vec = None

# visualize as three vectors from pos
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(0, len(waypoints), 1):
    wp = waypoints[i]
    pos = wp["pos"]
    rpy = wp["rpy"]

    # get rotation x vector
    R = np.eye(3)
    roll, pitch, yaw = rpy

    # compute rotation matrices
    R = R @ rotzyx(roll, 0, 0)
    R = R @ rotzyx(0, pitch, 0)
    R = R @ rotzyx(0, 0, yaw)
    

    # this vector must remain the same
    x_vec = R[:, 0]

    y_vec = R[:, 1]

    # get rotation z vector
    z_vec = R[:, 2]

    # ax.quiver(*pos, *x_vec, color='r', length=0.01)
    # ax.quiver(*pos, *y_vec, color='g', length=0.01)
    # ax.quiver(*pos, *z_vec, color='b', length=0.01)


    print("Difference from previous vectors:")
    print("X:", None if previous_x_vec is None else x_vec - previous_x_vec)
    print("Y:", None if previous_y_vec is None else y_vec - previous_y_vec)
    print("Z:", None if previous_z_vec is None else z_vec - previous_z_vec)

    if previous_z_vec is not None:
        # check if z_vec flipped direction compared to previous
        if np.dot(z_vec, previous_z_vec) < 0:
                    
            ax.quiver(*pos, *x_vec, color='r', length=0.01)
            ax.quiver(*pos, *y_vec, color='g', length=0.01)
            ax.quiver(*pos, *z_vec, color='b', length=0.01)


            # flipped, correct by rotating 180 degrees around x_vec
            R_correction = rotx(np.pi)
            R = R @ R_correction

            x_vec_edit = R[:, 0]
            y_vec_edit = R[:, 1]
            z_vec_edit = R[:, 2]

            ax.quiver(*pos, *x_vec_edit, color='r', length=0.02)
            ax.quiver(*pos, *y_vec_edit, color='g', length=0.02)
            ax.quiver(*pos, *z_vec_edit, color='b', length=0.02)
            
            
            # update z_vec after correction
            z_vec = z_vec_edit

    previous_x_vec = x_vec
    previous_y_vec = y_vec
    previous_z_vec = z_vec
    
    # convert to list for JSON serialization
    wp["mat"] = R.tolist()  # for debugging

# same scale
plt.axis('equal')  # does not work for 3D
plt.show()

with open("waypoints/waypoints_new_E.json", "w") as f:
    json.dump(waypoints, f, indent=4)