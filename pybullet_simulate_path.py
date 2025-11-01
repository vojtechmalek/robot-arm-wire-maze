import pybullet as p
import pybullet_data
import time
import os
import numpy as np
from ctu_crs import CRS97
import threading
import utils

# 1. Start PyBullet
p.connect(p.GUI, options="--verbose")
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setAdditionalSearchPath(os.path.join(os.path.dirname(__file__)))  # add path to your robot URDFs

# 2. Load plane (optional)
plane = p.loadURDF("plane.urdf")

# 3. Load robot
# robot = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

# load custom robot URDF
robot_urdf = p.loadURDF("robot/crs97.urdf", useFixedBase=True)
# set robot rotation if needed
# p.resetBasePositionAndOrientation(robot_urdf, [0, 0, 0], p.getQuaternionFromEuler([0, 0, np.pi]))
print("Base position and orientation:", p.getBasePositionAndOrientation(robot_urdf))
p.resetBasePositionAndOrientation(robot_urdf, [0, 0, 0.1 - 0.0250 + 0.0004], p.getQuaternionFromEuler([0, 0, 0])) # adjust for correct base height

# 4. Load an object (e.g., a cube)
object_pos = [0.6, 0, 0.1]
object_urdf = "cube_small.urdf"
obj = p.loadURDF(object_urdf, object_pos)

# Enable collision detection
p.setGravity(0, 0, 0)

def get_first_n_revolute_joints(body, n=6):
    joints = []
    for i in range(p.getNumJoints(body)):
        info = p.getJointInfo(body, i)
        joint_type = info[2]
        # REVOLUTE joint type constant is 0 in pybullet
        if joint_type == p.JOINT_REVOLUTE or joint_type == 0:
            joints.append(i)
        if len(joints) >= n:
            break
    return joints

# cache the first six revolute joint indices
joint_indices = get_first_n_revolute_joints(robot_urdf, 6)
if len(joint_indices) < 6:
    print(f"Warning: found only {len(joint_indices)} revolute joints: {joint_indices}")

# set joint positions: [0, 0, -45, 0, -45, 0]
joint_positions = [0, 0, np.deg2rad(-45), 0, np.deg2rad(-45), 0]
for jidx, pos in zip(joint_indices, joint_positions):
    p.setJointMotorControl2(bodyIndex=robot_urdf, jointIndex=jidx, controlMode=p.POSITION_CONTROL, targetPosition=pos, force=500)

def move_robot_to_q(q_target):
    q_target = q_target * np.array([1, -1, -1, 1, -1, -1])  # adjust for joint direction differences
    for jidx, pos in zip(joint_indices, q_target):
        p.setJointMotorControl2(bodyIndex=robot_urdf, jointIndex=jidx, controlMode=p.POSITION_CONTROL, targetPosition=pos, force=500)

def get_robot_q():
    q = []
    for jidx in joint_indices:
        joint_state = p.getJointState(robot_urdf, jidx)
        q.append(joint_state[0])  # position
    return np.array(q)

def get_robot_last_arm_position():
    # get position of joint named 'robot_end_pos'
    robot_end_pos_state = p.getLinkState(robot_urdf, 8)
    robot_end_pos_position = robot_end_pos_state[0]  # position
    robot_end_pos_orientation = robot_end_pos_state[1]  # orientation
    return robot_end_pos_position, robot_end_pos_orientation

def simulation_loop(stop_event):
    while True:
        if p.isConnected() == 0 or stop_event.is_set():
            break

        p.stepSimulation()

        # 6. Check for collision
        contacts = p.getContactPoints(bodyA=robot_urdf, bodyB=obj)
        if contacts:
            print("=== COLLISION DETECTED ===")
            for c in contacts:
                print("Link:", c.linkIndexA,
                        "Object link:", c.linkIndexB,
                        "Contact distance:", c.contactDistance,
                        "Normal force:", c.normalForce)

        time.sleep(1/240)

stop_event = threading.Event()
sim_thread = threading.Thread(target=simulation_loop, args=(stop_event,), daemon=True)
sim_thread.start()


##### OUR CODE HERE TO CONTROL THE ROBOT #####


robot = CRS97(tty_dev=None)  # set argument tty_dev=None if you are not connected to robot,









#################################################


# keep main thread alive while GUI is open; allow Ctrl+C to stop
try:
    while p.isConnected():
        time.sleep(0.5)
except KeyboardInterrupt:
    stop_event.set()
    sim_thread.join()
