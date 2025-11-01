import pybullet as p
import pybullet_data
import time
import os
import numpy as np

# 1. Start PyBullet
p.connect(p.GUI, options="--verbose")
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setAdditionalSearchPath(os.path.join(os.path.dirname(__file__)))  # add path to your robot URDFs

# 2. Load plane (optional)
plane = p.loadURDF("plane.urdf")

# 3. Load robot
# robot = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

# load custom robot URDF
robot = p.loadURDF("robot/crs97.urdf", useFixedBase=True)
# set robot rotation if needed
p.resetBasePositionAndOrientation(robot, [0, 0, 0], p.getQuaternionFromEuler([0, 0, np.pi]))

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
joint_indices = get_first_n_revolute_joints(robot, 6)
if len(joint_indices) < 6:
    print(f"Warning: found only {len(joint_indices)} revolute joints: {joint_indices}")

# set joint positions: [0, 0, -45, 0, -45, 0]
joint_positions = [0, 0, np.deg2rad(-45), 0, np.deg2rad(-45), 0]
for jidx, pos in zip(joint_indices, joint_positions):
    p.setJointMotorControl2(bodyIndex=robot, jointIndex=jidx, controlMode=p.POSITION_CONTROL, targetPosition=pos, force=500)

# 5. Simulation loop
while True:
    if (p.isConnected() == 0):
        break

    p.stepSimulation()

    # 6. Check for collision
    contacts = p.getContactPoints(bodyA=robot, bodyB=obj)

    if contacts:
        print("=== COLLISION DETECTED ===")
        for c in contacts:
            print("Link:", c.linkIndexA,
                  "Object link:", c.linkIndexB,
                  "Contact distance:", c.contactDistance,
                  "Normal force:", c.normalForce)

    time.sleep(1/240)
