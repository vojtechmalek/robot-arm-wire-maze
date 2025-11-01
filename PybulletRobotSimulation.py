
from pathlib import Path
import cv2
import yaml
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc
import time
import numpy as np
import threading
import inspect


class PybulletRobotSimulation():
    def __init__(self, maze_urdf: str = "robot/mazes/C.urdf",
                 maze_position: list[float] = [0.409188095097534, 0.07017433, 0.032],
                 maze_rotation_deg: float = -16.288773 + 180):
        p_direct = bc.BulletClient(connection_mode=p.DIRECT)

        self._bullet_client = p_direct  # store for later use in other methods
        
        # 1. Start PyBullet
        # p.connect(p.GUI, options="--verbose")
        # p_gui.connect(p.GUI)
        p_direct.setAdditionalSearchPath(pybullet_data.getDataPath())
        # p.setAdditionalSearchPath(os.path.join(os.path.dirname(__file__)))  # add path to your robot URDFs

        # 3. Load robot
        self._robot_urdf = p_direct.loadURDF("robot/crs97.urdf", useFixedBase=True, physicsClientId=p_direct._client)
        # set robot rotation if needed
        # p.resetBasePositionAndOrientation(robot_urdf, [0, 0, 0], p.getQuaternionFromEuler([0, 0, np.pi]))
        print("Base position and orientation:", p_direct.getBasePositionAndOrientation(self._robot_urdf, physicsClientId=p_direct._client))
        p_direct.resetBasePositionAndOrientation(self._robot_urdf, [0, 0, 0.1 - 0.0250 + 0.0004], p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=p_direct._client) # adjust for correct base height

        # 4. Load maze
        object_pos = maze_position
        object_rot = p.getQuaternionFromEuler([0, 0, np.deg2rad(maze_rotation_deg)])
        object_urdf = maze_urdf
        self._maze_urdf = p_direct.loadURDF(object_urdf, object_pos, object_rot)

        # Enable collision detection
        p_direct.setGravity(0, 0, 0)

        def get_first_n_revolute_joints(body, n=6):
            joints = []
            for i in range(p_direct.getNumJoints(body, physicsClientId=p_direct._client)):
                info = p_direct.getJointInfo(body, i, physicsClientId=p_direct._client)
                joint_type = info[2]
                # REVOLUTE joint type constant is 0 in pybullet
                if joint_type == p.JOINT_REVOLUTE or joint_type == 0:
                    joints.append(i)
                if len(joints) >= n:
                    break
            return joints

        # cache the first six revolute joint indices
        self._joint_indices = get_first_n_revolute_joints(self._robot_urdf, 6)
        if len(self._joint_indices) < 6:
            print(f"Warning: found only {len(self._joint_indices)} revolute joints: {self._joint_indices}")

        for i in range(p_direct.getNumJoints(self._robot_urdf)):
            # disable default motor control
            p.setJointMotorControl2(self._robot_urdf, i, p.VELOCITY_CONTROL, force=0, physicsClientId=p_direct._client)

        # p_gui.setRealTimeSimulation(0)
        # Disable collisions between robot and maze
        p_direct.setPhysicsEngineParameter(numSolverIterations=0, contactSlop=0, physicsClientId=p_direct._client)


        # set joint positions: [0, 0, -45, 0, -45, 0]
        joint_positions = [0, 0, np.deg2rad(-45), 0, np.deg2rad(-45), 0]
        self._set_q(np.array(joint_positions))

        time.sleep(1.0)  # wait a bit for everything to settle

    def _set_q(self, q_target: np.ndarray):
        q_target = q_target * np.array([1, -1, -1, 1, -1, 1])  # adjust for joint direction differences

        for jidx, pos in zip(self._joint_indices, q_target):
            # p.setJointMotorControl2(bodyIndex=self._robot_urdf, jointIndex=jidx, controlMode=p.POSITION_CONTROL, targetPosition=pos, force=500)
            p.resetJointState(self._robot_urdf, jidx, pos, physicsClientId=self._bullet_client._client)

    def get_robot_q(self):
        q = []
        for jidx in self._joint_indices:
            joint_state = p.getJointState(self._robot_urdf, jidx, physicsClientId=self._bullet_client._client)
            q.append(joint_state[0])  # position
        return np.array(q) * np.array([1, -1, -1, 1, -1, 1])  # adjust for joint direction differences

    def get_robot_last_arm_position(self):
        # get position of joint named 'robot_end_pos'
        robot_end_pos_state = p.getLinkState(self._robot_urdf, 8, physicsClientId=self._bullet_client._client)
        robot_end_pos_position = robot_end_pos_state[0]  # position
        robot_end_pos_orientation = robot_end_pos_state[1]  # orientation
        return robot_end_pos_position, robot_end_pos_orientation


    def close(self):
        # Stop the simulation thread
        p.disconnect()

    def __del__(self):
        self.close()

    def move_to_q(self, q_target: np.ndarray):
        self._set_q(q_target)

    def get_q(self) -> np.ndarray:
        return self.get_robot_q()
    
    def step_simulation(self):
        self._bullet_client.stepSimulation(physicsClientId=self._bullet_client._client)
    
    def is_colliding_with_maze(self) -> bool:
        contacts = self._bullet_client.getContactPoints(bodyA=self._robot_urdf, bodyB=self._maze_urdf, physicsClientId=self._bullet_client._client)
        return len(contacts) > 0