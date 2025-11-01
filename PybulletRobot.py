
from pathlib import Path
import cv2
import yaml
from ctu_crs.crs_robot import CRSRobot

import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc
import time
import numpy as np
import threading
import inspect

import base_detection


class PybulletRobot(CRSRobot):
    def __init__(self, tty_dev: str | None = "/dev/mars", baudrate: int = 19200,
                 maze_urdf: str = "robot/mazes/C.urdf",
                 aruco_image_path: str = "aruco_images/aruco_6.png"):
        tty_dev = None  # force None, we don't use real robot connection

        self.aruco_image_path = aruco_image_path
        
        yaml_path = Path(inspect.getfile(CRSRobot)).parent / "params_crs97.yaml"
        with open(yaml_path, "r") as f:
            crs_params = yaml.safe_load(f)
        super().__init__(tty_dev, baudrate, **crs_params)


        p_gui = bc.BulletClient(connection_mode=p.GUI)
        
        # 1. Start PyBullet
        # p.connect(p.GUI, options="--verbose")
        # p_gui.connect(p.GUI)
        p_gui.setAdditionalSearchPath(pybullet_data.getDataPath())
        # p.setAdditionalSearchPath(os.path.join(os.path.dirname(__file__)))  # add path to your robot URDFs

        # 2. Load plane (optional)
        plane = p_gui.loadURDF("plane.urdf")

        # 3. Load robot
        self._robot_urdf = p_gui.loadURDF("robot/crs97.urdf", useFixedBase=True)
        # set robot rotation if needed
        # p.resetBasePositionAndOrientation(robot_urdf, [0, 0, 0], p.getQuaternionFromEuler([0, 0, np.pi]))
        print("Base position and orientation:", p.getBasePositionAndOrientation(self._robot_urdf))
        p_gui.resetBasePositionAndOrientation(self._robot_urdf, [0, 0, 0.1 - 0.0250 + 0.0004], p.getQuaternionFromEuler([0, 0, 0])) # adjust for correct base height


        aruco_img = cv2.imread(aruco_image_path)
        aruco_marker_position, aruco_marker_rotation = base_detection.detect_base_position(aruco_img, False)

        base_world_position = [aruco_marker_position[0], aruco_marker_position[1], 0.032]
        # 4. Load maze
        object_pos = base_world_position
        object_rot = p.getQuaternionFromEuler([0, 0, aruco_marker_rotation + np.pi/2])
        object_urdf = maze_urdf
        self._maze_urdf = p_gui.loadURDF(object_urdf, object_pos, object_rot, flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)

        # Enable collision detection
        p_gui.setGravity(0, 0, 0)

        def get_first_n_revolute_joints(body, n=6):
            joints = []
            for i in range(p_gui.getNumJoints(body)):
                info = p_gui.getJointInfo(body, i)
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

        for i in range(p_gui.getNumJoints(self._robot_urdf)):
            # disable default motor control
            p.setJointMotorControl2(self._robot_urdf, i, p.VELOCITY_CONTROL, force=0)

        # p_gui.setRealTimeSimulation(0)
        # Disable collisions between robot and maze
        p_gui.setPhysicsEngineParameter(numSolverIterations=0, contactSlop=0)


        # set joint positions: [0, 0, -45, 0, -45, 0]
        joint_positions = [0, 0, np.deg2rad(-45), 0, np.deg2rad(-45), 0]
        self._set_q(np.array(joint_positions))

        def simulation_loop(stop_event):
            while True:
                if p_gui.isConnected() == 0 or stop_event.is_set():
                    break

                p_gui.stepSimulation()

                # contacts = p_gui.getContactPoints(bodyA=self._robot_urdf, bodyB=self._maze_urdf)
                # if contacts:
                #     print("=== COLLISION DETECTED ===")
                #     for c in contacts:
                #         linkA_name = p_gui.getJointInfo(self._robot_urdf, c[3])[12].decode('utf-8') if c[3] != -1 else "base"
                #         linkB_name = p_gui.getJointInfo(self._maze_urdf, c[4])[12].decode('utf-8') if c[4] != -1 else "base"
                #         print("LinkA:", linkA_name,
                #             "LinkB:", linkB_name)
                #         print("LinkIndexA:", c[3],
                #             "LinkIndexB:", c[4],
                #             "Contact distance:", c[8],
                #             "Normal force:", c[9])

                time.sleep(1/240)

        self._stop_event = threading.Event()
        self._sim_thread = threading.Thread(target=simulation_loop, args=(self._stop_event,), daemon=True)
        self._sim_thread.start()

        time.sleep(1.0)  # wait a bit for everything to settle

    def _set_q(self, q_target: np.ndarray):
        q_target = q_target * np.array([1, -1, -1, 1, -1, 1])  # adjust for joint direction differences

        for jidx, pos in zip(self._joint_indices, q_target):
            # p.setJointMotorControl2(bodyIndex=self._robot_urdf, jointIndex=jidx, controlMode=p.POSITION_CONTROL, targetPosition=pos, force=500)
            p.resetJointState(self._robot_urdf, jidx, pos)

    def _draw_frame(self, se3_matrix, length=0.1, width=0.005):
        origin = se3_matrix[:3, 3]
        x_axis = se3_matrix[:3, 0]
        y_axis = se3_matrix[:3, 1]
        z_axis = se3_matrix[:3, 2]

        p.addUserDebugLine(origin, origin + length * x_axis, [1, 0, 0], width * 100, 2.0)
        p.addUserDebugLine(origin, origin + length * y_axis, [0, 1, 0], width * 100, 2.0)
        p.addUserDebugLine(origin, origin + length * z_axis, [0, 0, 1], width * 100, 2.0)

    def move_robot_to_q(self, q_target):
        prev_q = self.get_robot_q()

        q_norm = np.linalg.norm(q_target - prev_q)
        move_time = q_norm * 2  # at least 0.5s, more for larger moves

        step_count = 50
        for i in range(1, step_count + 1):
            t = i / step_count
            q_interp = prev_q + t * (q_target - prev_q)
            self._set_q(q_interp)
            time.sleep(move_time / step_count)

        time.sleep(0.2)  # wait a bit for the motion to start

    def get_robot_q(self):
        q = []
        for jidx in self._joint_indices:
            joint_state = p.getJointState(self._robot_urdf, jidx)
            q.append(joint_state[0])  # position
        return np.array(q) * np.array([1, -1, -1, 1, -1, 1])  # adjust for joint direction differences

    def get_robot_last_arm_position(self):
        # get position of joint named 'robot_end_pos'
        robot_end_pos_state = p.getLinkState(self._robot_urdf, 8)
        robot_end_pos_position = robot_end_pos_state[0]  # position
        robot_end_pos_orientation = robot_end_pos_state[1]  # orientation
        return robot_end_pos_position, robot_end_pos_orientation


    def close(self):
        # Stop the simulation thread
        if self._sim_thread is not None:
            self._stop_event.set()
            self._sim_thread.join()
        p.disconnect()

    def move_to_q(self, q_target: np.ndarray):
        assert self.in_limits(q_target), "Joint limits violated."

        self.move_robot_to_q(q_target)

    def initialize(self, home = True):
        pass

    def wait_for_motion_stop(self):
        time.sleep(1.0)  # simple wait, improve if needed

    def grab_image(self) -> np.ndarray:
        # read image file
        img_path = self.aruco_image_path
        img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

        # simple placeholder implementation
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(640, 480)
        img = np.array(rgbImg).reshape((height, width, 4))[:, :, :3]  # remove alpha channel
        return img
    
    def get_q(self) -> np.ndarray:
        return self.get_robot_q()