import numpy as np
from typing import Optional, Union
from . import utils


class Pose:
    """
    Represents a 6D pose (position + rotation) in 3D space.
    """
    
    def __init__(self, 
                 R: Optional[np.ndarray] = None,  # 3x3 rotation matrix
                 p: Optional[np.ndarray] = None,  # 3D position vector
                 T: Optional[np.ndarray] = None,  # 4x4 SE3 matrix
                 rpy: Optional[np.ndarray] = None,  # [roll, pitch, yaw] in radians (ZYX)
                 quaternion: Optional[np.ndarray] = None,  # [w, x, y, z] quaternion
                 ):
        # Initialize to default values
        self._R = np.eye(3)
        self._p = np.zeros(3)
        
        # Handle each argument independently (later arguments override earlier ones)
        if T is not None:
            self._R = T[:3, :3].copy()
            self._p = T[:3, 3].copy()
        if R is not None:
            self._R = np.array(R, dtype=float)
        if rpy is not None:
            self._R = utils.rotzyx(rpy[0], rpy[1], rpy[2])
        if quaternion is not None:
            w, x, y, z = quaternion
            self._R = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
            ], dtype=float)
        if p is not None:
            self._p = np.array(p, dtype=float)
        
        # Validate dimensions
        if self._R.shape != (3, 3):
            raise ValueError(f"Rotation matrix must be 3x3, got {self._R.shape}")
        if self._p.shape != (3,):
            raise ValueError(f"Position must be 3D, got {self._p.shape}")
    
    @property
    def R(self) -> np.ndarray:
        """Get 3x3 rotation matrix."""
        return self._R.copy()
    
    @property
    def p(self) -> np.ndarray:
        """Get 3D position vector."""
        return self._p.copy()
    
    @property
    def position(self) -> np.ndarray:
        """Get 3D position vector (alias for p)."""
        return self._p.copy()
    
    @property
    def rotation(self) -> np.ndarray:
        """Get 3x3 rotation matrix (alias for R)."""
        return self._R.copy()
    
    @property
    def T(self) -> np.ndarray:
        """Get 4x4 SE3 transformation matrix."""
        T = np.eye(4)
        T[:3, :3] = self._R
        T[:3, 3] = self._p
        return T
    
    @property
    def se3(self) -> np.ndarray:
        """Get 4x4 SE3 transformation matrix (alias for T)."""
        return self.T
    
    @property
    def rpy(self) -> np.ndarray:
        """
        Get RPY Euler angles [roll, pitch, yaw] in radians (ZYX convention).
        """
        # Extract RPY from rotation matrix (ZYX convention)
        R = self._R
        # Handle singularities at pitch = ±π/2
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        
        if not singular:
            roll = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw = 0
        
        return np.array([roll, pitch, yaw])
    
    @property
    def quaternion(self) -> np.ndarray:
        """
        Get quaternion [w, x, y, z] (PyBullet format).
        """
        R = self._R
        
        # Convert rotation matrix to quaternion
        trace = np.trace(R)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        
        return np.array([w, x, y, z])
    
    def __matmul__(self, other: Union['Pose', np.ndarray]) -> Union['Pose', np.ndarray]:
        """
        Compose poses: self @ other
        """
        if isinstance(other, Pose):
            # Compose two poses
            T_result = self.T @ other.T
            return Pose(T=T_result)
        elif isinstance(other, np.ndarray):
            # Transform a point/vector
            if other.shape == (3,):
                # Point
                return self._R @ other + self._p
            elif other.shape == (4,):
                # Homogeneous point
                return self.T @ other
            else:
                raise ValueError(f"Cannot transform array of shape {other.shape}")
    
    def __rmatmul__(self, other: np.ndarray) -> 'Pose':
        """
        Left multiplication: other @ self (for SE3 matrix composition)
        """
        if isinstance(other, np.ndarray) and other.shape == (4, 4):
            T_result = other @ self.T
            return Pose(T=T_result)
        return NotImplemented
    
    def inverse(self) -> 'Pose':
        """
        Get the inverse pose.
        
        """
        # Inverse: R_inv = R^T, p_inv = -R^T @ p
        R_inv = self._R.T
        p_inv = -R_inv @ self._p
        return Pose(R=R_inv, p=p_inv)
    
    def __repr__(self) -> str:
        """String representation of the pose."""
        return f"Pose(p={self._p}, R=[[{self._R[0,0]:.3f}, ...]])"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        rpy = self.rpy
        return f"Pose(position={self._p}, rpy=[{rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f}])"



