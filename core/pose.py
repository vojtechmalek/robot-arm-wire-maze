import numpy as np
from typing import Optional
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
        Get RPY Euler angles [roll, pitch, yaw] in radians.
        """
        # Extract RPY from rotation matrix
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
      
    def __matmul__(self, other: 'Pose') -> 'Pose':
        """
        self @ other
        """
        T_result = self.T @ other.T
        return Pose(T=T_result)

    def __rmatmul__(self, other: 'Pose') -> 'Pose':
        """
        other @ self
        """
        T_result = other.T @ self.T
        return Pose(T=T_result)

