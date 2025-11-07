import cv2
import numpy as np
from pathlib import Path

BASE_HEIGHT = 0.037  # meters
DISTANCE_FROM_MARKER_TO_CENTER = 0.0375  # meters
id1_distance_from_center = (DISTANCE_FROM_MARKER_TO_CENTER, DISTANCE_FROM_MARKER_TO_CENTER)
id2_distance_from_center = (-DISTANCE_FROM_MARKER_TO_CENTER, -DISTANCE_FROM_MARKER_TO_CENTER)

def detect_base_position(img, display=False):
    # Load homography matrix
    homography_path = Path(__file__).parent.parent / "homography" / "homography.npy"
    H = np.load(homography_path)
    
    print(f"[BASE_DETECTION] Homography matrix loaded from {homography_path}")
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    corners, ids, _ = detector.detectMarkers(img)

    print(f"[BASE_DETECTION] Detected {len(ids)} markers")
    
    if ids is not None:
        if len(ids) == 2:
            # Find which marker is id1 and which is id2
            ids_flat = ids.flatten()
            idx1 = np.where(ids_flat == 1)[0]
            idx2 = np.where(ids_flat == 2)[0]
            
            if len(idx1) == 0 or len(idx2) == 0:
                raise ValueError("Could not find both marker id1 and id2")
            
            # Calculate the center points of the two markers
            # id1 is bottom left, id2 is top right
            center1 = np.mean(corners[idx1[0]][0], axis=0)  # id1 - bottom left
            center2 = np.mean(corners[idx2[0]][0], axis=0)  # id2 - top right

            # Calculate base center
            base_center = (center1 + center2) / 2

            # Calculate rotation from id1 (bottom left) to id2 (top right)
            # Vector from id1 to id2 points along the diagonal
            vector = center2 - center1

            # Calculate angle in radians (atan2 returns angle in radians)
            angle = -np.arctan2(vector[1], vector[0])

            # offset the angle to use robot coordinate system
            base_rotation = angle + np.pi / 4
            
            # Normalize angle to [-π, π] range
            base_rotation = ((base_rotation + np.pi) % (2 * np.pi)) - np.pi

            uv = np.array([base_center[0], base_center[1], 1.0])
            xy = H @ uv
            xy /= xy[2]

            base_center_robot = np.array([xy[0], xy[1], BASE_HEIGHT])
            
            # Calculate angles for each marker (for display)
            marker_angles = {}
            for i, marker_id in enumerate(ids_flat):
                if marker_id == 1:
                    idx = idx1[0]
                elif marker_id == 2:
                    idx = idx2[0]
                else:
                    continue
                # Calculate marker orientation: vector from top-left to top-right corner
                marker_vector = corners[idx][0][1] - corners[idx][0][0]
                marker_angle = np.arctan2(marker_vector[1], marker_vector[0])
                marker_angles[marker_id] = marker_angle
        
        elif len(ids) == 1:
            # Determine which marker ID was detected and use corresponding offset
            marker_id = ids.flatten()[0]
            if marker_id == 1:
                offset_from_center = np.array(id1_distance_from_center)
            elif marker_id == 2:
                offset_from_center = np.array(id2_distance_from_center)
            else:
                raise ValueError(f"Unknown marker ID: {marker_id}. Expected 1 or 2.")
            
            base_center = np.mean(corners[0][0], axis=0)
            vector = corners[0][0][1] - corners[0][0][0]
            base_rotation = -np.arctan2(vector[1], vector[0])
                 
            # Normalize angle to [-π, π] range
            base_rotation = ((base_rotation + np.pi) % (2 * np.pi)) - np.pi

            # Rotate the offset vector by the detected angle
            cos_a = np.cos(base_rotation)
            sin_a = np.sin(base_rotation)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated_offset = rotation_matrix @ offset_from_center

            uv = np.array([base_center[0], base_center[1], 1.0])
            xy = H @ uv
            xy /= xy[2]
            base_center_robot = np.array([xy[0] + rotated_offset[0], xy[1] + rotated_offset[1], BASE_HEIGHT])

        elif len(ids) == 0:
            raise ValueError("No markers detected.")
        else:
            raise ValueError("Unknown number of markers detected.")
        
        if display:
            img_markers = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)
            angle_deg = base_rotation * 180 / np.pi
            cv2.putText(img_markers, f"Base Rotation: {angle_deg:.1f} deg ({base_rotation:.3f} rad)", 
                        tuple(base_center.astype(int) - np.array([0, 10])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.circle(img_markers, tuple(base_center.astype(int)), 5, (0, 255, 0), -1)
            cv2.imshow("Detected Aruco Markers", img_markers)
            cv2.waitKey(0)
    
    print(f"[BASE_DETECTION] Base center in robot frame: {base_center_robot} with rotation {base_rotation:.3f} rad")
    # Return Pose instance with position (xyz) and rotation (yaw around Z axis)
    from .pose import Pose
    return Pose(p=base_center_robot, rpy=np.array([0, 0, base_rotation]))
