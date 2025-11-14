import cv2
import numpy as np
from pathlib import Path

BASE_HEIGHT = 0.037  # distance from ground to the base center in meters
DISTANCE_FROM_MARKER_TO_CENTER = 0.0375  # distance from the center of aruco marker to the center of the base
id1_distance_from_center = (DISTANCE_FROM_MARKER_TO_CENTER, DISTANCE_FROM_MARKER_TO_CENTER)
id2_distance_from_center = (-DISTANCE_FROM_MARKER_TO_CENTER, -DISTANCE_FROM_MARKER_TO_CENTER)

def detect_base_position(img, display=False):
    """
    Detect the base position and rotation from the image (from aruco markers)
    """
    
    # Load homography matrix
    homography_path = Path(__file__).parent.parent / "homography" / "homography.npy"
    H = np.load(homography_path)
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50) # 4x4 aruco dictionary with 50 markers
    detector = cv2.aruco.ArucoDetector(aruco_dict) # detector for aruco markers

    corners, ids, _ = detector.detectMarkers(img) # detect markers in the image

    if ids is None: # If we didnt detect any markers
        print("[BASE_DETECTION] No markers detected.")
        return None

    print(f"[BASE_DETECTION] Detected {len(ids)} markers")
    
    if len(ids) == 2:
        # We can see both markers, average their centers to get the base center and calculate the rotation based on their positions
        # Find which marker is id1 and which is id2
        ids_flat = ids.flatten()
        idx1 = np.where(ids_flat == 1)[0]
        idx2 = np.where(ids_flat == 2)[0]
        
        if len(idx1) == 0 or len(idx2) == 0: # If the ids are not what we expected
            raise ValueError("Could not find both marker id1 and id2")
        
        # Calculate the center points of the two markers (average the corners of the markers)
        # id1 is bottom left, id2 is top right
        center1 = np.mean(corners[idx1[0]][0], axis=0)  # id1 - bottom left
        center2 = np.mean(corners[idx2[0]][0], axis=0)  # id2 - top right

        # Calculate the position of the base center
        base_center = (center1 + center2) / 2

        # Calculate rotation from id1 (bottom left) to id2 (top right)
        # Vector from id1 to id2 points along the diagonal
        vector = center2 - center1

        # Calculate angle in radians
        angle = -np.arctan2(vector[1], vector[0])

        # Offset the angle to use robot coordinate system
        base_rotation = angle + np.pi / 4
        
        # Normalize angle to [-pi, pi] range
        base_rotation = ((base_rotation + np.pi) % (2 * np.pi)) - np.pi

        # Convert the base center to the robot coordinate system
        uv = np.array([base_center[0], base_center[1], 1.0])
        xy = H @ uv
        xy /= xy[2]

        # Add the z axis
        base_center_robot = np.array([xy[0], xy[1], BASE_HEIGHT])
    
    elif len(ids) == 1:
        # Determine which marker ID was detected and use corresponding offset
        marker_id = ids.flatten()[0]
        if marker_id == 1:
            offset_from_center = np.array(id1_distance_from_center)
        elif marker_id == 2:
            offset_from_center = np.array(id2_distance_from_center)
        else:
            raise ValueError(f"Unknown marker ID: {marker_id}. Expected 1 or 2.")
        
        aruco_center = np.mean(corners[0][0], axis=0)
        vector = corners[0][0][1] - corners[0][0][0] # vector from top-left to top-right corner
        base_rotation = -np.arctan2(vector[1], vector[0])
                
        # Normalize angle to [-pi, pi] range
        base_rotation = ((base_rotation + np.pi) % (2 * np.pi)) - np.pi

        # Rotate the offset vector by the detected angle
        cos_a = np.cos(base_rotation)
        sin_a = np.sin(base_rotation)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_offset = rotation_matrix @ offset_from_center

        # Convert the aruco center to the robot coordinate system
        uv = np.array([aruco_center[0], aruco_center[1], 1.0])
        xy = H @ uv
        xy /= xy[2]

        # Add the offset to the aruco center
        offset_robot = np.array([xy[0] + rotated_offset[0], xy[1] + rotated_offset[1]])

        # Add the z axis
        base_center_robot = np.array([offset_robot[0], offset_robot[1], BASE_HEIGHT])

    elif len(ids) == 0:
        raise ValueError("No markers detected.")
    else:
        raise ValueError("Unknown number of markers detected.")
    
    if display: # display the detected markers and the base center on the image
        img_markers = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids) # draw the detected markers on the image
        angle_deg = base_rotation * 180 / np.pi # convert the rotation to degrees
        cv2.putText(img_markers, f"Base Rotation: {angle_deg:.1f} deg ({base_rotation:.3f} rad)", 
                    tuple(base_center.astype(int) - np.array([0, 10])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(img_markers, tuple(base_center.astype(int)), 5, (0, 255, 0), -1) # add the base center to the image
        cv2.imshow("Detected Aruco Markers", img_markers)
        cv2.waitKey(0)
    
    print(f"[BASE_DETECTION] Base center in robot frame: {base_center_robot} with rotation {base_rotation:.3f} rad")
    
    # return our position and rotation as Pose object with the rotation around the z axis
    from .pose import Pose
    return Pose(p=base_center_robot, rpy=np.array([0, 0, base_rotation]))
