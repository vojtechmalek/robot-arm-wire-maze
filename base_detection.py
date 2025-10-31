import cv2
import numpy as np

default_angle_offset = -np.pi / 2  # -90 degrees in radians
diagonal_to_base_rotation_offset = -np.pi / 4  # -45 degrees in radians
distance_from_marker_to_center = 0.0375  # meters
id1_distance_from_center = (-distance_from_marker_to_center, distance_from_marker_to_center)
id2_distance_from_center = (distance_from_marker_to_center, -distance_from_marker_to_center)

H = np.load("homography.npy")

def detect_base_position(img, display=False):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    corners, ids, rejectedImgPoints = detector.detectMarkers(img)

    print(f"Detected {len(ids)} markers")
    
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
            diagonal_angle = -np.arctan2(vector[1], vector[0])
            base_rotation = diagonal_angle + diagonal_to_base_rotation_offset
            
            # Normalize angle to [-π, π] range
            base_rotation = ((base_rotation + np.pi) % (2 * np.pi)) - np.pi

            uv = np.array([base_center[0], base_center[1], 1.0])
            xy = H @ uv
            xy /= xy[2]
            base_center_robot = np.array([xy[0], xy[1], 0.054])
        
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
            # atan2 returns angle in radians
            angle = np.arctan2(vector[1], vector[0])
            angle = -angle + default_angle_offset
            
            # Normalize angle to [-π, π] range
            angle = ((angle + np.pi) % (2 * np.pi)) - np.pi
            
            # Rotate the offset vector by the detected angle
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated_offset = rotation_matrix @ offset_from_center
            
            # Apply rotated offset to marker center to get base center
            base_rotation = angle

            uv = np.array([base_center[0], base_center[1], 1.0])
            xy = H @ uv
            xy /= xy[2]
            print(f"arucu marker position: {xy}")
            base_center_robot = np.array([xy[0] + rotated_offset[0], xy[1] + rotated_offset[1], 0.054])
            print(f"base center robot: {base_center_robot}")

        elif len(ids) == 0:
            raise ValueError("No markers detected.")
        else:
            raise ValueError("Unknown number of markers detected.")
        
        if display:
            img_markers = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)
            # Convert to degrees for display (optional - can change to show radians)
            angle_deg = base_rotation * 180 / np.pi
            cv2.putText(img_markers, f"Rotation: {angle_deg:.1f} deg ({base_rotation:.3f} rad)", 
                        tuple(base_center.astype(int) - np.array([0, 10])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.circle(img_markers, tuple(base_center.astype(int)), 5, (0, 255, 0), -1)
            cv2.imshow("Detected Aruco Markers", img_markers)
            cv2.waitKey(0)
    
    return base_center_robot, base_rotation
