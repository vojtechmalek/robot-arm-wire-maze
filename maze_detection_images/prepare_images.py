import os
import cv2
import numpy as np

# maze type directories
MAZE_TYPE_DIRS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "NONE",
]

DISTANCE_FROM_MARKER_TO_CENTER = 0.0375  # meters
id1_distance_from_center = (DISTANCE_FROM_MARKER_TO_CENTER, DISTANCE_FROM_MARKER_TO_CENTER)
id2_distance_from_center = (-DISTANCE_FROM_MARKER_TO_CENTER, -DISTANCE_FROM_MARKER_TO_CENTER)

def get_detection_image(img):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    corners, ids, _ = detector.detectMarkers(img)

    if ids is None:
        print("[BASE_DETECTION] No markers detected.")
        return None

    print(f"[BASE_DETECTION] Detected {len(ids)} markers")
    
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

        diag_corner_vector = (corners[0][0][2] - corners[0][0][0])
        start_corner = corners[0][0][0]

        if marker_id == 1:
            offset_from_center = np.array(id1_distance_from_center)
        elif marker_id == 2:
            offset_from_center = np.array(id2_distance_from_center)
            diag_corner_vector = -diag_corner_vector
            start_corner = corners[0][0][2]
        else:
            raise ValueError(f"Unknown marker ID: {marker_id}. Expected 1 or 2.")
        
        vector = corners[0][0][1] - corners[0][0][0]
        base_rotation = -np.arctan2(vector[1], vector[0])
                
        # Normalize angle to [-π, π] range
        base_rotation = ((base_rotation + np.pi) % (2 * np.pi)) - np.pi

        base_center = start_corner + diag_corner_vector * 1.4375


    elif len(ids) == 0:
        raise ValueError("No markers detected.")
    else:
        raise ValueError("Unknown number of markers detected.")
    
    # img_markers = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)
    angle_deg = base_rotation * 180 / np.pi
    # # cv2.putText(img_markers, f"Base Rotation: {angle_deg:.1f} deg ({base_rotation:.3f} rad)", 
    # #             tuple(base_center.astype(int) - np.array([0, 10])),
    # #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    # cv2.circle(img_markers, tuple(base_center.astype(int)), 5, (0, 255, 0), -1)
    # cv2.imshow("Detected Aruco Markers", img_markers)
    # # cv2.waitKey(0)

    # overlay white over aruco markers to hide them
    for i in range(len(ids)):
        corner_pts = corners[i][0].astype(int)
        cv2.fillConvexPoly(img, corner_pts, (255, 255, 255))

    # move aruco marker to the center of the image and rotate
    img_height, img_width = img.shape[:2]
    translation = np.array([img_width / 2, img_height / 2]) - base_center[:2]
    # M = cv2.getRotationMatrix2D((img_width / 2, img_height / 2), angle_deg, 1)
    # M = cv2.getRotationMatrix2D((0, 0), angle_deg, 1)
    M = np.eye(2, 3)
    M[:, 2] = [translation[0], translation[1]]
    img_transformed = cv2.warpAffine(
        img, M, (img_width, img_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    M_rot = cv2.getRotationMatrix2D((img_width / 2, img_height / 2), -angle_deg, 1)
    img_transformed = cv2.warpAffine(
        img_transformed, M_rot, (img_width, img_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    # # overlay white over aruco markers to hide them
    # rect1 = (img_transformed.shape[0] // 2 - 50, img_transformed.shape[1] // 2 - 50, 100, 100)
    # cv2.rectangle(img_transformed, rect1, (255, 255, 255), -1)
    # rect2 = (img_transformed.shape[0] // 2 - 150, img_transformed.shape[1] // 2 - 150, 100, 100)
    # cv2.rectangle(img_transformed, rect2, (255, 255, 255), -1)

    # convert to grayscale
    img_transformed = cv2.cvtColor(img_transformed, cv2.COLOR_BGR2GRAY)

    # apply Gaussian blur
    img_transformed = cv2.GaussianBlur(img_transformed, (5, 5), 0)

    img_height, img_width = img_transformed.shape[:2]

    # keep only the maze area
    crop_size = 500
    img_transformed = img_transformed[img_height // 2 - crop_size // 2:img_height // 2 + crop_size // 2,
                                        img_width // 2 - crop_size // 2:img_width // 2 + crop_size // 2]

    cv2.rectangle(img_transformed, (15, 15), (180, 180), (255, 255, 255), -1)
    cv2.rectangle(img_transformed, (320, 320), (320 + 165, 320 + 165), (255, 255, 255), -1)

    # quantize to 4 levels
    img_transformed = (img_transformed // 64) * 64

    # scale to 32x32
    img_transformed = cv2.resize(img_transformed, (32, 32))

    return img_transformed

if __name__ == "__main__":
    for dir in MAZE_TYPE_DIRS:
        print(f"Processing maze type directory: {dir}")
        
        full_dir = os.path.join("maze_detection_images", dir)

        # Get the list of image files in the directory
        image_files = [f for f in os.listdir(full_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        for counter, image_file in enumerate(image_files):
            image_path = os.path.join(full_dir, image_file)
            print(f"  Processing image: {image_file}")

            # Read the image
            img = cv2.imread(image_path)

            img_transformed = get_detection_image(img)

            # scale up for visualization
            # img_transformed = cv2.resize(img_transformed, (256, 256), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Transformed Image", img_transformed)
            # cv2.waitKey(0)

            cv2.imwrite(os.path.join("maze_detection_images/processed", f"{dir}_{counter:02d}.png"), img_transformed)
