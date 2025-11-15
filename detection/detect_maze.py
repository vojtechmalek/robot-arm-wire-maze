import os
import cv2
import numpy as np

from maze_detection_images.prepare_images import get_detection_image

maze_learning_data_dir = "maze_detection_images/processed"

# images have filenames like A_00.png, B_01.png, etc. where letter indicates maze type
image_files = [f for f in os.listdir(maze_learning_data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

image_vectors = []
image_ids = []
for image_file in image_files:
    image_path = os.path.join(maze_learning_data_dir, image_file)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_vector = img.flatten() / 255.0  # normalize pixel values to [0, 1]
    image_vectors.append(img_vector)

    # Extract maze type from filename (e.g., A_00.png -> A)
    maze_type = image_file.split('_')[0]
    image_ids.append(maze_type)

image_tuples = list(zip(image_vectors, image_ids))

def get_maze_type(img, k = 5):
    detection_img = get_detection_image(img)
    if detection_img is None:
        return None
        
    img_vector = detection_img.flatten() / 255.0
    
    nearest_neighbors = sorted(
        image_tuples,
        key=lambda x: np.linalg.norm(img_vector - x[0])
    )[:k]

    type_counts = {}
    for neighbor in nearest_neighbors:
        maze_type = neighbor[1]
        if maze_type in type_counts:
            type_counts[maze_type] += 1
        else:
            type_counts[maze_type] = 1

    predicted_type = max(type_counts, key=type_counts.get)

    detected_maze = predicted_type
    confidence = type_counts[predicted_type] / k

    if detected_maze is None or detected_maze == "NONE" or confidence < 0.7:
        print(f"[MAIN] Could not detect maze type from image, what maze is this?, confidence={confidence:.2f}")
        detected_maze = input("Enter maze type (A, B, C, D, E): ").strip().upper()

    # Debug image saving removed to reduce file size
    # Uncomment below if debugging is needed:
    # debug_dir = "maze_detection_images/debug"
    # os.makedirs(debug_dir, exist_ok=True)
    # timestamp = int(np.random.rand() * 1e9)
    # cv2.imwrite(f"{debug_dir}/{detected_maze}_{timestamp}.png", detection_img)
    # cv2.imwrite(f"{debug_dir}/orig_{detected_maze}_{timestamp}.png", img)

    return detected_maze