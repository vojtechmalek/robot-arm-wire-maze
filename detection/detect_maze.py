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

def get_maze_type(img, k = 3):
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
    return predicted_type