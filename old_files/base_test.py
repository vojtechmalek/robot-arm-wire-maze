import cv2
from pathlib import Path
from base_detection import detect_base_position

# Get the directory containing aruco images
aruco_images_dir = Path("aruco_images")

# Iterate over all images in the directory, sorted numerically
def extract_number(path):
    """Extract numeric part from filename like 'aruco_1.png' -> 1"""
    name = path.stem  # Get filename without extension
    try:
        # Extract number after underscore
        return int(name.split('_')[-1])
    except (ValueError, IndexError):
        return 0

image_files = sorted(aruco_images_dir.glob("*.png"), key=extract_number)
for image_path in image_files:
    print(f"\nProcessing: {image_path.name}")
    img = cv2.imread(str(image_path))
    if img is not None:
        detect_base_position(img, display=True)
    else:
        print(f"Failed to load image: {image_path}")