import os
from pathlib import Path

import cv2
from tqdm import tqdm


def resize_images_in_place(folder_path, target_size=640):
    """
    Resize all images in the folder and overwrite them
    """
    # Convert to Path object
    folder_path = Path(folder_path)
    print(f"Processing directory: {folder_path}")

    # Get all images with different possible extensions
    image_files = []
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        image_files.extend(folder_path.glob(f"*{ext}"))

    image_files = list(set(image_files))  # Remove duplicates

    print(f"Found {len(image_files)} images")

    for img_path in tqdm(image_files, desc="Resizing images"):
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Could not read image: {img_path}")
                continue

            # Get original size
            original_size = img.shape[:2]

            # Resize image
            resized = cv2.resize(
                img, (target_size, target_size), interpolation=cv2.INTER_AREA
            )

            # Save image back to the same path (overwrite)
            cv2.imwrite(str(img_path), resized)

            print(
                f"Resized {img_path.name}: {original_size} -> {target_size}x{target_size}"
            )

        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")


# Set paths for all directories
base_path = Path("/Users/kaloyan/Code/Project3.1/Image_Data_Augmentation/yolo/data")
train_path = base_path / "train" / "images"
val_path = base_path / "val" / "images"
test_path = base_path / "test" / "images"

# Resize training images
print("Processing training images...")
resize_images_in_place(train_path, target_size=640)

# Resize validation images
print("\nProcessing validation images...")
resize_images_in_place(val_path, target_size=640)

# Resize test images
print("\nProcessing test images...")
resize_images_in_place(test_path, target_size=640)

print("\nAll done! All images have been resized to 640x640")
