from pathlib import Path

import cv2
from ultralytics import YOLO

# Load your trained model - replace with your trained model path
# It should be in runs/detect/your_experiment_name/weights/best.pt
model = YOLO(
    "runs/detect/weed_detection/weights/best.pt"
)  # adjust path to your trained model

# Path to test image
test_image_path = "/Users/kaloyan/Code/Project3.1/Image_Data_Augmentation/yolo/data/test/images"  # adjust to your test image path

# Run inference on the test image
results = model(test_image_path, conf=0.25)  # adjust confidence threshold if needed

# Process results
for r in results:
    # Plot results image
    im_array = r.plot()  # plot a BGR numpy array of predictions

    # Save the result
    output_path = Path("runs/detect/test_results")
    output_path.mkdir(exist_ok=True)

    # Save image with predictions
    cv2.imwrite(str(output_path / "prediction.jpg"), im_array)

    # Print detection information
    boxes = r.boxes
    for box in boxes:
        # Get class name
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # Get confidence
        confidence = float(box.conf[0])

        # Get coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        print(f"Detected {class_name} with confidence: {confidence:.2f}")
        print(f"Bounding box: ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})")
        print("---")

print(f"\nResults saved to {output_path}")
