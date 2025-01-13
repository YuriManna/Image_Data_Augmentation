from ultralytics import YOLO
import torch

def train_model(weights, data, img_size, epochs, batch_size):
    """
    Train a YOLO model.
    
    Args:
        weights (str): Path to pre-trained weights or 'yolov11s.pt' (or other variants).
        data (str): Path to dataset.yaml.
        img_size (int): Image size for training.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.
    """
    # Load the YOLO model
    model = YOLO(weights)
    
    # Train the model
    model.train(
        data=data,          
        imgsz=img_size,     
        epochs=epochs,     
        batch=batch_size,   
        name="weed_detection" 
    )
    print("Training completed. Check 'runs/train/weed_detection' for results.")

def validate_model(weights, data, img_size):
    """
    Validate the YOLO model on the validation dataset.
    
    Args:
        weights (str): Path to the trained weights file (e.g., best.pt).
        data (str): Path to dataset.yaml.
        img_size (int): Image size for validation.
    """
    # Load the model
    model = YOLO(weights)
    
    # Validate the model
    results = model.val(
        data=data,          
        imgsz=img_size,     
        split="val"       
    )
    print("Validation completed. Results:")
    print(f"Precision: {results.metrics.precision:.3f}")
    print(f"Recall: {results.metrics.recall:.3f}")
    print(f"mAP@0.5: {results.metrics.map50:.3f}")
    print(f"mAP@0.5:0.95: {results.metrics.map:.3f}")

def test_model(weights, data, img_size, conf_thresh, iou_thresh):
    """
    Test a YOLO model on the test dataset.
    
    Args:
        weights (str): Path to the trained weights file (e.g., best.pt).
        data (str): Path to dataset.yaml.
        img_size (int): Image size for inference.
        conf_thresh (float): Confidence threshold.
        iou_thresh (float): IOU threshold for NMS.
    """
    # Load the model
    model = YOLO(weights)
    
    # Run the test
    results = model.val(
        data=data,           
        imgsz=img_size,     
        conf=conf_thresh,   
        iou=iou_thresh,     
        split="test"        
    )
    print("Test completed. Results:")
    print(f"Precision: {results.metrics.precision:.3f}")
    print(f"Recall: {results.metrics.recall:.3f}")
    print(f"mAP@0.5: {results.metrics.map50:.3f}")
    print(f"mAP@0.5:0.95: {results.metrics.map:.3f}")

def print_logo():
    logo = """
  ___    ___ ________  ___       ________     
 |\  \  /  /|\   __  \|\  \     |\   __  \    
 \ \  \/  / | \  \|\  \ \  \    \ \  \|\  \   
  \ \    / / \ \  \\\   \ \  \    \ \  \\\   \  
   \/  /  /   \ \  \\\   \ \  \____\ \  \\\   \ 
 __/  / /      \ \_______\ \_______\ \_______\\
|\___/ /        \|_______|\|_______|\|_______|
\|___|/                                       
            """
    print(logo)

if __name__ == "__main__":
    print_logo()

    # Get the mode from the user
    mode = input("Please enter the mode to run (train, val, test): ").strip().lower()
    # Validate the input
    while mode not in ["train", "val", "test"]:
        if mode == "exit":
            print("Exiting the program.")
            exit()
        print("Invalid mode. Please choose from 'train', 'val', or 'test'.")
        mode = input("Please enter the mode to run (train, val, test): ").strip().lower()
    

    # Common parameters
    weights = "yolo\yolov11n.pt"  # Path to weights or pre-trained model
    data = "WeedCrop\data.yaml"  # Path to dataset configuration
    img_size = 416  # Image size for training and testing

    print(f"Running in {mode} mode...\n")

    if mode == "train":     
        epochs = 50
        batch_size = 32
        train_model(weights, data, img_size, epochs, batch_size)
    elif mode == "val":
        validate_model(weights, data, img_size)
    elif mode == "test":
        conf_thresh = 0.25
        iou_thresh = 0.45
        test_model(weights, data, img_size, conf_thresh, iou_thresh)
