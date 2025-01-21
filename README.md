# Image_Data_Augmentation

## Overview
This repository focuses on data augmentation for image datasets with bounding boxes, tailored for object detection tasks. The augmentation process enhances the diversity of training data, improving model robustness and performance.

### Dataset
To use the dataset:
1. Visit [Roboflow WeedCrop Dataset](https://universe.roboflow.com/new-workspace-csmgu/weedcrop-waifl).
2. Go to the "datasets" section.
3. Download the YOLOv11 formatted dataset.

### Important Configuration
Ensure that inside the Ultralytics settings.json the directory for the dataset correspond to the one where you downloaded the dataset. You can find the file here:
```
C:\Users\(username)\AppData\Roaming\Ultralytics\settings.json
```
Replace `(username)` with your Windows user directory name.

## Repository Structure

### `Data_augmentation.ipynb`
This notebook contains the pipeline for augmenting the dataset on the fly. It demonstrates various augmentation techniques designed to preserve bounding box integrity while increasing data variability. Use this notebook to preprocess your dataset before training.

### `yolo` Folder
- **YOLO Pipeline**: The folder includes the YOLO training pipeline, detailing the steps to train models using the augmented dataset.
- **Results**: Contains the results from our experiments and tests, providing insights into the effectiveness of the augmentation strategies.

## Getting Started
1. Download the dataset from the provided link.
2. Check and update the Ultralytics configuration file path.
3. Run `Data_augmentation.ipynb` to preprocess your dataset.
4. Use the pipeline in the `yolo` folder to train your YOLO model and review the test results.