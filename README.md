# Deep Learning-Based Lung Cancer Classification

## Project Overview

This project implements and compares two deep learning architectures — **AlexNet** and **YOLOv9** — for classifying lung cancer using histopathological images. The goal is to leverage AlexNet for high-accuracy image classification and YOLOv9 for real-time object detection, demonstrating the strengths of each model in medical image analysis.

## Objectives

- Classify lung cancer types with high accuracy using AlexNet.
- Achieve real-time detection and classification using YOLOv9.
- Improve generalization through data augmentation.
- Compare performance with state-of-the-art models.

## Models and Methodology

### 🔹 AlexNet
- **Framework**: PyTorch
- **Architecture**: 5 convolutional layers + ReLU, max pooling, dropout, and fully connected layers
- **Output**: 3-class classification
- **Training Configuration**:
  - Optimizer: Adam
  - Loss: CrossEntropyLoss
  - Epochs: 25
  - Learning Rate: 0.0001
  - Batch Size: 32

### 🔸 YOLOv9
- **Framework**: Ultralytics
- **Input Size**: 640x640
- **Annotations**: Manually labeled bounding boxes
- **Training Configuration**:
  - Optimizer: SGD
  - Epochs: 50
  - Confidence Threshold: 0.25
  - IoU Threshold: 0.5
  - Anchor Boxes: Auto-calculated

## Results

| Model     | Accuracy | F1 Score | mAP  | IoU  | FPS (Real-Time) |
|-----------|----------|----------|------|------|-----------------|
| AlexNet   | 100%     | 1.00     | —    | —    | Low             |
| YOLOv9    | 80%      | 0.79     | 0.75 | 0.65 | 30+             |

- **AlexNet**: High accuracy and precision, ideal for static analysis  
- **YOLOv9**: Real-time inference with trade-off in accuracy

## Challenges

- Manual annotation of bounding boxes  
- Hardware (GPU) limitations  
- Resolution and batch size tuning for resource efficiency

## Visuals

- Accuracy/Loss training graphs  
- Confusion matrices for both models  
- Screenshots of YOLOv9 detection results

## Dataset

The dataset used in this project is based on histopathological images of lung cancer and can be accessed here:  
🔗 [Lung and Colon Cancer Histopathological Images Dataset on Kaggle](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)

This dataset contains annotated images for three categories:  
- Lung Adenocarcinoma  
- Lung Squamous Cell Carcinoma  
- Benign Lung Tissue

## Conclusion

- **AlexNet** excels in classification accuracy but lacks real-time capability.  
- **YOLOv9** is suitable for real-time deployment but requires more refinement for medical-grade accuracy.

