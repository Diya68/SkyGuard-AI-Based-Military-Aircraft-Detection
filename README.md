# SkyGuard-AI-Based-Military-Aircraft-Detection
AI-based military aircraft detection system using YOLOv8 for real-time object detection from images, videos, and webcam. Includes dataset preprocessing, model training, evaluation, and GUI interface for live monitoring.
# Overview
SkyGuard AI is a deep learning–based system designed to detect and classify military aircraft from images, videos, and live webcam streams. The project uses the YOLOv8 object detection model to provide fast and accurate real-time detection of multiple aircraft types.
The main objective of this project is to automate the process of aerial image monitoring, which is traditionally time-consuming and prone to human error. By leveraging transfer learning and a pretrained YOLOv8 model, the system is trained on a military aircraft dataset obtained from Kaggle. The dataset annotations were processed and converted into YOLO format, followed by model training, evaluation, and performance optimization.
The trained model is capable of identifying aircraft locations using bounding boxes and predicting their classes with good accuracy. In addition, a user-friendly graphical interface (GUI) was developed to allow real-time detection through webcam, image upload, and video input.
This project demonstrates the practical application of computer vision and deep learning in defense, surveillance, and security-related scenarios. It also highlights the complete machine learning pipeline, including data preprocessing, model training, performance evaluation, and deployment.

# Problem Statement 
Monitoring aerial images manually to identify military aircraft is time-consuming, inefficient, and prone to human error, especially when large volumes of data must be analyzed. There is a need for an automated and reliable system that can accurately detect and classify aircraft in real time. This project aims to develop an AI-based solution using deep learning to improve detection accuracy and support surveillance and defense applications.

# Technical Details
1. Language: Python
2. Framework: PyTorch
3. Model: YOLOv8 (Ultralytics)
4. Dataset: Kaggle Military Aircraft Detection Dataset
5. Annotation Format: CSV converted to YOLO format
6. Image Processing: OpenCV
8. GUI: Tkinter
9. Training Method: Transfer Learning with pretrained YOLOv8
8. Evaluation Metrics: Precision, Recall, mAP
9. Detection Modes: Image, Video, and Live Webcam

# Challenges Faced
Dataset annotation conversion from CSV to YOLO format
Handling train, validation, and test splits
Managing GPU memory limitations during training
Optimizing hyperparameters to improve model accuracy
Fixing dataset path and file structure issues

# Installation and Setup
### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/your-repository-name.git
cd your-repository-name
```
### 2. Install Dependencies

```bash
pip install ultralytics opencv-python pillow
```
### 3. Run Detection

```bash
python detect.py
```
## Usage
Once executed, the program displays options to perform different aircraft detection tasks:

Detect aircraft in an image
Analyze aircraft in a video file
Start real-time aircraft detection using webcam
Launch the GUI for easy interaction
View detection results with bounding boxes and predicted aircraft classes
Exit the application

Follow the on-screen instructions to select an option, provide the required input (image, video, or webcam), and view the detection results. The system processes the input and displays detected aircraft with labeled bounding boxes and confidence scores.

## Workflow
```
+------------------------------+
|     Dataset Collection       |
| (Kaggle Aircraft Dataset)    |
+--------------+---------------+
               |
               v
+------------------------------+
|      Data Preprocessing      |
| (CSV → YOLO Format, Split)   |
+--------------+---------------+
               |
               v
+------------------------------+
|    Dataset Configuration     |
|        (aircraft.yaml)       |
+--------------+---------------+
               |
               v
+------------------------------+
|     Load Pretrained Model    |
|           YOLOv8             |
+--------------+---------------+
               |
               v
+------------------------------+
|        Model Training        |
|      (Transfer Learning)     |
+--------------+---------------+
               |
               v
+------------------------------+
|        Model Evaluation      |
|   Precision | Recall | mAP   |
+--------------+---------------+
               |
               v
+------------------------------+
|        Save Best Model       |
|           best.pt            |
+--------------+---------------+
               |
               v
+------------------------------+
|      Aircraft Detection      |
|   Image / Video / Webcam     |
+--------------+---------------+
               |
               v
+------------------------------+
|      Result Visualization    |
| Bounding Boxes + Labels      |
+------------------------------+
```
## Advantages
1. Enables automatic detection of military aircraft using deep learning.
2. Supports real-time detection from images, videos, and webcam.
3. Uses YOLOv8 model, which provides fast and accurate object detection.
4. Reduces manual effort in monitoring aerial imagery.
5. Provides visual results with bounding boxes and confidence scores.

## Limitations
1. Model accuracy depends on the quality and size of the dataset.
2. May struggle with small or partially visible aircraft.
3. Performance can decrease in low lighting or poor image quality.
4. Requires GPU resources for efficient training and faster inference.

## Conclusion

This project demonstrates an AI-based approach for detecting military aircraft using the YOLOv8 object detection model. The system successfully processes images, videos, and live webcam input to identify aircraft with good accuracy. By automating aircraft detection, the project highlights the practical use of computer vision and deep learning in surveillance and defense-related applications.
