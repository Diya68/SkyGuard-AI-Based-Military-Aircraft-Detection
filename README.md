# SkyGuard-AI-Based-Military-Aircraft-Detection
A state-of-the-art military aircraft detection system using Hybrid Architecture that combines YOLOv8 for real-time object detection with a Convolutional Neural Network (CNN) for attribute prediction. The system is capable of accurately identifying military aircraft from images, videos, and live webcam feeds.
It incorporates comprehensive dataset preprocessing, including annotation, augmentation, and normalization, followed by model training and evaluation to ensure high detection accuracy and robustness. The YOLOv8 model efficiently detects aircraft in real time, while the CNN module predicts additional attributes such as aircraft type, category, or threat level.
A user-friendly GUI interface is integrated for live monitoring, enabling seamless visualization of detection results and system performance in real-world scenarios.
# Overview
SkyGuard AI is a deep learning–based system designed to detect and classify military aircraft from images, videos, and live webcam streams. The system leverages a hybrid architecture that combines the strengths of real-time object detection and attribute prediction for enhanced analysis.

- In Stage 1, the YOLOv8 model is used to detect aircraft in real time, generating accurate bounding boxes and class labels for multiple aircraft types. This ensures fast and efficient identification even in complex visual environments.

- In Stage 2, a Convolutional Neural Network (CNN) processes the cropped aircraft regions obtained from YOLOv8 detections to predict continuous attributes such as wing span, length, speed, altitude, and crew capacity. This additional layer of analysis provides deeper insights beyond basic classification.
  
The system incorporates a complete machine learning pipeline, including dataset preprocessing (annotation conversion to YOLO format, augmentation, normalization), model training, evaluation, and performance optimization. Transfer learning is utilized with a pretrained YOLOv8 model, trained on a military aircraft dataset sourced from Kaggle.
A user-friendly Graphical User Interface (GUI) enables real-time monitoring through webcam streams, image uploads, and video inputs, allowing seamless interaction and visualization of detection results.
Overall, SkyGuard AI demonstrates a practical and scalable application of computer vision and deep learning in defense, surveillance, and security domains, highlighting an end-to-end workflow from data preparation to deployment with enhanced predictive intelligence.

# Problem Statement 
Detecting and analyzing military aircraft from images, videos, and live streams is a challenging task due to complex real-world conditions and limitations of traditional systems. Manual monitoring is time-consuming and error-prone, while existing automated systems often focus only on detection without providing deeper insights.

- Key Challenges:

-- High visual similarity between different aircraft types
-- Variations in viewpoint, scale, and orientation
-- Impact of environmental conditions like lighting, weather, and occlusion
-- Lack of systems that provide detailed attribute information (e.g., speed, wingspan, etc.)

Therefore, there is a need for an intelligent and real-time solution that can both detect and analyze aircraft effectively. The proposed SkyGuard AI addresses this by using a hybrid architecture (YOLOv8 + CNN) for accurate detection and advanced attribute prediction.

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
1. Dataset annotation conversion from CSV to YOLO format
2. Handling train, validation, and test splits
3. Managing GPU memory limitations during training
4. Optimizing hyperparameters to improve model accuracy
5. Fixing dataset path and file structure issues

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

1. Detect aircraft in an image
2. Analyze aircraft in a video file
3. Start real-time aircraft detection using webcam
4. Launch the GUI for easy interaction
5. View detection results with bounding boxes and predicted aircraft classes
6. Exit the application

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
