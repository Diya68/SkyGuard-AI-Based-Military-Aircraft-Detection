"""
Hybrid YOLO + Attribute Predictor Model
======================================
Uses YOLOv8 for aircraft detection (localization + classification)
and a separate Neural Network for aircraft attribute prediction
(wing span, length, max speed, max altitude, crew count)

This is a 2-stage pipeline:
1. YOLO detects aircraft and returns bounding boxes + aircraft class
2. Attribute predictor takes the cropped aircraft image and predicts continuous attributes
"""

import os
import json
import pickle
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# PyTorch for attribute predictor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# YOLO for detection
from ultralytics import YOLO

# Configuration
AIRCRAFT_DATA_ROOT = "aircraft_data"
ARCHIVE_ROOT = "archive (4)"
CROP_ROOT = os.path.join(AIRCRAFT_DATA_ROOT, "crop")
OUTPUT_DIR = "hybrid_model_output"

# Aircraft attributes database (for training the attribute predictor)
AIRCRAFT_ATTRIBUTES = {
    'A10': [17.5, 16.3, 720, 13700, 1],
    'B1': [41.8, 44.5, 1320, 18000, 4],
    'B2': [52.4, 21.0, 1010, 15200, 2],
    'B52': [56.4, 48.5, 957, 16700, 5],
    'C17': [51.7, 53.0, 830, 13700, 3],
    'C5': [67.9, 75.3, 932, 10800, 7],
    'E2': [24.6, 17.6, 626, 11200, 5],
    'EF2000': [10.9, 15.9, 2495, 16700, 1],
    'F14': [19.5, 19.1, 2485, 15200, 2],
    'F15': [13.1, 19.4, 3017, 20000, 1],
    'F16': [9.8, 15.0, 2175, 15200, 1],
    'F18': [12.3, 17.1, 1915, 15200, 1],
    'F22': [13.6, 18.9, 2410, 20000, 1],
    'F35': [10.7, 15.7, 1930, 18200, 1],
    'F117': [13.2, 20.1, 1040, 13700, 1],
    'SR71': [16.9, 32.7, 3540, 26000, 2],
    'U2': [31.4, 19.2, 805, 21300, 1],
    'V22': [14.0, 17.5, 565, 7600, 3],
    'XB70': [32.0, 59.1, 3300, 21300, 2],
    'YF23': [13.3, 21.0, 2655, 19800, 1],
}

ATTRIBUTE_NAMES = ['wing_span_m', 'length_m', 'max_speed_kmh', 'max_altitude_m', 'crew_count']


def ensure_output_dir():
    """Create output directory structure"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "results"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "cropped"), exist_ok=True)


# =============================================================================
# ATTRIBUTE PREDICTOR NEURAL NETWORK
# =============================================================================

class AircraftAttributeDataset(Dataset):
    """Dataset for training attribute predictor on cropped aircraft images"""

    def __init__(self, image_paths, labels, attributes, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.attributes = np.array(attributes, dtype=np.float32)
        self.transform = transform

        # Normalize attributes
        self.attr_scaler = StandardScaler()
        self.attributes_scaled = self.attr_scaler.fit_transform(self.attributes)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and preprocess image
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0  # Normalize to [0, 1]
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = torch.FloatTensor(img)

        attr = torch.FloatTensor(self.attributes_scaled[idx])

        return img, attr


class AircraftAttributePredictor(nn.Module):
    """
    CNN for predicting aircraft attributes from cropped images
    Uses transfer learning style architecture
    """

    def __init__(self, num_attributes=5):
        super(AircraftAttributePredictor, self).__init__()

        # Feature extraction layers (similar to VGG-style)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
        )

        # Calculate flattened size: 224 -> 112 -> 56 -> 28 -> 14
        self.flat_size = 512 * 14 * 14

        # Fully connected layers for regression
        self.regressor = nn.Sequential(
            nn.Linear(self.flat_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_attributes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.regressor(x)
        return x


# =============================================================================
# HYBRID MODEL CLASS
# =============================================================================

class HybridAircraftModel:
    """
    Hybrid Model combining YOLO for detection and Neural Network for attributes
    """

    def __init__(self, yolo_model_path=None, attr_model_path=None, device=None):
        """
        Initialize hybrid model

        Args:
            yolo_model_path: Path to trained YOLO model (or None to use pretrained)
            attr_model_path: Path to trained attribute predictor
            device: 'cuda', 'cpu', or None for auto
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize YOLO detector
        self.yolo_model = None
        self.yolo_model_path = yolo_model_path

        # Initialize attribute predictor
        self.attr_model = None
        self.attr_scaler = None
        self.attr_model_path = attr_model_path

        if yolo_model_path and os.path.exists(yolo_model_path):
            self.load_yolo(yolo_model_path)

        if attr_model_path and os.path.exists(attr_model_path):
            self.load_attribute_predictor(attr_model_path)

    def load_yolo(self, model_path):
        """Load YOLO model for detection"""
        print(f"Loading YOLO model from {model_path}...")
        self.yolo_model = YOLO(model_path)
        print("YOLO model loaded successfully!")
        return self.yolo_model

    def train_yolo(self, data_yaml, epochs=100, imgsz=800, batch=16):
        """
        Train YOLO model for aircraft detection

        Args:
            data_yaml: Path to data.yaml file
            epochs: Number of training epochs
            imgsz: Image size
            batch: Batch size
        """
        print("\n" + "="*60)
        print("Stage 1: Training YOLO for Aircraft Detection")
        print("="*60)

        # Load pretrained YOLOv8m
        self.yolo_model = YOLO("yolov8m.pt")

        # Train
        results = self.yolo_model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            optimizer="AdamW",
            lr0=0.001,
            weight_decay=0.0005,

            # Augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10,
            translate=0.1,
            scale=0.5,
            shear=2,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.2,

            patience=20,
            device=self.device if self.device != 'cpu' else 'cpu',

            # Save options
            save=True,
            project="hybrid_aircraft_detection",
            name="yolo_train",
        )

        self.yolo_model_path = os.path.join("hybrid_aircraft_detection", "yolo_train", "weights", "best.pt")
        print(f"\nYOLO training complete! Model saved to: {self.yolo_model_path}")

        return results

    def train_attribute_predictor(self, image_paths, labels, attributes, epochs=50, batch_size=32):
        """
        Train the attribute prediction neural network

        Args:
            image_paths: List of cropped aircraft image paths
            labels: List of aircraft class names
            attributes: List of attribute arrays [wing_span, length, speed, altitude, crew]
            epochs: Training epochs
            batch_size: Batch size
        """
        print("\n" + "="*60)
        print("Stage 2: Training Attribute Predictor")
        print("="*60)

        if len(image_paths) == 0:
            print("ERROR: No training data provided!")
            return None

        # Create dataset
        dataset = AircraftAttributeDataset(image_paths, labels, attributes)
        self.attr_scaler = dataset.attr_scaler

        # Split train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Create model
        self.attr_model = AircraftAttributePredictor(num_attributes=5).to(self.device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.attr_model.parameters(), lr=0.001, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

        # Training loop
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            # Training
            self.attr_model.train()
            train_loss = 0.0

            for images, attrs in train_loader:
                images = images.to(self.device)
                attrs = attrs.to(self.device)

                optimizer.zero_grad()
                outputs = self.attr_model(images)
                loss = criterion(outputs, attrs)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.attr_model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for images, attrs in val_loader:
                    images = images.to(self.device)
                    attrs = attrs.to(self.device)
                    outputs = self.attr_model(images)
                    loss = criterion(outputs, attrs)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            scheduler.step(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_attribute_predictor()

            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        print(f"\nAttribute predictor training complete! Best val loss: {best_val_loss:.4f}")
        return history

    def save_attribute_predictor(self, path=None):
        """Save attribute predictor model"""
        if path is None:
            path = os.path.join(OUTPUT_DIR, "models", "attribute_predictor.pt")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        save_dict = {
            'model_state_dict': self.attr_model.state_dict(),
            'scaler': self.attr_scaler,
            'attribute_names': ATTRIBUTE_NAMES
        }
        torch.save(save_dict, path)
        print(f"Attribute predictor saved to {path}")
        return path

    def load_attribute_predictor(self, path):
        """Load attribute predictor model"""
        print(f"Loading attribute predictor from {path}...")

        checkpoint = torch.load(path, map_location=self.device)

        self.attr_model = AircraftAttributePredictor(num_attributes=5).to(self.device)
        self.attr_model.load_state_dict(checkpoint['model_state_dict'])
        self.attr_model.eval()

        self.attr_scaler = checkpoint['scaler']

        print("Attribute predictor loaded successfully!")
        return self.attr_model

    def predict_attributes(self, image):
        """
        Predict attributes from a cropped aircraft image

        Args:
            image: Cropped aircraft image (numpy array)

        Returns:
            Dictionary of predicted attributes
        """
        if self.attr_model is None:
            print("WARNING: Attribute predictor not loaded!")
            return None

        # Preprocess
        img = cv2.resize(image, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.FloatTensor(img).unsqueeze(0).to(self.device)

        # Predict
        self.attr_model.eval()
        with torch.no_grad():
            pred_scaled = self.attr_model(img).cpu().numpy()[0]
            pred = self.attr_scaler.inverse_transform([pred_scaled])[0]

        return {
            'wing_span_m': pred[0],
            'length_m': pred[1],
            'max_speed_kmh': pred[2],
            'max_altitude_m': pred[3],
            'crew_count': max(1, round(pred[4]))  # Round crew to integer
        }

    def predict(self, image_path, conf=0.25, save_crops=True):
        """
        Full pipeline: Detect aircraft with YOLO, then predict attributes

        Args:
            image_path: Path to input image
            conf: Confidence threshold for YOLO
            save_crops: Whether to save cropped detections

        Returns:
            List of detection results with attributes
        """
        if self.yolo_model is None:
            print("ERROR: YOLO model not loaded!")
            return None

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"ERROR: Could not load image: {image_path}")
            return None

        # Step 1: YOLO Detection
        results = self.yolo_model(image_path, conf=conf, verbose=False)

        detections = []

        for result in results:
            boxes = result.boxes

            for i, box in enumerate(boxes):
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = result.names[class_id]

                # Crop aircraft
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Save crop if requested
                if save_crops:
                    crop_dir = os.path.join(OUTPUT_DIR, "cropped")
                    os.makedirs(crop_dir, exist_ok=True)
                    crop_path = os.path.join(crop_dir, f"{Path(image_path).stem}_det{i}.jpg")
                    cv2.imwrite(crop_path, crop)

                # Step 2: Predict attributes
                attributes = None
                if self.attr_model is not None:
                    attributes = self.predict_attributes(crop)

                detection = {
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'attributes': attributes
                }
                detections.append(detection)

        return detections

    def predict_and_visualize(self, image_path, conf=0.25, save=True):
        """
        Predict and visualize results on the image
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"ERROR: Could not load image: {image_path}")
            return None

        detections = self.predict(image_path, conf=conf)

        # Draw results
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            conf_score = det['confidence']

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{class_name}: {conf_score:.2%}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Draw attributes if available
            if det['attributes']:
                attrs = det['attributes']
                attr_texts = [
                    f"Wing: {attrs['wing_span_m']:.1f}m",
                    f"Len: {attrs['length_m']:.1f}m",
                    f"Speed: {attrs['max_speed_kmh']:.0f}km/h",
                    f"Alt: {attrs['max_altitude_m']:.0f}m",
                    f"Crew: {attrs['crew_count']}"
                ]

                for j, text in enumerate(attr_texts):
                    y_pos = y2 + 20 + j * 20
                    cv2.putText(img, text, (x1, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Save result
        if save:
            result_dir = os.path.join(OUTPUT_DIR, "results")
            os.makedirs(result_dir, exist_ok=True)
            result_path = os.path.join(result_dir, f"hybrid_{Path(image_path).name}")
            cv2.imwrite(result_path, img)
            print(f"Result saved to: {result_path}")

        return img, detections


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def load_training_data():
    """
    Load training data for attribute predictor
    Uses aircraft_data/crop dataset
    """
    print("\nLoading training data for attribute predictor...")

    image_paths = []
    labels = []
    attributes = []

    if not os.path.exists(CROP_ROOT):
        print(f"ERROR: {CROP_ROOT} not found!")
        return None, None, None

    class_names = sorted([d for d in os.listdir(CROP_ROOT)
                         if os.path.isdir(os.path.join(CROP_ROOT, d))])

    print(f"Found {len(class_names)} aircraft classes")

    for class_name in class_names:
        class_path = os.path.join(CROP_ROOT, class_name)
        class_images = [f for f in os.listdir(class_path)
                       if f.endswith(('.jpg', '.jpeg', '.png'))]

        # Get base attributes for this class
        base_attrs = AIRCRAFT_ATTRIBUTES.get(class_name, [20.0, 30.0, 1500, 15000, 2])

        for img_file in class_images:
            img_path = os.path.join(class_path, img_file)

            # Add variation to attributes for robustness
            variation = np.random.normal(0, 0.05, 5)  # 5% variation
            attrs = np.array(base_attrs) * (1 + variation)

            image_paths.append(img_path)
            labels.append(class_name)
            attributes.append(attrs)

    print(f"Loaded {len(image_paths)} training samples")
    return image_paths, labels, attributes


def train_hybrid_model():
    """
    Main training function - trains both YOLO and Attribute Predictor
    """
    print("="*70)
    print("HYBRID AIRCRAFT MODEL TRAINING")
    print("Stage 1: YOLO for Detection | Stage 2: Neural Network for Attributes")
    print("="*70)

    ensure_output_dir()

    model = HybridAircraftModel()

    # Stage 1: Train YOLO (if data.yaml exists)
    data_yaml = "yolo_dataset/data.yaml"
    if os.path.exists(data_yaml):
        yolo_results = model.train_yolo(data_yaml, epochs=100, imgsz=800, batch=16)
    else:
        print(f"WARNING: {data_yaml} not found. Skipping YOLO training.")
        print("Run 'python 1_prepare_dataset.py' first.")

    # Stage 2: Train Attribute Predictor
    image_paths, labels, attributes = load_training_data()

    if image_paths:
        attr_history = model.train_attribute_predictor(
            image_paths, labels, attributes,
            epochs=50, batch_size=32
        )

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Models saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("\nTo use the model:")
    print("  python mixed_model.py --mode predict --image <image_path>")
    print("="*70)

    return model


def predict_hybrid(image_path, yolo_path=None, attr_path=None, conf=0.25):
    """
    Run inference with hybrid model
    """
    # Auto-detect model paths
    if yolo_path is None:
        # Try to find trained YOLO model
        possible_paths = [
            "hybrid_aircraft_detection/yolo_train/weights/best.pt",
            "aircraft_detection_runs/train_run/weights/best.pt",
            "runs/detect/train/weights/best.pt"
        ]
        for p in possible_paths:
            if os.path.exists(p):
                yolo_path = p
                break

    if attr_path is None:
        attr_path = os.path.join(OUTPUT_DIR, "models", "attribute_predictor.pt")

    if yolo_path is None or not os.path.exists(yolo_path):
        print(f"ERROR: YOLO model not found!")
        print("Please train first or provide --yolo-model path")
        return None

    # Load model
    print(f"\n{'='*70}")
    print("HYBRID MODEL INFERENCE")
    print(f"{'='*70}")

    model = HybridAircraftModel(
        yolo_model_path=yolo_path,
        attr_model_path=attr_path if os.path.exists(attr_path) else None
    )

    # Run prediction
    print(f"\nProcessing: {image_path}")
    img, detections = model.predict_and_visualize(image_path, conf=conf)

    # Print results
    print(f"\n{'='*70}")
    print("DETECTION RESULTS")
    print(f"{'='*70}")

    for i, det in enumerate(detections, 1):
        print(f"\nDetection {i}:")
        print(f"  Class: {det['class']}")
        print(f"  Confidence: {det['confidence']:.2%}")
        print(f"  Bounding Box: ({det['bbox'][0]}, {det['bbox'][1]}) - ({det['bbox'][2]}, {det['bbox'][3]})")

        if det['attributes']:
            print(f"  Predicted Attributes:")
            for attr_name, value in det['attributes'].items():
                print(f"    - {attr_name}: {value:.2f}")

    if not detections:
        print("No aircraft detected in the image.")

    return detections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Hybrid YOLO + Attribute Predictor Model')
    parser.add_argument('--mode', '-m', type=str, default='train',
                       choices=['train', 'predict'], help='Mode: train or predict')
    parser.add_argument('--image', '-i', type=str, help='Image path for prediction')
    parser.add_argument('--yolo-model', type=str, help='Path to YOLO model')
    parser.add_argument('--attr-model', type=str, help='Path to attribute predictor')
    parser.add_argument('--conf', '-c', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')

    args = parser.parse_args()

    if args.mode == 'train':
        train_hybrid_model()
    elif args.mode == 'predict':
        if not args.image:
            print("ERROR: Please provide --image for prediction")
        else:
            predict_hybrid(args.image, args.yolo_model, args.attr_model, args.conf)
