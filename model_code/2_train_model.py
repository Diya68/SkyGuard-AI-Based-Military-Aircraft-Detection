"""
Train YOLOv8 Model for Aircraft Detection
Fixed version that works locally with proper paths
"""

import os
from ultralytics import YOLO
import torch

def check_device():
    """Check device - always use CPU"""
    print("Running in CPU-only mode (CUDA disabled)")
    return False

def train_model():
    """Train YOLOv8 model"""

    # Configuration
    DATA_YAML = "yolo_dataset/data.yaml"
    MODEL_SIZE = "yolov8m.pt"  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

    # Training hyperparameters
    EPOCHS = 100
    IMAGE_SIZE = 800  # Increased for aircraft detection
    BATCH_SIZE = 16   # Reduce if you get OOM errors
    PATIENCE = 20     # Early stopping patience

    # Check if data.yaml exists
    if not os.path.exists(DATA_YAML):
        print(f"ERROR: {DATA_YAML} not found!")
        print("Please run: python 1_prepare_dataset.py")
        return

    # Use CPU only
    use_cuda = check_device()
    device = 'cpu'

    print(f"\n{'='*50}")
    print("Starting YOLOv8 Training")
    print(f"{'='*50}")
    print(f"Model: {MODEL_SIZE}")
    print(f"Data: {DATA_YAML}")
    print(f"Epochs: {EPOCHS}")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Device: {device}")
    print(f"{'='*50}\n")

    # Load pretrained model
    # It will automatically download if not present
    print(f"Loading model {MODEL_SIZE}...")
    model = YOLO(MODEL_SIZE)

    # Train the model
    print("Starting training...")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,

        # Augmentation parameters for aircraft detection
        hsv_h=0.015,    # HSV Hue augmentation
        hsv_s=0.7,      # HSV Saturation augmentation
        hsv_v=0.4,      # HSV Value augmentation
        degrees=10,     # Rotation (+/- degrees)
        translate=0.1,  # Translation (+/- fraction)
        scale=0.5,      # Scale gain
        shear=2,        # Shear (+/- degrees)
        fliplr=0.5,     # Flip left-right probability
        mosaic=1.0,     # Mosaic augmentation probability
        mixup=0.2,      # MixUp augmentation probability

        patience=PATIENCE,
        device=device,

        # Save options
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs

        # Validation
        val=True,

        # Project name
        project="aircraft_detection_runs",
        name="train_run",
    )

    print(f"\n{'='*50}")
    print("Training completed!")
    print(f"{'='*50}")
    print(f"Best model saved to: {os.path.abspath('aircraft_detection_runs/train_run/weights/best.pt')}")
    print(f"Last model saved to: {os.path.abspath('aircraft_detection_runs/train_run/weights/last.pt')}")

    return model, results

def evaluate_model(model):
    """Evaluate the trained model on validation set"""
    print(f"\n{'='*50}")
    print("Running Model Evaluation")
    print(f"{'='*50}")

    # Validate
    metrics = model.val()

    print(f"\nValidation Metrics:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")

    return metrics

def export_model(model):
    """Export model to different formats"""
    print(f"\n{'='*50}")
    print("Exporting Model")
    print(f"{'='*50}")

    # Export to ONNX (for deployment)
    print("Exporting to ONNX format...")
    model.export(format="onnx", dynamic=True)

    # Export to TorchScript
    print("Exporting to TorchScript format...")
    model.export(format="torchscript")

    print("\nExport completed!")

if __name__ == "__main__":
    # Install required packages if needed
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "ultralytics", "torch", "torchvision", "opencv-python"])
        from ultralytics import YOLO

    # Train model
    model, results = train_model()

    # Evaluate
    evaluate_model(model)

    # Export
    export_model(model)

    print(f"\n{'='*50}")
    print("All done! Next steps:")
    print("  - Test the model: python 3_predict.py")
    print("  - Best model: aircraft_detection_runs/train_run/weights/best.pt")
    print(f"{'='*50}")
