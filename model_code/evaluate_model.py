"""
Model Evaluation Script
Tests and displays accuracy metrics for both YOLO and Mixed models
"""

import os
import json
import pickle
import numpy as np
import cv2
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
OUTPUT_DIR = "evaluation_results"
MIXED_MODEL_PATH = "mixed_model_output/models/mixed_model.pkl"
YOLO_MODEL_PATH = "aircraft_detection_runs/train_run/weights/best.pt"

def ensure_output_dir():
    """Create output directory"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

def evaluate_mixed_model():
    """Evaluate the mixed classification + prediction model"""
    print("="*60)
    print("Evaluating Mixed Classification + Prediction Model")
    print("="*60)

    if not os.path.exists(MIXED_MODEL_PATH):
        print(f"\nModel not found: {MIXED_MODEL_PATH}")
        print("Please train first: python mixed_model.py")
        return None

    # Load the model
    with open(MIXED_MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)

    label_encoder = model_data['label_encoder']
    scaler = model_data['scaler']
    attr_scaler = model_data['attr_scaler']
    classifier = model_data['classifier']
    regressor = model_data['regressor']
    class_names = model_data['class_names']

    print(f"\nLoaded model from: {MIXED_MODEL_PATH}")
    print(f"Classes: {len(class_names)}")

    # Load test data from mixed_model.py's output
    # We need to re-run data loading and split to get test set
    from mixed_model import load_aircraft_data_crop, load_archive_data, extract_image_features

    print("\nLoading datasets for evaluation...")
    crop_features, crop_labels, crop_attrs, _ = load_aircraft_data_crop()
    archive_features, archive_labels, archive_attrs, _ = load_archive_data()

    # Combine datasets
    all_features = crop_features + archive_features
    all_labels = crop_labels + archive_labels

    if crop_attrs:
        import numpy as np
        archive_attrs_synth = [np.array([20.0, 30.0, 1500, 15000, 2]) * (1 + np.random.normal(0, 0.2, 5))
                              for _ in range(len(archive_features))]
        all_attrs = crop_attrs + archive_attrs_synth
    else:
        all_attrs = None

    if len(all_features) == 0:
        print("No data loaded for evaluation")
        return None

    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(all_labels)
    y_true = le.transform(all_labels)

    # Split (same as training)
    from sklearn.model_selection import train_test_split
    X = np.array(all_features)
    X_scaled = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_true, test_size=0.2, random_state=42, stratify=y_true
    )

    if all_attrs is not None:
        attrs = np.array(all_attrs)
        attrs_scaled = attr_scaler.transform(attrs)
        _, _, attr_train, attr_test = train_test_split(
            X_scaled, attrs_scaled, test_size=0.2, random_state=42, stratify=y_true
        )

    results = {}

    # Evaluate Classification
    print("\n" + "-"*60)
    print("CLASSIFICATION EVALUATION")
    print("-"*60)

    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(le.classes_):
        class_mask = y_test == i
        if class_mask.sum() > 0:
            class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
            count = class_mask.sum()
            print(f"  {class_name:15s}: {class_acc:.4f} ({count:3d} samples)")

    # Classification report
    print("\nDetailed Classification Report:")
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    print(report)

    # Save report
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), 'w') as f:
        f.write(f"Mixed Model Classification Report\n")
        f.write(f"="*60 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write(report)

    # Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred)
    # Show only top classes if too many
    if len(le.classes_) > 20:
        # Get top 20 classes by frequency
        class_counts = np.bincount(y_test)
        top_indices = np.argsort(class_counts)[-20:]
        cm = cm[top_indices][:, top_indices]
        labels = [le.classes_[i] for i in top_indices]
    else:
        labels = le.classes_

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix (Top Classes)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", "confusion_matrix.png"), dpi=150)
    print(f"\nConfusion matrix saved to: {os.path.join(OUTPUT_DIR, 'plots', 'confusion_matrix.png')}")
    plt.close()

    results['classification'] = {
        'accuracy': accuracy,
        'num_classes': len(le.classes_),
        'num_test_samples': len(y_test)
    }

    # Evaluate Prediction (if available)
    if regressor is not None and all_attrs is not None:
        print("\n" + "-"*60)
        print("PREDICTION EVALUATION")
        print("-"*60)

        attr_pred = regressor.predict(X_test)
        attr_true_orig = attr_scaler.inverse_transform(attr_test)
        attr_pred_orig = attr_scaler.inverse_transform(attr_pred)

        attr_names = ['Wing Span (m)', 'Length (m)', 'Max Speed (km/h)',
                     'Max Altitude (m)', 'Crew Count']

        print("\nAttribute Prediction Metrics:")
        print(f"{'Attribute':<20} {'MAE':>12} {'RMSE':>12} {'R²':>12}")
        print("-"*60)

        mae_list = []
        rmse_list = []
        r2_list = []

        for i, name in enumerate(attr_names):
            mae = mean_absolute_error(attr_true_orig[:, i], attr_pred_orig[:, i])
            rmse = np.sqrt(mean_squared_error(attr_true_orig[:, i], attr_pred_orig[:, i]))
            r2 = r2_score(attr_true_orig[:, i], attr_pred_orig[:, i])

            mae_list.append(mae)
            rmse_list.append(rmse)
            r2_list.append(r2)

            print(f"{name:<20} {mae:>12.2f} {rmse:>12.2f} {r2:>12.4f}")

        # Overall metrics
        avg_mae = np.mean(mae_list)
        avg_rmse = np.mean(rmse_list)
        avg_r2 = np.mean(r2_list)

        print("-"*60)
        print(f"{'AVERAGE':<20} {avg_mae:>12.2f} {avg_rmse:>12.2f} {avg_r2:>12.4f}")

        # Save prediction report
        with open(os.path.join(OUTPUT_DIR, "prediction_report.txt"), 'w') as f:
            f.write(f"Mixed Model Prediction Report\n")
            f.write(f"="*60 + "\n\n")
            f.write(f"{'Attribute':<25} {'MAE':>12} {'RMSE':>12} {'R²':>12}\n")
            f.write("-"*60 + "\n")
            for i, name in enumerate(attr_names):
                f.write(f"{name:<25} {mae_list[i]:>12.2f} {rmse_list[i]:>12.2f} {r2_list[i]:>12.4f}\n")
            f.write("-"*60 + "\n")
            f.write(f"{'AVERAGE':<25} {avg_mae:>12.2f} {avg_rmse:>12.2f} {avg_r2:>12.4f}\n")

        # Plot prediction vs actual for each attribute
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, name in enumerate(attr_names):
            ax = axes[i]
            ax.scatter(attr_true_orig[:, i], attr_pred_orig[:, i], alpha=0.5)
            ax.plot([attr_true_orig[:, i].min(), attr_true_orig[:, i].max()],
                   [attr_true_orig[:, i].min(), attr_true_orig[:, i].max()],
                   'r--', lw=2)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{name}\nR² = {r2_list[i]:.4f}')

        # Remove extra subplot
        axes[5].remove()

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "plots", "prediction_scatter.png"), dpi=150)
        print(f"Prediction plots saved to: {os.path.join(OUTPUT_DIR, 'plots', 'prediction_scatter.png')}")
        plt.close()

        results['prediction'] = {
            'avg_mae': avg_mae,
            'avg_rmse': avg_rmse,
            'avg_r2': avg_r2
        }

    return results

def evaluate_yolo_model():
    """Evaluate YOLO model on test set"""
    print("\n" + "="*60)
    print("Evaluating YOLO Detection Model")
    print("="*60)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics not installed. Install with: pip install ultralytics")
        return None

    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"\nModel not found: {YOLO_MODEL_PATH}")
        print("Please train first: python 2_train_model.py")
        return None

    print(f"\nLoading model from: {YOLO_MODEL_PATH}")
    model = YOLO(YOLO_MODEL_PATH)

    # Check if validation data exists
    data_yaml = "yolo_dataset/data.yaml"
    if not os.path.exists(data_yaml):
        print(f"Dataset not found: {data_yaml}")
        return None

    print("\nRunning validation on test set...")
    results = model.val(data=data_yaml, split='test', device='cpu')

    print("\n" + "-"*60)
    print("YOLO VALIDATION RESULTS")
    print("-"*60)
    print(f"mAP50:    {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"mAP75:    {results.box.map75:.4f}")

    if hasattr(results.box, 'mp'):
        print(f"Precision: {results.box.mp:.4f}")
    if hasattr(results.box, 'mr'):
        print(f"Recall:    {results.box.mr:.4f}")

    # Save results
    with open(os.path.join(OUTPUT_DIR, "yolo_evaluation.txt"), 'w') as f:
        f.write(f"YOLO Model Evaluation Report\n")
        f.write(f"="*60 + "\n\n")
        f.write(f"mAP50:     {results.box.map50:.4f}\n")
        f.write(f"mAP50-95:  {results.box.map:.4f}\n")
        f.write(f"mAP75:     {results.box.map75:.4f}\n")
        if hasattr(results.box, 'mp'):
            f.write(f"Precision: {results.box.mp:.4f}\n")
        if hasattr(results.box, 'mr'):
            f.write(f"Recall:    {results.box.mr:.4f}\n")

    return {
        'map50': results.box.map50,
        'map': results.box.map,
        'map75': results.box.map75
    }

def compare_models():
    """Compare different model approaches"""
    print("\n" + "="*60)
    print("Model Comparison Summary")
    print("="*60)

    comparison = {
        'Model': ['Mixed Model (sklearn)', 'YOLOv8 (CPU)'],
        'Task': ['Classification + Prediction', 'Object Detection'],
        'Speed': ['Fast (~100ms)', 'Slow (~1-5s)'],
        'Hardware': ['CPU Only', 'CPU/GPU'],
        'Best For': ['Quick analysis, attributes', 'Precise localization']
    }

    print("\n" + "-"*60)
    for i, model in enumerate(comparison['Model']):
        print(f"\n{model}")
        print(f"  Task:       {comparison['Task'][i]}")
        print(f"  Speed:      {comparison['Speed'][i]}")
        print(f"  Hardware:   {comparison['Hardware'][i]}")
        print(f"  Best For:   {comparison['Best For'][i]}")

    print("\n" + "="*60)
    print("Recommendation:")
    print("-"*60)
    print("For quick classification & attribute prediction:")
    print("  Use: python mixed_model.py")
    print("\nFor precise aircraft detection with bounding boxes:")
    print("  Use: python 2_train_model.py (requires more resources)")
    print("="*60)

def main():
    """Main evaluation function"""
    ensure_output_dir()

    print("="*60)
    print("SKYGUARD MODEL EVALUATION")
    print("="*60)
    print(f"\nResults will be saved to: {os.path.abspath(OUTPUT_DIR)}")

    # Evaluate mixed model
    mixed_results = evaluate_mixed_model()

    # Evaluate YOLO model
    yolo_results = evaluate_yolo_model()

    # Comparison
    compare_models()

    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    if mixed_results:
        print("\nMixed Model Results:")
        if 'classification' in mixed_results:
            print(f"  Classification Accuracy: {mixed_results['classification']['accuracy']*100:.2f}%")
        if 'prediction' in mixed_results:
            print(f"  Prediction R² Score: {mixed_results['prediction']['avg_r2']:.4f}")

    if yolo_results:
        print("\nYOLO Model Results:")
        print(f"  mAP50: {yolo_results['map50']*100:.2f}%")
        print(f"  mAP50-95: {yolo_results['map']*100:.2f}%")

    print(f"\nDetailed reports saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("="*60)

if __name__ == "__main__":
    main()
