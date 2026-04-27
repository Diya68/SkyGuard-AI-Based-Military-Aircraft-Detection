"""
Inference/Prediction Script for Aircraft Detection
Run inference on images or video using trained YOLOv8 model
"""

import os
import cv2
from ultralytics import YOLO
import glob
from pathlib import Path

import argparse


class AircraftDetector:
    def __init__(self, model_path="aircraft_detection_runs/train_run/weights/best.pt", conf=0.25):
        """
        Initialize detector with trained model

        Args:
            model_path: Path to trained YOLOv8 model (.pt file)
            conf: Confidence threshold (0-1)
        """
        self.conf = conf

        # Load model
        if not os.path.exists(model_path):
            print(f"ERROR: Model not found at {model_path}")
            print("Please run training first: python 2_train_model.py")
            exit(1)

        print(f"Loading model from {model_path}...")
        self.model = YOLO(model_path)
        print("Model loaded successfully!")

        # Load class names
        self.class_names = self.model.names
        print(f"Classes: {len(self.class_names)} aircraft types")

    def predict_image(self, image_path, save=True, show=True):
        """
        Run inference on a single image

        Args:
            image_path: Path to image file
            save: Whether to save the output image
            show: Whether to display the result

        Returns:
            Detection results
        """
        if not os.path.exists(image_path):
            print(f"ERROR: Image not found: {image_path}")
            return None

        print(f"\nProcessing: {image_path}")

        # Run inference
        results = self.model(image_path, conf=self.conf, verbose=False)

        # Process results
        for result in results:
            boxes = result.boxes
            num_detections = len(boxes)
            print(f"  Detected {num_detections} aircraft")

            # Print detection details
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                class_name = self.class_names[class_id]
                confidence = float(box.conf[0])
                print(f"    {i+1}. {class_name}: {confidence:.2%}")

            # Save results
            if save:
                output_dir = "predictions"
                os.makedirs(output_dir, exist_ok=True)

                # Get filename
                filename = Path(image_path).name
                output_path = os.path.join(output_dir, f"pred_{filename}")

                # Save annotated image
                annotated_frame = result.plot()
                cv2.imwrite(output_path, annotated_frame)
                print(f"  Saved to: {output_path}")

            # Show results
            if show:
                annotated_frame = result.plot()
                cv2.imshow("Aircraft Detection", annotated_frame)
                print("  Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        return results

    def predict_directory(self, directory, save=True):
        """
        Run inference on all images in a directory

        Args:
            directory: Path to directory containing images
            save: Whether to save output images
        """
        if not os.path.exists(directory):
            print(f"ERROR: Directory not found: {directory}")
            return

        # Supported image extensions
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']

        # Find all images
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(directory, ext)))
            image_paths.extend(glob.glob(os.path.join(directory, ext.upper())))

        if not image_paths:
            print(f"No images found in {directory}")
            return

        print(f"\nFound {len(image_paths)} images in {directory}")
        print(f"Processing with confidence threshold: {self.conf}\n")

        results_list = []
        for img_path in image_paths:
            result = self.predict_image(img_path, save=save, show=False)
            if result:
                results_list.append(result)

        print(f"\n{'='*50}")
        print(f"Processed {len(results_list)} images")
        print(f"Predictions saved to: predictions/")

    def predict_video(self, video_path, save=True):
        """
        Run inference on video

        Args:
            video_path: Path to video file or 0 for webcam
            save: Whether to save output video
        """
        # Open video
        if video_path == 0 or video_path == "0":
            cap = cv2.VideoCapture(0)
            source_name = "webcam"
        else:
            if not os.path.exists(video_path):
                print(f"ERROR: Video not found: {video_path}")
                return
            cap = cv2.VideoCapture(video_path)
            source_name = Path(video_path).name

        if not cap.isOpened():
            print(f"ERROR: Could not open video source")
            return

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\nProcessing video: {source_name}")
        print(f"Resolution: {width}x{height}, FPS: {fps}")

        # Video writer
        out = None
        if save and video_path != 0:
            output_dir = "predictions"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"pred_{source_name}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        print("Press 'q' to quit\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference
            results = self.model(frame, conf=self.conf, verbose=False)

            # Process results
            for result in results:
                annotated_frame = result.plot()

                # Count detections
                num_detections = len(result.boxes)
                if num_detections > 0:
                    print(
                        f"Frame {frame_count}: {num_detections} aircraft detected", end='\r')

                # Write frame
                if out:
                    out.write(annotated_frame)

                # Display
                cv2.imshow("Aircraft Detection", annotated_frame)

            frame_count += 1

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

        print(f"\nProcessed {frame_count} frames")
        if save and out:
            print(f"Output saved to: {output_path}")


def test_model_on_dataset():
    """Test model on test set images"""
    print("Testing on dataset test images...")

    detector = AircraftDetector()

    # Test on a few images from yolo_dataset/test/images
    test_dir = "yolo_dataset/images/test"
    if os.path.exists(test_dir):
        detector.predict_directory(test_dir, save=True)
    else:
        print(f"Test directory not found: {test_dir}")


def interactive_mode():
    """Interactive mode for user input"""
    print("\n" + "="*50)
    print("Aircraft Detection - Interactive Mode")
    print("="*50)

    # Load detector
    detector = AircraftDetector()

    while True:
        print("\nOptions:")
        print("  1. Predict on single image")
        print("  2. Predict on directory")
        print("  3. Predict on video")
        print("  4. Test on dataset")
        print("  5. Exit")

        choice = input("\nEnter choice (1-5): ").strip()

        if choice == '1':
            path = input("Enter image path: ").strip()
            detector.predict_image(path)

        elif choice == '2':
            path = input("Enter directory path: ").strip()
            detector.predict_directory(path)

        elif choice == '3':
            path = input("Enter video path (or 0 for webcam): ").strip()
            detector.predict_video(path)

        elif choice == '4':
            test_model_on_dataset()

        elif choice == '5':
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Aircraft Detection Inference')
    parser.add_argument('--source', '-s', type=str, default=None,
                        help='Path to image, directory, or video')
    parser.add_argument('--model', '-m', type=str,
                        default='aircraft_detection_runs/train_run/weights/best.pt',
                        help='Path to trained model')
    parser.add_argument('--conf', '-c', type=float, default=0.25,
                        help='Confidence threshold (default: 0.25)')
    parser.add_argument('--mode', type=str, default='auto',
                        choices=['auto', 'image', 'dir', 'video', 'test'],
                        help='Inference mode')

    args = parser.parse_args()

    # Install ultralytics if needed
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Installing ultralytics...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'ultralytics'])

    # Run based on mode
    if args.source is None:
        # Interactive mode
        interactive_mode()
    else:
        # Command line mode
        detector = AircraftDetector(args.model, args.conf)

        if args.mode == 'auto':
            # Auto detect type
            if os.path.isdir(args.source):
                detector.predict_directory(args.source)
            elif args.source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                detector.predict_video(args.source)
            else:
                detector.predict_image(args.source)

        elif args.mode == 'image':
            detector.predict_image(args.source)

        elif args.mode == 'dir':
            detector.predict_directory(args.source)

        elif args.mode == 'video':
            detector.predict_video(args.source)

        elif args.mode == 'test':
            test_model_on_dataset()
