"""
Dataset Preparation Script
Converts PASCAL VOC XML annotations to YOLO format and organizes dataset
"""

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import random

# Configuration
DATASET_ROOT = "archive (4)"
IMAGES_DIR = os.path.join(DATASET_ROOT, "JPEGImages")
ANNOTATIONS_DIR = os.path.join(DATASET_ROOT, "Annotations", "Horizontal Bounding Boxes")
IMAGESETS_DIR = os.path.join(DATASET_ROOT, "ImageSets", "Main")

OUTPUT_DIR = "yolo_dataset"
TRAIN_SPLIT_FILE = os.path.join(IMAGESETS_DIR, "train.txt")
TEST_SPLIT_FILE = os.path.join(IMAGESETS_DIR, "test.txt")

def parse_xml_annotation(xml_path):
    """Parse PASCAL VOC XML annotation file"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Get image dimensions
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    # Parse objects
    objects = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        objects.append({
            'class': class_name,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        })

    return width, height, objects

def convert_to_yolo_format(xmin, ymin, xmax, ymax, img_w, img_h):
    """
    Convert PASCAL VOC bbox to YOLO format
    YOLO: <class_id> <x_center> <y_center> <width> <height> (all normalized 0-1)
    """
    x_center = ((xmin + xmax) / 2) / img_w
    y_center = ((ymin + ymax) / 2) / img_h
    box_w = (xmax - xmin) / img_w
    box_h = (ymax - ymin) / img_h

    # Clamp values to [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    box_w = max(0, min(1, box_w))
    box_h = max(0, min(1, box_h))

    return x_center, y_center, box_w, box_h

def create_dataset_structure():
    """Create YOLO dataset directory structure"""
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)
    print(f"Created directory structure in {OUTPUT_DIR}")

def load_split_ids():
    """Load train and test IDs from split files"""
    # Load train IDs
    with open(TRAIN_SPLIT_FILE, 'r') as f:
        train_ids = [line.strip() for line in f.readlines()]

    # Load test IDs
    with open(TEST_SPLIT_FILE, 'r') as f:
        test_ids = [line.strip() for line in f.readlines()]

    # Create validation split from train (20%)
    random.seed(42)
    random.shuffle(train_ids)
    val_split = int(len(train_ids) * 0.2)
    val_ids = train_ids[:val_split]
    train_ids = train_ids[val_split:]

    return train_ids, val_ids, test_ids

def process_dataset():
    """Main function to process the dataset"""
    print("Starting dataset preparation...")

    # Create output directories
    create_dataset_structure()

    # Load split IDs
    train_ids, val_ids, test_ids = load_split_ids()
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # Collect all class names
    all_classes = set()
    annotation_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.xml')]

    for ann_file in annotation_files:
        xml_path = os.path.join(ANNOTATIONS_DIR, ann_file)
        try:
            _, _, objects = parse_xml_annotation(xml_path)
            for obj in objects:
                all_classes.add(obj['class'])
        except Exception as e:
            print(f"Error parsing {ann_file}: {e}")

    class_names = sorted(list(all_classes))
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    print(f"Found {len(class_names)} classes: {class_names[:10]}...")  # Show first 10

    # Save class names
    with open(os.path.join(OUTPUT_DIR, 'classes.txt'), 'w') as f:
        f.write('\n'.join(class_names))
    print("Saved class names to classes.txt")

    # Create data.yaml for YOLOv8
    yaml_content = f"""path: {os.path.abspath(OUTPUT_DIR).replace('\\', '/')}
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}
names: {class_names}
"""
    with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
        f.write(yaml_content)
    print("Created data.yaml")

    # Process each split
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }

    total_annotations = 0
    for split_name, ids in splits.items():
        print(f"\nProcessing {split_name} split ({len(ids)} images)...")
        split_annotations = 0

        for img_id in ids:
            img_name = f"{img_id}.jpg"
            ann_name = f"{img_id}.xml"

            img_src = os.path.join(IMAGES_DIR, img_name)
            ann_src = os.path.join(ANNOTATIONS_DIR, ann_name)

            # Skip if files don't exist
            if not os.path.exists(img_src):
                print(f"  Warning: Image not found: {img_src}")
                continue
            if not os.path.exists(ann_src):
                print(f"  Warning: Annotation not found: {ann_src}")
                continue

            # Copy image
            img_dst = os.path.join(OUTPUT_DIR, 'images', split_name, img_name)
            shutil.copy2(img_src, img_dst)

            # Parse annotation and convert to YOLO
            try:
                img_w, img_h, objects = parse_xml_annotation(ann_src)

                # Write YOLO format annotation
                label_dst = os.path.join(OUTPUT_DIR, 'labels', split_name, f"{img_id}.txt")
                with open(label_dst, 'w') as f:
                    for obj in objects:
                        class_id = class_to_id[obj['class']]
                        x_center, y_center, box_w, box_h = convert_to_yolo_format(
                            obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'],
                            img_w, img_h
                        )
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")
                        split_annotations += 1
            except Exception as e:
                print(f"  Error processing {ann_name}: {e}")

        total_annotations += split_annotations
        print(f"  {split_name}: {split_annotations} objects annotated")

    print(f"\n{'='*50}")
    print("Dataset preparation completed!")
    print(f"Total objects annotated: {total_annotations}")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Classes: {len(class_names)}")
    print(f"\nYou can now run: python 2_train_model.py")

if __name__ == "__main__":
    process_dataset()
