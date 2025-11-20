#!/usr/bin/env python3
"""
Convert YOLO-format bounding box labels to RSPrompter format.

Usage:
    # Keep original class labels from YOLO
    python scripts/convert_yolo_to_rsprompter.py \
        --dataset-root /zfs/ai4good/student/yuqyan/treeAI/0_RGB_fL \
        --output-dir ./rsprompter_prompts/0_RGB_fL \
        --splits train val \
        --label-mode original

    # Set all labels to 1 (ignore semantic classes)
    python scripts/convert_yolo_to_rsprompter.py \
        --dataset-root /zfs/ai4good/student/yuqyan/treeAI/0_RGB_fL \
        --output-dir ./rsprompter_prompts/0_RGB_fL \
        --splits train val \
        --label-mode all_ones

Input format (YOLO .txt):
    Each line: class_id x_center y_center width height (normalized 0-1)

Output format (RSPrompter .json):
    {"labels": [class_id, ...], "scores": [], "bboxes": [[x1, y1, x2, y2], ...]}
"""
import argparse
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert YOLO labels to RSPrompter format.')
    parser.add_argument('--dataset-root', required=True, type=Path,
                        help='Root directory of dataset (contains train/, val/, etc.)')
    parser.add_argument('--output-dir', required=True, type=Path,
                        help='Output base directory for RSPrompter prompts')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'],
                        help='Splits to process (e.g., train val test)')
    parser.add_argument('--format', choices=['yolo', 'xyxy'], default='yolo',
                        help='Input bbox format: yolo (x_center y_center w h, normalized) or xyxy (x1 y1 x2 y2, absolute)')
    parser.add_argument('--label-mode', choices=['original', 'all_ones'], default='original',
                        help='Label mode: original (keep class IDs from YOLO) or all_ones (set all to 1)')
    return parser.parse_args()


def parse_voc_xml(xml_path: Path) -> Tuple[List[List[float]], List[int]]:
    """
    Parse Pascal VOC XML format to extract bboxes and labels.
    
    Args:
        xml_path: Path to XML file
        
    Returns:
        bboxes: List of [x1, y1, x2, y2] coordinates
        labels: List of class IDs
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    bboxes = []
    labels = []
    
    for obj in root.findall('object'):
        # Get class name/number
        name_elem = obj.find('name')
        if name_elem is None:
            continue
        
        try:
            class_id = int(name_elem.text)
        except ValueError:
            # If name is not a number, skip or map to default
            continue
        
        # Get bbox
        bndbox = obj.find('bndbox')
        if bndbox is None:
            continue
        
        try:
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_id)
        except (AttributeError, ValueError) as e:
            continue
    
    return bboxes, labels


def yolo_to_xyxy(x_center: float, y_center: float, width: float, height: float, 
                 img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Convert YOLO format (normalized center + size) to XYXY (absolute coordinates).
    
    Args:
        x_center, y_center, width, height: Normalized values (0-1)
        img_width, img_height: Image dimensions in pixels
    
    Returns:
        x1, y1, x2, y2: Absolute pixel coordinates
    """
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    width_abs = width * img_width
    height_abs = height * img_height
    
    x1 = x_center_abs - width_abs / 2.0
    y1 = y_center_abs - height_abs / 2.0
    x2 = x_center_abs + width_abs / 2.0
    y2 = y_center_abs + height_abs / 2.0
    
    return x1, y1, x2, y2


def convert_split(
    dataset_root: Path,
    split: str,
    output_dir: Path,
    bbox_format: str = 'yolo',
    label_mode: str = 'original'
) -> Dict[str, int]:
    """
    Convert one split from YOLO to RSPrompter format.
    
    Args:
        label_mode: 'original' to keep class IDs, 'all_ones' to set all to 1
    
    Returns:
        Dict with success/failure stats
    """
    # Paths
    images_dir = dataset_root / split / 'images'
    # Try labels_txt first (YOLO format), then labels as fallback
    labels_dir = dataset_root / split / 'labels_txt'
    if not labels_dir.exists():
        labels_dir = dataset_root / split / 'labels'
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir} (tried labels_txt and labels)")
    
    # Output directory
    split_output_dir = output_dir / split
    split_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    log_file = split_output_dir / f'conversion_log_{split}.log'
    logger = logging.getLogger(f'convert_{split}')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.propagate = False
    
    logger.info(f"Converting {split} split")
    logger.info(f"Images: {images_dir}")
    logger.info(f"Labels: {labels_dir}")
    logger.info(f"Output: {split_output_dir}")
    logger.info(f"Format: {bbox_format}")
    logger.info(f"Label mode: {label_mode}")
    logger.info("="*80)
    
    # Find all images
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_files.extend(images_dir.glob(ext))
    image_files = sorted(image_files)
    
    print(f"\n📦 Processing {split} split")
    print(f"   Found {len(image_files)} images")
    
    # Process each image
    successful = 0
    skipped_no_label = 0
    skipped_empty_label = 0
    failed = []
    
    for img_path in tqdm(image_files, desc=f'  Converting {split}'):
        img_stem = img_path.stem
        label_path = labels_dir / f"{img_stem}.txt"
        output_file = split_output_dir / f"{img_stem}.json"
        
        # Skip if output already exists
        if output_file.exists():
            logger.debug(f"{img_path.name} | SKIPPED | Output file already exists")
            continue
        
        # Check if label file exists (try .txt first, then .xml)
        label_path_txt = labels_dir / f"{img_stem}.txt"
        label_path_xml = labels_dir / f"{img_stem}.xml"
        
        if label_path_txt.exists():
            label_path = label_path_txt
            label_format = 'yolo'
        elif label_path_xml.exists():
            label_path = label_path_xml
            label_format = 'voc'
        else:
            logger.warning(f"{img_path.name} | NO_LABEL | Label file not found: {label_path_txt.name} or {label_path_xml.name}")
            skipped_no_label += 1
            # Create empty prompt file
            with open(output_file, 'w') as f:
                json.dump({'labels': [], 'scores': [], 'bboxes': []}, f)
            continue
        
        try:
            # Read image to get dimensions
            with Image.open(img_path) as img:
                img_width, img_height = img.size
            
            # Read label file based on format
            bboxes = []
            labels = []
            
            if label_format == 'voc':
                # Parse XML (Pascal VOC format)
                try:
                    bboxes, labels = parse_voc_xml(label_path)
                except Exception as e:
                    logger.error(f"{img_path.name} | XML_PARSE_ERROR | {e}")
                    failed.append(str(img_path))
                    continue
            else:
                # Parse YOLO format (.txt)
                with open(label_path, 'r') as f:
                    for line_idx, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        if len(parts) < 5:
                            logger.warning(f"{img_path.name} | INVALID_LINE | Line {line_idx+1}: expected at least 5 values (class + bbox), got {len(parts)}")
                            continue
                        
                        try:
                            # First column is always class ID in YOLO format
                            class_id = int(parts[0])
                            coords = [float(x) for x in parts[1:5]]
                            
                            if bbox_format == 'yolo':
                                # Convert YOLO (x_center, y_center, w, h, normalized) to XYXY (absolute)
                                x_center, y_center, width, height = coords
                                x1, y1, x2, y2 = yolo_to_xyxy(x_center, y_center, width, height, img_width, img_height)
                            else:
                                # Already in XYXY format
                                x1, y1, x2, y2 = coords
                            
                            bboxes.append([x1, y1, x2, y2])
                            labels.append(class_id)
                        
                        except (ValueError, IndexError) as e:
                            logger.warning(f"{img_path.name} | PARSE_ERROR | Line {line_idx+1}: {e}")
                            continue
            
            # Clamp all bboxes to image bounds and validate
            valid_bboxes = []
            valid_labels = []
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox
                x1 = max(0, min(x1, img_width))
                y1 = max(0, min(y1, img_height))
                x2 = max(0, min(x2, img_width))
                y2 = max(0, min(y2, img_height))
                
                # Validate bbox
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"{img_path.name} | INVALID_BBOX | Bbox {i}: invalid bbox ({x1}, {y1}, {x2}, {y2})")
                    continue
                
                valid_bboxes.append([x1, y1, x2, y2])
                valid_labels.append(labels[i])
            
            bboxes = valid_bboxes
            labels = valid_labels
            
            if len(bboxes) == 0:
                logger.warning(f"{img_path.name} | EMPTY_LABEL | No valid bboxes found")
                skipped_empty_label += 1
            else:
                # Get unique class counts for logging
                unique_classes = set(labels)
                logger.info(f"{img_path.name} | SUCCESS | Converted {len(bboxes)} bboxes (classes: {sorted(unique_classes)})")
            
            # Determine final labels based on mode
            if label_mode == 'all_ones':
                final_labels = [1] * len(bboxes)
            else:  # original
                final_labels = labels
            
            # Save to JSON (RSPrompter format)
            output_data = {
                'labels': final_labels,
                'scores': [],  # No scores available from ground truth
                'bboxes': bboxes
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f)
            
            successful += 1
        
        except Exception as e:
            logger.error(f"{img_path.name} | ERROR | {type(e).__name__}: {e}")
            failed.append(str(img_path))
            continue
    
    # Summary
    logger.info("="*80)
    logger.info(f"SUMMARY for {split}")
    logger.info(f"Total images: {len(image_files)}")
    logger.info(f"Successfully converted: {successful}")
    logger.info(f"Skipped (no label): {skipped_no_label}")
    logger.info(f"Skipped (empty label): {skipped_empty_label}")
    logger.info(f"Failed: {len(failed)}")
    if failed:
        logger.info(f"Failed files: {failed}")
    logger.info("="*80)
    
    # Close logger
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    
    return {
        'successful': successful,
        'skipped_no_label': skipped_no_label,
        'skipped_empty_label': skipped_empty_label,
        'failed': len(failed),
        'total': len(image_files)
    }


def main():
    args = parse_args()
    
    print("="*60)
    print("YOLO to RSPrompter Format Converter")
    print("="*60)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Output directory: {args.output_dir}")
    print(f"Splits: {args.splits}")
    print(f"Format: {args.format}")
    print(f"Label mode: {args.label_mode}")
    
    total_stats = {
        'successful': 0,
        'skipped_no_label': 0,
        'skipped_empty_label': 0,
        'failed': 0
    }
    
    for split in args.splits:
        try:
            stats = convert_split(
                dataset_root=args.dataset_root,
                split=split,
                output_dir=args.output_dir,
                bbox_format=args.format,
                label_mode=args.label_mode
            )
            
            print(f"  ✅ {split}: {stats['successful']}/{stats['total']} converted, "
                  f"{stats['skipped_no_label']} no label, "
                  f"{stats['skipped_empty_label']} empty, "
                  f"{stats['failed']} failed")
            
            total_stats['successful'] += stats['successful']
            total_stats['skipped_no_label'] += stats['skipped_no_label']
            total_stats['skipped_empty_label'] += stats['skipped_empty_label']
            total_stats['failed'] += stats['failed']
        
        except FileNotFoundError as e:
            print(f"  ⚠️  {split}: {e}")
            continue
    
    print("\n" + "="*60)
    print("✅ Conversion complete!")
    print(f"   Total converted: {total_stats['successful']}")
    print(f"   Total skipped (no label): {total_stats['skipped_no_label']}")
    print(f"   Total skipped (empty): {total_stats['skipped_empty_label']}")
    print(f"   Total failed: {total_stats['failed']}")
    print(f"   Output directory: {args.output_dir}/*/")
    print("="*60)


if __name__ == '__main__':
    main()

