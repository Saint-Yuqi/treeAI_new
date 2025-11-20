#!/usr/bin/env python3
"""
Export bounding box prompts from trained RSPrompter checkpoint.

Usage:
    conda activate sam2
    python scripts/export_rsprompter_bbox_prompts.py \
        --config /path/to/rsprompter_config.py \
        --checkpoint /path/to/checkpoint.pth \
        --dataset-root /path/to/dataset \
        --split train \
        --output-dir ./rsprompter_prompts/train
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

# Add RSPrompter to path
RSPROMPTER_ROOT = Path(__file__).parent.parent.parent / 'RSPrompter'
sys.path.insert(0, str(RSPROMPTER_ROOT))

from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export RSPrompter bbox prompts.')
    parser.add_argument('--config', required=True, type=Path,
                        help='Path to RSPrompter config file')
    parser.add_argument('--checkpoint', required=True, type=Path,
                        help='Path to RSPrompter checkpoint')
    parser.add_argument('--dataset-root', required=True, type=Path,
                        help='Root of dataset (contains images/train, images/test, etc.)')
    parser.add_argument('--split', choices=['train', 'test', 'val', 'pick'], default='train',
                        help='Which split to process (pick for pick dataset)')
    parser.add_argument('--score-thr', type=float, default=0.3,
                        help='Minimum score threshold to keep predictions')
    parser.add_argument('--max-prompts', type=int, default=None,
                        help='Optional cap on number of prompts per image')
    parser.add_argument('--output-dir', required=True, type=Path,
                        help='Output directory for JSON files')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device for inference')
    parser.add_argument('--pick-dir', type=Path, default=None,
                        help='Path to pick dataset directory (only used if split=pick)')
    return parser.parse_args()


def list_images(image_dir: Path):
    """List all images in directory."""
    patterns = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    files = []
    for pattern in patterns:
        files.extend(image_dir.glob(pattern))
    return sorted(files)


def main():
    args = parse_args()
    
    # Load RSPrompter model
    print(f"Loading RSPrompter from {args.checkpoint}")
    cfg = Config.fromfile(str(args.config))
    model = init_detector(cfg, str(args.checkpoint), device=args.device)
    
    # Find images
    if args.split == 'pick':
        # For pick dataset, use pick-dir or dataset_root/pick/images
        if args.pick_dir:
            image_dir = Path(args.pick_dir) / 'images'
        else:
            image_dir = args.dataset_root / 'pick' / 'images'
        if not image_dir.is_dir():
            raise FileNotFoundError(
                f'Pick image directory not found: {image_dir}. '
                f'Use --pick-dir to specify pick dataset directory.')
    else:
        # Standard splits: train, test, val
        image_dir = args.dataset_root / 'images' / args.split
        if not image_dir.is_dir():
            # TreeAI datasets ship as dataset_root/split/images; fall back when found.
            alt_dir = args.dataset_root / args.split / 'images'
            if alt_dir.is_dir():
                image_dir = alt_dir
            else:
                raise FileNotFoundError(
                    f'Image directory not found. Tried: {image_dir} and {alt_dir}')
    
    image_files = list_images(image_dir)
    print(f"Found {len(image_files)} images in {image_dir}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image with error handling
    successful = 0
    skipped = 0
    failed = []
    
    for img_path in tqdm(image_files, desc=f'Exporting {args.split} prompts'):
        # Check if file exists and is readable
        if not img_path.exists():
            print(f"\n⚠️  Warning: Image file does not exist: {img_path}")
            failed.append(str(img_path))
            skipped += 1
            continue
        
        try:
            # Run inference
            result = inference_detector(model, str(img_path))
            pred_instances = result.pred_instances
            
            # Extract predictions
            if hasattr(pred_instances, 'scores'):
                scores = pred_instances.scores.detach().cpu().numpy()
            else:
                scores = np.ones(len(pred_instances.labels))
            
            labels = pred_instances.labels.detach().cpu().numpy()
            
            if pred_instances.bboxes is not None:
                bboxes = pred_instances.bboxes.detach().cpu().numpy()
            else:
                # Save empty prompt file if no bboxes
                bboxes = np.zeros((0, 4))
            
            # Filter by score and sort
            valid_indices = scores >= args.score_thr
            scores = scores[valid_indices]
            labels = labels[valid_indices]
            bboxes = bboxes[valid_indices]
            
            # Sort by score (descending)
            if len(scores) > 0:
                order = np.argsort(-scores)
                scores = scores[order]
                labels = labels[order]
                bboxes = bboxes[order]
                
                # Limit number of prompts
                if args.max_prompts is not None and len(scores) > args.max_prompts:
                    scores = scores[:args.max_prompts]
                    labels = labels[:args.max_prompts]
                    bboxes = bboxes[:args.max_prompts]
            else:
                # No valid predictions
                scores = np.array([])
                labels = np.array([])
                bboxes = np.zeros((0, 4))
            
            # Save to JSON (format compatible with prompt_loader.py)
            output_data = {
                'labels': labels.tolist(),
                'scores': scores.tolist(),
                'bboxes': bboxes.tolist()  # XYXY format
            }
            
            output_file = args.output_dir / f"{img_path.stem}.json"
            with open(output_file, 'w') as f:
                json.dump(output_data, f)
            
            successful += 1
            
        except Exception as e:
            print(f"\n⚠️  Warning: Failed to process {img_path.name}: {e}")
            failed.append(str(img_path))
            skipped += 1
            continue
    
    print(f"\n✅ Export complete!")
    print(f"   Successfully exported: {successful}/{len(image_files)}")
    if skipped > 0:
        print(f"   Skipped/Failed: {skipped}")
        if len(failed) <= 10:
            print(f"   Failed files: {failed}")
        else:
            print(f"   Failed files (first 10): {failed[:10]}...")


if __name__ == '__main__':
    main()
