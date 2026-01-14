#!/usr/bin/env python3
"""
Export bounding box prompts from trained RSPrompter checkpoint.

Usage (single dataset):
    conda activate RSPrompter
    python scripts/export_rsprompter_bbox_prompts.py \
        --config /home/c/yuqyan/code/RSPrompter/configs/treeai/rsprompter_anchor_treeai.py \
        --checkpoint /home/c/yuqyan/code/RSPrompter/work_dirs/treeai/rsprompter_0_12_fully_labeled/best_coco_bbox_mAP_epoch_50.pth\
        --dataset-root /home/c/yuqyan/data/TreeAI/12_RGB_SemSegm_640_fL\
        --splits train val test pick\
        --output-base-dir ./rsprompter_prompts/12_RGB_SemSegm_640_fL

Usage (batch processing):
    python scripts/export_rsprompter_bbox_prompts.py \
        --config /home/c/yuqyan/code/RSPrompter/configs/treeai/rsprompter_anchor_treeai.py\
        --checkpoint /home/c/yuqyan/code/RSPrompter/work_dirs/treeai/rsprompter_0_12_fully_labeled/best_coco_bbox_mAP_epoch_50.pth\
        --datasets-root /zfs/ai4good/student/yuqyan/treeAI \
        --dataset-names 0_RGB_fL 12_RGB_ObjDet_640_fL \
        --splits train test \
        --output-base-dir ./rsprompter_prompts
"""
import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm

# Add RSPrompter to path
RSPROMPTER_ROOT = Path(__file__).parent.parent.parent / 'RSPrompter'
sys.path.insert(0, str(RSPROMPTER_ROOT))

from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector


def extract_dataset_name(dataset_root: Path) -> str:
    """
    Extract dataset name from path.
    
    Examples:
        "/path/to/34_RGB_SemSegm_640_pL" -> "34_RGB_SemSegm_640_pL"
        "/path/to/12_RGB_ObjDet_640_fL" -> "12_RGB_ObjDet_640_fL"
    """
    path_str = str(dataset_root)
    parts = Path(path_str).parts
    for part in reversed(parts):
        # Match dataset name pattern: number(s)_RGB_...
        if re.match(r'\d+_RGB_', part):
            return part
    # Fallback: use the last directory name
    return dataset_root.name


def get_default_max_prompts() -> Optional[int]:
    """
    Get default max_prompts from configs/configs.yaml if available.
    Falls back to 50 if config file exists but value is missing, or None if file doesn't exist.
    """
    config_path = Path(__file__).parent.parent / 'configs' / 'configs.yaml'
    if not config_path.exists():
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Try to get max_prompts from rsprompter section
        if 'rsprompter' in config and 'max_prompts' in config['rsprompter']:
            return config['rsprompter']['max_prompts']
        
        # If rsprompter section exists but max_prompts is missing, use default 50
        if 'rsprompter' in config:
            return 50
        
        return None
    except Exception:
        # If any error reading config, return None (no default)
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export RSPrompter bbox prompts.')
    parser.add_argument('--config', required=True, type=Path,
                        help='Path to RSPrompter config file')
    parser.add_argument('--checkpoint', required=True, type=Path,
                        help='Path to RSPrompter checkpoint')
    
    # Single dataset mode
    parser.add_argument('--dataset-root', type=Path, default=None,
                        help='Root of single dataset (contains images/train, images/test, etc.)')
    
    # Batch processing mode
    parser.add_argument('--datasets-root', type=Path, default=None,
                        help='Root directory containing multiple datasets')
    parser.add_argument('--dataset-names', nargs='*', type=str, default=None,
                        help='List of dataset names to process (if not provided, auto-detect)')
    
    parser.add_argument('--split', choices=['train', 'test', 'val', 'pick'], default='train',
                        help='Which split to process (pick for pick dataset)')
    parser.add_argument('--splits', nargs='*', type=str, default=None,
                        help='List of splits to process (for batch mode)')
    
    parser.add_argument('--score-thr', type=float, default=0.2,
                        help='Minimum score threshold to keep predictions')
    
    # Get default max_prompts from config file (defaults to 50 if config exists)
    default_max_prompts = get_default_max_prompts()
    parser.add_argument('--max-prompts', type=int, default=default_max_prompts,
                        help=f'Optional cap on number of prompts per image (default: {default_max_prompts if default_max_prompts else "None"})')
    
    # Output directory: now uses base directory + dataset name
    parser.add_argument('--output-base-dir', type=Path, default=Path('./rsprompter_prompts'),
                        help='Base output directory (will create {dataset_name}/{split} subdirectories)')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Direct output directory (overrides --output-base-dir, for backward compatibility)')
    
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


def setup_logger(output_dir: Path, dataset_name: str, split: str) -> logging.Logger:
    """Setup logger for debug information."""
    log_file = output_dir / f'rsprompt_debug_{dataset_name}_{split}.log'
    
    # Create logger
    logger = logging.getLogger(f'rsprompter_{dataset_name}_{split}')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()  # Clear existing handlers
    
    # File handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.propagate = False
    
    return logger


def process_split(
    model,
    dataset_root: Path,
    dataset_name: str,
    split: str,
    output_dir: Path,
    score_thr: float,
    max_prompts: Optional[int],
    pick_dir: Optional[Path] = None
) -> Dict[str, int]:
    """
    Process a single split of a dataset.
    
    Returns:
        Dict with 'successful', 'skipped', 'failed' counts
    """
    # Create output directory first (needed for logger)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(output_dir, dataset_name, split)
    logger.info(f"Starting prompt export for {dataset_name}/{split}")
    logger.info(f"Dataset root: {dataset_root}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Score threshold: {score_thr}")
    logger.info(f"Max prompts per image: {max_prompts}")
    logger.info("="*80)
    # Find images
    if split == 'pick':
        # For pick dataset, use pick-dir or dataset_root/pick/images
        if pick_dir:
            image_dir = Path(pick_dir) / 'images'
        else:
            image_dir = dataset_root / 'pick' / 'images'
        if not image_dir.is_dir():
            raise FileNotFoundError(
                f'Pick image directory not found: {image_dir}. '
                f'Use --pick-dir to specify pick dataset directory.')
    else:
        # Standard splits: train, test, val
        image_dir = dataset_root / 'images' / split
        if not image_dir.is_dir():
            # TreeAI datasets ship as dataset_root/split/images; fall back when found.
            alt_dir = dataset_root / split / 'images'
            if alt_dir.is_dir():
                image_dir = alt_dir
            else:
                raise FileNotFoundError(
                    f'Image directory not found. Tried: {image_dir} and {alt_dir}')
    
    image_files = list_images(image_dir)
    print(f"  Found {len(image_files)} images in {image_dir}")
    
    # Process each image with error handling
    successful = 0
    skipped_existing = 0
    skipped_no_detections = 0
    skipped_below_threshold = 0
    failed_image_error = []
    failed_inference_error = []
    
    for img_path in tqdm(image_files, desc=f'  Exporting {split} prompts', leave=False):
        img_name = img_path.name
        img_stem = img_path.stem
        
        # Check if file exists and is readable
        if not img_path.exists():
            logger.error(f"{img_name} | IMAGE_NOT_FOUND | File does not exist: {img_path}")
            failed_image_error.append(str(img_path))
            continue
        
        # Check if prompt already exists (individual file)
        output_file = output_dir / f"{img_stem}.json"
        if output_file.exists():
            logger.debug(f"{img_name} | SKIPPED_EXISTS | Prompt file already exists")
            skipped_existing += 1
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
                # No bboxes detected
                logger.warning(f"{img_name} | NO_DETECTIONS | RSPrompter found no objects")
                skipped_no_detections += 1
                # Save empty prompt file
                output_data = {'labels': [], 'scores': [], 'bboxes': []}
                with open(output_file, 'w') as f:
                    json.dump(output_data, f)
                successful += 1
                continue
            
            # Count detections before and after filtering
            num_detections_before = len(scores)
            
            # Filter by score
            valid_indices = scores >= score_thr
            scores_filtered = scores[valid_indices]
            labels_filtered = labels[valid_indices]
            bboxes_filtered = bboxes[valid_indices]
            
            num_detections_after = len(scores_filtered)
            num_filtered = num_detections_before - num_detections_after
            
            # Sort by score (descending)
            if len(scores_filtered) > 0:
                order = np.argsort(-scores_filtered)
                scores_filtered = scores_filtered[order]
                labels_filtered = labels_filtered[order]
                bboxes_filtered = bboxes_filtered[order]
                
                # Log if prompts were truncated
                if max_prompts is not None and len(scores_filtered) > max_prompts:
                    num_truncated = len(scores_filtered) - max_prompts
                    logger.info(
                        f"{img_name} | TRUNCATED | "
                        f"Kept top {max_prompts} prompts, discarded {num_truncated} lower-scored prompts"
                    )
                    scores_filtered = scores_filtered[:max_prompts]
                    labels_filtered = labels_filtered[:max_prompts]
                    bboxes_filtered = bboxes_filtered[:max_prompts]
                
                # Log successful export with details
                max_score = scores_filtered[0]
                min_score = scores_filtered[-1]
                logger.info(
                    f"{img_name} | SUCCESS | "
                    f"Exported {len(scores_filtered)} prompts "
                    f"(filtered {num_filtered} below threshold {score_thr:.2f}, "
                    f"score range: {min_score:.3f}-{max_score:.3f})"
                )
            else:
                # All detections filtered out by threshold
                max_score = scores.max() if len(scores) > 0 else 0.0
                logger.warning(
                    f"{img_name} | ALL_BELOW_THRESHOLD | "
                    f"All {num_detections_before} detections below threshold {score_thr:.2f} "
                    f"(max score: {max_score:.3f})"
                )
                skipped_below_threshold += 1
                scores_filtered = np.array([])
                labels_filtered = np.array([])
                bboxes_filtered = np.zeros((0, 4))
            
            # Save to JSON (format compatible with prompt_loader.py)
            output_data = {
                'labels': labels_filtered.tolist(),
                'scores': scores_filtered.tolist(),
                'bboxes': bboxes_filtered.tolist()  # XYXY format
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f)
            
            successful += 1
            
            # Clear GPU cache to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"{img_name} | INFERENCE_ERROR | {type(e).__name__}: {e}")
            failed_inference_error.append(str(img_path))
            
            # Clear GPU cache even on error to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            continue
    
    # Summary logging
    logger.info("="*80)
    logger.info(f"SUMMARY for {dataset_name}/{split}")
    logger.info(f"Total images: {len(image_files)}")
    logger.info(f"Successfully exported: {successful}")
    logger.info(f"Skipped (already exists): {skipped_existing}")
    logger.info(f"Skipped (no detections): {skipped_no_detections}")
    logger.info(f"Skipped (all below threshold): {skipped_below_threshold}")
    logger.info(f"Failed (image errors): {len(failed_image_error)}")
    logger.info(f"Failed (inference errors): {len(failed_inference_error)}")
    
    if failed_image_error:
        logger.info(f"Images with errors: {failed_image_error}")
    if failed_inference_error:
        logger.info(f"Images with inference errors: {failed_inference_error}")
    
    logger.info("="*80)
    
    # Close logger handlers
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    
    total_skipped = skipped_existing + skipped_no_detections + skipped_below_threshold
    total_failed = len(failed_image_error) + len(failed_inference_error)
    
    return {
        'successful': successful,
        'skipped': total_skipped,
        'skipped_existing': skipped_existing,
        'skipped_no_detections': skipped_no_detections,
        'skipped_below_threshold': skipped_below_threshold,
        'failed': failed_image_error + failed_inference_error,
        'failed_image_error': len(failed_image_error),
        'failed_inference_error': len(failed_inference_error),
        'total': len(image_files)
    }


def main():
    args = parse_args()
    
    # Load RSPrompter model (only once)
    print(f"Loading RSPrompter from {args.checkpoint}")
    cfg = Config.fromfile(str(args.config))
    model = init_detector(cfg, str(args.checkpoint), device=args.device)
    print("✅ Model loaded")
    
    # Determine processing mode
    if args.dataset_root is not None:
        # Single dataset mode
        dataset_root = Path(args.dataset_root)
        dataset_name = extract_dataset_name(dataset_root)
        
        # Determine splits to process
        splits = args.splits if args.splits else [args.split]
        
        print(f"\n📦 Processing dataset: {dataset_name}")
        print(f"   Root: {dataset_root}")
        print(f"   Splits: {splits}")
        print(f"   Output base: {args.output_base_dir}")
        
        total_stats = {
            'successful': 0,
            'skipped': 0,
            'skipped_existing': 0,
            'skipped_no_detections': 0,
            'skipped_below_threshold': 0,
            'failed': [],
            'failed_image_error': 0,
            'failed_inference_error': 0
        }
        
        for split in splits:
            # Determine output directory
            if args.output_dir is not None and len(splits) == 1:
                # Only use custom output_dir when processing single split
                output_dir = args.output_dir
            else:
                output_dir = args.output_base_dir / dataset_name / split
            
            print(f"\n{'='*60}")
            print(f"Processing split: {split}")
            print(f"{'='*60}")
            
            try:
                stats = process_split(
                    model=model,
                    dataset_root=dataset_root,
                    dataset_name=dataset_name,
                    split=split,
                    output_dir=output_dir,
                    score_thr=args.score_thr,
                    max_prompts=args.max_prompts,
                    pick_dir=args.pick_dir if split == 'pick' else None
                )
                
                print(f"\n✅ Export complete for {dataset_name}/{split}!")
                print(f"   Successfully exported: {stats['successful']}/{stats['total']} images")
                if stats['skipped'] > 0:
                    print(f"   Skipped: {stats['skipped']} (existing: {stats['skipped_existing']}, "
                          f"no detections: {stats['skipped_no_detections']}, "
                          f"below threshold: {stats['skipped_below_threshold']})")
                if stats['failed_image_error'] > 0 or stats['failed_inference_error'] > 0:
                    print(f"   Failed: {len(stats['failed'])} (image errors: {stats['failed_image_error']}, "
                          f"inference errors: {stats['failed_inference_error']})")
                    if len(stats['failed']) <= 10:
                        print(f"   Failed files: {stats['failed']}")
                    else:
                        print(f"   Failed files (first 10): {stats['failed'][:10]}...")
                print(f"   Output directory: {output_dir}/")
                print(f"   Debug log: {output_dir}/rsprompt_debug_{dataset_name}_{split}.log")
                
                # Accumulate stats
                total_stats['successful'] += stats['successful']
                total_stats['skipped'] += stats['skipped']
                total_stats['skipped_existing'] += stats['skipped_existing']
                total_stats['skipped_no_detections'] += stats['skipped_no_detections']
                total_stats['skipped_below_threshold'] += stats['skipped_below_threshold']
                total_stats['failed'].extend(stats['failed'])
                total_stats['failed_image_error'] += stats['failed_image_error']
                total_stats['failed_inference_error'] += stats['failed_inference_error']
            
            except FileNotFoundError as e:
                print(f"\n⚠️  Warning: {e}, skipping {split}...")
                continue
        
        if len(splits) > 1:
            print(f"\n{'='*60}")
            print(f"✅ All splits complete for {dataset_name}!")
            print(f"   Total images exported: {total_stats['successful']}")
            print(f"   Total skipped: {total_stats['skipped']} "
                  f"(existing: {total_stats['skipped_existing']}, "
                  f"no detections: {total_stats['skipped_no_detections']}, "
                  f"below threshold: {total_stats['skipped_below_threshold']})")
            if total_stats['failed_image_error'] > 0 or total_stats['failed_inference_error'] > 0:
                print(f"   Total failed: {len(total_stats['failed'])} "
                      f"(image errors: {total_stats['failed_image_error']}, "
                      f"inference errors: {total_stats['failed_inference_error']})")
            print(f"{'='*60}")
    
    elif args.datasets_root is not None:
        # Batch processing mode
        datasets_root = Path(args.datasets_root)
        
        # Auto-detect datasets if not specified
        if args.dataset_names is None:
            dataset_names = [d.name for d in datasets_root.iterdir() 
                           if d.is_dir() and not d.name.startswith('.')]
            print(f"🔍 Auto-detected {len(dataset_names)} datasets: {dataset_names}")
        else:
            dataset_names = args.dataset_names
        
        # Determine splits to process
        splits = args.splits if args.splits else [args.split]
        
        print(f"\n📦 Batch processing mode")
        print(f"   Datasets root: {datasets_root}")
        print(f"   Datasets: {dataset_names}")
        print(f"   Splits: {splits}")
        print(f"   Output base: {args.output_base_dir}")
        
        total_stats = {
            'successful': 0, 
            'skipped': 0, 
            'skipped_existing': 0,
            'skipped_no_detections': 0,
            'skipped_below_threshold': 0,
            'failed': [],
            'failed_image_error': 0,
            'failed_inference_error': 0
        }
        
        for dataset_name in dataset_names:
            dataset_root = datasets_root / dataset_name
            if not dataset_root.exists():
                print(f"\n⚠️  Warning: Dataset not found: {dataset_root}, skipping...")
                continue
            
            print(f"\n{'='*60}")
            print(f"Processing dataset: {dataset_name}")
            print(f"{'='*60}")
            
            for split in splits:
                output_dir = args.output_base_dir / dataset_name / split
                
                try:
                    stats = process_split(
                        model=model,
                        dataset_root=dataset_root,
                        dataset_name=dataset_name,
                        split=split,
                        output_dir=output_dir,
                        score_thr=args.score_thr,
                        max_prompts=args.max_prompts,
                        pick_dir=None  # Not supported in batch mode
                    )
                    
                    print(f"  ✅ {split}: {stats['successful']}/{stats['total']} exported, "
                          f"{stats['skipped']} skipped, {len(stats['failed'])} failed")
                    total_stats['successful'] += stats['successful']
                    total_stats['skipped'] += stats['skipped']
                    total_stats['skipped_existing'] += stats['skipped_existing']
                    total_stats['skipped_no_detections'] += stats['skipped_no_detections']
                    total_stats['skipped_below_threshold'] += stats['skipped_below_threshold']
                    total_stats['failed'].extend(stats['failed'])
                    total_stats['failed_image_error'] += stats['failed_image_error']
                    total_stats['failed_inference_error'] += stats['failed_inference_error']
                
                except FileNotFoundError as e:
                    print(f"  ⚠️  {split}: {e}, skipping...")
                    continue
        
        print(f"\n{'='*60}")
        print(f"✅ Batch export complete!")
        print(f"   Total images exported: {total_stats['successful']}")
        print(f"   Total skipped: {total_stats['skipped']} "
              f"(existing: {total_stats['skipped_existing']}, "
              f"no detections: {total_stats['skipped_no_detections']}, "
              f"below threshold: {total_stats['skipped_below_threshold']})")
        print(f"   Total failed: {len(total_stats['failed'])} "
              f"(image errors: {total_stats['failed_image_error']}, "
              f"inference errors: {total_stats['failed_inference_error']})")
        if total_stats['failed'] and len(total_stats['failed']) <= 20:
            print(f"   Failed files: {total_stats['failed']}")
        elif total_stats['failed']:
            print(f"   Failed files (first 20): {total_stats['failed'][:20]}...")
        print(f"   Output directory: {args.output_base_dir}/*/")
        print(f"   Debug logs: {args.output_base_dir}/*/rsprompt_debug_*.log")
    
    else:
        print("❌ Error: Must specify either --dataset-root (single mode) or --datasets-root (batch mode)")
        sys.exit(1)


if __name__ == '__main__':
    main()
