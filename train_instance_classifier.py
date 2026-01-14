#!/usr/bin/env python3
"""
Train Instance-level Tree Species Classifier

Simple training loop with visualization and wandb integration.

1. Load instances from manifest
2. Train classifier
3. Evaluate and visualize
4. Save model

Generates qualitative visualizations and logs to wandb every N epochs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from PIL import Image
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.instance_dataset import InstanceTransform, TreeInstanceDataset, create_instance_dataloaders
from utils.instance_classifier import create_instance_classifier
from utils.instance_visualizer import (
    load_treeai_classes,
    instances_to_semantic_map,
    predict_instances_for_image,
    visualize_instance_predictions,
    plot_confusion_matrix_semantic,
    compute_group_f1_scores,
)
from utils.instance_evaluation import evaluate_instance_segmentation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train instance-level tree species classifier')
    
    # Config file argument (highest priority)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file (if provided, overrides other args)')
    
    # Data arguments
    parser.add_argument('--train-manifest', type=str, default=None,
                        help='Path to training manifest JSONL file')
    parser.add_argument('--val-manifest', type=str, default=None,
                        help='Path to validation manifest JSONL file')
    parser.add_argument('--test-manifest', type=str, default=None,
                        help='Path to test manifest JSONL file')
    parser.add_argument('--output-root', type=str, default=None,
                        help='Root directory for instance data (contains mask files)')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of tree species classes')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, default=None,
                        choices=['simple', 'masked'],
                        help='Type of classifier model')
    parser.add_argument('--encoder', type=str, default=None,
                        help='Encoder architecture (timm model name)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='Use ImageNet pretrained weights (overrides config)')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                        help='Disable ImageNet pretrained weights (overrides config)')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout rate')
    parser.add_argument('--use-mask-features', dest='use_mask_features', action='store_true',
                        help='Use mask features (only for masked model)')
    parser.add_argument('--no-use-mask-features', dest='use_mask_features', action='store_false',
                        help='Disable mask features (overrides config)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=None,
                        help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of data loading workers')
    
    # Data filtering
    parser.add_argument('--min-purity', type=float, default=None,
                        help='Minimum purity threshold for instances')
    parser.add_argument('--min-area', type=int, default=None,
                        help='Minimum mask area in pixels')
    parser.add_argument('--image-size', type=int, nargs=2, default=None,
                        help='Image size (H W)')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for checkpoints')
    parser.add_argument('--save-freq', type=int, default=None,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Visualization arguments
    parser.add_argument('--viz-freq', type=int, default=None,
                        help='Generate visualizations every N epochs (0 to disable)')
    parser.add_argument('--pick-root', type=str, default=None,
                        help='Path to pick dataset for visualization')
    parser.add_argument('--classes-yaml', type=str, default=None,
                        help='Path to treeAI_classes.yaml')
    
    # Test arguments
    parser.add_argument('--auto-test-on-finish', dest='auto_test_on_finish', action='store_true', default=None,
                        help='Automatically test on test set when training finishes')
    parser.add_argument('--no-auto-test-on-finish', dest='auto_test_on_finish', action='store_false',
                        help='Disable automatic testing on finish')
    parser.add_argument('--test-image-dir', type=str, default=None,
                        help='Directory containing test images')
    parser.add_argument('--test-label-dir', type=str, default=None,
                        help='Directory containing test labels')
    
    # Full Validation arguments
    parser.add_argument('--val-image-dir', type=str, default=None,
                        help='Directory containing validation images (for full evaluation)')
    parser.add_argument('--val-label-dir', type=str, default=None,
                        help='Directory containing validation labels (for full evaluation)')
    
    # Wandb arguments
    parser.add_argument('--use-wandb', dest='use_wandb', action='store_true', default=None,
                        help='Enable wandb logging')
    parser.add_argument('--no-wandb', dest='use_wandb', action='store_false',
                        help='Disable wandb logging')
    parser.add_argument('--wandb-project', type=str, default=None,
                        help='Wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='Wandb entity/team name')
    parser.add_argument('--wandb-name', type=str, default=None,
                        help='Wandb run name (default: auto-generated)')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    
    parser.set_defaults(pretrained=None, use_mask_features=None)
    
    return parser.parse_args()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    model_type: str,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        if model_type == 'masked':
            masks = batch['mask'].to(device)
            logits = model(images, masks)
        else:
            logits = model(images)
        
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'acc': total_correct / total_samples,
        })
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
    }


def generate_visualizations(
    model: nn.Module,
    pick_root: Path,
    output_root: Path,
    output_dir: Path,
    class_names: Dict[int, str],
    class_colors: Dict[int, Tuple[int, int, int]],
    class_groups: Dict[str, List[int]],
    model_type: str,
    device: torch.device,
    image_size: Tuple[int, int],
    use_wandb: bool = False,
    epoch: Optional[int] = None,
):
    """Generate qualitative visualizations for pick dataset."""
    model.eval()
    
    # Find pick images and manifest
    pick_image_dir = pick_root / 'images'
    pick_label_dir = pick_root / 'labels'
    pick_manifest = output_root / 'pick' / 'instances_manifest.jsonl'
    
    if not pick_manifest.exists():
        print(f"⚠️  Pick manifest not found: {pick_manifest}")
        return
    
    # Load pick instances
    instances_by_image = {}
    with pick_manifest.open('r') as f:
        for line in f:
            if line.strip():
                inst = json.loads(line)
                image_name = inst['image']
                if image_name not in instances_by_image:
                    instances_by_image[image_name] = []
                instances_by_image[image_name].append(inst)
    
    # Setup transform
    transform = InstanceTransform(image_size=image_size, augment=False)
    
    # Process images
    all_images = []
    all_gt_masks = []
    all_pred_masks = []
    all_bboxes = []
    
    image_names = sorted(instances_by_image.keys())[:10]  # First 10 images
    
    for image_name in tqdm(image_names, desc='Generating predictions'):
        image_path = list(pick_image_dir.glob(f"{image_name}.*"))[0]
        label_path = list(pick_label_dir.glob(f"{image_name}.*"))[0]
        
        # Load GT
        gt_mask = np.array(Image.open(label_path)).astype(np.int32)
        
        # Predict instances
        image_rgb, predictions, scores, masks, bboxes = predict_instances_for_image(
            model=model,
            image_path=str(image_path),
            manifest_instances=instances_by_image[image_name],
            output_root=output_root,
            transform=transform,
            device=device,
            model_type=model_type,
            mask_base_path=pick_manifest.parent,
        )
        
        # Convert to semantic map
        pred_mask = instances_to_semantic_map(
            image_shape=gt_mask.shape,
            instance_masks=masks,
            instance_labels=predictions,
            instance_scores=scores,
        )
        
        all_images.append(image_rgb)
        all_gt_masks.append(gt_mask)
        all_pred_masks.append(pred_mask)
        all_bboxes.append(bboxes)
    
    # Visualize first 5 and last 5
    visualize_instance_predictions(
        images=all_images,
        gt_masks=all_gt_masks,
        pred_masks=all_pred_masks,
        bboxes_list=all_bboxes,
        class_names=class_names,
        class_colors=class_colors,
        num_samples=5,
        which='first',
        save_path=output_dir / 'qual_first.jpg',
    )
    
    visualize_instance_predictions(
        images=all_images,
        gt_masks=all_gt_masks,
        pred_masks=all_pred_masks,
        bboxes_list=all_bboxes,
        class_names=class_names,
        class_colors=class_colors,
        num_samples=5,
        which='last',
        save_path=output_dir / 'qual_last.jpg',
    )
    
    # Compute metrics
    y_true = np.concatenate([mask.flatten() for mask in all_gt_masks])
    y_pred = np.concatenate([mask.flatten() for mask in all_pred_masks])
    
    # Confusion matrices
    plot_confusion_matrix_semantic(
        y_true, y_pred, class_names,
        output_dir / 'confusion.jpg',
        normalize=None,
        title='Confusion Matrix (Counts)'
    )
    plot_confusion_matrix_semantic(
        y_true, y_pred, class_names,
        output_dir / 'confusion_norm_true.jpg',
        normalize='true',
        title='Confusion Matrix (Normalized by True)'
    )
    plot_confusion_matrix_semantic(
        y_true, y_pred, class_names,
        output_dir / 'confusion_norm_pred.jpg',
        normalize='pred',
        title='Confusion Matrix (Normalized by Pred)'
    )
    plot_confusion_matrix_semantic(
        y_true, y_pred, class_names,
        output_dir / 'confusion_norm_all.jpg',
        normalize='all',
        title='Confusion Matrix (Normalized All)'
    )
    
    # Compute group F1 scores
    # Exclude order/genus level classes from group F1 calculation
    non_species_classes = [
        37, 60,  # order level (coniferous, deciduous)
        5, 11, 43, 50, 56, 58, 59, 61,  # genus level (betula sp., picea sp., etc.)
    ]
    group_f1 = compute_group_f1_scores(
        y_true, y_pred, class_groups, 
        ignore_classes=non_species_classes
    )
    
    print("\n📊 Group F1 Scores (pick dataset):")
    for group_name, f1 in group_f1.items():
        print(f"  {group_name:20s}: {f1:.4f}")
    
    # Log to wandb
    if use_wandb:
        import wandb
        
        # Log images
        log_dict = {
            'qualitative/first': wandb.Image(str(output_dir / 'qual_first.jpg')),
            'qualitative/last': wandb.Image(str(output_dir / 'qual_last.jpg')),
            'confusion/raw': wandb.Image(str(output_dir / 'confusion.jpg')),
            'confusion/norm_true': wandb.Image(str(output_dir / 'confusion_norm_true.jpg')),
        }
        
        # Log group F1 scores
        for group_name, f1 in group_f1.items():
            log_dict[f'group_f1/{group_name}'] = f1
        
        if epoch is not None:
            wandb.log(log_dict, step=epoch)
        else:
            wandb.log(log_dict)
    
    print(f"✅ Visualizations saved to: {output_dir}")


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_type: str,
    num_classes: int,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # Per-class accuracy
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    
    pbar = tqdm(dataloader, desc='Validation')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        if model_type == 'masked':
            masks = batch['mask'].to(device)
            logits = model(images, masks)
        else:
            logits = model(images)
        
        loss = criterion(logits, labels)
        
        # Metrics
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)
        
        # Per-class accuracy
        for i in range(len(labels)):
            label = labels[i].item()
            class_total[label] += 1
            if preds[i] == labels[i]:
                class_correct[label] += 1
        
        pbar.set_postfix({
            'loss': loss.item(),
            'acc': total_correct / total_samples,
        })
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    # Compute per-class accuracy
    per_class_acc = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            per_class_acc[i] = class_correct[i] / class_total[i]
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'per_class_accuracy': per_class_acc,
        'mean_class_accuracy': np.mean([acc for acc in per_class_acc.values()]),
    }


def validate_full_segmentation(
    model: nn.Module,
    instances_by_image: Dict[str, List[Dict]],
    image_dir: Path,
    label_dir: Path,
    output_root: Path,
    class_names: Dict[int, str],
    class_groups: Dict[str, List[int]],
    device: torch.device,
    model_type: str,
    image_size: Tuple[int, int],
    epoch: int,
    output_dir: Path,
    use_wandb: bool = False,
    excluded_classes: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Run full semantic segmentation validation (matches test protocol)."""
    print(f"\n🧪 Running Full Image Validation (Epoch {epoch + 1})...")
    
    # Create epoch output directory
    val_out_dir = output_dir / f'val_epoch_{epoch + 1}'
    val_out_dir.mkdir(parents=True, exist_ok=True)
    
    results = evaluate_instance_segmentation(
        model=model,
        instances_by_image=instances_by_image,
        image_dir=image_dir,
        label_dir=label_dir,
        output_root=output_root,
        class_names=class_names,
        class_groups=class_groups,
        device=device,
        model_type=model_type,
        image_size=image_size,
        max_samples=None, # Validate on all available validation images
        desc=f"Val Epoch {epoch+1}",
        excluded_classes=excluded_classes,
    )
    
    if not results:
        return {}

    # Extract arrays for visualization/confusion matrix but don't keep them in memory if not needed
    y_true = results.pop('y_true')
    y_pred = results.pop('y_pred')
    all_images = results.pop('all_images')
    all_gt_masks = results.pop('all_gt_masks')
    all_pred_masks = results.pop('all_pred_masks')
    all_bboxes = results.pop('all_bboxes')
    
    # Save metrics
    with (val_out_dir / 'val_results.json').open('w') as f:
        json.dump(results, f, indent=2)
        
    # Generate confusion matrix
    plot_confusion_matrix_semantic(
        y_true, y_pred, class_names,
        val_out_dir / 'confusion_norm_true.jpg',
        normalize='true',
        title=f'Val Epoch {epoch+1} Confusion Matrix'
    )
    plot_confusion_matrix_semantic(
        y_true, y_pred, class_names,
        val_out_dir / 'confusion_with_bg.jpg',
        normalize='true',  # 0-100 percentage normalized by true labels
        title=f'Val Epoch {epoch+1} Confusion Matrix (with Background)',
        include_background=True
    )
    
    # Log metrics to wandb
    if use_wandb:
        import wandb
        log_dict = {
            'val_full/F1-avg': results['F1-avg'],
            'val_full/F1-avg-wo0': results['F1-avg-wo0'],
            'val_full/mean_iou': results['mean_iou'],
            'val_full/pixel_accuracy': results['pixel_accuracy_incl_bg'],
            'val_full/species_only_f1': results['species_only_f1'],
            'val_full/species_only_iou': results['species_only_iou'],
        }
        # Add group F1s
        for k, v in results['group_f1_scores'].items():
            log_dict[f'val_full/group_f1/{k}'] = v
            
        wandb.log(log_dict, step=epoch+1)
        
    print(f"  F1-avg: {results['F1-avg']:.4f} | F1-avg-wo0: {results['F1-avg-wo0']:.4f} | mIoU: {results['mean_iou']:.4f} | Species F1: {results['species_only_f1']:.4f}")
    
    return results


@torch.no_grad()
def test_on_test_set(
    checkpoint_path: Path,
    test_manifest: str,
    output_root: str,
    test_image_dir: str,
    test_label_dir: str,
    classes_yaml: str,
    output_dir: Path,
    device: torch.device,
    num_viz_samples: int = 20,
):
    """Test trained model on actual test set using full evaluation."""
    print("\n" + "=" * 80)
    print("Testing on True Test Set")
    print("=" * 80)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_args = checkpoint['args']
    
    # Load class information
    class_names, class_colors, class_groups = load_treeai_classes(classes_yaml)
    
    # Create model
    model = create_instance_classifier(
        num_classes=checkpoint_args['num_classes'],
        model_type=checkpoint_args['model_type'],
        encoder_name=checkpoint_args['encoder'],
        pretrained=False,
        dropout=checkpoint_args['dropout'],
        use_mask_features=checkpoint_args.get('use_mask_features', False),
    )
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    # Load test instances
    print(f"\nLoading test instances from: {test_manifest}")
    instances_by_image = {}
    with open(test_manifest, 'r') as f:
        for line in f:
            if line.strip():
                inst = json.loads(line)
                image_name = inst['image']
                if image_name not in instances_by_image:
                    instances_by_image[image_name] = []
                instances_by_image[image_name].append(inst)
    
    # Run evaluation
    output_dir.mkdir(parents=True, exist_ok=True)
    image_size = checkpoint_args.get('image_size', [224, 224])
    excluded_classes = checkpoint_args.get('excluded_classes', None)
    
    results = evaluate_instance_segmentation(
        model=model,
        instances_by_image=instances_by_image,
        image_dir=Path(test_image_dir),
        label_dir=Path(test_label_dir),
        output_root=Path(output_root),
        class_names=class_names,
        class_groups=class_groups,
        device=device,
        model_type=checkpoint_args['model_type'],
        image_size=tuple(image_size),
        desc="Testing",
        excluded_classes=excluded_classes,
    )
    
    if not results:
        print("Evaluation failed or returned no results.")
        return

    # Extract arrays for visualization
    y_true = results.pop('y_true')
    y_pred = results.pop('y_pred')
    all_images = results.pop('all_images')
    all_gt_masks = results.pop('all_gt_masks')
    all_pred_masks = results.pop('all_pred_masks')
    all_bboxes = results.pop('all_bboxes')
    
    # Add checkpoint info to results
    results['checkpoint'] = str(checkpoint_path)
    
    # Print Summary
    print(f"\n📊 Overall Metrics:")
    print(f"  F1-avg (with bg):    {results['F1-avg']:.4f}")
    print(f"  F1-avg-wo0 (no bg):  {results['F1-avg-wo0']:.4f}")
    print(f"  Mean Precision:      {results['mean_precision']:.4f}")
    print(f"  Mean Recall:         {results['mean_recall']:.4f}")
    print(f"  Mean IoU (mIoU):     {results['mean_iou']:.4f}")
    
    if results['species_only_f1'] > 0:
        print(f"\n🌲 Species-Only Metrics ({results['num_species_classes']} species):")
        print(f"  Mean F1 Score:    {results['species_only_f1']:.4f}")
        print(f"  Mean IoU (mIoU):  {results['species_only_iou']:.4f}")

    # Save results
    with (output_dir / 'test_results.json').open('w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {output_dir / 'test_results.json'}")
    
    # Generate confusion matrices
    print("\nGenerating confusion matrices...")
    plot_confusion_matrix_semantic(
        y_true, y_pred, class_names,
        output_dir / 'confusion.jpg',
        normalize=None,
        title='Test Set Confusion Matrix (Counts)'
    )
    plot_confusion_matrix_semantic(
        y_true, y_pred, class_names,
        output_dir / 'confusion_with_bg.jpg',
        normalize='true',  # 0-100 percentage normalized by true labels
        title='Test Set Confusion Matrix (with Background)',
        include_background=True
    )
    plot_confusion_matrix_semantic(
        y_true, y_pred, class_names,
        output_dir / 'confusion_norm_true.jpg',
        normalize='true',
        title='Test Set Confusion Matrix (Normalized by True)'
    )
    
    # Generate visualizations
    if len(all_images) > 0:
        print("\nGenerating visualizations...")
        num_viz = min(num_viz_samples, len(all_images))
        
        # First N samples
        visualize_instance_predictions(
            images=all_images[:num_viz],
            gt_masks=all_gt_masks[:num_viz],
            pred_masks=all_pred_masks[:num_viz],
            bboxes_list=all_bboxes[:num_viz],
            class_names=class_names,
            class_colors=class_colors,
            num_samples=min(10, num_viz),
            which='first',
            save_path=output_dir / 'qualitative_first.jpg',
        )
        
        # Last N samples
        if len(all_images) > num_viz:
            visualize_instance_predictions(
                images=all_images[-num_viz:],
                gt_masks=all_gt_masks[-num_viz:],
                pred_masks=all_pred_masks[-num_viz:],
                bboxes_list=all_bboxes[-num_viz:],
                class_names=class_names,
                class_colors=class_colors,
                num_samples=min(10, num_viz),
                which='last',
                save_path=output_dir / 'qualitative_last.jpg',
            )
    
    print("\n" + "=" * 80)
    print("Testing complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


DEFAULT_ARGS = {
    'model_type': 'simple',
    'encoder': 'resnet50',
    'pretrained': True,
    'dropout': 0.3,
    'use_mask_features': False,
    'epochs': 50,
    'batch_size': 32,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'num_workers': 4,
    'min_purity': 0.7,
    'min_area': 100,
    'image_size': [224, 224],
    'output_dir': './outputs/instance_classifier',
    'save_freq': 5,
    'viz_freq': 0,
    'use_wandb': False,
    'device': 'cuda',
    'auto_test_on_finish': True,
}


def _apply_default_args(args: argparse.Namespace) -> argparse.Namespace:
    for key, value in DEFAULT_ARGS.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    # Ensure image_size stored as list (coming from argparse tuple)
    if args.image_size is not None:
        args.image_size = list(args.image_size)
    return args


def load_config_and_merge_args(args: argparse.Namespace) -> argparse.Namespace:
    """Load config from YAML and merge with command-line args."""
    if args.config is None:
        # No config file, validate required args
        required = ['train_manifest', 'val_manifest', 'output_root', 'num_classes', 'classes_yaml']
        missing = [arg for arg in required if getattr(args, arg.replace('-', '_')) is None]
        if missing:
            raise ValueError(f"Missing required arguments: {missing}")
        return _apply_default_args(args)
    
    # Load config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Helper to set arg if not provided on command line
    def get_config_value(path: str):
        value = config
        for key in path.split('.'):
            if not isinstance(value, dict) or key not in value:
                return None
            value = value[key]
        return value

    def set_if_none(arg_name, config_path):
        if getattr(args, arg_name) is not None:
            return
        value = get_config_value(config_path)
        if value is not None:
            setattr(args, arg_name, value)
    
    # Dataset config
    set_if_none('train_manifest', 'dataset.train_manifest')
    set_if_none('val_manifest', 'dataset.val_manifest')
    set_if_none('test_manifest', 'dataset.test_manifest')
    set_if_none('output_root', 'dataset.output_root')
    set_if_none('num_classes', 'dataset.num_classes')
    set_if_none('classes_yaml', 'dataset.classes_yaml')
    set_if_none('min_purity', 'dataset.min_purity')
    set_if_none('min_area', 'dataset.min_area')
    set_if_none('batch_size', 'dataset.batch_size')
    set_if_none('num_workers', 'dataset.num_workers')
    
    # Handle image_size (special case: list)
    if args.image_size is None:
        image_size = config.get('dataset', {}).get('image_size', [224, 224])
        args.image_size = image_size if isinstance(image_size, list) else [image_size, image_size]
    
    # Handle excluded_classes (special case: list)
    if not hasattr(args, 'excluded_classes') or args.excluded_classes is None:
        excluded_classes = config.get('dataset', {}).get('excluded_classes', None)
        args.excluded_classes = excluded_classes
    
    # Model config
    set_if_none('model_type', 'model.model_type')
    set_if_none('encoder', 'model.encoder')
    set_if_none('pretrained', 'model.pretrained')
    set_if_none('dropout', 'model.dropout')
    set_if_none('use_mask_features', 'model.use_mask_features')
    
    # Training config
    set_if_none('epochs', 'training.epochs')
    set_if_none('lr', 'training.lr')
    set_if_none('weight_decay', 'training.weight_decay')
    set_if_none('save_freq', 'training.save_freq')
    set_if_none('output_dir', 'training.output_dir')
    set_if_none('resume', 'training.resume_from')
    
    # Visualization config
    if config.get('visualization', {}).get('enabled', True):
        set_if_none('viz_freq', 'visualization.viz_freq')
        set_if_none('pick_root', 'visualization.pick_root')
    
    # Wandb config
    if args.use_wandb is None:
        args.use_wandb = config.get('wandb', {}).get('enabled', False)
    
    if args.use_wandb:
        set_if_none('wandb_project', 'wandb.project')
        set_if_none('wandb_entity', 'wandb.entity')
        set_if_none('wandb_name', 'wandb.run_name')
    
    # Device
    set_if_none('device', 'device')
    
    # Test config
    if args.auto_test_on_finish is None:
        args.auto_test_on_finish = config.get('testing', {}).get('auto_test_on_finish', True)
    set_if_none('test_image_dir', 'testing.test_image_dir')
    set_if_none('test_label_dir', 'testing.test_label_dir')
    set_if_none('val_image_dir', 'validation.val_image_dir')
    set_if_none('val_label_dir', 'validation.val_label_dir')
    
    return _apply_default_args(args)


def main():
    args = parse_args()
    
    # Load config and merge with command-line args
    args = load_config_and_merge_args(args)
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Instance-level Tree Species Classification")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model type: {args.model_type}")
    print(f"Encoder: {args.encoder}")
    print(f"Num classes: {args.num_classes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Min purity: {args.min_purity}")
    print(f"Min area: {args.min_area}")
    if args.use_wandb:
        print(f"Wandb: enabled (project: {args.wandb_project}, entity: {args.wandb_entity})")
    print("=" * 80)
    
    # Save arguments
    with (output_dir / 'args.json').open('w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load class information
    class_names, class_colors, class_groups = load_treeai_classes(args.classes_yaml)
    print(f"\nLoaded {len(class_names)} classes from {args.classes_yaml}")
    print(f"Class groups: {list(class_groups.keys())}")
    
    # Initialize wandb
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args),
        )
        print("✅ Wandb initialized")
    
    # Prepare full validation data
    val_instances_by_image = {}
    do_full_validation = False
    
    # Infer validation directories if not provided
    if not args.val_image_dir or not args.val_label_dir:
        # Try to infer from pick_root or output_root
        if args.pick_root:
            # Typical structure: .../dataset_name/pick -> .../dataset_name/val
            val_root = Path(args.pick_root).parent / 'val'
            if (val_root / 'images').exists() and (val_root / 'labels').exists():
                args.val_image_dir = args.val_image_dir or str(val_root / 'images')
                args.val_label_dir = args.val_label_dir or str(val_root / 'labels')
                print(f"Inferred validation dirs: {args.val_image_dir}, {args.val_label_dir}")
        elif args.output_root:
            # Try output_root (dataset root)
             val_root = Path(args.output_root) / 'val'
             if (val_root / 'images').exists() and (val_root / 'labels').exists():
                args.val_image_dir = args.val_image_dir or str(val_root / 'images')
                args.val_label_dir = args.val_label_dir or str(val_root / 'labels')
                print(f"Inferred validation dirs from output_root: {args.val_image_dir}, {args.val_label_dir}")

    if args.val_image_dir and args.val_label_dir and Path(args.val_image_dir).exists() and Path(args.val_label_dir).exists():
        print(f"\nLoading validation instances for full evaluation from {args.val_manifest}...")
        with open(args.val_manifest, 'r') as f:
            for line in f:
                if line.strip():
                    inst = json.loads(line)
                    image_name = inst['image']
                    if image_name not in val_instances_by_image:
                        val_instances_by_image[image_name] = []
                    val_instances_by_image[image_name].append(inst)
        print(f"Loaded {len(val_instances_by_image)} validation images for full evaluation")
        do_full_validation = True
    else:
        print("\n⚠️  Full validation disabled (missing val_image_dir/val_label_dir)")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_instance_dataloaders(
        train_manifest=Path(args.train_manifest),
        val_manifest=Path(args.val_manifest),
        output_root=Path(args.output_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_purity=args.min_purity,
        min_area=args.min_area,
        image_size=tuple(args.image_size),
        excluded_classes=getattr(args, 'excluded_classes', None),
    )
    
    # Create model
    print("\nCreating model...")
    model = create_instance_classifier(
        num_classes=args.num_classes,
        model_type=args.model_type,
        encoder_name=args.encoder,
        pretrained=args.pretrained,
        dropout=args.dropout,
        use_mask_features=args.use_mask_features,
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        print(f"\nResuming from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
    
    # Training loop
    print("\nStarting training...")
    print("=" * 80)
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 80)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, args.model_type
        )
        
        print(f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Train Acc: {train_metrics['accuracy']:.4f}")
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, args.model_type, args.num_classes
        )
        
        print(f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val Mean Class Acc: {val_metrics['mean_class_accuracy']:.4f}")
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'val/mean_class_accuracy': val_metrics['mean_class_accuracy'],
                'lr': optimizer.param_groups[0]['lr'],
            }, step=epoch + 1)
        
        # Update learning rate
        scheduler.step()
        
        # Generate visualizations
        if args.viz_freq > 0 and (epoch + 1) % args.viz_freq == 0 and args.pick_root:
            print(f"\nGenerating visualizations (epoch {epoch + 1})...")
            generate_visualizations(
                model=model,
                pick_root=Path(args.pick_root),
                output_root=Path(args.output_root),
                output_dir=output_dir / f'qualitative/epoch_{epoch + 1:03d}',
                class_names=class_names,
                class_colors=class_colors,
                class_groups=class_groups,
                model_type=args.model_type,
                device=device,
                image_size=tuple(args.image_size),
                use_wandb=args.use_wandb,
                epoch=epoch + 1,
            )

        # Run Full Validation (matches test set protocol)
        if do_full_validation and args.viz_freq > 0 and (epoch + 1) % args.viz_freq == 0:
             validate_full_segmentation(
                model=model,
                instances_by_image=val_instances_by_image,
                image_dir=Path(args.val_image_dir),
                label_dir=Path(args.val_label_dir),
                output_root=Path(args.output_root),
                class_names=class_names,
                class_groups=class_groups,
                device=device,
                model_type=args.model_type,
                image_size=tuple(args.image_size),
                epoch=epoch,
                output_dir=output_dir,
                use_wandb=args.use_wandb,
                excluded_classes=getattr(args, 'excluded_classes', None),
            )
        
        # Save checkpoint
        is_best = val_metrics['accuracy'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['accuracy']
        
        # Always prepare checkpoint dict (for best/final/last)
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_val_acc': best_val_acc,
            'args': vars(args),
        }
        
        # Save best model (automatically when validation accuracy improves)
        if is_best:
            best_path = output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path} (acc: {best_val_acc:.4f})")
        
        # Save last checkpoint (for resume capability, overwrites previous)
        last_path = output_dir / 'last_checkpoint.pt'
        torch.save(checkpoint, last_path)
    
    # Save final model
    final_checkpoint = {
        'epoch': args.epochs - 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_val_acc': best_val_acc,
        'args': vars(args),
    }
    final_path = output_dir / 'final_model.pt'
    torch.save(final_checkpoint, final_path)
    print(f"\nSaved final model: {final_path}")
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("=" * 80)
    
    # Test on test set
    if args.auto_test_on_finish and args.test_manifest:
        print("\n🧪 Auto-testing on test set...")
        
        # Infer test image/label directories if not provided
        if not args.test_image_dir or not args.test_label_dir:
            # Try to infer from pick_root pattern
            if args.pick_root:
                # /path/to/12_RGB_SemSegm_640_fL/pick -> /path/to/12_RGB_SemSegm_640_fL/test
                test_root = Path(args.pick_root).parent / 'test'
                args.test_image_dir = args.test_image_dir or str(test_root / 'images')
                args.test_label_dir = args.test_label_dir or str(test_root / 'labels')
            else:
                print("⚠️  Test image/label directories not specified, skipping test.")
                args.auto_test_on_finish = False
        
        if args.auto_test_on_finish:
            try:
                test_on_test_set(
                    checkpoint_path=output_dir / 'best_model.pt',
                    test_manifest=args.test_manifest,
                    output_root=args.output_root,
                    test_image_dir=args.test_image_dir,
                    test_label_dir=args.test_label_dir,
                    classes_yaml=args.classes_yaml,
                    output_dir=output_dir / 'test_results',
                    device=device,
                    num_viz_samples=20,
                )
            except Exception as e:
                print(f"❌ Test failed: {e}")
                import traceback
                traceback.print_exc()
    elif args.auto_test_on_finish and not args.test_manifest:
        print("⚠️  Auto-test enabled but test_manifest not specified, skipping test.")
    
    # Close wandb
    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()
