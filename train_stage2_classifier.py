#!/usr/bin/env python3
"""
Stage 2: Decoupled Classifier Training

Freeze backbone from Stage 1 checkpoint, re-train classifier head with
class-balanced sampling to fix tail class performance.

This follows Kang et al. (ICLR 2020) "Decoupling Representation and Classifier":
- Stage 1: Normal training learns good features (backbone)
- Stage 2: Freeze backbone, retrain classifier with balanced sampling

Usage:
    python train_stage2_classifier.py \
        --stage1-checkpoint ./outputs/instance_classifier/best_model.pt \
        --output-dir ./outputs/instance_classifier_stage2
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.instance_dataset import InstanceTransform, TreeInstanceDataset
from utils.instance_classifier import create_instance_classifier
from utils.instance_visualizer import load_treeai_classes

# Import test function from Stage 1 script
from train_instance_classifier import test_on_test_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Stage 2: Re-train classifier with class-balanced sampling'
    )
    
    # Required: Stage 1 checkpoint
    parser.add_argument(
        '--stage1-checkpoint', type=str, required=True,
        help='Path to Stage 1 best_model.pt checkpoint'
    )
    
    # Dataset (can override from checkpoint)
    parser.add_argument('--train-manifest', type=str, default=None)
    parser.add_argument('--val-manifest', type=str, default=None)
    parser.add_argument('--output-root', type=str, default=None)
    parser.add_argument('--classes-yaml', type=str, default=None)
    
    # Stage 2 training params
    parser.add_argument('--epochs', type=int, default=30,
                        help='Stage 2 epochs (default: 30, usually shorter)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for classifier head (higher than Stage 1)')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    
    # Output
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for Stage 2 checkpoints')
    parser.add_argument('--save-freq', type=int, default=5)
    
    # Sampling strategy
    parser.add_argument('--sampling', type=str, default='smoothed',
                        choices=['balanced', 'sqrt', 'effective', 'smoothed'],
                        help='Class-balanced sampling strategy (smoothed recommended)')
    parser.add_argument('--effective-beta', type=float, default=0.999,
                        help='Beta for effective number sampling (Cui et al.)')
    parser.add_argument('--smoothing-alpha', type=float, default=0.5,
                        help='Smoothing exponent for "smoothed" strategy: w=1/n^alpha. '
                             '0=no rebalancing, 0.5=sqrt, 1=full balanced. Default 0.5')
    
    # Optional: reinitialize classifier head
    parser.add_argument('--reinit-classifier', action='store_true',
                        help='Reinitialize classifier weights (recommended)')
    
    # Wandb
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='tree_instance_stage2')
    parser.add_argument('--wandb-entity', type=str, default=None)
    
    # Test arguments (same as Stage 1)
    parser.add_argument('--auto-test-on-finish', dest='auto_test_on_finish',
                        action='store_true', default=True,
                        help='Automatically test on test set when training finishes')
    parser.add_argument('--no-auto-test-on-finish', dest='auto_test_on_finish',
                        action='store_false',
                        help='Disable automatic testing on finish')
    parser.add_argument('--test-manifest', type=str, default=None,
                        help='Path to test manifest (inferred from train_manifest if not specified)')
    parser.add_argument('--test-image-dir', type=str, default=None,
                        help='Directory containing test images')
    parser.add_argument('--test-label-dir', type=str, default=None,
                        help='Directory containing test labels')
    
    parser.add_argument('--device', type=str, default='cuda')
    
    return parser.parse_args()


def compute_class_weights(
    dataset: TreeInstanceDataset,
    strategy: str = 'smoothed',
    beta: float = 0.999,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Compute per-sample weights for class-balanced sampling.
    
    Strategies and their sources:
    - 'balanced': w = 1/n (classic inverse frequency, very aggressive)
    - 'sqrt': w = 1/√n (common heuristic, moderate)
    - 'effective': w = (1-β)/(1-β^n) from Cui et al. CVPR 2019
      "Class-Balanced Loss Based on Effective Number of Samples"
    - 'smoothed': w = 1/n^α (recommended, α controls strength)
      α=0: no rebalancing, α=0.5: sqrt, α=1: full balanced
      From Kang et al. ICLR 2020 "Decoupling Representation and Classifier"
    
    Returns sample weights (one per instance).
    """
    # Count classes
    labels = [inst['label'] - 1 for inst in dataset.instances]  # 0-indexed
    class_counts = Counter(labels)
    num_classes = max(labels) + 1
    
    # Compute class weights
    class_weights = torch.zeros(num_classes)
    for cls, count in class_counts.items():
        if strategy == 'balanced':
            class_weights[cls] = 1.0 / count
        elif strategy == 'sqrt':
            class_weights[cls] = 1.0 / np.sqrt(count)
        elif strategy == 'effective':
            # Effective number: E_n = (1 - beta^n) / (1 - beta)
            # From Cui et al. CVPR 2019
            effective_n = (1 - beta ** count) / (1 - beta)
            class_weights[cls] = 1.0 / effective_n
        elif strategy == 'smoothed':
            # Smoothed rebalancing: w = 1 / n^alpha
            # alpha=0: uniform (no rebalancing)
            # alpha=0.5: sqrt (moderate)
            # alpha=1: fully balanced (aggressive)
            # From Kang et al. ICLR 2020
            class_weights[cls] = 1.0 / (count ** alpha)
    
    # Normalize so weights sum to num_samples
    class_weights = class_weights / class_weights.sum() * len(labels)
    
    # Create per-sample weights
    sample_weights = torch.tensor([class_weights[label].item() for label in labels])
    
    # Print detailed stats
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    bulk_cls, bulk_count = sorted_classes[0]
    tail_cls, tail_count = sorted_classes[-1]
    
    # Expected sampling ratio after rebalancing
    bulk_weight = class_weights[bulk_cls].item()
    tail_weight = class_weights[tail_cls].item()
    
    print(f"\n📊 Class-balanced sampling ({strategy}):")
    if strategy == 'smoothed':
        print(f"  Smoothing alpha: {alpha} (0=none, 0.5=sqrt, 1=full)")
    elif strategy == 'effective':
        print(f"  Effective beta: {beta}")
    print(f"  Total samples: {len(labels)}")
    print(f"  Classes with samples: {len(class_counts)}")
    print(f"\n  Class distribution:")
    print(f"    Bulk class {bulk_cls}: {bulk_count} samples")
    print(f"    Tail class {tail_cls}: {tail_count} samples")
    print(f"    Original ratio (bulk:tail): {bulk_count}:{tail_count} = {bulk_count/tail_count:.1f}:1")
    
    # After rebalancing
    effective_bulk = bulk_weight * bulk_count
    effective_tail = tail_weight * tail_count
    print(f"\n  After rebalancing (expected samples per epoch):")
    print(f"    Bulk class {bulk_cls}: ~{effective_bulk:.0f} samples")
    print(f"    Tail class {tail_cls}: ~{effective_tail:.0f} samples")
    print(f"    New ratio (bulk:tail): {effective_bulk:.0f}:{effective_tail:.0f} = {effective_bulk/effective_tail:.1f}:1")
    
    return sample_weights


def freeze_backbone(model: nn.Module) -> int:
    """
    Freeze all backbone parameters, keep classifier trainable.
    Returns count of frozen parameters.
    """
    frozen_count = 0
    trainable_count = 0
    
    for name, param in model.named_parameters():
        if 'classifier' in name:
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()
    
    print(f"\n🔒 Backbone frozen:")
    print(f"  Frozen params: {frozen_count:,}")
    print(f"  Trainable params (classifier): {trainable_count:,}")
    
    return frozen_count


def reinitialize_classifier(model: nn.Module) -> None:
    """Reinitialize classifier head weights (Xavier/Kaiming)."""
    print("\n🔄 Reinitializing classifier head...")
    for name, module in model.named_modules():
        if 'classifier' in name:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                print(f"  Reinitialized: {name}")


def train_epoch_stage2(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    model_type: str,
) -> Dict[str, float]:
    """Train for one epoch (classifier head only)."""
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc='Stage 2 Training')
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
        
        # Backward pass (only classifier gradients)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{total_correct / total_samples:.4f}',
        })
    
    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
    }


@torch.no_grad()
def validate_stage2(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_type: str,
    num_classes: int,
) -> Dict[str, float]:
    """Validate model with per-class metrics."""
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc='Validation'):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        if model_type == 'masked':
            masks = batch['mask'].to(device)
            logits = model(images, masks)
        else:
            logits = model(images)
        
        loss = criterion(logits, labels)
        
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        for i in range(len(labels)):
            label = labels[i].item()
            class_total[label] += 1
            if preds[i] == labels[i]:
                class_correct[label] += 1
    
    # Compute metrics
    per_class_acc = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            per_class_acc[i] = class_correct[i] / class_total[i]
    
    # Mean class accuracy (key metric for tail classes)
    mean_class_acc = np.mean([acc for acc in per_class_acc.values()])
    
    # Separate bulk/tail metrics
    class_counts = Counter(all_labels)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Top 10% = bulk, bottom 10% = tail
    n_classes_with_data = len(sorted_classes)
    n_bulk = max(1, n_classes_with_data // 10)
    n_tail = max(1, n_classes_with_data // 10)
    
    bulk_classes = [c for c, _ in sorted_classes[:n_bulk]]
    tail_classes = [c for c, _ in sorted_classes[-n_tail:]]
    
    bulk_acc = np.mean([per_class_acc.get(c, 0) for c in bulk_classes])
    tail_acc = np.mean([per_class_acc.get(c, 0) for c in tail_classes])
    
    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'mean_class_accuracy': mean_class_acc,
        'bulk_accuracy': bulk_acc,
        'tail_accuracy': tail_acc,
        'per_class_accuracy': per_class_acc,
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # Load Stage 1 checkpoint
    print(f"\n📦 Loading Stage 1 checkpoint: {args.stage1_checkpoint}")
    checkpoint = torch.load(args.stage1_checkpoint, map_location=device, weights_only=False)
    stage1_args = checkpoint['args']
    
    # Merge args (use Stage 1 args as defaults)
    if args.train_manifest is None:
        args.train_manifest = stage1_args.get('train_manifest')
    if args.val_manifest is None:
        args.val_manifest = stage1_args.get('val_manifest')
    if args.output_root is None:
        args.output_root = stage1_args.get('output_root')
    if args.classes_yaml is None:
        args.classes_yaml = stage1_args.get('classes_yaml')
    
    # Extract model config from Stage 1
    num_classes = stage1_args['num_classes']
    model_type = stage1_args['model_type']
    encoder_name = stage1_args['encoder']
    dropout = stage1_args['dropout']
    use_mask_features = stage1_args.get('use_mask_features', False)
    image_size = tuple(stage1_args.get('image_size', [224, 224]))
    excluded_classes = stage1_args.get('excluded_classes', None)
    min_purity = stage1_args.get('min_purity', 0.7)
    min_area = stage1_args.get('min_area', 100)
    
    print(f"\n📋 Stage 1 config:")
    print(f"  Encoder: {encoder_name}")
    print(f"  Model type: {model_type}")
    print(f"  Num classes: {num_classes}")
    print(f"  Image size: {image_size}")
    
    # Create model and load Stage 1 weights
    model = create_instance_classifier(
        num_classes=num_classes,
        model_type=model_type,
        encoder_name=encoder_name,
        pretrained=False,  # Don't need ImageNet, we load Stage 1
        dropout=dropout,
        use_mask_features=use_mask_features,
    )
    model.load_state_dict(checkpoint['model'])
    
    # Freeze backbone
    freeze_backbone(model)
    
    # Optionally reinitialize classifier head
    if args.reinit_classifier:
        reinitialize_classifier(model)
    
    model.to(device)
    
    # Create datasets
    print(f"\n📂 Loading datasets...")
    train_transform = InstanceTransform(image_size=image_size, augment=True)
    val_transform = InstanceTransform(image_size=image_size, augment=False)
    
    train_dataset = TreeInstanceDataset(
        Path(args.train_manifest),
        Path(args.output_root),
        min_purity=min_purity,
        min_area=min_area,
        transform=train_transform,
        crop_to_bbox=True,
        excluded_classes=excluded_classes,
    )
    
    val_dataset = TreeInstanceDataset(
        Path(args.val_manifest),
        Path(args.output_root),
        min_purity=min_purity,
        min_area=min_area,
        transform=val_transform,
        crop_to_bbox=True,
        excluded_classes=excluded_classes,
    )
    
    # Create class-balanced sampler
    sample_weights = compute_class_weights(
        train_dataset,
        strategy=args.sampling,
        beta=args.effective_beta,
        alpha=args.smoothing_alpha,
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,  # Must be True for weighted sampling
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,  # Use balanced sampler instead of shuffle
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Optimizer (only for classifier params)
    classifier_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        classifier_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Simple LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Stage 2 config
    config = {
        'stage1_checkpoint': args.stage1_checkpoint,
        'sampling_strategy': args.sampling,
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'reinit_classifier': args.reinit_classifier,
    }
    with (output_dir / 'stage2_config.json').open('w') as f:
        json.dump(config, f, indent=2)
    
    # Wandb
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"stage2_{args.sampling}",
            config=vars(args),
        )
    
    # Training loop
    print("\n" + "=" * 80)
    print("🚀 Stage 2: Classifier Re-training with Class-Balanced Sampling")
    print("=" * 80)
    
    best_mean_class_acc = 0.0
    best_tail_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_metrics = train_epoch_stage2(
            model, train_loader, criterion, optimizer, device, model_type
        )
        
        # Validate
        val_metrics = validate_stage2(
            model, val_loader, criterion, device, model_type, num_classes
        )
        
        # Update LR
        scheduler.step()
        
        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val Mean Class Acc: {val_metrics['mean_class_accuracy']:.4f}")
        print(f"Val Bulk Acc: {val_metrics['bulk_accuracy']:.4f} | Val Tail Acc: {val_metrics['tail_accuracy']:.4f}")
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'val/mean_class_accuracy': val_metrics['mean_class_accuracy'],
                'val/bulk_accuracy': val_metrics['bulk_accuracy'],
                'val/tail_accuracy': val_metrics['tail_accuracy'],
                'lr': optimizer.param_groups[0]['lr'],
            })
        
        # Save best by mean class accuracy (not overall accuracy!)
        is_best = val_metrics['mean_class_accuracy'] > best_mean_class_acc
        if is_best:
            best_mean_class_acc = val_metrics['mean_class_accuracy']
            best_tail_acc = val_metrics['tail_accuracy']
            
            # Merge args with stage1_args for compatibility with test_on_test_set
            # test_on_test_set needs: num_classes, model_type, encoder, dropout, etc.
            merged_args = {**stage1_args, **vars(args)}
            
            checkpoint_dict = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'args': merged_args,  # Merged args for test compatibility
                'stage2_args': vars(args),
                'stage1_args': stage1_args,
            }
            torch.save(checkpoint_dict, output_dir / 'best_model_stage2.pt')
            print(f"✅ New best! Mean Class Acc: {best_mean_class_acc:.4f}")
        
        # Periodic save
        if (epoch + 1) % args.save_freq == 0:
            merged_args = {**stage1_args, **vars(args)}
            checkpoint_dict = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val_metrics': val_metrics,
                'args': merged_args,
                'stage2_args': vars(args),
                'stage1_args': stage1_args,
            }
            torch.save(checkpoint_dict, output_dir / f'checkpoint_epoch_{epoch+1}.pt')
    
    # Final save
    merged_args = {**stage1_args, **vars(args)}
    final_checkpoint = {
        'epoch': args.epochs - 1,
        'model': model.state_dict(),
        'best_mean_class_acc': best_mean_class_acc,
        'best_tail_acc': best_tail_acc,
        'args': merged_args,
        'stage2_args': vars(args),
        'stage1_args': stage1_args,
    }
    torch.save(final_checkpoint, output_dir / 'final_model_stage2.pt')
    
    print("\n" + "=" * 80)
    print("🎉 Stage 2 Training Complete!")
    print(f"  Best Mean Class Acc: {best_mean_class_acc:.4f}")
    print(f"  Best Tail Acc: {best_tail_acc:.4f}")
    print(f"  Saved to: {output_dir}")
    print("=" * 80)
    
    # Test on test set (same as Stage 1)
    if args.auto_test_on_finish:
        # Infer test_manifest from train_manifest if not specified
        if args.test_manifest is None:
            train_manifest_path = Path(args.train_manifest)
            inferred_test = train_manifest_path.parent.parent / 'test' / 'instances_manifest.jsonl'
            if inferred_test.exists():
                args.test_manifest = str(inferred_test)
                print(f"\n📍 Inferred test manifest: {args.test_manifest}")
        
        # Infer test image/label directories if not specified
        if args.test_manifest and (not args.test_image_dir or not args.test_label_dir):
            # Try to infer from stage1_args
            args.test_image_dir = args.test_image_dir or stage1_args.get('test_image_dir')
            args.test_label_dir = args.test_label_dir or stage1_args.get('test_label_dir')
        
        if args.test_manifest and args.test_image_dir and args.test_label_dir:
            print("\n🧪 Auto-testing on test set...")
            try:
                test_output_dir = output_dir / 'test_results'
                test_on_test_set(
                    checkpoint_path=output_dir / 'best_model_stage2.pt',
                    test_manifest=args.test_manifest,
                    output_root=args.output_root,
                    test_image_dir=args.test_image_dir,
                    test_label_dir=args.test_label_dir,
                    classes_yaml=args.classes_yaml,
                    output_dir=test_output_dir,
                    device=device,
                )
            except Exception as e:
                print(f"❌ Test failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            missing = []
            if not args.test_manifest:
                missing.append('test_manifest')
            if not args.test_image_dir:
                missing.append('test_image_dir')
            if not args.test_label_dir:
                missing.append('test_label_dir')
            print(f"\n⚠️  Auto-test skipped. Missing: {', '.join(missing)}")
            print("   Use --test-manifest, --test-image-dir, --test-label-dir to specify.")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()

