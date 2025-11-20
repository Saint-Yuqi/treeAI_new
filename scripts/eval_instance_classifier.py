#!/usr/bin/env python3
"""
Evaluate Instance-level Tree Species Classifier

Computes accuracy, per-class metrics, and confusion matrix.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.instance_dataset import InstanceTransform, TreeInstanceDataset
from utils.instance_classifier import create_instance_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate instance classifier')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test-manifest', type=str, required=True,
                        help='Path to test manifest JSONL file')
    parser.add_argument('--output-root', type=str, required=True,
                        help='Root directory for instance data')
    parser.add_argument('--output-dir', type=str, default='./outputs/eval',
                        help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    return parser.parse_args()


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    model_type: str,
    num_classes: int,
) -> Dict:
    """Evaluate model and return all predictions."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_purities = []
    
    pbar = tqdm(dataloader, desc='Evaluating')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label']
        purities = batch['purity'].numpy()
        
        # Forward pass
        if model_type == 'masked':
            masks = batch['mask'].to(device)
            logits = model(images, masks)
        else:
            logits = model(images)
        
        # Get predictions and probabilities
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)
        all_purities.extend(purities)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_purities = np.array(all_purities)
    
    # Compute metrics
    accuracy = (all_preds == all_labels).mean()
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    
    # Per-class metrics
    report = classification_report(
        all_labels,
        all_preds,
        labels=list(range(num_classes)),
        output_dict=True,
        zero_division=0,
    )
    
    # Stratify by purity
    high_purity = all_purities > 0.9
    if high_purity.sum() > 0:
        high_purity_acc = (all_preds[high_purity] == all_labels[high_purity]).mean()
    else:
        high_purity_acc = 0.0
    
    low_purity = all_purities <= 0.7
    if low_purity.sum() > 0:
        low_purity_acc = (all_preds[low_purity] == all_labels[low_purity]).mean()
    else:
        low_purity_acc = 0.0
    
    return {
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'purities': all_purities,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'high_purity_accuracy': high_purity_acc,
        'low_purity_accuracy': low_purity_acc,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: Path,
    normalize: bool = False,
    title: str = 'Confusion Matrix',
):
    """Plot confusion matrix."""
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        square=True,
        cbar_kws={'label': 'Normalized Count' if normalize else 'Count'},
    )
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix: {output_path}")


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Instance Classifier Evaluation")
    print("=" * 80)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    checkpoint_args = checkpoint['args']
    
    print(f"Model type: {checkpoint_args['model_type']}")
    print(f"Encoder: {checkpoint_args['encoder']}")
    print(f"Num classes: {checkpoint_args['num_classes']}")
    
    # Create model
    print("\nCreating model...")
    model = create_instance_classifier(
        num_classes=checkpoint_args['num_classes'],
        model_type=checkpoint_args['model_type'],
        encoder_name=checkpoint_args['encoder'],
        pretrained=False,  # Don't need pretrained for eval
        dropout=checkpoint_args['dropout'],
        use_mask_features=checkpoint_args.get('use_mask_features', False),
    )
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    # Create dataloader
    print("\nLoading test data...")
    test_transform = InstanceTransform(
        image_size=tuple(checkpoint_args['image_size']),
        augment=False,
    )
    
    test_dataset = TreeInstanceDataset(
        manifest_path=Path(args.test_manifest),
        output_root=Path(args.output_root),
        min_purity=0.0,  # Include all for evaluation
        min_area=0,
        transform=test_transform,
        crop_to_bbox=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Evaluate
    print("\nEvaluating...")
    results = evaluate_model(
        model,
        test_loader,
        device,
        checkpoint_args['model_type'],
        checkpoint_args['num_classes'],
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print(f"High Purity (>0.9) Accuracy: {results['high_purity_accuracy']:.4f}")
    print(f"Low Purity (<=0.7) Accuracy: {results['low_purity_accuracy']:.4f}")
    
    # Per-class metrics
    print("\nPer-class Metrics:")
    print("-" * 80)
    for class_id in range(checkpoint_args['num_classes']):
        class_key = str(class_id)
        if class_key in results['classification_report']:
            metrics = results['classification_report'][class_key]
            print(f"Class {class_id}: "
                  f"Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, "
                  f"F1={metrics['f1-score']:.3f}, "
                  f"Support={int(metrics['support'])}")
    
    # Macro averages
    macro_metrics = results['classification_report']['macro avg']
    print("\nMacro Average:")
    print(f"  Precision: {macro_metrics['precision']:.4f}")
    print(f"  Recall: {macro_metrics['recall']:.4f}")
    print(f"  F1-score: {macro_metrics['f1-score']:.4f}")
    
    # Save results
    results_to_save = {
        'accuracy': float(results['accuracy']),
        'high_purity_accuracy': float(results['high_purity_accuracy']),
        'low_purity_accuracy': float(results['low_purity_accuracy']),
        'classification_report': results['classification_report'],
        'confusion_matrix': results['confusion_matrix'].tolist(),
    }
    
    results_path = output_dir / 'evaluation_results.json'
    with results_path.open('w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nSaved results: {results_path}")
    
    # Plot confusion matrices
    print("\nGenerating visualizations...")
    plot_confusion_matrix(
        results['confusion_matrix'],
        output_dir / 'confusion_matrix.jpg',
        normalize=False,
        title='Confusion Matrix (Counts)',
    )
    
    plot_confusion_matrix(
        results['confusion_matrix'],
        output_dir / 'confusion_matrix_normalized.jpg',
        normalize=True,
        title='Confusion Matrix (Normalized)',
    )
    
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
