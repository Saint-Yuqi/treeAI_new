#!/usr/bin/env python3
"""
Test Instance Classifier on True Test Set

This script evaluates the trained instance classifier on the actual test dataset,
not the pick dataset used for visualization.

Generates:
1. Full semantic segmentation predictions for test images
2. Confusion matrices (all normalization variants)
3. Per-class and group F1 scores
4. Qualitative visualizations
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.instance_classifier import create_instance_classifier
from utils.instance_visualizer import (
    load_treeai_classes,
    visualize_instance_predictions,
    plot_confusion_matrix_semantic,
)
from utils.instance_evaluation import evaluate_instance_segmentation


def parse_args():
    parser = argparse.ArgumentParser(description='Test instance classifier on test set')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test-manifest', type=str, required=True,
                        help='Path to test instances manifest')
    parser.add_argument('--output-root', type=str, required=True,
                        help='Root directory for instance data')
    parser.add_argument('--classes-yaml', type=str, required=True,
                        help='Path to treeAI_classes.yaml')
    parser.add_argument('--output-dir', type=str, default='./outputs/test_results',
                        help='Output directory for results')
    parser.add_argument('--test-image-dir', type=str, required=True,
                        help='Directory containing test images')
    parser.add_argument('--test-label-dir', type=str, required=True,
                        help='Directory containing test labels')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--num-viz-samples', type=int, default=20,
                        help='Number of samples to visualize')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Testing Instance Classifier on Test Set")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test manifest: {args.test_manifest}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    
    # Load checkpoint
    print("\nLoading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    checkpoint_args = checkpoint['args']
    
    print(f"Model: {checkpoint_args['model_type']} + {checkpoint_args['encoder']}")
    print(f"Num classes: {checkpoint_args['num_classes']}")
    print(f"Trained for {checkpoint['epoch']+1} epochs")
    
    # Load class information
    print("\nLoading class information...")
    class_names, class_colors, class_groups = load_treeai_classes(args.classes_yaml)
    print(f"Loaded {len(class_names)} classes")
    print(f"Class groups: {list(class_groups.keys())}")
    
    # Create model
    print("\nCreating model...")
    model = create_instance_classifier(
        num_classes=checkpoint_args['num_classes'],
        model_type=checkpoint_args['model_type'],
        encoder_name=checkpoint_args['encoder'],
        pretrained=False,
        dropout=checkpoint_args['dropout'],
        use_mask_features=checkpoint_args.get('use_mask_features', False),
    )
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    # Load test instances
    print("\nLoading test instances...")
    instances_by_image = {}
    with open(args.test_manifest, 'r') as f:
        for line in f:
            if line.strip():
                inst = json.loads(line)
                image_name = inst['image']
                if image_name not in instances_by_image:
                    instances_by_image[image_name] = []
                instances_by_image[image_name].append(inst)
    
    print(f"Found {len(instances_by_image)} test images")
    
    # Run evaluation
    image_size = checkpoint_args.get('image_size', [224, 224])
    excluded_classes = checkpoint_args.get('excluded_classes', None)
    
    results = evaluate_instance_segmentation(
        model=model,
        instances_by_image=instances_by_image,
        image_dir=Path(args.test_image_dir),
        label_dir=Path(args.test_label_dir),
        output_root=Path(args.output_root),
        class_names=class_names,
        class_groups=class_groups,
        device=device,
        model_type=checkpoint_args['model_type'],
        image_size=tuple(image_size),
        desc="Predicting",
        excluded_classes=excluded_classes,
    )
    
    if not results:
        print("Evaluation failed or returned no results.")
        return

    # Extract arrays
    y_true = results.pop('y_true')
    y_pred = results.pop('y_pred')
    all_images = results.pop('all_images')
    all_gt_masks = results.pop('all_gt_masks')
    all_pred_masks = results.pop('all_pred_masks')
    all_bboxes = results.pop('all_bboxes')
    
    # Add checkpoint info
    results['checkpoint'] = str(args.checkpoint)
    
    # Print Metrics
    print(f"\n📊 Overall Metrics (macro average across {len(results['per_class_metrics'])} classes):")
    print(f"  F1-avg (with bg):    {results['F1-avg']:.4f}")
    print(f"  F1-avg-wo0 (no bg):  {results['F1-avg-wo0']:.4f}")
    print(f"  Mean Precision:      {results['mean_precision']:.4f}")
    print(f"  Mean Recall:         {results['mean_recall']:.4f}")
    print(f"  Mean IoU (mIoU):     {results['mean_iou']:.4f}")
    
    if results['species_only_f1'] > 0:
        print(f"\n🌲 Species-Only Metrics ({results['num_species_classes']} species):")
        print(f"  Mean F1 Score:    {results['species_only_f1']:.4f}")
        print(f"  Mean Precision:   {results['species_only_precision']:.4f}")
        print(f"  Mean Recall:      {results['species_only_recall']:.4f}")
        print(f"  Mean IoU (mIoU):  {results['species_only_iou']:.4f}")

    # Per-class metrics summary
    print(f"\n📋 Per-Class Metrics:")
    print(f"{'Class ID':<10} {'Name':<30} {'F1':<8} {'Precision':<10} {'Recall':<8} {'IoU':<8}")
    print("-" * 90)
    for cls_id, metrics in sorted(results['per_class_metrics'].items()):
        print(f"{cls_id:<10} {metrics['name']:<30} {metrics['f1']:<8.4f} {metrics['precision']:<10.4f} {metrics['recall']:<8.4f} {metrics['iou']:<8.4f}")
    
    # Group F1 Scores
    print("\n📊 Group F1 Scores:")
    for group_name, f1 in results['group_f1_scores'].items():
        print(f"  {group_name:20s}: {f1:.4f}")

    # Save results
    with (output_dir / 'test_results.json').open('w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_dir / 'test_results.json'}")
    
    # Save prediction arrays
    print(f"\n💾 Saving prediction arrays for evaluation comparison...")
    np.save(output_dir / 'y_true.npy', y_true)
    np.save(output_dir / 'y_pred.npy', y_pred)
    print(f"   Saved y_true.npy ({y_true.shape})")
    print(f"   Saved y_pred.npy ({y_pred.shape})")

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
    plot_confusion_matrix_semantic(
        y_true, y_pred, class_names,
        output_dir / 'confusion_norm_pred.jpg',
        normalize='pred',
        title='Test Set Confusion Matrix (Normalized by Pred)'
    )
    plot_confusion_matrix_semantic(
        y_true, y_pred, class_names,
        output_dir / 'confusion_norm_all.jpg',
        normalize='all',
        title='Test Set Confusion Matrix (Normalized All)'
    )
    
    # Generate visualizations
    if len(all_images) > 0:
        print("\nGenerating visualizations...")
        num_viz = min(args.num_viz_samples, len(all_images))
        
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


if __name__ == '__main__':
    main()
