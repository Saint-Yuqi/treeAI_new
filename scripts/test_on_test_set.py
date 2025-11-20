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
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.instance_classifier import create_instance_classifier
from utils.instance_visualizer import (
    load_treeai_classes,
    instances_to_semantic_map,
    predict_instances_for_image,
    visualize_instance_predictions,
    plot_confusion_matrix_semantic,
    compute_group_f1_scores,
)


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
    total_instances = sum(len(insts) for insts in instances_by_image.values())
    print(f"Total instances: {total_instances}")
    
    # Setup transform
    image_size = checkpoint_args.get('image_size', [224, 224])
    transform = transforms.Compose([
        transforms.Resize(tuple(image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Process all test images
    print("\nProcessing test images...")
    all_images = []
    all_gt_masks = []
    all_pred_masks = []
    all_bboxes = []
    all_image_names = []
    
    test_image_dir = Path(args.test_image_dir)
    test_label_dir = Path(args.test_label_dir)
    
    for image_name in tqdm(sorted(instances_by_image.keys()), desc='Predicting'):
        # Find image and label files
        image_files = list(test_image_dir.glob(f"{image_name}.*"))
        label_files = list(test_label_dir.glob(f"{image_name}.*"))
        
        if not image_files or not label_files:
            continue
        
        image_path = image_files[0]
        label_path = label_files[0]
        
        # Load GT
        gt_mask = np.array(Image.open(label_path)).astype(np.int32)
        
        # Predict instances
        image_rgb, predictions, scores, masks, bboxes = predict_instances_for_image(
            model=model,
            image_path=str(image_path),
            manifest_instances=instances_by_image[image_name],
            output_root=Path(args.output_root),
            transform=transform,
            device=device,
            model_type=checkpoint_args['model_type'],
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
        all_image_names.append(image_name)
    
    print(f"\nProcessed {len(all_images)} images")
    
    # Compute pixel-level metrics
    print("\nComputing metrics...")
    y_true = np.concatenate([mask.flatten() for mask in all_gt_masks])
    y_pred = np.concatenate([mask.flatten() for mask in all_pred_masks])
    
    # Overall accuracy
    valid_mask = y_true != 0  # Exclude background
    if valid_mask.sum() > 0:
        accuracy = (y_true[valid_mask] == y_pred[valid_mask]).mean()
    else:
        accuracy = 0.0
    
    print(f"Overall Pixel Accuracy (excluding background): {accuracy:.4f}")
    
    # Group F1 scores
    print("\nComputing group F1 scores...")
    group_f1 = compute_group_f1_scores(y_true, y_pred, class_groups)
    
    print("\n📊 Group F1 Scores:")
    for group_name, f1 in group_f1.items():
        print(f"  {group_name:20s}: {f1:.4f}")
    
    # Save results
    results = {
        'checkpoint': str(args.checkpoint),
        'num_test_images': len(all_images),
        'num_instances': total_instances,
        'pixel_accuracy': float(accuracy),
        'group_f1_scores': {k: float(v) for k, v in group_f1.items()},
    }
    
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


if __name__ == '__main__':
    main()

