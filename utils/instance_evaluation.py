import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

from utils.instance_dataset import InstanceTransform
from utils.instance_visualizer import (
    instances_to_semantic_map,
    predict_instances_for_image,
    compute_group_f1_scores,
)

def evaluate_instance_segmentation(
    model: torch.nn.Module,
    instances_by_image: Dict[str, List[Dict]],
    image_dir: Path,
    label_dir: Path,
    output_root: Path,
    class_names: Dict[int, str],
    class_groups: Dict[str, List[int]],
    device: torch.device,
    model_type: str = 'simple',
    image_size: Tuple[int, int] = (224, 224),
    max_samples: Optional[int] = None,
    desc: str = "Evaluating",
    excluded_classes: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Evaluate instance segmentation model using full image reconstruction.
    
    Matches logic from scripts/test_on_test_set.py.
    """
    # Setup transform
    transform = InstanceTransform(
        image_size=image_size,
        augment=False,
    )
    
    # Process images
    all_images = []
    all_gt_masks = []
    all_pred_masks = []
    all_bboxes = []
    all_image_names = []
    
    image_names = sorted(instances_by_image.keys())
    if max_samples is not None:
        image_names = image_names[:max_samples]
    
    # Ensure directories exist
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
        
    for image_name in tqdm(image_names, desc=desc):
        # Find image and label files
        image_files = list(image_dir.glob(f"{image_name}.*"))
        label_files = list(label_dir.glob(f"{image_name}.*"))
        
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
            output_root=output_root,
            transform=transform,
            device=device,
            model_type=model_type,
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
    
    if not all_images:
        print(f"Warning: No images processed in {desc}")
        return {}

    # Compute pixel-level metrics
    y_true = np.concatenate([mask.flatten() for mask in all_gt_masks])
    y_pred = np.concatenate([mask.flatten() for mask in all_pred_masks])
    
    # Overall accuracy (including background)
    accuracy_incl_bg = (y_true == y_pred).mean()

    # Overall accuracy (excluding background)
    valid_mask = y_true != 0
    if valid_mask.sum() > 0:
        accuracy_excl_bg = (y_true[valid_mask] == y_pred[valid_mask]).mean()
    else:
        accuracy_excl_bg = 0.0
    
    # Define non-species classes to exclude (order and genus level)
    # These should NOT be included in F1-avg-wo0 calculation
    # If not provided, use default list
    if excluded_classes is None:
        non_species_classes = [
            37, 60,  # order level (coniferous, deciduous)
            5, 11, 43, 50, 56, 58, 59, 61,  # genus level (betula sp., picea sp., etc.)
        ]
    else:
        non_species_classes = excluded_classes
    
    # Get unique classes
    unique_classes_with_bg = sorted([c for c in set(y_true) | set(y_pred)])  # All classes including bg
    unique_classes_no_bg = sorted([c for c in unique_classes_with_bg if c != 0])  # Exclude background
    species_classes = sorted([c for c in unique_classes_no_bg if c not in non_species_classes])  # Species only
    
    # Per-class F1, Precision, Recall for ALL non-background classes (for per_class_metrics)
    per_class_f1_all = f1_score(y_true, y_pred, labels=unique_classes_no_bg, average=None, zero_division=0)
    per_class_precision_all = precision_score(y_true, y_pred, labels=unique_classes_no_bg, average=None, zero_division=0)
    per_class_recall_all = recall_score(y_true, y_pred, labels=unique_classes_no_bg, average=None, zero_division=0)
    
    # Per-class IoU for ALL non-background classes
    per_class_iou = {}
    for i, cls in enumerate(unique_classes_no_bg):
        mask_true = (y_true == cls)
        mask_pred = (y_pred == cls)
        intersection = np.logical_and(mask_true, mask_pred).sum()
        union = np.logical_or(mask_true, mask_pred).sum()
        iou = intersection / union if union > 0 else 0.0
        per_class_iou[cls] = iou
    
    # F1-avg: Mean F1 including background but excluding order/genus classes
    # Classes to include: background (0) + species classes
    classes_for_f1_avg = sorted([0] + species_classes) if 0 in unique_classes_with_bg else species_classes
    per_class_f1_with_bg = f1_score(y_true, y_pred, labels=classes_for_f1_avg, average=None, zero_division=0)
    mean_f1_with_bg = np.mean(per_class_f1_with_bg) if len(per_class_f1_with_bg) > 0 else 0.0
    
    # F1-avg-wo0: Mean F1 for SPECIES ONLY (exclude background AND order/genus classes)
    # This is the main metric we care about!
    species_f1 = f1_score(y_true, y_pred, labels=species_classes, average=None, zero_division=0)
    species_precision = precision_score(y_true, y_pred, labels=species_classes, average=None, zero_division=0)
    species_recall = recall_score(y_true, y_pred, labels=species_classes, average=None, zero_division=0)
    
    mean_f1_wo0 = np.mean(species_f1) if len(species_f1) > 0 else 0.0
    mean_precision = np.mean(species_precision) if len(species_precision) > 0 else 0.0
    mean_recall = np.mean(species_recall) if len(species_recall) > 0 else 0.0
    
    # Species IoU
    species_iou_values = [per_class_iou[cls] for cls in species_classes if cls in per_class_iou]
    mean_iou = np.mean(species_iou_values) if species_iou_values else 0.0
    
    # Species-only metrics (already computed above, just assign for clarity)
    mean_species_f1 = mean_f1_wo0
    mean_species_precision = mean_precision
    mean_species_recall = mean_recall
    mean_species_iou = mean_iou
    
    # Group F1 scores
    # The class_groups loaded from treeAI_classes.yaml already contain 'bulk', 'medium', 'tail'
    # so we use them directly without re-calculating from instance counts.
    # Exclude order/genus level classes from group F1 calculation
    group_f1 = compute_group_f1_scores(
        y_true, y_pred, class_groups, 
        ignore_classes=non_species_classes
    )
    
    # Per-class metrics dictionary (species only, excluding background and order/genus)
    per_class_metrics = {}
    for i, cls in enumerate(species_classes):
        # Find index in species arrays
        per_class_metrics[int(cls)] = {
            'name': class_names.get(cls, f"Class_{cls}"),
            'f1': float(species_f1[i]),
            'precision': float(species_precision[i]),
            'recall': float(species_recall[i]),
            'iou': float(per_class_iou.get(cls, 0.0)),
        }

    results = {
        'num_images': len(all_images),
        'pixel_accuracy_incl_bg': float(accuracy_incl_bg),
        'pixel_accuracy_excl_bg': float(accuracy_excl_bg),
        # F1 metrics (0-1 scale), both exclude order/genus classes
        'F1-avg': float(mean_f1_with_bg),  # Species + background (excludes order/genus)
        'F1-avg-wo0': float(mean_f1_wo0),  # Species only (excludes bg + order/genus)
        # Legacy keys (0-1 scale, species only)
        'mean_f1': float(mean_f1_wo0),
        'mean_precision': float(mean_precision),
        'mean_recall': float(mean_recall),
        'mean_iou': float(mean_iou),
        'species_only_f1': float(mean_species_f1),
        'species_only_precision': float(mean_species_precision),
        'species_only_recall': float(mean_species_recall),
        'species_only_iou': float(mean_species_iou),
        'num_species_classes': len(species_classes),
        'excluded_classes': non_species_classes,  # Order/genus classes that were excluded
        'per_class_metrics': per_class_metrics,
        'group_f1_scores': {k: float(v) for k, v in group_f1.items()},
        'y_true': y_true,
        'y_pred': y_pred,
        'all_images': all_images,
        'all_gt_masks': all_gt_masks,
        'all_pred_masks': all_pred_masks,
        'all_bboxes': all_bboxes,
    }
    
    return results

