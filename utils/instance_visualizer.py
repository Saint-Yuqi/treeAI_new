#!/usr/bin/env python3
"""
Instance Classification Visualizer

Converts instance predictions back to semantic segmentation format for visualization.
Reuses the original visualization pipeline from semantic segmentation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.patches import Patch, Rectangle
from PIL import Image, ImageDraw
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def load_treeai_classes(yaml_path: str) -> Tuple[Dict[int, str], Dict[int, Tuple[int, int, int]], Dict[str, List[int]]]:
    """
    Load TreeAI class names, colors, and groups from YAML.
    
    Returns
    -------
    class_names : Dict[int, str]
        Mapping from class ID to name
    class_colors : Dict[int, Tuple[int, int, int]]
        Mapping from class ID to RGB color
    class_groups : Dict[str, List[int]]
        Mapping from group name to list of class IDs
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    class_names = {}
    class_colors = {}
    
    for class_id, info in data['classes'].items():
        class_id = int(class_id)
        class_names[class_id] = info['name']
        
        # Convert hex color to RGB
        hex_color = info['color'].lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        class_colors[class_id] = rgb
    
    class_groups = data.get('class_groups', {})
    
    return class_names, class_colors, class_groups


def instances_to_semantic_map(
    image_shape: Tuple[int, int],
    instance_masks: List[np.ndarray],
    instance_labels: List[int],
    instance_scores: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Convert instance predictions to semantic segmentation map.
    
    When instances overlap, use the one with higher score (or later in list).
    
    Parameters
    ----------
    image_shape : Tuple[int, int]
        (H, W) shape of output segmentation map
    instance_masks : List[np.ndarray]
        List of binary masks, each (H, W)
    instance_labels : List[int]
        List of class labels for each instance
    instance_scores : List[float], optional
        List of confidence scores for each instance (higher = higher priority)
    
    Returns
    -------
    semantic_map : np.ndarray
        Semantic segmentation map, shape (H, W), dtype int
    """
    H, W = image_shape
    semantic_map = np.zeros((H, W), dtype=np.int32)
    
    # If no scores provided, use order (later instances overwrite earlier ones)
    if instance_scores is None:
        instance_scores = list(range(len(instance_masks)))
    
    # Sort by score (ascending), so higher scores overwrite lower scores
    sorted_indices = sorted(range(len(instance_masks)), key=lambda i: instance_scores[i])
    
    for idx in sorted_indices:
        mask = instance_masks[idx]
        label = instance_labels[idx]
        
        # Resize mask if needed
        if mask.shape != image_shape:
            mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
            mask_pil = mask_pil.resize((W, H), Image.NEAREST)
            mask = (np.array(mask_pil) > 127).astype(bool)
        
        semantic_map[mask] = label
    
    return semantic_map


def predict_instances_for_image(
    model: torch.nn.Module,
    image_path: str,
    manifest_instances: List[Dict],
    output_root: Path,
    transform,
    device: torch.device,
    model_type: str,
    mask_base_path: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int], List[float], List[np.ndarray], List[List[float]]]:
    """
    Predict all instances for a given image.
    
    Returns
    -------
    image_rgb : np.ndarray
        Original image (H, W, 3)
    predictions : List[int]
        Predicted labels for each instance
    scores : List[float]
        Confidence scores for each instance
    masks : List[np.ndarray]
        Binary masks for each instance
    bboxes : List[List[float]]
        Bounding boxes for each instance [x1, y1, x2, y2]
    """
    # Load original image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    output_root = Path(output_root)
    base_mask_dir = Path(mask_base_path) if mask_base_path else output_root
    predictions = []
    scores = []
    masks = []
    bboxes = []
    
    for inst in manifest_instances:
        # Load mask
        mask_rel = Path(inst['mask_path'])
        if mask_rel.is_absolute():
            mask_path = mask_rel
        else:
            mask_path = base_mask_dir / mask_rel
            if not mask_path.exists():
                split = inst.get('split')
                if split:
                    alt_path = output_root / split / mask_rel
                    if alt_path.exists():
                        mask_path = alt_path
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found for instance {inst['image']} at {mask_path}")
        mask = Image.open(mask_path).convert('L')
        mask_np = (np.array(mask) > 127)
        
        # Get bbox
        bbox = inst['bbox']
        x1, y1, x2, y2 = [int(x) for x in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_np.shape[1], x2), min(image_np.shape[0], y2)
        
        # Crop to bbox
        image_crop = image_np[y1:y2, x1:x2]
        mask_crop = mask_np[y1:y2, x1:x2]
        
        # Transform
        image_pil = Image.fromarray(image_crop)
        mask_pil = Image.fromarray((mask_crop * 255).astype(np.uint8))
        
        if transform:
            # Transform expects to handle both image and mask
            image_tensor, mask_tensor = transform(image_pil, mask_pil)
        else:
            import torchvision.transforms.functional as TF
            image_tensor = TF.to_tensor(image_pil)
            mask_tensor = TF.to_tensor(mask_pil)

        image_tensor = image_tensor.unsqueeze(0).to(device)
        mask_tensor = mask_tensor.unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            if model_type == 'masked':
                logits = model(image_tensor, mask_tensor)
            else:
                logits = model(image_tensor)
            
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            score = probs[0, pred].item()
        
        predictions.append(pred + 1)  # Convert back to 1-indexed
        scores.append(score)
        masks.append(mask_np.astype(bool))
        bboxes.append(bbox)
    
    return image_np, predictions, scores, masks, bboxes


def visualize_instance_predictions(
    images: List[np.ndarray],
    gt_masks: List[np.ndarray],
    pred_masks: List[np.ndarray],
    bboxes_list: List[List[List[float]]],
    class_names: Dict[int, str],
    class_colors: Dict[int, Tuple[int, int, int]],
    num_samples: int = 5,
    which: str = 'first',
    save_path: Optional[Path] = None,
):
    """
    Visualize instance predictions in semantic segmentation format.
    
    Layout: 4 columns
    1. Original Image
    2. Image + Red BBoxes
    3. Ground Truth (colored)
    4. Prediction (colored)
    
    Parameters
    ----------
    images : List[np.ndarray]
        Original RGB images, each (H, W, 3)
    gt_masks : List[np.ndarray]
        Ground truth semantic masks, each (H, W)
    pred_masks : List[np.ndarray]
        Predicted semantic masks, each (H, W)
    bboxes_list : List[List[List[float]]]
        Bounding boxes for each image, each a list of [x1, y1, x2, y2]
    class_names : Dict[int, str]
        Mapping from class ID to name
    class_colors : Dict[int, Tuple[int, int, int]]
        Mapping from class ID to RGB color
    num_samples : int
        Number of samples to visualize
    which : str
        'first' or 'last'
    save_path : Path, optional
        Path to save visualization
    """
    # Select indices
    n = len(images)
    if which == 'first':
        indices = list(range(min(num_samples, n)))
    elif which == 'last':
        indices = list(range(max(0, n - num_samples), n))
    else:
        indices = list(range(min(num_samples, n)))
    
    # Create custom colormap
    max_class = max(class_colors.keys())
    colors_array = np.zeros((max_class + 1, 3), dtype=np.uint8)
    for class_id, color in class_colors.items():
        colors_array[class_id] = color
    
    custom_cmap = matplotlib.colors.ListedColormap(colors_array / 255.0)
    
    # Create legend patches
    legend_patches = []
    for class_id in sorted(class_colors.keys()):
        if class_id == 0:  # Skip background
            continue
        color = np.array(class_colors[class_id]) / 255.0
        patch = Patch(facecolor=color, edgecolor='black', label=f'{class_id}: {class_names[class_id]}')
        legend_patches.append(patch)
    
    # Create figure
    num_cols = 4
    fig = plt.figure(figsize=(20, len(indices) * 5))
    
    for count, i in enumerate(indices):
        # Column 1: Original Image
        plt.subplot(len(indices), num_cols, count * num_cols + 1)
        if count == 0:
            plt.title('Image', fontsize=22)
        plt.imshow(images[i])
        plt.axis('off')
        
        # Column 2: Image + Red BBoxes
        img_with_bbox = images[i].copy()
        if bboxes_list[i]:
            img_pil = Image.fromarray(img_with_bbox)
            draw = ImageDraw.Draw(img_pil)
            for bbox in bboxes_list[i]:
                x1, y1, x2, y2 = bbox
                draw.rectangle([int(x1), int(y1), int(x2), int(y2)], outline=(255, 0, 0), width=2)
            img_with_bbox = np.array(img_pil)
        
        plt.subplot(len(indices), num_cols, count * num_cols + 2)
        if count == 0:
            plt.title('Image + BBoxes', fontsize=22)
        plt.imshow(img_with_bbox)
        plt.axis('off')
        
        # Column 3: Ground Truth
        plt.subplot(len(indices), num_cols, count * num_cols + 3)
        if count == 0:
            plt.title('Reference', fontsize=22)
        plt.imshow(gt_masks[i], cmap=custom_cmap, vmin=0, vmax=max_class)
        plt.axis('off')
        
        # Column 4: Prediction
        plt.subplot(len(indices), num_cols, count * num_cols + 4)
        if count == 0:
            plt.title('Prediction', fontsize=22)
        plt.imshow(pred_masks[i], cmap=custom_cmap, vmin=0, vmax=max_class)
        plt.axis('off')
    
    # Add legend
    plt.figlegend(
        handles=legend_patches,
        loc='center left',
        bbox_to_anchor=(1.0, 0.5),
        ncol=2,
        fontsize=16,
        borderaxespad=0.
    )
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization: {save_path}")
    
    plt.close()


def plot_confusion_matrix_semantic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Dict[int, str],
    save_path: Path,
    normalize: Optional[str] = None,
    title: str = 'Confusion Matrix',
    include_background: bool = False,
):
    """
    Plot confusion matrix for semantic segmentation (pixel-level).
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels, flattened
    y_pred : np.ndarray
        Predicted labels, flattened
    class_names : Dict[int, str]
        Mapping from class ID to name
    save_path : Path
        Path to save plot
    normalize : str, optional
        'true', 'pred', 'all', or None
    title : str
        Plot title
    include_background : bool
        If True, include class 0 (background) in the matrix
    """
    # Get all known classes from class_names to ensure fixed matrix size
    labels = sorted(class_names.keys())
    
    if not include_background:
        labels = [l for l in labels if l != 0]  # Exclude background
    
    # Filter labels to only those present in data if we want a compact matrix? 
    # The user specifically requested seeing ALL classes ("61 kinds"), so we stick to `labels` 
    # derived from `class_names`. However, sklearn's confusion_matrix will complain or work fine 
    # if labels are provided. It works fine.
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Normalize if requested
    if normalize == 'true':
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
        cm = cm * 100  # Convert to percentage
    elif normalize == 'pred':
        cm = cm.astype('float') / (cm.sum(axis=0, keepdims=True) + 1e-8)
        cm = cm * 100  # Convert to percentage
    elif normalize == 'all':
        cm = cm.astype('float') / (cm.sum() + 1e-8)
        cm = cm * 100  # Convert to percentage
    
    # Plot
    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Labels
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=[class_names.get(l, str(l)) for l in labels],
        yticklabels=[class_names.get(l, str(l)) for l in labels],
        title=title,
        ylabel='True label',
        xlabel='Predicted label'
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Annotate cells
    fmt = '.0f' if normalize else 'd'  # Use integer format for percentage
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha='center', va='center',
                   color='white' if cm[i, j] > thresh else 'black',
                   fontsize=8)
    
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close()
    
    print(f"Saved confusion matrix: {save_path}")


def compute_group_f1_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_groups: Dict[str, List[int]],
    ignore_classes: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute F1 scores for class groups (macro average).
    
    For each group, compute F1 for each class in the group, then average.
    This gives equal weight to each class regardless of frequency.
    Classes in ignore_classes are excluded from calculation.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels (can include background 0)
    y_pred : np.ndarray
        Predicted labels (can include background 0)
    class_groups : Dict[str, List[int]]
        Mapping from group name to list of class IDs
    ignore_classes : List[int], optional
        Classes to ignore (e.g., order/genus level classes).
        These will be excluded from group F1 calculation.
    
    Returns
    -------
    group_f1_scores : Dict[str, float]
        Macro-averaged F1 score for each group (uses nanmean to ignore NaN)
    """
    from sklearn.metrics import f1_score
    
    if ignore_classes is None:
        ignore_classes = []
    
    group_f1_scores = {}
    
    for group_name, class_ids in class_groups.items():
        # Filter out ignored classes (order/genus level)
        valid_class_ids = [c for c in class_ids if c not in ignore_classes]
        
        # Get classes that actually appear in the data
        present_classes = [c for c in valid_class_ids if c in y_true or c in y_pred]
        
        if len(present_classes) == 0:
            # No valid classes present in this group
            group_f1_scores[group_name] = 0.0
            continue
        
        # Compute per-class F1 for classes in this group
        # zero_division=np.nan: classes with no samples get NaN (excluded from mean)
        per_class_f1 = f1_score(
            y_true, 
            y_pred, 
            labels=present_classes, 
            average=None, 
            zero_division=np.nan  # Use NaN for missing classes
        )
        
        # Macro average: use nanmean to ignore NaN values (classes without samples)
        group_f1_scores[group_name] = float(np.nanmean(per_class_f1))
    
    return group_f1_scores


__all__ = [
    'load_treeai_classes',
    'instances_to_semantic_map',
    'predict_instances_for_image',
    'visualize_instance_predictions',
    'plot_confusion_matrix_semantic',
    'compute_group_f1_scores',
]
