#!/usr/bin/env python3
"""
Evaluation utilities matching treeAI-segmentation evaluation methodology.

This module provides identical evaluation methods as used in treeAI-segmentation
to ensure fair comparison and reproducibility.
"""
import os
import json
import random
from typing import Dict, Optional, Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import cv2

from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix
from torchmetrics import JaccardIndex
from sklearn.metrics import ConfusionMatrixDisplay
from utils.prompt_loader import get_prompts_for_image


def softmax_entropy(probabilities: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute softmax entropy.
    
    Parameters:
    -----------
    probabilities : torch.Tensor
        Probability tensor
    dim : int
        Dimension to compute entropy over
    
    Returns:
    --------
    torch.Tensor
        Entropy values
    """
    return -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=dim)


def convert_ndarray_to_list(obj: Any) -> Any:
    """Convert numpy arrays to lists recursively for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


def denormalize(img: np.ndarray, mean: Tuple[float, float, float] = (0.485, 0.456, 0.406), 
                std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
    """
    Denormalize image from ImageNet normalization.
    
    Parameters:
    -----------
    img : np.ndarray
        Normalized image, shape (H, W, 3) or (3, H, W)
    mean : tuple
        Mean values used for normalization
    std : tuple
        Std values used for normalization
    
    Returns:
    --------
    np.ndarray
        Denormalized image in [0, 1] range
    """
    if img.ndim == 3 and img.shape[0] == 3:
        # CHW format
        img = img.transpose(1, 2, 0)
    
    mean = np.array(mean)
    std = np.array(std)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_classes: int,
    device: torch.device,
    ignore_index: int = -1,
    boost_pr: Optional[str] = None,
    var_th: float = 0.5,
    custom_avg: Optional[Dict[str, list]] = None,
    save_dir: Optional[str] = None,
    image_names: Optional[list] = None
) -> Dict:
    """
    Evaluate semantic segmentation model using confusion matrix.
    
    This function matches the evaluation methodology from treeAI-segmentation exactly.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The trained segmentation model
    dataloader : torch.utils.data.DataLoader
        DataLoader containing the evaluation dataset
    num_classes : int
        Number of classes in the segmentation task
    device : torch.device
        Device to run evaluation on
    ignore_index : int
        Index to ignore in evaluation
    boost_pr : str, optional
        Strategy to boost performance:
        - "recall" or "r": Increase recall by replacing uncertain background pixels
        - "precision" or "p": Increase precision by setting uncertain pixels to background
    var_th : float
        Variance threshold for boosting (default: 0.5)
    custom_avg : dict, optional
        Dictionary of class group names to class indices for custom averaging
    save_dir : str, optional
        Directory to save evaluation metrics
    image_names : list, optional
        List of image names for batch identification
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'conf': Confusion matrix (numpy array)
        - 'semseg': Dictionary of semantic segmentation metrics
    """
    model.eval()
    model.to(device)
    
    # Initialize confusion matrix
    conf_ = MulticlassConfusionMatrix(num_classes, ignore_index=ignore_index).to(device)
    
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Get image names if available
            batch_names = batch.get('name', [f"img_{i}" for i in range(imgs.size(0))])
            if isinstance(batch_names, str):
                batch_names = [batch_names]
            elif isinstance(batch_names, tuple):
                batch_names = list(batch_names)
            
            # Run model
            outputs = model(imgs, image_names=batch_names)
            
            # Get top-2 class indices for each pixel
            _, top2_indices = torch.topk(outputs, 2, dim=1)  # Shape: (batch, 2, H, W)
            preds = top2_indices[:, 0, :, :]  # Get top-1 class as prediction
            var = softmax_entropy(torch.softmax(outputs, dim=1), dim=1)
            
            # Apply precision/recall boosting if requested
            if boost_pr:
                if boost_pr.lower().startswith("r"):  # Recall boosting
                    boost_mask = (preds == 0) & (var > var_th)
                    preds[boost_mask] = top2_indices[:, 1, :, :][boost_mask]
                elif boost_pr.lower().startswith("p"):  # Precision boosting
                    boost_mask = (preds != 0) & (var > var_th)
                    preds[boost_mask] = 0
            
            # Update confusion matrix
            conf_.update(preds, masks)
    
    # Compute confusion matrix
    conf = conf_.compute().cpu().numpy()
    TP = np.diag(conf)
    FP = conf.sum(axis=0) - TP
    FN = conf.sum(axis=1) - TP
    
    # Compute metrics
    oa = np.sum(TP) / np.sum(conf)
    ua = np.divide(TP, (TP+FP), out=np.full_like(TP, np.nan, dtype=float), where=(TP+FP) != 0)  # Precision
    pa = np.divide(TP, (TP+FN), out=np.full_like(TP, np.nan, dtype=float), where=(TP+FN) != 0)  # Recall
    iou = np.divide(TP, (TP+FP+FN), out=np.full_like(TP, np.nan, dtype=float), where=(TP+FP+FN) != 0)
    f1 = np.divide(TP, TP+0.5*(FN+FP), out=np.full_like(TP, np.nan, dtype=float), where=(TP+FP+FN) != 0)
    
    # Build metrics dictionary (matching treeAI-segmentation format exactly)
    metrics_semseg = {
        'OA': oa,
        'F1-avg': np.nanmean(f1),
        'F1-avg-wo0': np.nanmean(f1[1:]),
        'F1': f1,
        'IoU-avg': np.nanmean(iou),
        'IoU-avg-wo0': np.nanmean(iou[1:]),
        'IoU': iou,
        'Precision-avg': np.nanmean(ua),
        'Precision-avg-wo0': np.nanmean(ua[1:]),
        'Precision': ua,
        'Recall-avg': np.nanmean(pa),
        'Recall-avg-wo0': np.nanmean(pa[1:]),
        'Recall': pa,
    }
    
    # Add custom averages if provided
    if custom_avg:
        for group_name, indices in custom_avg.items():
            metrics_semseg[f'F1-avg-{group_name}'] = np.nanmean(f1[np.array(indices, dtype=int)])
            metrics_semseg[f'IoU-avg-{group_name}'] = np.nanmean(iou[np.array(indices, dtype=int)])
            metrics_semseg[f'Precision-avg-{group_name}'] = np.nanmean(ua[np.array(indices, dtype=int)])
            metrics_semseg[f'Recall-avg-{group_name}'] = np.nanmean(pa[np.array(indices, dtype=int)])
    
    # Save metrics if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'metrics_semseg.json'), 'w') as f:
            json.dump(convert_ndarray_to_list(metrics_semseg), f, indent=4)
    
    metrics = {
        'conf': conf,
        'semseg': metrics_semseg,
    }
    
    return metrics


def visualize_save_confusion(
    confusion: np.ndarray,
    save_name: str,
    display_labels: Optional[list] = None,
    fontsize: int = 3
):
    """Visualize and save confusion matrix."""
    disp = ConfusionMatrixDisplay(confusion, display_labels=display_labels)
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 8))
    disp.plot(ax=ax, cmap='Blues', values_format='.0f', colorbar=False)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
    for text in ax.texts:
        text.set_fontsize(fontsize)
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.tight_layout()
    plt.savefig(save_name, dpi=500, bbox_inches='tight')
    plt.close()


def visualize_save_confusions(
    confusion: np.ndarray,
    save_dir: str,
    display_labels: Optional[list] = None
):
    """
    Create and save confusion matrices with different normalizations.
    
    Matches treeAI-segmentation visualization exactly.
    """
    # Create confusion matrices with different norms
    conf_all = confusion
    conf_norm_all = np.round(np.nan_to_num(confusion/np.sum(confusion), nan=0)*100, 0)
    conf_norm_pred = np.round(np.nan_to_num(confusion/np.sum(confusion, axis=0)[np.newaxis, :], nan=0)*100, 0)
    conf_norm_true = np.round(np.nan_to_num(confusion/np.sum(confusion, axis=1)[:, np.newaxis], nan=0)*100, 0)
    
    # Visualize and save confusion matrices
    os.makedirs(save_dir, exist_ok=True)
    visualize_save_confusion(conf_all, os.path.join(save_dir, 'confusion.jpg'), display_labels)
    visualize_save_confusion(conf_norm_all, os.path.join(save_dir, 'confusion_norm_all.jpg'), display_labels)
    visualize_save_confusion(conf_norm_pred, os.path.join(save_dir, 'confusion_norm_pred.jpg'), display_labels)
    visualize_save_confusion(conf_norm_true, os.path.join(save_dir, 'confusion_norm_true.jpg'), display_labels)


def visualize_scores_per_class(
    scores: Dict[str, np.ndarray],
    save_name: str,
    display_labels: list,
    fontsize: int = 5
):
    """
    Visualize per-class scores.
    
    Matches treeAI-segmentation visualization exactly.
    """
    num_classes = len(display_labels)
    score_names = list(scores.keys())
    num_scores = len(score_names)
    
    # Define the positions of the bars for each class
    x = np.arange(num_classes)
    width = 0.5 / num_scores
    group_offset = (num_scores - 1) * width / 2
    
    # Define a colorblind-friendly palette
    colors = plt.get_cmap("Set2")(np.linspace(0, 1, num_scores))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot each score as a separate set of bars
    for i, (score_name, color) in enumerate(zip(score_names, colors)):
        score_values = scores[score_name]
        bar_positions = x - group_offset + i * width
        ax.bar(bar_positions, score_values, width, label=score_name, color=color)
    
    # Place the legend at the top center
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=num_scores)
    
    # Add style, labels, and title
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=45, ha='right')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_model_samplewise(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_classes: int,
    device: torch.device,
    ignore_index: int = -1,
    boost_pr: Optional[str] = None,
    var_th: float = 0.5,
    save_metrics_batch_limit: Optional[int] = None,
    save_imgs_batch_limit: int = 10,
    image_names: Optional[List[str]] = None
) -> Tuple[Dict, Dict, Dict]:
    """
    Evaluate model sample-wise, returning per-sample metrics and images.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The trained segmentation model
    dataloader : torch.utils.data.DataLoader
        DataLoader containing the evaluation dataset
    num_classes : int
        Number of classes in the segmentation task
    device : torch.device
        Device to run evaluation on
    ignore_index : int
        Index to ignore in evaluation
    boost_pr : str, optional
        Strategy to boost performance (recall/precision)
    var_th : float
        Variance threshold for boosting
    save_metrics_batch_limit : int, optional
        Maximum number of batches to process for metrics
    save_imgs_batch_limit : int
        Maximum number of batches to save images from
    image_names : list, optional
        List of image names (deprecated, uses batch['name'] instead)
    
    Returns:
    --------
    all_imgs : dict
        Dictionary with 'imgs', 'preds', 'masks', 'var', 'name'
    all_metrics : dict
        Dictionary with per-sample metrics: 'name', 'acc', 'iou', 'f1', 'conf'
    flt_metrics : dict
        Filtered metrics (NaN where class not present): 'name', 'acc', 'iou', 'f1'
    """
    model.eval()
    all_imgs = []
    all_preds = []
    all_masks = []
    all_var = []
    all_class_presence = []
    all_acc = []
    all_iou = []
    all_f1 = []
    all_conf = []
    all_names = []
    all_original_imgs = []  # Store original images
    all_sam2_masks = []
    
    acc = MulticlassAccuracy(num_classes=num_classes, average=None, ignore_index=ignore_index).to(device)
    iou = JaccardIndex(task='multiclass', num_classes=num_classes, average=None, ignore_index=ignore_index).to(device)
    f1 = MulticlassF1Score(num_classes=num_classes, average=None, ignore_index=ignore_index).to(device)
    conf = MulticlassConfusionMatrix(num_classes, ignore_index=ignore_index).to(device)
    model.to(device)
    
    if save_metrics_batch_limit is not None and save_metrics_batch_limit < save_imgs_batch_limit:
        save_metrics_batch_limit = save_imgs_batch_limit
    processed_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if save_metrics_batch_limit is not None and processed_batches >= save_metrics_batch_limit:
                break
            
            imgs = batch['image'].to(device)
            masks = batch['mask'].to(device)
            names = batch.get('name', [f"img_{i}" for i in range(imgs.size(0))])
            if isinstance(names, str):
                names = [names]
            elif isinstance(names, tuple):
                names = list(names)
            
            # Get original images if available (for visualization)
            original_imgs = batch.get('original_image', None)
            if original_imgs is None:
                # Fallback: convert normalized images back (not ideal but works)
                original_imgs = []
                for img in imgs:
                    img_np = img.cpu().permute(1, 2, 0).numpy()
                    # Denormalize
                    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    img_np = np.clip(img_np, 0, 1)
                    img_np = (img_np * 255).astype(np.uint8)
                    original_imgs.append(img_np)
                original_imgs = np.stack(original_imgs)
            else:
                # Convert list of arrays to numpy array
                if isinstance(original_imgs, (list, tuple)):
                    original_imgs = np.stack([np.array(img) if isinstance(img, torch.Tensor) else img 
                                            for img in original_imgs])
                elif isinstance(original_imgs, torch.Tensor):
                    # Convert from CHW to HWC and ensure uint8
                    if original_imgs.dim() == 4:  # (B, C, H, W)
                        original_imgs = original_imgs.permute(0, 2, 3, 1).cpu().numpy()
                    else:
                        original_imgs = original_imgs.permute(1, 2, 0).cpu().numpy()
                    if original_imgs.max() <= 1.0:
                        original_imgs = (original_imgs * 255).astype(np.uint8)
                    else:
                        original_imgs = original_imgs.astype(np.uint8)
            
            # Get image names if available
            batch_names = batch.get('name', [f"img_{i}" for i in range(imgs.size(0))])
            if isinstance(batch_names, str):
                batch_names = [batch_names]
            elif isinstance(batch_names, tuple):
                batch_names = list(batch_names)
            
            # Run model
            outputs = model(imgs, image_names=batch_names)
            
            # Get top-2 class indices for each pixel
            # --- DEBUGGING: Save outputs to files for debugging ---
            import os

            debug_dir = "debug_outputs"
            os.makedirs(debug_dir, exist_ok=True)

            # Save model softmax probabilities
            probs = torch.softmax(outputs, dim=1)
            

            np.save(os.path.join(debug_dir, f"probs_batch{processed_batches}.npy"),
                    probs.cpu().numpy())


            _, top2_indices = torch.topk(outputs, 2, dim=1)
            preds = top2_indices[:, 0, :, :]
            var = softmax_entropy(torch.softmax(outputs, dim=1), dim=1)
            
            if boost_pr:
                if boost_pr.lower().startswith("r"):  # Recall boosting
                    boost_mask = (preds == 0) & (var > var_th)
                    preds[boost_mask] = top2_indices[:, 1, :, :][boost_mask]
                elif boost_pr.lower().startswith("p"):  # Precision boosting
                    boost_mask = (preds != 0) & (var > var_th)
                    preds[boost_mask] = 0
            
            # Save images if within limit
            if processed_batches < save_imgs_batch_limit:
                all_imgs.append(imgs.cpu())
                all_preds.append(preds.cpu())
                all_masks.append(masks.cpu())
                all_var.append(var.cpu())
                all_original_imgs.append(original_imgs)  # Store original images
                
                # Collect SAM2 mask coverage for visualization
                bbox_prompts_batch = None
                if hasattr(model, '_bbox_prompt_cache'):
                    bbox_prompts_batch = []
                    for name in batch_names:
                        if model._bbox_prompt_cache:
                            bbox_prompts_batch.append(get_prompts_for_image(model._bbox_prompt_cache, name))
                        else:
                            bbox_prompts_batch.append(None)
                    if bbox_prompts_batch and all(p is None for p in bbox_prompts_batch):
                        bbox_prompts_batch = None
                sam2_masks = model.get_sam2_masks_for_training(
                    imgs,
                    image_names=batch_names,
                    bbox_prompts=bbox_prompts_batch
                )
                all_sam2_masks.append(sam2_masks.cpu())
            
            # Save names
            all_names.extend(names)
            
            # Check class presence
            preds_one_hot = F.one_hot(preds, num_classes=num_classes).any(dim=(1, 2))
            masks_ = masks.clone()
            masks_[masks_ == ignore_index] = 0
            masks_one_hot = F.one_hot(masks_, num_classes=num_classes).any(dim=(1, 2))
            class_presence = preds_one_hot | masks_one_hot
            all_class_presence.append(class_presence.cpu())
            
            # Compute per-sample metrics
            for i in range(imgs.size(0)):
                all_acc.append(acc(preds[i], masks[i]).unsqueeze(0))
                all_iou.append(iou(preds[i], masks[i]).unsqueeze(0))
                all_f1.append(f1(preds[i], masks[i]).unsqueeze(0))
                all_conf.append(conf(preds[i], masks[i]).unsqueeze(0))
            
            processed_batches += 1
    
    # Convert to numpy
    if save_imgs_batch_limit > 0 and len(all_imgs) > 0:
        all_imgs_tensor = torch.cat(all_imgs)
        # Convert from CHW to HWC for visualization
        all_imgs_np = all_imgs_tensor.permute(0, 2, 3, 1).cpu().numpy()
        all_preds = torch.cat(all_preds).cpu().numpy()
        all_masks = torch.cat(all_masks).cpu().numpy()
        all_var = torch.cat(all_var).cpu().numpy()
        all_sam2_masks_np = torch.cat(all_sam2_masks).cpu().numpy() if all_sam2_masks else np.array([])
        # Use original images if available
        if len(all_original_imgs) > 0:
            all_original_imgs_np = np.concatenate(all_original_imgs, axis=0)
        else:
            all_original_imgs_np = all_imgs_np  # Fallback to normalized images
    else:
        all_imgs_np = np.array([])
        all_preds = np.array([])
        all_masks = np.array([])
        all_var = np.array([])
        all_sam2_masks_np = np.array([])
        all_original_imgs_np = np.array([])
    
    all_class_presence = torch.cat(all_class_presence).cpu().numpy()
    all_acc = torch.cat(all_acc).cpu().numpy()
    all_iou = torch.cat(all_iou).cpu().numpy()
    all_f1 = torch.cat(all_f1).cpu().numpy()
    all_conf = torch.cat(all_conf).cpu().numpy()
    
    # Filter metrics: NaN where class not present
    flt_acc = np.where(all_class_presence, all_acc, np.nan)
    flt_iou = np.where(all_class_presence, all_iou, np.nan)
    flt_f1 = np.where(all_class_presence, all_f1, np.nan)
    
    # Organize results
    num_saved_imgs = all_imgs_np.shape[0] if isinstance(all_imgs_np, np.ndarray) and all_imgs_np.size > 0 else 0
    all_imgs_dict = {
        'name': all_names[:num_saved_imgs],
        'imgs': all_imgs_np,
        'original_imgs': all_original_imgs_np,  # Add original images
        'sam2_masks': all_sam2_masks_np,
        'preds': all_preds,
        'masks': all_masks,
        'var': all_var
    }
    
    all_metrics_dict = {
        'name': all_names,
        'acc': all_acc,
        'iou': all_iou,
        'f1': all_f1,
        'conf': all_conf
    }
    
    flt_metrics_dict = {
        'name': all_names,
        'acc': flt_acc,
        'iou': flt_iou,
        'f1': flt_f1
    }
    
    return all_imgs_dict, all_metrics_dict, flt_metrics_dict


def visualize_results(
    imgs: np.ndarray,
    masks: np.ndarray,
    preds: np.ndarray,
    colors: List,
    variances: Optional[np.ndarray] = None,
    num_samples: int = 5,
    which: str = 'random',
    save_dir: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    original_imgs: Optional[np.ndarray] = None,
    bbox_prompts: Optional[Dict[str, np.ndarray]] = None,
    image_names: Optional[List[str]] = None,
    sam2_masks: Optional[np.ndarray] = None,
):
    """
    Visualize segmentation results.
    
    Parameters:
    -----------
    imgs : np.ndarray
        Images, shape (N, H, W, 3) or (N, 3, H, W)
    masks : np.ndarray
        Ground truth masks, shape (N, H, W)
    preds : np.ndarray
        Predictions, shape (N, H, W)
    colors : list
        List of colors for each class
    variances : np.ndarray, optional
        Variance/uncertainty maps, shape (N, H, W)
    num_samples : int
        Number of samples to visualize
    which : str
        'first', 'last', 'random', or int (starting index)
    save_dir : str, optional
        Directory to save visualizations
    class_names : list, optional
        List of class names
    original_imgs : np.ndarray, optional
        Original images (uint8, [0, 255]), shape (N, H, W, 3)
        If provided, will be used instead of denormalized imgs
    bbox_prompts : dict, optional
        Dictionary mapping image names to bbox arrays (N, 4) in XYXY format
    image_names : list, optional
        List of image names corresponding to imgs
    sam2_masks : np.ndarray, optional
        SAM2 mask coverage maps (N, H, W)
    """
    # Select indices
    if which == 'first':
        vis_idx = list(range(min(num_samples, imgs.shape[0])))
    elif which == 'last':
        vis_idx = list(range(max(0, imgs.shape[0] - num_samples), imgs.shape[0]))
    elif which == 'random':
        vis_idx = sorted(random.sample(range(imgs.shape[0]), min(num_samples, imgs.shape[0])))
    elif isinstance(which, int):
        start_idx = min(which, imgs.shape[0] - num_samples)
        vis_idx = list(range(start_idx, start_idx + num_samples))
    else:
        vis_idx = list(range(min(num_samples, imgs.shape[0])))
    
    # Use original images if available, otherwise use denormalized images
    use_original = original_imgs is not None and original_imgs.size > 0
    
    # Create colormaps
    custom_cmap = ListedColormap(colors)
    custom_cmap0 = ListedColormap(colors[1:])  # Without background
    
    if class_names:
        legend_patches = [mpatches.Patch(color=colors[1:][i], label=class_names[1:][i]) 
                         for i in range(len(class_names) - 1)]
    
    # Helper function to draw bboxes on image
    def draw_bboxes(img, bboxes, color=(255, 0, 0), thickness=2):
        """Draw bounding boxes on image."""
        img_copy = img.copy()
        if bboxes is not None and len(bboxes) > 0:
            for bbox in bboxes:
                x1, y1, x2, y2 = map(int, bbox[:4])
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
        return img_copy
    
    # Determine number of columns: Image, [Image+BBoxes], [SAM2 Mask], Reference, Prediction, and optionally Variance
    has_bboxes = bbox_prompts is not None and image_names is not None
    has_sam2_masks = sam2_masks is not None and getattr(sam2_masks, 'size', 0) > 0
    num_cols = 3  # Image, Reference, Prediction (base)
    if has_bboxes:
        num_cols += 1  # Add Image+BBoxes column
    if has_sam2_masks:
        num_cols += 1  # Add SAM2 mask overlay column
    if variances is not None:
        num_cols += 1  # Add Variance column
    
    # DEBUG: Print bbox visualization info
    print(f"\n[DEBUG visualize_results]")
    print(f"  bbox_prompts: {type(bbox_prompts)}, len={len(bbox_prompts) if bbox_prompts else 0}")
    print(f"  image_names: {type(image_names)}, len={len(image_names) if image_names else 0}")
    print(f"  has_bboxes: {has_bboxes}")
    print(f"  has_sam2_masks: {has_sam2_masks}")
    print(f"  num_cols: {num_cols}")
    if image_names:
        print(f"  Sample image_names: {image_names[:5]}")
    if bbox_prompts and image_names:
        matched = [name for name in image_names[:10] if name in bbox_prompts]
        print(f"  Matched prompts in first 10: {len(matched)}/10 - {matched}")
    
    # Plot 1: RGB - Image - Image+BBoxes - Reference - Prediction - [Variance]
    plt.figure(figsize=(num_cols * 5, len(vis_idx) * 5), dpi=300)
    for count, i in enumerate(vis_idx):
        col_idx = 0
        
        # Get original or denormalized image
        if use_original:
            img = original_imgs[i].astype(np.float32) / 255.0  # Normalize to [0, 1] for display
            # Fix image shape: ensure (H, W, C) format for matplotlib
            if img.ndim == 3:
                # If shape is (C, H, W), convert to (H, W, C)
                if img.shape[0] == 3 and img.shape[0] < img.shape[1]:
                    img = img.transpose(1, 2, 0)
                # If shape is (H, C, W), convert to (H, W, C)
                elif img.shape[1] == 3 and img.shape[1] < img.shape[0]:
                    img = img.transpose(0, 2, 1)
            # Ensure last dimension is 3 (RGB)
            if img.ndim == 3 and img.shape[-1] != 3:
                if img.shape[0] == 3:
                    img = img.transpose(1, 2, 0)
                elif img.shape[2] == 3:
                    img = img.transpose(0, 1, 2)
        else:
            img = denormalize(imgs[i])
        
        # Column 1: Original image (always show)
        plt.subplot(len(vis_idx), num_cols, count * num_cols + col_idx + 1)
        if count == 0:
            plt.title('Image', fontsize=22)
        plt.imshow(img)
        plt.axis('off')
        col_idx += 1
        
        # Column 2: Image with bboxes (always show if we have bbox_prompts and image_names)
        if has_bboxes:
            # Get original image for bbox drawing (uint8 format)
            if use_original:
                img_for_bbox = original_imgs[i].copy()
            else:
                img_for_bbox = (img * 255).astype(np.uint8)
            
            # CRITICAL FIX: Ensure image is (H, W, C) format BEFORE drawing bboxes
            # cv2.rectangle requires (H, W, C) format
            if img_for_bbox.ndim == 3:
                if img_for_bbox.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                    img_for_bbox = img_for_bbox.transpose(1, 2, 0)
                elif img_for_bbox.shape[1] == 3 and img_for_bbox.shape[1] < img_for_bbox.shape[0]:  # (H, C, W) -> (H, W, C)
                    img_for_bbox = img_for_bbox.transpose(0, 2, 1)
            
            # Check if this image has prompts
            image_name = image_names[i] if i < len(image_names) else None
            if image_name and image_name in bbox_prompts:
                bboxes = bbox_prompts[image_name]
                if count == 0:  # DEBUG: Only print for first image
                    print(f"  [DEBUG] Drawing bboxes for '{image_name}': {bboxes.shape[0]} boxes")
                    print(f"  [DEBUG] img_for_bbox shape (after fix): {img_for_bbox.shape}, dtype: {img_for_bbox.dtype}")
                img_with_bboxes = draw_bboxes(img_for_bbox, bboxes, color=(255, 0, 0), thickness=2)
                if count == 0:
                    print(f"  [DEBUG] After draw_bboxes: shape={img_with_bboxes.shape}, same_as_input={np.array_equal(img_with_bboxes, img_for_bbox)}")
            else:
                # No prompts for this image, just show original
                if count == 0:  # DEBUG: Only print for first image
                    print(f"  [DEBUG] No prompts for '{image_name}'")
                img_with_bboxes = img_for_bbox
            
            plt.subplot(len(vis_idx), num_cols, count * num_cols + col_idx + 1)
            if count == 0:
                plt.title('Image + BBoxes', fontsize=22)
            # Image is already in (H, W, C) format from above fix
            plt.imshow(img_with_bboxes.astype(np.float32) / 255.0)
            plt.axis('off')
            col_idx += 1
        
        if has_sam2_masks:
            mask_overlay = sam2_masks[i]
            if mask_overlay.ndim > 2:
                mask_overlay = np.squeeze(mask_overlay)
            mask_overlay = mask_overlay.astype(np.float32)
            mask_overlay = mask_overlay - mask_overlay.min()
            if mask_overlay.max() > 0:
                mask_overlay = mask_overlay / (mask_overlay.max() + 1e-8)
            plt.subplot(len(vis_idx), num_cols, count * num_cols + col_idx + 1)
            if count == 0:
                plt.title('SAM2 Mask', fontsize=22)
            plt.imshow(img)
            plt.imshow(mask_overlay, cmap='plasma', alpha=0.45)
            plt.axis('off')
            col_idx += 1
        
        # Reference
        plt.subplot(len(vis_idx), num_cols, count * num_cols + col_idx + 1)
        if count == 0:
            plt.title('Reference', fontsize=22)
        plt.imshow(masks[i], cmap=custom_cmap, vmin=0, vmax=len(colors) - 1)
        plt.axis('off')
        col_idx += 1
        
        # Prediction
        plt.subplot(len(vis_idx), num_cols, count * num_cols + col_idx + 1)
        if count == 0:
            plt.title('Prediction', fontsize=22)
        plt.imshow(preds[i], cmap=custom_cmap, vmin=0, vmax=len(colors) - 1)
        plt.axis('off')
        col_idx += 1
        
        # Variance (if available)
        if variances is not None:
            plt.subplot(len(vis_idx), num_cols, count * num_cols + col_idx + 1)
            if count == 0:
                plt.title('Variance', fontsize=22)
            plt.imshow(variances[i], cmap=matplotlib.colormaps['viridis'], vmin=0)
            plt.axis('off')
            col_idx += 1
    
    if class_names:
        plt.figlegend(
            handles=legend_patches,
            loc='center left',
            bbox_to_anchor=(1.0, 0.5),
            ncol=2,
            fontsize=16,
            borderaxespad=0.
        )
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'qual_' + str(which) + '.jpg'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # NOTE: Removed redundant "overlay" visualization (grayscale with overlaid masks)
    # The detailed "qual" visualization above already shows everything needed:
    # Image, Image+BBoxes, Reference, Prediction, and Variance


__all__ = [
    'evaluate_model',
    'evaluate_model_samplewise',
    'visualize_results',
    'visualize_save_confusions',
    'visualize_scores_per_class',
    'softmax_entropy',
    'convert_ndarray_to_list',
    'denormalize',
]
