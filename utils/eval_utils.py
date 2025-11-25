#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:52:49 2024

@author: lukas
"""
import sys, os, io, time
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import random

from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix
from torchmetrics import JaccardIndex
from sklearn.metrics import ConfusionMatrixDisplay


# =============================================================================
# Utility functions
# =============================================================================
def denormalize(img: np.ndarray, mean: list = [0.485, 0.456, 0.406], std: list = [0.229, 0.224, 0.225]) -> np.ndarray:
    """
    Denormalize an image tensor back to [0, 1] range.
    
    Parameters:
    -----------
    img : np.ndarray
        Normalized image of shape (C, H, W) or (H, W, C)
    mean : list
        Mean values used for normalization
    std : list
        Std values used for normalization
    
    Returns:
    --------
    denorm_img : np.ndarray
        Denormalized image in [0, 1] range with shape (H, W, C)
    """
    if img.ndim == 3:
        if img.shape[0] == 3:  # (C, H, W)
            img = np.transpose(img, (1, 2, 0))  # -> (H, W, C)
    
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    
    denorm = img * std + mean
    denorm = np.clip(denorm, 0, 1)
    
    return denorm


def convert_ndarray_to_list(obj):
    """
    Recursively convert numpy arrays to lists for JSON serialization.
    
    Parameters:
    -----------
    obj : any
        Object to convert (dict, list, np.ndarray, or scalar)
    
    Returns:
    --------
    converted : any
        Object with all numpy arrays converted to lists
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_ndarray_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


# =============================================================================
def evaluate_model(model, dataloader, num_classes, device, ignore_index=-1, boost_pr=None, var_th=0.5, custom_avg=None, save_dir=None):
    '''
    Evaluates a semantic segmentation model using a confusion matrix and patch-based classification metrics.
    
    Parameters:
    ----------
    model : torch.nn.Module
        The trained segmentation model.
    dataloader : torch.utils.data.DataLoader
        DataLoader containing the evaluation dataset.
    num_classes : int
        Number of classes in the segmentation task.
    device : torch.device
        Device to run the evaluation on (e.g., 'cuda' or 'cpu').
    boost_pr : str, optional
        Strategy to boost model performance. 
        - "recall" (or variants like "r", "Recall") increases recall by replacing uncertain background (0) pixels with the second most likely class.
        - "precision" (or variants like "p", "Precision") increases precision by setting uncertain non-background pixels to background (0).
        Default is None (no boosting).
    var_th : float, optional
        Variance threshold for applying recall or precision boosting. Predictions with uncertainty higher than this threshold are modified.
        Default is 0.5.
    custom_avg: dict, optional
        Contains (multiple) lists of class indices (keys: name of the particular data subset) over which the scores are to be averaged.
    save_dir : str, optional
        Directory to save evaluation metrics as JSON files.
    
    Returns:
    -------
    conf : np.ndarray
        Confusion matrix for semantic segmentation evaluation.
    metrics_semseg : dict
        Dictionary containing overall accuracy (OA) + per-class and averaged precision (=ua: users accuracy), recall (=pa: producers accuracy), IoU, and F1 scores for semantic segmentation.
    '''
    
    model.eval()
    model.to(device)
    
    # # Initialize confusion matrix counters for semseg and classification (each patch size and class)
    conf_ = MulticlassConfusionMatrix(num_classes, ignore_index=ignore_index).to(device)

    with torch.no_grad():
        for batch in dataloader:
            imgs = batch['image'].to(device)
            masks = batch['mask'].to(device)
            outputs = model(imgs) # run model and update confusion matrix
            _, top2_indices = torch.topk(outputs, 2, dim=1) # Get top-2 class indices for each pixel- Shape: (batch, 2, H, W)
            preds = top2_indices[:, 0, :, :] # Get top-1 class as prediction - Shape: (batch, H, W)
            var = softmax_entropy(outputs, dim=1)
                
            if boost_pr: # Boost Precision or Recall         
                if boost_pr.lower().startswith("r"):  # Recall boosting
                    boost_mask = (preds == 0) & (var > var_th)
                    preds[boost_mask] = top2_indices[:, 1, :, :][boost_mask]
                elif boost_pr.lower().startswith("p"):  # Precision boosting
                    boost_mask = (preds != 0) & (var > var_th)
                    preds[boost_mask] = 0
            
            # # update confusion matrix
            conf_.update(preds, masks)
        
    # # Semantic segmentation metrics
    conf = conf_.compute().cpu().numpy()
    TP = np.diag(conf)
    FP = conf.sum(axis=0) - TP
    FN = conf.sum(axis=1) - TP
    
    oa = np.sum(TP)/np.sum(conf)
    ua = np.divide(TP, (TP+FP), out=np.full_like(TP, np.nan, dtype=float), where=(TP+FP) != 0)
    pa = np.divide(TP, (TP+FN), out=np.full_like(TP, np.nan, dtype=float), where=(TP+FN) != 0)
    iou = np.divide(TP, (TP+FP+FN), out=np.full_like(TP, np.nan, dtype=float), where=(TP+FP+FN) != 0)
    f1 = np.divide(TP, TP+0.5*(FN+FP), out=np.full_like(TP, np.nan, dtype=float), where=(TP+FP+FN) != 0)

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
    
    if save_dir:
        with open(os.path.join(save_dir,'metrics_semseg.json'), "w") as f:
            json.dump(convert_ndarray_to_list(metrics_semseg), f, indent=4)
            
    metrics = {'conf': conf,
               'semseg': metrics_semseg,
               }       

    return metrics


# =============================================================================
def visualize_scores_per_class(scores, save_name, display_labels, fontsize=5):
    # Number of classes and scores
    num_classes = len(display_labels)
    score_names = list(scores.keys())
    num_scores = len(score_names)
    
    # Define the positions of the bars for each class
    x = np.arange(num_classes)  # the label locations
    width = 0.5 / num_scores  # width of each bar, adjusted to fit all scores for each class
    group_offset = (num_scores - 1) * width / 2  # Offset to center each group on the tick

    # Define a colorblind-friendly palette from Color Brewer
    colors = plt.get_cmap("Set2")(np.linspace(0, 1, num_scores))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot each score as a separate set of bars
    for i, (score_name, color) in enumerate(zip(score_names, colors)):
        score_values = scores[score_name]
        bar_positions = x - group_offset + i * width
        ax.bar(bar_positions, score_values, width, label=score_name, color=color)
        
    # Place the legend at the top center, outside the plot
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

    
# =============================================================================
def visualize_save_confusion(confusion, save_name, display_labels, fontsize=3):
    disp = ConfusionMatrixDisplay(confusion,display_labels=display_labels)
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 8))  # Optionally, specify the size of the plot
    disp.plot(ax=ax, cmap='Blues', values_format='.0f', colorbar=False)  # Adjust format as needed
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
    for text in ax.texts:
        text.set_fontsize(fontsize)
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.tight_layout()
    plt.savefig(save_name, dpi=500, bbox_inches='tight')
    plt.close()


# =============================================================================
def visualize_save_confusions(confusion, save_dir, display_labels=None):
    # # create confusion matrices with different norms
    conf_all = confusion
    conf_norm_all = np.round(np.nan_to_num(confusion/np.sum(confusion),nan=0)*100,0)
    conf_norm_pred = np.round(np.nan_to_num(confusion/np.sum(confusion,axis=0)[np.newaxis,:],nan=0)*100,0)
    conf_norm_true = np.round(np.nan_to_num(confusion/np.sum(confusion,axis=1)[:,np.newaxis],nan=0)*100,0)
    
    # # visualize and save confusion matrices (without normalization)
    visualize_save_confusion(conf_all, os.path.join(save_dir,'confusion.jpg'), display_labels)
    visualize_save_confusion(conf_norm_all, os.path.join(save_dir,'confusion_norm_all.jpg'), display_labels)
    visualize_save_confusion(conf_norm_pred, os.path.join(save_dir,'confusion_norm_pred.jpg'), display_labels)
    visualize_save_confusion(conf_norm_true, os.path.join(save_dir,'confusion_norm_true.jpg'), display_labels)
    

# =============================================================================
def evaluate_model_samplewise(model, dataloader, num_classes, device, ignore_index=-1, boost_pr=None, var_th=0.5, save_metrics_batch_limit=None, save_imgs_batch_limit=10):
    
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
                break  # Stop processing if batch limit is reached
            imgs = batch['image'].to(device)
            masks = batch['mask'].to(device)
            name = batch['name']
            outputs = model(imgs) # run model and update confusion matrix
            _, top2_indices = torch.topk(outputs, 2, dim=1) # Get top-2 class indices for each pixel- Shape: (batch, 2, H, W)
            preds = top2_indices[:, 0, :, :] # Get top-1 class as prediction - Shape: (batch, H, W)
            var = softmax_entropy(outputs, dim=1)
            
            if boost_pr: # Boost Precision or Recall        
                if boost_pr.lower().startswith("r"):  # Recall boosting
                    boost_mask = (preds == 0) & (var > var_th)
                    preds[boost_mask] = top2_indices[:, 1, :, :][boost_mask]
                elif boost_pr.lower().startswith("p"):  # Precision boosting
                    boost_mask = (preds != 0) & (var > var_th)
                    preds[boost_mask] = 0
                    
            # # save imgs and masks
            if processed_batches < save_imgs_batch_limit:
                all_imgs.append(imgs)
                all_preds.append(preds)
                all_masks.append(masks)
                all_var.append(var)
            
            # # save names
            all_names.append(name)

            # # check samplewise if classes occur in preds OR masks (=TRUE), if not (=FALSE)
            # # Then combine presence information from both predictions and masks
            preds_one_hot = F.one_hot(preds, num_classes=num_classes).any(dim=(1, 2))
            masks_ = masks 
            masks_[masks_ == ignore_index] = 0 # little hack to avoid CUDA error
            masks_one_hot = F.one_hot(masks_, num_classes=num_classes).any(dim=(1, 2))
            class_presence = preds_one_hot | masks_one_hot
            all_class_presence.append(class_presence)
            
            # # for some metrics, there are no parameters to calculate them samplewise, so its done manually by iterating over the batch dimension
            for i in range(imgs.size(0)):
                all_acc.append(acc(preds[i], masks[i]).unsqueeze(0))
                all_iou.append(iou(preds[i], masks[i]).unsqueeze(0))
                all_f1.append(f1(preds[i], masks[i]).unsqueeze(0))
                all_conf.append(conf(preds[i], masks[i]).unsqueeze(0))
            
            processed_batches += 1

    # # cat list of batches to numpy
    if save_imgs_batch_limit>0:
        all_imgs = torch.cat(all_imgs).cpu().numpy()
        all_preds = torch.cat(all_preds).cpu().numpy()
        all_masks = torch.cat(all_masks).cpu().numpy()
        all_var = torch.cat(all_var).cpu().numpy()
    else:
        all_imgs = all_preds = all_masks = all_var = []
    all_class_presence = torch.cat(all_class_presence).cpu().numpy()
    all_acc = torch.cat(all_acc).cpu().numpy()
    all_iou = torch.cat(all_iou).cpu().numpy()
    all_f1 = torch.cat(all_f1).cpu().numpy()
    all_conf = torch.cat(all_conf).cpu().numpy()
    all_names = sum(all_names, [])
    
    # # if class IS NOT in mask OR pred, so all_class_presence = False for this sample, set nan in filtered (flt) metric list
    flt_acc = np.where(all_class_presence, all_acc, torch.tensor(float('nan')))
    flt_iou = np.where(all_class_presence, all_iou, torch.tensor(float('nan')))
    flt_f1 = np.where(all_class_presence, all_f1, torch.tensor(float('nan')))
    
    # # summarize metrics in dictionaries
    all_imgs = {
        'name': all_names[:all_imgs.shape[0]] if isinstance(all_imgs, np.ndarray) and all_imgs.size > 0 else [],
        'imgs': all_imgs,
        'preds': all_preds,
        'masks': all_masks,
        'var': all_var}
    
    all_metrics = {
        'name': all_names,
        'acc': all_acc,
        'iou': all_iou,
        'f1': all_f1,
        'conf': all_conf}
    
    flt_metrics = {
        'name': all_names,
        'acc': flt_acc,
        'iou': flt_iou,
        'f1': flt_f1}
    
    return all_imgs, all_metrics, flt_metrics

# =============================================================================
def visualize_results(imgs, masks, preds, colors, variances=None, num_samples=5, which='random', save_dir=None, class_names=None):
    if which=='first':
        vis_idx = range(num_samples)
    elif which=='last':
        vis_idx = range(imgs.shape[0]-num_samples,imgs.shape[0])
    elif which=='random':
        vis_idx = np.sort(random.sample(range(imgs.shape[0]), num_samples))
    elif isinstance(which, int):
        if which>imgs.shape[0]-num_samples:
            which = imgs.shape[0]-num_samples
            print('-which- was chosen too large: It was overwritten to just visualize the last num_sample images')
        vis_idx = range(which,which+num_samples)
    
    # Create a custom colormap
    custom_cmap = ListedColormap(colors)
    custom_cmap0 = ListedColormap(colors[1:]) # without background class
    
    if class_names:
        legend_patches = [mpatches.Patch(color=colors[1:][i], label=class_names[1:][i]) for i in range(len(class_names)-1)]
    
    # 1. plot: RGB - Reference - Prediction - [Variance]
    if variances is not None:
        plt.figure(figsize=(15, num_samples * 5), dpi=300)
        count=0
        for i in vis_idx:
            img = denormalize(imgs[i])
    
            plt.subplot(num_samples, 4, count * 4 + 1)
            if count == 0: plt.title('Image',fontsize=22)
            plt.imshow(img)
            plt.axis('off')
    
            plt.subplot(num_samples, 4, count * 4 + 2)
            if count == 0: plt.title('Reference',fontsize=22)
            plt.imshow(masks[i], cmap=custom_cmap, vmin=0, vmax=len(colors)-1)
            plt.axis('off')
    
            plt.subplot(num_samples, 4, count * 4 + 3)
            if count == 0: plt.title('Prediction',fontsize=22)
            plt.imshow(preds[i], cmap=custom_cmap, vmin=0, vmax=len(colors)-1)
            plt.axis('off')
            
            plt.subplot(num_samples, 4, count * 4 + 4)
            if count == 0: plt.title('Variance',fontsize=22)
            plt.imshow(variances[i], cmap=matplotlib.colormaps['viridis'], vmin=0) # other nice cmap: afmhot
            plt.axis('off')
            
            count += 1
    else:
        plt.figure(figsize=(15, num_samples * 5), dpi=300)
        count=0
        for i in vis_idx:
            img = denormalize(imgs[i])
    
            plt.subplot(num_samples, 3, count * 3 + 1)
            if count == 0: plt.title('Image',fontsize=22)
            plt.imshow(img)
            plt.axis('off')
    
            plt.subplot(num_samples, 3, count * 3 + 2)
            if count == 0: plt.title('Reference',fontsize=22)
            plt.imshow(masks[i], cmap=custom_cmap, vmin=0, vmax=len(colors)-1)
            plt.axis('off')
    
            plt.subplot(num_samples, 3, count * 3 + 3)
            if count == 0: plt.title('Prediction',fontsize=22)
            plt.imshow(preds[i], cmap=custom_cmap, vmin=0, vmax=len(colors)-1)
            plt.axis('off')
            
            count += 1
            
    if class_names:
        plt.figlegend(
            handles=legend_patches,
            loc='center left',                # Anchor the legend to the left-center of bbox
            bbox_to_anchor=(1.0, 0.5),      # Place it outside the figure to the right
            ncol=2,                           # Two columns
            fontsize=16,                      # Adjust font size
            borderaxespad=0.
        )
    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir,'qual_'+str(which)+'.jpg'), dpi=300, bbox_inches='tight')
    plt.close()
    
    
    # 2. plot: Grayscaled image with overlaid masks (excluding zero values)
    plt.figure(figsize=(10, num_samples * 5), dpi=300)
    count = 0
    for i in vis_idx:
        img = denormalize(imgs[i])
    
        # Convert the image to grayscale
        gray_image = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    
        # Prepare the masks with zero values masked out
        ref_mask = np.ma.masked_where(masks[i] == 0, masks[i])  # Mask out zero values in the reference mask
        pred_mask = np.ma.masked_where(preds[i] == 0, preds[i])  # Mask out zero values in the prediction mask
    
        # Plot the true image with overlayed reference mask
        plt.subplot(num_samples, 2, count * 2 + 1)
        if count == 0: plt.title('Reference',fontsize=22)
        plt.imshow(gray_image, cmap='gray')
        plt.imshow(ref_mask, cmap=custom_cmap0, vmin=1, vmax=len(colors)-1, alpha=0.5)  # Overlay reference mask with transparency
        plt.axis('off')
    
        # Plot the true image with overlayed prediction mask
        plt.subplot(num_samples, 2, count * 2 + 2)
        if count == 0: plt.title('Prediction',fontsize=22)
        plt.imshow(gray_image, cmap='gray')
        plt.imshow(pred_mask, cmap=custom_cmap0, vmin=1, vmax=len(colors)-1, alpha=0.5)  # Overlay prediction mask with transparency
        plt.axis('off')
    
        count += 1
    
    if class_names:
        plt.figlegend(
            handles=legend_patches,
            loc='center left',                # Anchor the legend to the left-center of bbox
            bbox_to_anchor=(1.0, 0.5),      # Place it outside the figure to the right
            ncol=2,                           # Two columns
            fontsize=22,                      # Adjust font size
            borderaxespad=0.
        )
    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir,'overlay_'+str(which)+'.jpg'), dpi=300, bbox_inches='tight')
    plt.close()
    
    
# =============================================================================
def softmax_entropy(probabilities, dim):
    return -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=dim)


# =============================================================================
# Simple Dataset for Testing
# =============================================================================
class SimpleImageMaskDataset(Dataset):
    """
    Simple dataset for loading images and masks from directories.
    Used for testing with evaluate_model().
    """
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        
        # Find all images
        self.image_files = sorted(list(self.image_dir.glob('*.jpg')) + 
                                  list(self.image_dir.glob('*.png')) + 
                                  list(self.image_dir.glob('*.tif')))
        
        # Match labels
        self.label_files = []
        for img_file in self.image_files:
            stem = img_file.stem
            label_candidates = (list(self.label_dir.glob(f"{stem}.*")))
            if label_candidates:
                self.label_files.append(label_candidates[0])
            else:
                print(f"Warning: No label found for {img_file.name}")
                self.label_files.append(None)
        
        # Filter out images without labels
        valid_pairs = [(img, lbl) for img, lbl in zip(self.image_files, self.label_files) if lbl is not None]
        self.image_files = [pair[0] for pair in valid_pairs]
        self.label_files = [pair[1] for pair in valid_pairs]
        
        print(f"Loaded {len(self.image_files)} image-label pairs")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Load mask
        mask_path = self.label_files[idx]
        mask = Image.open(mask_path)
        mask = np.array(mask).astype(np.int64)
        
        # Apply transform to image
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        # Convert mask to tensor
        mask = torch.from_numpy(mask).long()
        
        return {
            'image': image,
            'mask': mask,
            'name': img_path.stem
        }


# =============================================================================
def test_model_on_dataset(
    model,
    image_dir,
    label_dir,
    num_classes,
    device='cuda',
    batch_size=8,
    num_workers=4,
    ignore_index=-1,
    boost_pr=None,
    var_th=0.5,
    custom_avg=None,
    save_dir=None,
    image_transform=None,
    class_names=None,
    class_colors=None,
    visualize=True,
    num_viz_samples=10
):
    """
    Test a semantic segmentation model on a dataset directory.
    
    This is a convenience wrapper around evaluate_model() that:
    1. Creates a DataLoader from image/label directories
    2. Calls evaluate_model() with the DataLoader
    3. Optionally visualizes confusion matrices and qualitative results
    
    Parameters:
    ----------
    model : torch.nn.Module
        The trained segmentation model.
    image_dir : str or Path
        Directory containing test images.
    label_dir : str or Path
        Directory containing test labels (masks).
    num_classes : int
        Number of classes in the segmentation task.
    device : str or torch.device
        Device to run evaluation on (e.g., 'cuda' or 'cpu').
    batch_size : int, optional
        Batch size for DataLoader. Default is 8.
    num_workers : int, optional
        Number of workers for DataLoader. Default is 4.
    ignore_index : int, optional
        Index to ignore in metrics computation. Default is -1.
    boost_pr : str, optional
        Strategy to boost precision/recall. See evaluate_model() for details.
    var_th : float, optional
        Variance threshold for boosting. Default is 0.5.
    custom_avg : dict, optional
        Custom class groupings for averaging metrics.
    save_dir : str or Path, optional
        Directory to save evaluation results and visualizations.
    image_transform : torchvision.transforms, optional
        Transform to apply to images. If None, uses default normalization.
    class_names : dict, optional
        Dictionary mapping class indices to names (for visualization).
    class_colors : dict or list, optional
        Colors for each class (for visualization).
    visualize : bool, optional
        Whether to generate visualizations. Default is True.
    num_viz_samples : int, optional
        Number of samples to visualize. Default is 10.
    
    Returns:
    -------
    metrics : dict
        Dictionary containing:
        - 'conf': confusion matrix (np.ndarray)
        - 'semseg': semantic segmentation metrics (dict)
    
    Example:
    -------
    >>> results = test_model_on_dataset(
    ...     model=my_model,
    ...     image_dir='./data/test/images',
    ...     label_dir='./data/test/labels',
    ...     num_classes=61,
    ...     save_dir='./results/test',
    ...     class_names=class_names_dict
    ... )
    >>> print(f"Overall Accuracy: {results['semseg']['OA']:.4f}")
    >>> print(f"Mean F1: {results['semseg']['F1-avg']:.4f}")
    """
    
    # Setup save directory
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Default image transform (ImageNet normalization)
    if image_transform is None:
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Create dataset and dataloader
    print("Creating test dataset...")
    dataset = SimpleImageMaskDataset(image_dir, label_dir, transform=image_transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device != 'cpu' else False
    )
    
    # Run evaluation using existing evaluate_model function
    print("\nRunning evaluation with confusion matrix + torchmetrics...")
    print("=" * 80)
    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        num_classes=num_classes,
        device=device,
        ignore_index=ignore_index,
        boost_pr=boost_pr,
        var_th=var_th,
        custom_avg=custom_avg,
        save_dir=save_dir
    )
    
    # Print metrics
    print("\n" + "=" * 80)
    print("Test Results (using torchmetrics)")
    print("=" * 80)
    semseg = metrics['semseg']
    print(f"Overall Accuracy (OA):        {semseg['OA']:.4f}")
    print(f"Mean F1 Score:                {semseg['F1-avg']:.4f}")
    print(f"Mean F1 (excluding bg):       {semseg['F1-avg-wo0']:.4f}")
    print(f"Mean IoU (mIoU):              {semseg['IoU-avg']:.4f}")
    print(f"Mean IoU (excluding bg):      {semseg['IoU-avg-wo0']:.4f}")
    print(f"Mean Precision:               {semseg['Precision-avg']:.4f}")
    print(f"Mean Precision (excluding bg):{semseg['Precision-avg-wo0']:.4f}")
    print(f"Mean Recall:                  {semseg['Recall-avg']:.4f}")
    print(f"Mean Recall (excluding bg):   {semseg['Recall-avg-wo0']:.4f}")
    
    if custom_avg:
        print("\nCustom Group Averages:")
        for group_name in custom_avg.keys():
            print(f"  {group_name}:")
            print(f"    F1:        {semseg[f'F1-avg-{group_name}']:.4f}")
            print(f"    IoU:       {semseg[f'IoU-avg-{group_name}']:.4f}")
            print(f"    Precision: {semseg[f'Precision-avg-{group_name}']:.4f}")
            print(f"    Recall:    {semseg[f'Recall-avg-{group_name}']:.4f}")
    
    # Visualize if requested
    if visualize and save_dir:
        print("\nGenerating visualizations...")
        
        # Confusion matrices
        if class_names:
            display_labels = [class_names.get(i, f'Class_{i}') for i in range(num_classes)]
        else:
            display_labels = [f'Class_{i}' for i in range(num_classes)]
        
        visualize_save_confusions(metrics['conf'], save_dir, display_labels)
        
        # Per-class scores bar chart
        scores = {
            'F1': semseg['F1'],
            'IoU': semseg['IoU'],
            'Precision': semseg['Precision'],
            'Recall': semseg['Recall']
        }
        visualize_scores_per_class(
            scores,
            save_name=str(save_dir / 'scores_per_class.jpg'),
            display_labels=display_labels,
            fontsize=5
        )
        
        # Qualitative results (sample-wise visualization)
        print("\nGenerating qualitative visualizations...")
        all_imgs, all_metrics_dict, flt_metrics = evaluate_model_samplewise(
            model=model,
            dataloader=dataloader,
            num_classes=num_classes,
            device=device,
            ignore_index=ignore_index,
            boost_pr=boost_pr,
            var_th=var_th,
            save_metrics_batch_limit=None,
            save_imgs_batch_limit=min(num_viz_samples, len(dataset))
        )
        
        if class_colors:
            if isinstance(class_colors, dict):
                colors = [class_colors.get(i, [0, 0, 0]) for i in range(num_classes)]
            else:
                colors = class_colors
        else:
            # Generate default colors
            colors = plt.cm.tab20(np.linspace(0, 1, num_classes))[:, :3]
        
        # Visualize first N samples
        if len(all_imgs['imgs']) > 0:
            visualize_results(
                imgs=all_imgs['imgs'],
                masks=all_imgs['masks'],
                preds=all_imgs['preds'],
                colors=colors,
                variances=all_imgs['var'],
                num_samples=min(5, len(all_imgs['imgs'])),
                which='first',
                save_dir=save_dir,
                class_names=display_labels if class_names else None
            )
        
        print(f"\n✅ All results saved to: {save_dir}")
    
    print("=" * 80)
    return metrics
