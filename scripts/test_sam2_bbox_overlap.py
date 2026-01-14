#!/usr/bin/env python3
"""Test SAM2 with bbox prompts, especially overlapping bboxes.

Manually input one or multiple bboxes (can overlap) and see what masks
SAM2 outputs under the code constraints (restrict_to_bbox, bbox_margin, etc.).
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from omegaconf import OmegaConf
from hydra.utils import instantiate

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def build_sam2_from_config_path(config_path: str, ckpt_path: str, device: str = "cuda"):
    """Build SAM2 model from config path."""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    cfg = OmegaConf.load(config_path)
    
    if 'trainer' in cfg and 'model' in cfg.trainer:
        model_cfg = cfg.trainer.model
    elif 'model' in cfg:
        model_cfg = cfg.model
    else:
        raise ValueError(f"Cannot find 'model' config in {config_path}")
    
    try:
        OmegaConf.resolve(model_cfg)
    except Exception as e:
        print(f"Warning: Config resolution had issues: {e}")
        if 'image_size' in model_cfg and isinstance(model_cfg.image_size, str) and '$' in str(model_cfg.image_size):
            if 'scratch' in cfg and 'resolution' in cfg.scratch:
                model_cfg.image_size = cfg.scratch.resolution
    
    model = instantiate(model_cfg, _recursive_=True)
    
    from sam2.build_sam import _load_checkpoint
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    model.eval()
    
    return model


def apply_bbox_constraint(
    mask: np.ndarray,
    bbox: tuple,
    H: int,
    W: int,
    restrict_to_bbox: bool = True,
    bbox_margin: float = 0.15,
) -> np.ndarray:
    """Apply bbox constraint to mask (same logic as in precompute_sam2_instances.py).
    
    Args:
        mask: Boolean mask (H, W)
        bbox: (x1, y1, x2, y2)
        H, W: Image dimensions
        restrict_to_bbox: Whether to restrict mask to expanded bbox
        bbox_margin: Margin ratio for bbox expansion
    
    Returns:
        Constrained mask
    """
    if not restrict_to_bbox:
        return mask
    
    x1, y1, x2, y2 = bbox
    
    # Calculate expanded bbox with margin
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    margin_x = int(bbox_w * bbox_margin)
    margin_y = int(bbox_h * bbox_margin)
    
    exp_x1 = max(0, int(x1 - margin_x))
    exp_y1 = max(0, int(y1 - margin_y))
    exp_x2 = min(W - 1, int(x2 + margin_x))
    exp_y2 = min(H - 1, int(y2 + margin_y))
    
    # Create expanded bbox mask
    bbox_mask = np.zeros((H, W), dtype=bool)
    bbox_mask[exp_y1:exp_y2+1, exp_x1:exp_x2+1] = True
    
    # Apply constraint
    constrained_mask = mask & bbox_mask
    
    return constrained_mask, (exp_x1, exp_y1, exp_x2, exp_y2)


def visualize_bbox_predictions(
    image: np.ndarray,
    bboxes: list,
    all_masks: list,  # List of lists: each bbox has multiple candidate masks
    all_scores: list,  # List of lists: each bbox has multiple scores
    expanded_bboxes: list,
    output_path: Path,
    restrict_to_bbox: bool,
    bbox_margin: float,
):
    """Visualize SAM2 predictions for multiple bboxes with multimask output.
    
    Args:
        image: RGB image array (H, W, 3)
        bboxes: List of (x1, y1, x2, y2) tuples
        all_masks: List of lists - each bbox has multiple candidate masks
        all_scores: List of lists - each bbox has multiple scores
        expanded_bboxes: List of expanded bbox coordinates
        output_path: Path to save visualization
        restrict_to_bbox: Whether bbox constraint was applied
        bbox_margin: Margin ratio used
    """
    H, W = image.shape[:2]
    n_bboxes = len(bboxes)
    
    # Find max number of masks per bbox
    max_masks = max(len(masks) for masks in all_masks)
    
    # Create figure: one row per bbox, one column per candidate mask
    fig = plt.figure(figsize=(max_masks * 5, n_bboxes * 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_bboxes))
    
    # Plot: one row per bbox, one column per candidate mask
    plot_idx = 1
    for bbox_idx, (bbox, bbox_masks, bbox_scores) in enumerate(zip(bboxes, all_masks, all_scores)):
        x1, y1, x2, y2 = bbox
        
        for mask_idx, (mask, score) in enumerate(zip(bbox_masks, bbox_scores)):
            ax = plt.subplot(n_bboxes, max_masks, plot_idx)
            ax.imshow(image)
            
            # Draw original bbox
            rect1 = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=colors[bbox_idx], facecolor='none', linestyle='-'
            )
            ax.add_patch(rect1)
            
            # Draw expanded bbox if constraint was applied
            if restrict_to_bbox and expanded_bboxes and expanded_bboxes[bbox_idx]:
                exp_x1, exp_y1, exp_x2, exp_y2 = expanded_bboxes[bbox_idx]
                rect2 = patches.Rectangle(
                    (exp_x1, exp_y1), exp_x2 - exp_x1, exp_y2 - exp_y1,
                    linewidth=1, edgecolor='yellow', facecolor='none', linestyle='--'
                )
                ax.add_patch(rect2)
            
            # Draw mask in red
            mask_rgba = np.zeros((H, W, 4))
            mask_rgba[mask] = [1, 0, 0, 0.5]  # Red with transparency
            ax.imshow(mask_rgba)
            
            # Title
            title = f'BBox {bbox_idx+1}, Mask {mask_idx+1}\n'
            title += f'Score: {score:.3f}, Area: {mask.sum()} px'
            if mask_idx == 0:
                title += f'\n({x1},{y1})-({x2},{y2})'
            ax.set_title(title, fontsize=10)
            ax.axis('off')
            
            plot_idx += 1
    
    # If there are empty slots, hide them
    total_slots = n_bboxes * max_masks
    for idx in range(plot_idx, total_slots + 1):
        ax = plt.subplot(n_bboxes, max_masks, idx)
        ax.axis('off')
    
    plt.suptitle(f'SAM2 Multimask Output (restrict_to_bbox={restrict_to_bbox}, margin={bbox_margin:.0%})',
                fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to: {output_path}")
    plt.close()


def visualize_zorder_overlay(
    image: np.ndarray,
    bboxes: list,
    all_masks: list,
    all_scores: list,
    expanded_bboxes: list,
    output_path: Path,
    restrict_to_bbox: bool,
    bbox_margin: float,
):
    """Visualize Z-order overlay: large boxes first (red), small boxes later (green, can override).
    
    Args:
        image: RGB image array (H, W, 3)
        bboxes: List of (x1, y1, x2, y2) tuples
        all_masks: List of lists - each bbox has multiple candidate masks
        all_scores: List of lists - each bbox has multiple scores
        expanded_bboxes: List of expanded bbox coordinates
        output_path: Path to save visualization
        restrict_to_bbox: Whether bbox constraint was applied
        bbox_margin: Margin ratio used
    """
    H, W = image.shape[:2]
    n_bboxes = len(bboxes)
    
    if n_bboxes < 2:
        print("⚠️  Z-order visualization requires at least 2 bboxes")
        return
    
    # Calculate bbox areas for sorting
    bbox_areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in bboxes]
    
    # Sort by area (large first) - Z-order
    sorted_indices = sorted(range(n_bboxes), key=lambda i: bbox_areas[i], reverse=True)
    
    # Select best mask (highest score) for each bbox
    best_masks = []
    best_scores = []
    best_indices = []
    for idx in range(n_bboxes):
        bbox_masks = all_masks[idx]
        bbox_scores = all_scores[idx]
        best_mask_idx = np.argmax(bbox_scores)
        best_masks.append(bbox_masks[best_mask_idx])
        best_scores.append(bbox_scores[best_mask_idx])
        best_indices.append(best_mask_idx)
    
    # Create final overlay: process in Z-order (large first, small later can override)
    final_mask = np.zeros((H, W, 3), dtype=np.uint8)  # RGB for colored overlay
    final_mask_colored = np.zeros((H, W, 4))  # RGBA for visualization
    
    # Process in Z-order: large boxes first (red), small boxes later (green)
    for order_idx, bbox_idx in enumerate(sorted_indices):
        mask = best_masks[bbox_idx]
        bbox = bboxes[bbox_idx]
        x1, y1, x2, y2 = bbox
        
        # Color: first box (largest) = red, second box (smaller) = green
        if order_idx == 0:
            color = [1, 0, 0, 0.6]  # Red for large box
            color_name = "Red (Large)"
        elif order_idx == 1:
            color = [0, 1, 0, 0.6]  # Green for small box
            color_name = "Green (Small)"
        else:
            # If more than 2 boxes, use other colors
            colors_list = [[0, 0, 1, 0.6], [1, 1, 0, 0.6], [1, 0, 1, 0.6]]
            color = colors_list[min(order_idx - 2, len(colors_list) - 1)]
            color_name = f"Color {order_idx + 1}"
        
        # Overlay: small box can override large box
        mask_rgba = np.zeros((H, W, 4))
        mask_rgba[mask] = color
        final_mask_colored = np.maximum(final_mask_colored, mask_rgba)
        
        # Also store in RGB for final visualization
        final_mask[mask] = [int(c * 255) for c in color[:3]]
    
    # Create visualization figure
    fig = plt.figure(figsize=(20, 12))
    
    # View 1: Original image with bboxes
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(image)
    for idx, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        color = 'red' if idx == sorted_indices[0] else 'green' if idx == sorted_indices[1] else 'blue'
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none', linestyle='-',
            label=f'BBox {idx+1} (Area: {area})'
        )
        ax1.add_patch(rect)
    ax1.set_title('Original Image + BBoxes\n(Red=Large, Green=Small)', fontsize=12)
    ax1.legend()
    ax1.axis('off')
    
    # View 2: Individual masks (before overlay)
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(image)
    for idx, (bbox, mask, score) in enumerate(zip(bboxes, best_masks, best_scores)):
        x1, y1, x2, y2 = bbox
        color = 'red' if idx == sorted_indices[0] else 'green' if idx == sorted_indices[1] else 'blue'
        mask_rgba = np.zeros((H, W, 4))
        mask_rgba[mask] = [1 if color == 'red' else 0, 1 if color == 'green' else 0, 0, 0.5]
        ax2.imshow(mask_rgba)
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1, edgecolor=color, facecolor='none', linestyle='--'
        )
        ax2.add_patch(rect)
    ax2.set_title('Individual Masks (Best Score)\nBefore Overlay', fontsize=12)
    ax2.axis('off')
    
    # View 3: Z-order overlay result (final)
    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(image)
    ax3.imshow(final_mask_colored)
    for idx, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        color = 'red' if idx == sorted_indices[0] else 'green' if idx == sorted_indices[1] else 'blue'
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none', linestyle='-'
        )
        ax3.add_patch(rect)
    ax3.set_title('Z-Order Overlay Result\n(Red→Green, Small Overrides Large)', fontsize=12)
    ax3.axis('off')
    
    # View 4: Only large box (red)
    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(image)
    large_idx = sorted_indices[0]
    large_mask = best_masks[large_idx]
    large_bbox = bboxes[large_idx]
    mask_rgba = np.zeros((H, W, 4))
    mask_rgba[large_mask] = [1, 0, 0, 0.6]  # Red
    ax4.imshow(mask_rgba)
    x1, y1, x2, y2 = large_bbox
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2, edgecolor='red', facecolor='none', linestyle='-'
    )
    ax4.add_patch(rect)
    ax4.set_title(f'Large Box (Red)\nScore: {best_scores[large_idx]:.3f}, Area: {large_mask.sum()} px', fontsize=10)
    ax4.axis('off')
    
    # View 5: Only small box (green)
    ax5 = plt.subplot(2, 3, 5)
    ax5.imshow(image)
    small_idx = sorted_indices[1]
    small_mask = best_masks[small_idx]
    small_bbox = bboxes[small_idx]
    mask_rgba = np.zeros((H, W, 4))
    mask_rgba[small_mask] = [0, 1, 0, 0.6]  # Green
    ax5.imshow(mask_rgba)
    x1, y1, x2, y2 = small_bbox
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2, edgecolor='green', facecolor='none', linestyle='-'
    )
    ax5.add_patch(rect)
    ax5.set_title(f'Small Box (Green)\nScore: {best_scores[small_idx]:.3f}, Area: {small_mask.sum()} px', fontsize=10)
    ax5.axis('off')
    
    # View 6: Overlap regions
    ax6 = plt.subplot(2, 3, 6)
    ax6.imshow(image)
    # Show overlap
    overlap_mask = (best_masks[sorted_indices[0]] & best_masks[sorted_indices[1]])
    overlap_rgba = np.zeros((H, W, 4))
    overlap_rgba[overlap_mask] = [1, 1, 0, 0.7]  # Yellow for overlap
    ax6.imshow(overlap_rgba)
    # Show final overlay
    ax6.imshow(final_mask_colored)
    for idx, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        color = 'red' if idx == sorted_indices[0] else 'green' if idx == sorted_indices[1] else 'blue'
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1, edgecolor=color, facecolor='none', linestyle='--'
        )
        ax6.add_patch(rect)
    ax6.set_title('Overlap Analysis\n(Yellow=Overlap, Green Overrides Red)', fontsize=10)
    ax6.axis('off')
    
    plt.suptitle(f'Z-Order Overlay: Large (Red) → Small (Green)\n'
                f'Small box can override large box in overlap regions',
                fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved Z-order visualization to: {output_path}")
    plt.close()


def main():
    # Default image path
    image_path = Path("/home/c/yuqyan/code/treeAI_new/000000000080.png")
    
    # Get image path from user if different
    user_path = input(f"Image path (Enter for default: {image_path}): ").strip()
    if user_path:
        image_path = Path(user_path)
    
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load image
    print(f"📷 Loading image: {image_path}")
    image = np.array(Image.open(image_path).convert('RGB'))
    H, W = image.shape[:2]
    print(f"   Image size: {W} x {H}")
    
    # Get bboxes from user
    print("\n📦 Enter bboxes (one per line, format: x1 y1 x2 y2)")
    print("   Example: 100 100 300 300")
    print("   Example: 200 150 400 350  (overlapping with first)")
    print("   Enter empty line to finish")
    
    bboxes = []
    while True:
        bbox_input = input(f"   BBox {len(bboxes)+1}: ").strip()
        if not bbox_input:
            if len(bboxes) == 0:
                print("   ⚠️  Need at least one bbox")
                continue
            break
        
        try:
            coords = [int(x) for x in bbox_input.split()]
            if len(coords) != 4:
                print("   ❌ Need 4 numbers: x1 y1 x2 y2")
                continue
            x1, y1, x2, y2 = coords
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)
            
            if x2 <= x1 or y2 <= y1:
                print("   ❌ Invalid bbox: x2 > x1 and y2 > y1 required")
                continue
            
            bboxes.append((x1, y1, x2, y2))
            print(f"      Added: ({x1}, {y1}) -> ({x2}, {y2})")
        except ValueError:
            print("   ❌ Invalid input. All values must be integers")
    
    print(f"\n✅ Total {len(bboxes)} bbox(es) to test")
    
    # Get constraint settings
    print("\n⚙️  Constraint settings:")
    restrict_input = input("   restrict_to_bbox? (y/n, default=y): ").strip().lower()
    restrict_to_bbox = restrict_input != 'n'
    
    bbox_margin = 0.15
    if restrict_to_bbox:
        margin_input = input(f"   bbox_margin (default=0.15): ").strip()
        if margin_input:
            try:
                bbox_margin = float(margin_input)
            except ValueError:
                print("   ⚠️  Invalid margin, using default 0.15")
    
    # Load SAM2 model
    print("\n🔧 Loading SAM2 model...")
    config_path = "/home/c/yuqyan/code/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_b+_tree_finetune_full.yaml"
    ckpt_path = "/home/c/yuqyan/code/sam2/sam2_logs/configs/sam2.1_training/sam2.1_hiera_b+_tree_finetune_full/checkpoints/checkpoint.pt"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {device}")
    
    try:
        model = build_sam2_from_config_path(config_path, ckpt_path, device=device)
        predictor = SAM2ImagePredictor(model)
        print("✅ SAM2 loaded")
    except Exception as e:
        print(f"❌ Failed to load SAM2: {e}")
        return
    
    # Set image
    predictor.set_image(image)
    
    # Run predictions for each bbox with multimask
    print(f"\n🎯 Running SAM2 predictions for {len(bboxes)} bbox(es) (multimask mode)...")
    all_masks = []  # List of lists: each bbox has multiple candidate masks
    all_scores = []  # List of lists: each bbox has multiple scores
    expanded_bboxes = []
    
    for idx, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        print(f"   [{idx+1}/{len(bboxes)}] BBox ({x1}, {y1}) -> ({x2}, {y2})...", end=' ', flush=True)
        
        # Predict with bbox prompt (multimask mode)
        box = np.array([x1, y1, x2, y2], dtype=np.float32)
        
        pred_masks, pred_scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=True,  # Enable multimask to see all candidates
        )
        
        # Process all candidate masks
        bbox_masks = []
        bbox_scores = []
        
        # Calculate expanded bbox once (same for all masks of this bbox)
        if restrict_to_bbox:
            _, exp_bbox = apply_bbox_constraint(
                pred_masks[0].astype(bool), bbox, H, W, restrict_to_bbox=True, bbox_margin=bbox_margin
            )
            expanded_bboxes.append(exp_bbox)
        else:
            expanded_bboxes.append(None)
        
        for mask_idx, (pred_mask, pred_score) in enumerate(zip(pred_masks, pred_scores)):
            mask = pred_mask.astype(bool)
            score = float(pred_score) if pred_scores is not None else 0.0
            
            # Apply bbox constraint
            if restrict_to_bbox:
                mask, _ = apply_bbox_constraint(
                    mask, bbox, H, W, restrict_to_bbox=True, bbox_margin=bbox_margin
                )
            
            bbox_masks.append(mask)
            bbox_scores.append(score)
        
        all_masks.append(bbox_masks)
        all_scores.append(bbox_scores)
        
        print(f"Found {len(bbox_masks)} candidate masks")
        for mask_idx, (mask, score) in enumerate(zip(bbox_masks, bbox_scores)):
            print(f"      Mask {mask_idx+1}: Score={score:.3f}, Area={mask.sum()} px")
    
    # Visualize results
    output_dir = PROJECT_ROOT / "outputs" / "sam2_bbox_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    bbox_str = "_".join([f"{x1}_{y1}_{x2}_{y2}" for x1, y1, x2, y2 in bboxes])
    output_path = output_dir / f"{image_path.stem}_bboxes_{bbox_str}.png"
    zorder_output_path = output_dir / f"{image_path.stem}_zorder_overlay_{bbox_str}.png"
    
    print(f"\n📊 Generating visualizations...")
    
    # Standard multimask visualization
    visualize_bbox_predictions(
        image=image,
        bboxes=bboxes,
        all_masks=all_masks,
        all_scores=all_scores,
        expanded_bboxes=expanded_bboxes,
        output_path=output_path,
        restrict_to_bbox=restrict_to_bbox,
        bbox_margin=bbox_margin,
    )
    
    # Z-order overlay visualization (if 2+ bboxes)
    if len(bboxes) >= 2:
        visualize_zorder_overlay(
            image=image,
            bboxes=bboxes,
            all_masks=all_masks,
            all_scores=all_scores,
            expanded_bboxes=expanded_bboxes,
            output_path=zorder_output_path,
            restrict_to_bbox=restrict_to_bbox,
            bbox_margin=bbox_margin,
        )
    
    # Print summary
    print("\n" + "=" * 80)
    print("📈 Summary")
    print("=" * 80)
    print(f"Total bboxes tested: {len(bboxes)}")
    print(f"Constraint: restrict_to_bbox={restrict_to_bbox}, margin={bbox_margin:.0%}")
    for idx, (bbox, bbox_masks, bbox_scores) in enumerate(zip(bboxes, all_masks, all_scores)):
        x1, y1, x2, y2 = bbox
        print(f"\nBBox {idx+1}: ({x1}, {y1}) -> ({x2}, {y2})")
        print(f"  Found {len(bbox_masks)} candidate masks:")
        for mask_idx, (mask, score) in enumerate(zip(bbox_masks, bbox_scores)):
            print(f"    Mask {mask_idx+1}: Score={score:.3f}, Area={mask.sum()} px, "
                  f"Coverage={mask.sum() / ((x2-x1)*(y2-y1)):.1%}")
    
    # Overlap analysis (using best mask from each bbox)
    if len(bboxes) > 1:
        print("\n🔍 Overlap Analysis (using best mask from each bbox):")
        best_masks = [masks[0] for masks in all_masks]  # Use first (best) mask
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                overlap = (best_masks[i] & best_masks[j]).sum()
                union = (best_masks[i] | best_masks[j]).sum()
                iou = overlap / union if union > 0 else 0
                print(f"  BBox {i+1} (best) & BBox {j+1} (best): IoU = {iou:.3f}, Overlap = {overlap} px")
    
    print(f"\n✅ Results saved to:")
    print(f"   Multimask view: {output_path}")
    if len(bboxes) >= 2:
        print(f"   Z-order overlay: {output_dir / f'{image_path.stem}_zorder_overlay_{bbox_str}.png'}")


if __name__ == "__main__":
    main()

