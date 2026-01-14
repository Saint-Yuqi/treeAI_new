#!/usr/bin/env python3
"""
Visualize RSPrompter bbox prompts on images.

Usage:
    python scripts/visualize_bbox_prompts.py \
        --prompt-dir rsprompter_prompts/12_RGB_SemSegm_640_fL/pick \
        --image-dir /home/c/yuqyan/data/TreeAI/12_RGB_SemSegm_640_fL/pick/images \
        --output-dir outputs/bbox_visualizations
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def load_bbox_prompts(prompt_file: Path) -> Tuple[List[List[float]], List[int]]:
    """Load bboxes and labels from RSPrompter prompt file."""
    with open(prompt_file) as f:
        data = json.load(f)
    
    bboxes = data.get('bboxes', [])
    labels = data.get('labels', [])
    
    return bboxes, labels


def draw_bboxes(
    image: Image.Image,
    bboxes: List[List[float]],
    labels: List[int],
    color: Tuple[int, int, int] = (255, 0, 0),  # Red (RGB)
    thickness: int = 2,
) -> Image.Image:
    """Draw bboxes on image."""
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)
    
    # Try to load font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        
        # Draw label text
        label_text = f"Class {label}"
        
        # Get text bounding box
        if font:
            bbox_text = draw.textbbox((x1, y1), label_text, font=font)
        else:
            bbox_text = draw.textbbox((x1, y1), label_text)
        
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        
        # Draw text background
        draw.rectangle(
            [x1, y1 - text_height - 2, x1 + text_width + 2, y1],
            fill=color,
        )
        
        # Draw text
        draw.text(
            (x1 + 1, y1 - text_height - 1),
            label_text,
            fill=(255, 255, 255),  # White text
            font=font,
        )
    
    return vis_image


def find_image_file(image_dir: Path, image_name: str) -> Path:
    """Find image file with given name (any extension)."""
    extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    for ext in extensions:
        image_path = image_dir / f"{image_name}{ext}"
        if image_path.exists():
            return image_path
    return None


def visualize_prompts(
    prompt_dir: Path,
    image_dir: Path,
    output_dir: Path,
    show: bool = False,
):
    """Visualize all bbox prompts in prompt directory."""
    prompt_dir = Path(prompt_dir)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prompt_files = sorted(prompt_dir.glob('*.json'))
    
    if not prompt_files:
        print(f"❌ No JSON files found in {prompt_dir}")
        return
    
    print(f"📂 Found {len(prompt_files)} prompt files")
    print(f"📂 Image directory: {image_dir}")
    print(f"📂 Output directory: {output_dir}")
    
    for prompt_file in tqdm(prompt_files, desc="Visualizing"):
        image_name = prompt_file.stem
        
        # Load bboxes
        try:
            bboxes, labels = load_bbox_prompts(prompt_file)
        except Exception as e:
            print(f"⚠️  Error loading {prompt_file.name}: {e}")
            continue
        
        if not bboxes:
            print(f"⚠️  No bboxes in {prompt_file.name}")
            continue
        
        # Find image file
        image_path = find_image_file(image_dir, image_name)
        if image_path is None:
            print(f"⚠️  Image not found for {image_name}")
            continue
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"⚠️  Error loading image {image_path}: {e}")
            continue
        
        # Draw bboxes
        vis_image = draw_bboxes(image, bboxes, labels)
        
        # Save visualization
        output_path = output_dir / f"{image_name}_bboxes.jpg"
        vis_image.save(output_path, quality=95)
        
        # Show if requested
        if show:
            vis_image.show()
    
    print(f"\n✅ Visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize RSPrompter bbox prompts')
    parser.add_argument(
        '--prompt-dir',
        type=str,
        required=True,
        help='Directory containing bbox prompt JSON files'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        required=True,
        help='Directory containing source images'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs/bbox_visualizations',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display images interactively (requires X11)'
    )
    
    args = parser.parse_args()
    
    visualize_prompts(
        prompt_dir=Path(args.prompt_dir),
        image_dir=Path(args.image_dir),
        output_dir=Path(args.output_dir),
        show=args.show,
    )


if __name__ == '__main__':
    main()

