#!/usr/bin/env python3
"""
TreeAI Dataset Loader for Semantic Segmentation
Simple, clean data loader following SAM2 and treeAI-segmentation patterns
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image, ImageFile
import torchvision.transforms.functional as TF

# Allow PIL to load slightly corrupted/truncated PNGs that appear in TreeAI dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True


class TreeAISemanticDataset(Dataset):
    """
    Dataset for TreeAI semantic segmentation.
    
    Directory structure:
        dataset_root/
            images/
                000000000018.png
                ...
            labels/
                000000000018.png
                ...
    
    Parameters:
    -----------
    image_dir : str
        Path to images directory
    label_dir : str
        Path to labels directory
    num_classes : int
        Number of semantic classes (including background)
    ignore_index : int
        Index to ignore in loss computation (default: -1)
    transform : callable, optional
        Transform to apply to images
    """
    
    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        num_classes: int,
        ignore_index: int = -1,
        transform=None
    ):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.transform = transform
        
        # Get all image files
        self.image_files = sorted(list(self.image_dir.glob('*.png')))
        
        # Verify corresponding labels exist
        self.valid_samples = []
        for img_path in self.image_files:
            label_path = self.label_dir / img_path.name
            if label_path.exists():
                self.valid_samples.append((img_path, label_path))
        
        if len(self.valid_samples) == 0:
            raise ValueError(f"No valid image-label pairs found in {self.image_dir}")
        
        print(f"Found {len(self.valid_samples)} valid samples")
    
    def __len__(self) -> int:
        return len(self.valid_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, label_path = self.valid_samples[idx]
        
        # Load image and label with retry for permission/IO errors
        # Common in multi-worker DataLoader scenarios
        max_retries = 5
        import time
        
        for attempt in range(max_retries):
            try:
                # Use absolute path and ensure file exists
                img_path_abs = Path(img_path).resolve()
                label_path_abs = Path(label_path).resolve()
                
                if not img_path_abs.exists():
                    raise FileNotFoundError(f"Image file not found: {img_path_abs}")
                if not label_path_abs.exists():
                    raise FileNotFoundError(f"Label file not found: {label_path_abs}")
                
                # Open files
                image = Image.open(img_path_abs).convert('RGB')
                label = Image.open(label_path_abs)
                
                # Load immediately to avoid keeping file handles open
                image.load()
                label.load()
                break
                
            except (PermissionError, OSError, IOError) as e:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    wait_time = 0.1 * (2 ** attempt) + (attempt * 0.01)
                    time.sleep(wait_time)
                    continue
                else:
                    raise PermissionError(
                        f"Failed to open files after {max_retries} attempts. "
                        f"Image: {img_path}, Label: {label_path}. "
                        f"Original error: {e}. "
                        f"Check file permissions and ensure no other process is accessing these files."
                    )
        
        # Convert to tensors
        image = TF.to_tensor(image)  # Shape: (3, H, W), range [0, 1]
        label = torch.from_numpy(np.array(label, dtype=np.int64))  # Shape: (H, W)
        
        # Store original image for visualization (before transforms)
        original_image = np.array(Image.open(img_path_abs).convert('RGB'))  # (H, W, 3), uint8, [0, 255]
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'mask': label,
            'name': img_path.stem,
            'original_image': original_image,  # Original image for visualization
            'image_path': str(img_path_abs)  # Path to original image
        }


def create_dataloaders(
    dataset_root: Union[str, List[str]],
    num_classes: int,
    batch_size: int = 4,
    num_workers: int = 4,
    ignore_index: int = -1,
    train_transform=None,
    val_transform=None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation/test dataloaders.
    
    Parameters:
    -----------
    dataset_root : str
        Root directory containing train/val/test splits
    num_classes : int
        Number of semantic classes
    batch_size : int
        Batch size for dataloaders
    num_workers : int
        Number of worker processes for data loading
    ignore_index : int
        Index to ignore in labels
    train_transform : callable, optional
        Transform for training data
    val_transform : callable, optional
        Transform for validation data
    
    Returns:
    --------
    train_loader, test_loader : DataLoader, DataLoader
    """
    # Normalize to list of roots
    roots: List[Path] = []
    if isinstance(dataset_root, (list, tuple)):
        roots = [Path(r) for r in dataset_root]
    else:
        roots = [Path(dataset_root)]

    # Create datasets from all roots
    train_datasets: List[Dataset] = []
    test_datasets: List[Dataset] = []
    for root in roots:
        train_images = root / 'train' / 'images'
        train_labels = root / 'train' / 'labels'
        test_images = root / 'test' / 'images'
        test_labels = root / 'test' / 'labels'

        if train_images.exists() and train_labels.exists():
            dataset = TreeAISemanticDataset(
                str(train_images),
                str(train_labels),
                num_classes=num_classes,
                ignore_index=ignore_index,
                transform=train_transform
            )
            train_datasets.append(dataset)
            print(f"  ✓ Added training dataset from: {root}")
            print(f"    Samples: {len(dataset)}")

        if test_images.exists() and test_labels.exists():
            dataset = TreeAISemanticDataset(
                str(test_images),
                str(test_labels),
                num_classes=num_classes,
                ignore_index=ignore_index,
                transform=val_transform
            )
            test_datasets.append(dataset)
            print(f"  ✓ Added test dataset from: {root}")
            print(f"    Samples: {len(dataset)}")
        
        if not train_images.exists() or not train_labels.exists():
            print(f"  ⚠️  Warning: Training data not found at {root}/train/")
        if not test_images.exists() or not test_labels.exists():
            print(f"  ⚠️  Warning: Test data not found at {root}/test/")

    train_dataset = None if not train_datasets else (train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets))
    test_dataset = None if not test_datasets else (test_datasets[0] if len(test_datasets) == 1 else ConcatDataset(test_datasets))
    
    # Print summary
    if train_dataset:
        total_train_samples = len(train_dataset)
        print(f"\n📊 Training dataset summary:")
        print(f"   Total datasets: {len(train_datasets)}")
        print(f"   Total samples: {total_train_samples}")
        if len(train_datasets) > 1:
            print(f"   Dataset sizes: {[len(ds) for ds in train_datasets]}")
    
    if test_dataset:
        total_test_samples = len(test_dataset)
        print(f"\n📊 Test dataset summary:")
        print(f"   Total datasets: {len(test_datasets)}")
        print(f"   Total samples: {total_test_samples}")
        if len(test_datasets) > 1:
            print(f"   Dataset sizes: {[len(ds) for ds in test_datasets]}")
    
    # Create dataloaders
    train_loader = None
    if train_dataset:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between epochs
            prefetch_factor=2 if num_workers > 0 else None,  # Prefetch batches
        )
    
    test_loader = None
    if test_dataset:
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None,
        )
    
    return train_loader, test_loader


def create_pick_dataloader(
    pick_roots: Union[str, List[str]],
    num_classes: int,
    batch_size: int = 4,
    num_workers: int = 4,
    ignore_index: int = -1,
    transform=None
) -> Optional[torch.utils.data.DataLoader]:
    """
    Create dataloader for pick dataset (used for qualitative visualization).
    
    Pick dataset structure:
        pick_root/
            images/
                000000000018.png
                ...
            labels/
                000000000018.png
                ...
    
    Parameters:
    -----------
    pick_roots : str or list[str]
        Path(s) to pick dataset directories
    num_classes : int
        Number of semantic classes
    batch_size : int
        Batch size for dataloader
    num_workers : int
        Number of worker processes
    ignore_index : int
        Index to ignore in labels
    transform : callable, optional
        Transform to apply to images
    
    Returns:
    --------
    pick_loader : DataLoader or None
        DataLoader for pick dataset, or None if no valid pick directories found
    """
    # Normalize to list
    roots: List[Path] = []
    if isinstance(pick_roots, (list, tuple)):
        roots = [Path(r) for r in pick_roots]
    elif isinstance(pick_roots, str):
        # Handle string input - split by space/newline if multiple paths
        # But first check if it's a single path
        pick_roots_str = pick_roots.strip()
        if ' ' in pick_roots_str or '\n' in pick_roots_str:
            # Multiple paths separated by space or newline
            paths = [p.strip() for p in pick_roots_str.replace('\n', ' ').split() if p.strip()]
            roots = [Path(p) for p in paths]
        else:
            # Single path
            roots = [Path(pick_roots_str)]
    else:
        roots = [Path(str(pick_roots))]
    
    pick_datasets: List[Dataset] = []
    for root in roots:
        pick_images = root / 'images'
        pick_labels = root / 'labels'
        
        if pick_images.exists() and pick_labels.exists():
            pick_datasets.append(
                TreeAISemanticDataset(
                    str(pick_images),
                    str(pick_labels),
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    transform=transform
                )
            )
        else:
            print(f"⚠️  Warning: Pick directory not found or incomplete: {root}")
    
    if not pick_datasets:
        return None
    
    pick_dataset = pick_datasets[0] if len(pick_datasets) == 1 else ConcatDataset(pick_datasets)
    
    pick_loader = torch.utils.data.DataLoader(
        pick_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    return pick_loader
