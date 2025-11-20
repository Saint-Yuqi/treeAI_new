# Tree Species Instance Classification with SAM2

This project implements **instance-level tree species classification** using SAM2 (Segment Anything Model 2) for segmentation and a CNN classifier for species recognition.

## Overview

### Two Approaches

This repository contains two approaches for tree species classification:

#### **1. Instance-level Classification (Recommended)**

**Simple. Direct. Effective.**

- SAM2 segments individual tree crowns (excellent boundaries)
- CNN classifier predicts species for each tree instance
- Treats classification as an **instance problem**, not a pixel problem

```
Image → SAM2 (tree masks) → CNN Encoder → Classifier → Species per Tree
```

**Advantages:**
- Simple architecture (100 lines vs 1000+)
- Faster training and inference
- Uses global tree appearance (not just local pixels)
- Easy to debug and improve
- Natural fit for the problem

#### **2. Semantic Segmentation (Legacy)**

- SAM2 + pixel-level semantic head
- More complex, slower, harder to train
- Kept for comparison purposes

**The rest of this README focuses on the instance-level approach.**

### Key Features (Instance Approach)

- **Simple Architecture**: ~100 lines of core logic vs 1000+ for semantic approach
- **SAM2 Segmentation**: Excellent tree crown boundaries without training
- **Majority Vote Labels**: Bootstrap labels from semantic ground truth (automatic, no manual annotation)
- **Purity Filtering**: Train on confident instances (>70% label agreement)
- **Standard CNN Encoder**: Use any pretrained model (ResNet, EfficientNet, etc.)
- **Fast Training**: ~10x faster than semantic approach
- **Easy to Debug**: Simple forward pass, no complex loss weighting

### Why Instance Classification is Better

**Problem:** You want to identify tree species. Each tree has a species label.

**Instance approach (correct):**
1. Segment tree crowns (SAM2 does this perfectly)
2. Look at whole tree appearance
3. Classify each tree → one prediction per tree
4. Natural, simple, effective

**Semantic approach (wrong tool):**
1. Classify every pixel independently
2. Local context only
3. Millions of predictions per image
4. Complex aggregation needed
5. Slower, harder to train

**The data proves it:** SAM2 already gives you instance masks with `purity` scores. High purity = clean instance. Low purity = boundary confusion. Just filter and train.

### Label Quality

Your `precompute_sam2_instances.py` generates labels via majority voting:

```python
# For each SAM2 mask
pixels_inside_mask = ground_truth[mask]
most_common_label = mode(pixels_inside_mask)
purity = fraction_of_pixels_with_most_common_label
```

**Purity distribution matters:**
- `purity > 0.9`: Excellent, use for training
- `purity 0.7-0.9`: Good, use for training
- `purity < 0.7`: Noisy, filter out or use as validation

Check your purity distribution:
```bash
python -c "
import json
from pathlib import Path
purities = []
for line in open('sam2_instances/12_RGB_SemSegm_640_fL/train/instances_manifest.jsonl'):
    purities.append(json.loads(line)['purity'])
import numpy as np
print(f'Mean: {np.mean(purities):.3f}')
print(f'Median: {np.median(purities):.3f}')
print(f'>0.9: {np.mean([p>0.9 for p in purities]):.1%}')
print(f'>0.7: {np.mean([p>0.7 for p in purities]):.1%}')
"
```

## Project Structure

```
treeAI_segmentation/
├── configs/
│   └── configs.yaml              # Main configuration file
├── utils/
│   ├── dataset.py                # TreeAI dataset loader
│   ├── model_prompted.py         # SAM2 + semantic head model
│   ├── prompt_loader.py          # RSPrompter prompt loader
│   ├── eval_treeai.py            # Evaluation utilities
│   └── mask_generator.py         # Mask generation utilities
├── scripts/
│   └── export_rsprompter_bbox_prompts.py  # Export prompts from RSPrompter
├── train.py                      # Training script
├── test.py                       # Complete pipeline orchestrator
└── README.md                     # This file
```

## Environment Setup

### Prerequisites

1. **Conda Environment**: Create and activate the `sam2` conda environment
   ```bash
   conda activate sam2
   ```

2. **RSPrompter Environment**: Separate Python environment with RSPrompter dependencies (optional, for prompt export)

### Required Dependencies

- PyTorch
- SAM2 (Segment Anything Model 2)
- torchmetrics
- scikit-learn
- matplotlib
- pyyaml

## Quick Start (Instance Classification)

### Step 1: Precompute SAM2 Instances

First, generate individual tree crown masks using SAM2:

```bash
conda activate sam2
python scripts/precompute_sam2_instances.py \
  --config configs/configs.yaml
```

This creates:
- Individual mask PNGs for each tree
- Metadata JSON files with labels (from majority voting)
- Manifest JSONL files for easy loading

**Output structure:**
```
sam2_instances/
└── 12_RGB_SemSegm_640_fL/
    ├── train/
    │   ├── instances/
    │   │   ├── 000000000018/
    │   │   │   ├── mask_000.png
    │   │   │   ├── mask_001.png
    │   │   │   └── metadata.json
    │   │   └── ...
    │   └── instances_manifest.jsonl  # All instances in one file
    ├── val/
    └── test/
```

### Step 2: Train Instance Classifier

Train a simple CNN classifier on the precomputed instances:

```bash
conda activate sam2
python train_instance_classifier.py \
  --train-manifest sam2_instances/12_RGB_SemSegm_640_fL/train/instances_manifest.jsonl \
  --val-manifest sam2_instances/12_RGB_SemSegm_640_fL/val/instances_manifest.jsonl \
  --output-root sam2_instances/12_RGB_SemSegm_640_fL \
  --num-classes 50 \
  --model-type simple \
  --encoder resnet50 \
  --batch-size 32 \
  --lr 1e-4 \
  --epochs 50 \
  --min-purity 0.7 \
  --output-dir outputs/instance_classifier
```

**Key arguments:**
- `--model-type`: `simple` (crop to bbox) or `masked` (apply mask)
- `--encoder`: Any timm model (resnet50, efficientnet_b0, etc.)
- `--min-purity`: Filter instances by label purity (0.7 = 70% pixels agree)
- `--min-area`: Minimum mask area in pixels

### Step 3: Train with Visualization and Wandb

**Method 1: Using config file (Recommended)**

Edit `configs/configs_instance.yaml` to match your paths, then:

```bash
conda activate sam2
python train_instance_classifier.py --config configs/configs_instance.yaml
```

**Method 2: Using command-line arguments**

```bash
conda activate sam2
python train_instance_classifier.py \
  --train-manifest sam2_instances/12_RGB_SemSegm_640_fL/train/instances_manifest.jsonl \
  --val-manifest sam2_instances/12_RGB_SemSegm_640_fL/val/instances_manifest.jsonl \
  --output-root sam2_instances/12_RGB_SemSegm_640_fL \
  --num-classes 57 \
  --classes-yaml /home/c/yuqyan/code/treeAI-segmentation/configs/data/treeAI_classes.yaml \
  --pick-root /home/c/yuqyan/data/TreeAI/12_RGB_SemSegm_640_fL/pick \
  --batch-size 32 \
  --epochs 50 \
  --min-purity 0.7 \
  --viz-freq 5 \
  --use-wandb \
  --wandb-project tree_instance \
  --wandb-entity stefaniehurschler-uzh \
  --output-dir outputs/instance_classifier
```

**Method 3: Mix config file + command-line overrides**

```bash
# Use config but override specific parameters
python train_instance_classifier.py \
  --config configs/configs_instance.yaml \
  --epochs 100 \
  --encoder efficientnet_b4
```

**Visualizations generated every 5 epochs:**
- Original images with bounding boxes
- Ground truth vs predictions (colored by class)
- Confusion matrices (4 variants)
- Group F1 scores (bulk/medium/tail/dead-tree/invasive-alien)

### Step 4: Evaluate

Evaluate the trained model on test set:

```bash
conda activate sam2
python eval_instance_classifier.py \
  --checkpoint outputs/instance_classifier/best_model.pt \
  --test-manifest sam2_instances/12_RGB_SemSegm_640_fL/test/instances_manifest.jsonl \
  --output-root sam2_instances/12_RGB_SemSegm_640_fL \
  --output-dir outputs/eval \
  --batch-size 32
```

This generates:
- Overall accuracy
- Per-class metrics (precision, recall, F1)
- Confusion matrices
- Accuracy stratified by instance purity

## Model Selection

The instance classifier supports any CNN encoder from the [timm library](https://github.com/huggingface/pytorch-image-models).

**Recommended encoders:**

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `mobilenetv3_small` | 2.5M | Fast | Good | Deployment, edge devices |
| `efficientnet_b0` | 5.3M | Fast | Good | Good balance |
| `resnet50` | 25.6M | Medium | Better | **Default, good balance** |
| `efficientnet_b4` | 19.3M | Medium | Better | High accuracy |
| `convnext_small` | 50.2M | Slow | Best | Maximum accuracy |

**Example: Train with EfficientNet-B2 (faster, smaller)**
```bash
python train_instance_classifier.py \
  --train-manifest sam2_instances/.../train/instances_manifest.jsonl \
  --val-manifest sam2_instances/.../val/instances_manifest.jsonl \
  --output-root sam2_instances/... \
  --num-classes 50 \
  --encoder efficientnet_b2 \
  --batch-size 64 \
  --lr 1e-4 \
  --epochs 50
```

**Example: Train with ConvNeXt (maximum accuracy)**
```bash
python train_instance_classifier.py \
  --train-manifest sam2_instances/.../train/instances_manifest.jsonl \
  --val-manifest sam2_instances/.../val/instances_manifest.jsonl \
  --output-root sam2_instances/... \
  --num-classes 50 \
  --encoder convnext_small \
  --batch-size 16 \
  --lr 5e-5 \
  --epochs 100
```

## Dataset Structure

### Original Dataset (for SAM2 instance generation)

```
dataset_root/
├── train/
│   ├── images/
│   │   └── *.png
│   └── labels/
│       └── *.png  (semantic segmentation ground truth)
└── test/
    ├── images/
    └── labels/
```

### Generated Instance Dataset (for classifier training)

```
sam2_instances/
└── 12_RGB_SemSegm_640_fL/
    ├── train/
    │   ├── instances/
    │   │   ├── 000000000018/
    │   │   │   ├── mask_000.png  (binary mask)
    │   │   │   ├── mask_001.png
    │   │   │   └── metadata.json (labels, bboxes, purity)
    │   └── instances_manifest.jsonl  (all instances)
    ├── val/
    └── test/
```

## Complete Example Workflow

```bash
# Step 0: Activate environment
conda activate sam2

# Step 1: Generate instance masks (one-time, takes ~1-2 hours for large dataset)
python scripts/precompute_sam2_instances.py --config configs/configs.yaml

# Step 2: Check label quality
python -c "
import json
purities = [json.loads(l)['purity'] for l in open('sam2_instances/12_RGB_SemSegm_640_fL/train/instances_manifest.jsonl')]
import numpy as np
print(f'High quality (>0.9): {sum(p>0.9 for p in purities)} instances')
print(f'Good quality (>0.7): {sum(p>0.7 for p in purities)} instances')
print(f'Total: {len(purities)} instances')
"

# Step 3: Train classifier (fast, ~2-3 hours)
python train_instance_classifier.py \
  --train-manifest sam2_instances/12_RGB_SemSegm_640_fL/train/instances_manifest.jsonl \
  --val-manifest sam2_instances/12_RGB_SemSegm_640_fL/val/instances_manifest.jsonl \
  --output-root sam2_instances/12_RGB_SemSegm_640_fL \
  --num-classes 50 \
  --encoder resnet50 \
  --batch-size 32 \
  --lr 1e-4 \
  --epochs 50 \
  --min-purity 0.7 \
  --output-dir outputs/instance_classifier

# Step 4: Evaluate
python eval_instance_classifier.py \
  --checkpoint outputs/instance_classifier/best_model.pt \
  --test-manifest sam2_instances/12_RGB_SemSegm_640_fL/test/instances_manifest.jsonl \
  --output-root sam2_instances/12_RGB_SemSegm_640_fL \
  --output-dir outputs/eval
```

## Training Tips

### 1. Start Simple
- Use `--encoder resnet50` (default)
- Use `--model-type simple` (crop to bbox, no explicit mask)
- Train for 50 epochs first
- Only add complexity if needed

### 2. Data Quality
- Check purity distribution before training
- Start with `--min-purity 0.7` or `0.8`
- If you have few high-purity instances, lower threshold to 0.6

### 3. Hyperparameters
- **Learning rate**: `1e-4` for scratch, `5e-5` for fine-tuning
- **Batch size**: 32 for ResNet50, 64 for smaller models
- **Image size**: 224x224 (default), or 384x384 for more detail
- **Dropout**: 0.3 (default), increase to 0.5 if overfitting

### 4. Data Augmentation
Built-in augmentations in `create_instance_dataloaders()`:
- Random horizontal/vertical flips
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation)

### 5. If Accuracy is Low
1. **Check label quality**: Are purities too low?
2. **Visualize errors**: Which classes are confused?
3. **Try larger encoder**: `convnext_small` or `efficientnet_b4`
4. **Increase training time**: 100 epochs instead of 50
5. **Add more data**: Lower `min_purity` threshold carefully

### 6. If Training is Slow
1. **Use smaller encoder**: `efficientnet_b0` or `mobilenetv3_small`
2. **Reduce image size**: `--image-size 128 128`
3. **Increase batch size**: If GPU memory allows
4. **Use mixed precision**: (add to code if needed)

## Evaluation

### Metrics

The evaluation follows treeAI-segmentation methodology:

- **Overall Accuracy (OA)**
- **Per-class Metrics**:
  - Precision (User's Accuracy, UA)
  - Recall (Producer's Accuracy, PA)
  - Intersection over Union (IoU)
  - F1 Score
- **Averaged Metrics**:
  - Average over all classes
  - Average excluding background (wo0)
  - Custom group averages (if specified)

### Evaluation Functions

- `evaluate_model()`: Dataset-level evaluation with confusion matrix
- `evaluate_model_samplewise()`: Per-sample evaluation for detailed analysis
- `visualize_results()`: Generate qualitative visualizations

### Visualization Outputs

Visualizations include:
- RGB images with reference and prediction masks
- Variance/uncertainty maps
- Grayscale images with overlaid masks
- Confusion matrices (4 normalization variants)
- Per-class score bar charts

## Configuration Options

### Dataset Configuration

- `root`: Main dataset root directory
- `extra_roots`: Additional dataset roots to concatenate
- `classes_yaml`: Path to classes YAML file
- `batch_size`: Batch size for training
- `num_workers`: Data loading workers
- `ignore_index`: Index to ignore in loss computation

### Model Configuration

- `sam2_checkpoint`: Path to fine-tuned SAM2 checkpoint
- `sam2_config`: SAM2 model configuration name
- `freeze_sam2`: Whether to freeze SAM2 encoder (train only semantic head)
- `hidden_channels`: Hidden layer dimensionality in semantic head

### RSPrompter Configuration

- `checkpoint`: RSPrompter checkpoint path
- `config`: RSPrompter configuration file
- `bbox_prompt_dir`: Directory to store exported prompts
- `min_score`: Minimum confidence score for prompts
- `max_prompts`: Maximum number of prompts per image

### Training Configuration

- `epochs`: Number of training epochs
- `lr`: Learning rate
- `weight_decay`: Weight decay for optimizer
- `save_freq`: Save checkpoint every N epochs
- `output_dir`: Output directory for checkpoints and logs
- `resume_from`: Path to checkpoint to resume from

### Visualization Configuration

- `pick_roots`: List of pick dataset directories for qualitative visualization
- `viz_freq`: Generate visualizations every N epochs (0 to disable)

## Output Structure

```
outputs/
└── sam2_rsprompter_prompted/
    ├── checkpoints/
    │   ├── checkpoint_epoch_5.pt
    │   ├── checkpoint_epoch_10.pt
    │   ├── best_model.pt
    │   └── final_model.pt
    ├── qualitative/
    │   ├── epoch_005/
    │   │   └── top_1/
    │   │       ├── qual_first.jpg      # RGB visualization (first 5 samples)
    │   │       ├── qual_last.jpg       # RGB visualization (last 5 samples)
    │   │       ├── overlay_first.jpg   # Grayscale overlay (first 5 samples)
    │   │       └── overlay_last.jpg    # Grayscale overlay (last 5 samples)
    │   └── epoch_010/
    │       └── top_1/
    │           └── ...
    └── val_epoch_10/
        ├── metrics_semseg.json         # Evaluation metrics (JSON)
        ├── confusion.jpg                # Confusion matrix (raw counts)
        ├── confusion_norm_all.jpg      # Confusion matrix (normalized by all)
        ├── confusion_norm_pred.jpg     # Confusion matrix (normalized by predictions)
        ├── confusion_norm_true.jpg     # Confusion matrix (normalized by ground truth)
        └── scores_per_class.jpg        # Per-class score bar chart
```

### Visualization Formats

#### 1. Qualitative Visualizations (`qualitative/epoch_XXX/top_1/`)

Generated every `viz_freq` epochs using the pick dataset.

**File Format**: JPG images, 300 DPI, high resolution

**Content**:

- **`qual_first.jpg`** and **`qual_last.jpg`**:
  - **Layout**: 4 columns × N rows (N = number of samples, default 5)
  - **Columns**:
    1. **Image**: Original RGB image
    2. **Reference**: Ground truth semantic mask (colored by class)
    3. **Prediction**: Model prediction (colored by class)
    4. **Variance**: Uncertainty/variance map (viridis colormap)
  - **Figure size**: 15 × (N × 5) inches
  - **Legend**: Class names and colors shown on the right side

- **`overlay_first.jpg`** and **`overlay_last.jpg`**:
  - **Layout**: 2 columns × N rows
  - **Columns**:
    1. **Reference**: Grayscale image with ground truth mask overlay (semi-transparent)
    2. **Prediction**: Grayscale image with prediction mask overlay (semi-transparent)
  - **Figure size**: 10 × (N × 5) inches
  - **Legend**: Class names and colors shown on the right side

#### 2. Validation Visualizations (`val_epoch_XXX/`)

Generated during validation evaluation.

**File Formats**:

- **Confusion Matrices** (JPG, 500 DPI):
  - `confusion.jpg`: Raw confusion matrix with absolute counts
  - `confusion_norm_all.jpg`: Normalized by total number of pixels
  - `confusion_norm_pred.jpg`: Normalized by predicted class totals (Precision)
  - `confusion_norm_true.jpg`: Normalized by ground truth class totals (Recall)

- **Scores Per Class** (`scores_per_class.jpg`, 300 DPI):
  - Bar chart showing per-class metrics (IoU, F1, Precision, Recall)
  - Colorblind-friendly palette
  - Class labels on x-axis

- **Metrics JSON** (`metrics_semseg.json`):
  - All evaluation metrics in JSON format
  - Includes overall accuracy, per-class metrics, and averaged metrics

## Troubleshooting

### Common Issues

1. **"No valid instances found"**: Check that manifest path is correct and contains valid JSON lines
2. **"Label index out of range"**: Ensure `--num-classes` matches your dataset (check max label in manifest)
3. **CUDA Out of Memory**: Reduce `--batch-size` or `--image-size`
4. **Low accuracy**: Check purity distribution, try larger encoder, more epochs
5. **Slow training**: Use smaller encoder (`efficientnet_b0`), reduce image size

### Expected Performance

On a typical tree species dataset (50 classes, 10K instances):
- **ResNet50**: ~75-85% accuracy with high-purity instances
- **EfficientNet-B4**: ~80-90% accuracy
- **ConvNeXt-Small**: ~85-92% accuracy

Training time (single RTX 3090):
- ResNet50: ~2-3 hours for 50 epochs
- ConvNeXt-Small: ~4-5 hours for 50 epochs

## Legacy: Semantic Segmentation Approach

The old semantic segmentation code is still available in:
- `train.py` - Pixel-level semantic segmentation training
- `utils/model_prompted.py` - SAM2 + semantic head model
- `test.py` - Complete pipeline for semantic approach

**When to use semantic approach:**
- Never. Use instance classification instead.
- (Kept only for comparison/research purposes)

**Why it's worse:**
- 10x more complex code
- 10x slower training
- Harder to debug
- Pixel-level predictions when you need instance-level

## Project Structure

```
treeAI_new/
├── configs/
│   └── configs.yaml              # Config for SAM2 instance generation
├── scripts/
│   ├── precompute_sam2_instances.py  # Generate instance masks
│   └── export_rsprompter_bbox_prompts.py  # Export bbox prompts
├── utils/
│   ├── instance_dataset.py       # Instance dataset loader
│   ├── instance_classifier.py    # CNN classifier models
│   ├── model_prompted.py         # [Legacy] Semantic head
│   ├── dataset.py                # [Legacy] Semantic dataset
│   └── eval_treeai.py            # [Legacy] Evaluation utils
├── train_instance_classifier.py  # Train instance classifier
├── eval_instance_classifier.py   # Evaluate instance classifier
├── train.py                      # [Legacy] Train semantic model
├── test.py                       # [Legacy] Full semantic pipeline
└── README.md
```

## Citation

If you use this code, please cite:

- SAM2: [Segment Anything Model 2](https://github.com/facebookresearch/segment-anything-2)
- RSPrompter: [RSPrompter: Learn to Prompt for Remote Sensing Instance Segmentation](https://github.com/KyanChen/RSPrompter)




