#!/usr/bin/env python3
"""Pre-compute SAM2 instance masks from cached RSPrompter prompts.

This script refactors the old "head1 + semantic head" path into a pure
precomputation pipeline:

1. Collect dataset splits from ``cfg.dataset`` (root + extra_roots).
2. Use existing ``rsprompter_prompts/<dataset>/<split>`` JSONs as bbox prompts.
3. Run SAM2 once to harvest high-quality masks per prompt.
4. Save every mask as a standalone binary PNG and record metadata so later
   stages can train an instance-level classifier ("each SAM2 mask -> tree type").

The visualization stack (train.py qualitative plots) still reads the same
``rsprompter_prompts`` directory, so nothing changes there.
"""

from __future__ import annotations

import importlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

# Custom resolver for ${times:x,y} syntax in SAM2 configs
def times_resolver(a, b):
    try:
        # Prioritize integer multiplication if possible
        return int(a) * int(b)
    except (ValueError, TypeError):
        # Fallback to float multiplication
        return float(a) * float(b)

# Register the resolver only if it doesn't exist to avoid errors on re-runs
if not OmegaConf.has_resolver("times"):
    OmegaConf.register_new_resolver("times", times_resolver)


# Add project root to path so we can import SAM2 helpers shipped with the repo.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sam2.build_sam import build_sam2  # type: ignore
from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class InstancePrediction:
    """SAM2 prediction paired with the semantic majority label."""

    mask: np.ndarray  # boolean mask
    label: int  # 1-indexed class id
    area: int
    bbox: List[float]
    score: float
    purity: float


@dataclass
class DatasetSplitSpec:
    """Where to read images/labels/prompts and write outputs."""

    dataset_name: str
    dataset_root: Path
    split: str
    image_dir: Path
    label_dir: Path
    prompt_dir: Path
    output_dir: Path


# ---------------------------------------------------------------------------
# Path + configuration helpers
# ---------------------------------------------------------------------------


def _resolve_sam2_config_path(config_name: str) -> Tuple[str, Path, Path]:
    """Convert a friendly config key into (file name, parent dir, full path)."""

    cfg_key = config_name.strip()
    if not cfg_key.endswith('.yaml'):
        cfg_key = f"{cfg_key}.yaml"

    candidate_path = Path(cfg_key)
    if candidate_path.exists():
        return candidate_path.name, candidate_path.parent, candidate_path.resolve()

    sam2_module = importlib.import_module('sam2')
    sam2_root = Path(sam2_module.__file__).resolve().parent
    configs_root = sam2_root / 'configs'

    search_dirs: List[Path] = []
    if 'sam2.1' in cfg_key:
        # Prioritize sam2.1 configs, then fall back to training configs if needed.
        search_dirs.append(configs_root / 'sam2.1')
        search_dirs.append(configs_root / 'sam2.1_training')
    search_dirs.append(configs_root / 'sam2')

    for directory in search_dirs:
        config_path = directory / cfg_key
        if config_path.exists():
            return config_path.name, directory, config_path.resolve()

    raise FileNotFoundError(
        f"Could not locate SAM2 config '{config_name}'. Looked inside: "
        f"{', '.join(str(d) for d in search_dirs)}"
    )


def _first_existing_dir(candidates: Iterable[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


def build_sam2_from_config_path(config_path: str, ckpt_path: str, device: str = "cuda"):
    """
    Build SAM2 model from absolute config file path.
    This avoids Hydra's interpolation resolution issues (e.g., 'divide' type).
    
    Args:
        config_path: Absolute path to config YAML file
        ckpt_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        SAM2 model instance
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load config directly with OmegaConf (avoid Hydra's resolvers)
    cfg = OmegaConf.load(config_path)
    
    # Extract model config - it could be at cfg.model or cfg.trainer.model
    if 'trainer' in cfg and 'model' in cfg.trainer:
        # Training config format
        model_cfg = cfg.trainer.model
    elif 'model' in cfg:
        # Inference config format
        model_cfg = cfg.model
    else:
        raise ValueError(f"Cannot find 'model' config in {config_path}")
    
    # Resolve only the model config (avoid resolving training-specific interpolations)
    try:
        OmegaConf.resolve(model_cfg)
    except Exception as e:
        # If resolution fails, try to manually resolve common patterns
        print(f"Warning: Config resolution had issues: {e}")
        print("Attempting to resolve critical model params only...")
        # Manually set image_size if it references scratch.resolution
        if 'image_size' in model_cfg and isinstance(model_cfg.image_size, str) and '$' in str(model_cfg.image_size):
            if 'scratch' in cfg and 'resolution' in cfg.scratch:
                model_cfg.image_size = cfg.scratch.resolution
                print(f"Set image_size to {model_cfg.image_size}")
    
    # Instantiate model
    model = instantiate(model_cfg, _recursive_=True)
    
    # Load checkpoint
    from sam2.build_sam import _load_checkpoint
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    model.eval()
    
    return model


def discover_dataset_splits(
    cfg: DictConfig, 
    output_root: Path,
    dataset_filter: Optional[str] = None,
) -> List[DatasetSplitSpec]:
    """Build dataset split specs from cfg.dataset + rsprompter prompt cache.
    
    Args:
        cfg: Hydra config
        output_root: Root directory for output
        dataset_filter: Optional dataset name to filter (e.g., '12_RGB_SemSegm_640_fL')
                       If provided, only process this dataset.
    """

    dataset_cfg = OmegaConf.select(cfg, 'dataset', default=None)
    if dataset_cfg is None:
        raise ValueError("Config missing 'dataset' block")

    dataset_dict = OmegaConf.to_container(dataset_cfg, resolve=True) or {}
    root_value = dataset_dict.get('root')
    if not root_value:
        raise ValueError("dataset.root must be specified in configs/configs.yaml")

    dataset_roots: List[Path] = [Path(root_value).expanduser()]
    extra_roots = dataset_dict.get('extra_roots') or []
    for extra in extra_roots:
        if not extra:
            continue
        dataset_roots.append(Path(extra).expanduser())
    splits = dataset_dict.get('splits') or ['train', 'val', 'test', 'pick']
    bbox_root = Path(cfg.rsprompter.bbox_prompt_dir).expanduser().resolve()

    specs: List[DatasetSplitSpec] = []
    for dataset_root in dataset_roots:
        if not dataset_root.exists():
            print(f"⚠️  Dataset root not found, skipping: {dataset_root}")
            continue
        dataset_name = dataset_root.name
        
        # Filter by dataset name if specified
        if dataset_filter is not None and dataset_name != dataset_filter:
            continue
            
        for split in splits:
            image_dir = _first_existing_dir([
                dataset_root / split / 'images',
                dataset_root / 'images' / split,
                dataset_root / split,
            ])
            label_dir = _first_existing_dir([
                dataset_root / split / 'labels',
                dataset_root / split / 'masks',
                dataset_root / 'labels' / split,
                dataset_root / 'masks' / split,
            ])
            prompt_dir = bbox_root / dataset_name / split

            if image_dir is None or label_dir is None:
                continue
            if not prompt_dir.exists():
                continue

            output_dir = output_root / dataset_name / split
            specs.append(
                DatasetSplitSpec(
                    dataset_name=dataset_name,
                    dataset_root=dataset_root,
                    split=split,
                    image_dir=image_dir,
                    label_dir=label_dir,
                    prompt_dir=prompt_dir,
                    output_dir=output_dir,
                )
            )

    return specs


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


IMAGE_EXTS: Sequence[str] = ('.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG')
LABEL_EXTS: Sequence[str] = ('.png', '.PNG', '.tif', '.tiff', '.TIF', '.TIFF')


def find_image_file(image_dir: Path, stem: str) -> Optional[Path]:
    for ext in IMAGE_EXTS:
        path = image_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def load_bbox_prompts_for_image(prompt_dir: Path, image_name: str) -> np.ndarray:
    prompt_file = prompt_dir / f"{image_name}.json"
    if not prompt_file.exists():
        return np.empty((0, 4), dtype=np.float32)

    with prompt_file.open('r') as f:
        data = json.load(f)

    if isinstance(data, dict) and 'bboxes' in data:
        bboxes = data.get('bboxes', [])
    elif isinstance(data, list):
        bboxes = [item['bbox'] for item in data if 'bbox' in item]
    else:
        bboxes = []

    if not bboxes:
        return np.empty((0, 4), dtype=np.float32)

    return np.asarray(bboxes, dtype=np.float32)


# ---------------------------------------------------------------------------
# Core SAM2 instance generation
# ---------------------------------------------------------------------------


def sort_bboxes_by_area(
    bboxes: np.ndarray, 
    ascending: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """按面积排序bboxes，用于Z-Order处理。
    
    森林场景中树木经常重叠。先处理大框（背景），后处理小框（前景），
    这样小树的Mask会覆盖在大树之上，符合视觉遮挡关系。
    
    Args:
        bboxes: (N, 4) array of [x1, y1, x2, y2]
        ascending: True=小到大, False=大到小（推荐False：大框先处理）
    
    Returns:
        sorted_bboxes: 排序后的bboxes
        sort_indices: 排序索引（用于追踪原始顺序）
    """
    if bboxes.size == 0:
        return bboxes, np.array([], dtype=np.int64)
    
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    sort_indices = np.argsort(areas)
    if not ascending:
        sort_indices = sort_indices[::-1]
    return bboxes[sort_indices], sort_indices


def remove_overlapping_instances(
    predictions: List[InstancePrediction],
    iou_threshold: float = 0.5,
) -> List[InstancePrediction]:
    """Remove overlapping instances, keeping only the best one per region.
    
    Similar to bbox_to_mask_sam2.py's overlay strategy, but preserves instance
    granularity. Higher quality masks (purity, area, score) take precedence.
    
    Args:
        predictions: List of instance predictions to deduplicate.
        iou_threshold: If a mask overlaps with already-occupied pixels by more
                       than this ratio, it is discarded.
    
    Returns:
        Deduplicated list of instance predictions.
    """
    if len(predictions) <= 1:
        return predictions
    
    # Sort by quality: purity > area > score (descending)
    # Best masks are processed first and "claim" their pixels
    sorted_preds = sorted(
        predictions,
        key=lambda p: (p.purity, p.area, p.score),
        reverse=True,
    )
    
    H, W = sorted_preds[0].mask.shape
    occupied = np.zeros((H, W), dtype=bool)
    kept: List[InstancePrediction] = []
    
    for pred in sorted_preds:
        mask_area = pred.mask.sum()
        if mask_area == 0:
            continue
        
        # How much of this mask is already covered by better masks?
        overlap = (pred.mask & occupied).sum()
        overlap_ratio = overlap / mask_area
        
        if overlap_ratio > iou_threshold:
            # This mask is mostly covered by higher-quality masks, skip it
            continue
        
        # Keep this mask and mark its pixels as occupied
        kept.append(pred)
        occupied |= pred.mask
    
    return kept


def generate_sam2_instances(
    predictor: SAM2ImagePredictor,
    image: np.ndarray,
    bboxes: np.ndarray,
    gt_mask: np.ndarray,
    min_area: int,
    min_overlap: float,
    iou_threshold: float = 0.5,
    sort_by_area: bool = True,
    restrict_to_bbox: bool = True,
    bbox_margin: float = 0.15,
) -> List[InstancePrediction]:
    """Run SAM2 for the provided bbox prompts and filter valid instances.
    
    Includes deduplication: when multiple bboxes cover the same tree crown,
    only the highest-quality mask is kept (based on purity, area, score).
    
    Args:
        predictor: SAM2ImagePredictor instance
        image: RGB image as numpy array (H, W, 3)
        bboxes: (N, 4) array of [x1, y1, x2, y2] bbox prompts
        gt_mask: Ground truth semantic mask (H, W) for label assignment
        min_area: Minimum mask area in pixels to keep
        min_overlap: Minimum overlap ratio with gt_mask to be valid
        iou_threshold: IoU threshold for deduplication
        sort_by_area: If True, process bboxes from large to small (Z-order).
                      Large trees are processed first, small trees later.
                      This ensures small trees (foreground) can override 
                      large trees (background) in overlap handling.
        restrict_to_bbox: If True, limit SAM2 predictions within expanded bbox.
                          Prevents SAM2 from predicting far outside the prompt bbox.
        bbox_margin: Margin ratio to expand bbox (0.15 = 15%). Only used when
                     restrict_to_bbox=True. Allows SAM2 to predict slightly 
                     beyond the manually annotated bbox to compensate for 
                     imprecise annotations and tree branches extending outside.
    
    Returns:
        List of validated InstancePrediction objects.
    """

    if bboxes.size == 0:
        return []

    H, W = image.shape[:2]
    
    # Z-Order优化：按面积排序，大框先处理，小框后处理
    # 这样在后续的重叠处理中，小树可以优先保留（覆盖大树）
    if sort_by_area:
        bboxes, _ = sort_bboxes_by_area(bboxes, ascending=False)

    predictor.set_image(image)
    predictions: List[InstancePrediction] = []

    min_valid_pixels = max(int(min_area * min_overlap), 1)

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox[:4]
        
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox[None, :],
            multimask_output=False,
        )

        mask = masks[0].astype(bool)
        
        # 放宽BBox硬裁剪限制：不做硬裁剪，或增加margin余量
        # 人工标注的BBox往往不完美，SAM2可能预测出框外延伸的树枝
        if restrict_to_bbox:
            # 计算带margin的扩展bbox（允许SAM2预测超出原始bbox一定比例）
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            margin_x = int(bbox_w * bbox_margin)
            margin_y = int(bbox_h * bbox_margin)
            
            # 扩展后的bbox坐标（限制在图像范围内）
            exp_x1 = max(0, int(x1 - margin_x))
            exp_y1 = max(0, int(y1 - margin_y))
            exp_x2 = min(W - 1, int(x2 + margin_x))
            exp_y2 = min(H - 1, int(y2 + margin_y))
            
            # 创建扩展后的bbox区域mask
            bbox_mask = np.zeros((H, W), dtype=bool)
            bbox_mask[exp_y1:exp_y2+1, exp_x1:exp_x2+1] = True
            
            # 将SAM2预测限制在扩展后的bbox框内（软限制，带margin）
            mask = mask & bbox_mask
        
        area = int(mask.sum())
        if area < min_area:
            continue

        valid_pixels = gt_mask[mask]
        if valid_pixels.size < min_valid_pixels:
            continue

        class_counts = Counter(int(v) for v in valid_pixels.flatten())
        if not class_counts:
            continue

        most_common_class, count = class_counts.most_common(1)[0]
        if most_common_class == 0:
            continue

        purity = count / max(valid_pixels.size, 1)
        if purity < 0.3:
            continue

        bbox_list = bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox)
        predictions.append(
            InstancePrediction(
                mask=mask,
                label=int(most_common_class),
                area=area,
                bbox=[float(x) for x in bbox_list],
                score=float(scores[0]) if scores is not None else 0.0,
                purity=float(purity),
            )
        )

    # Remove overlapping instances - keep only the best mask per tree crown
    # This mimics bbox_to_mask_sam2.py's overlay logic but preserves instance info
    predictions = remove_overlapping_instances(predictions, iou_threshold=iou_threshold)

    return predictions


def rebuild_manifest(spec: DatasetSplitSpec) -> Tuple[int, Counter]:
    """Aggregate all per-image metadata into a manifest.jsonl file."""

    instances_root = spec.output_dir / 'instances'
    manifest_path = spec.output_dir / 'instances_manifest.jsonl'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_entries: List[dict] = []
    stats = Counter()

    if not instances_root.exists():
        with manifest_path.open('w') as f:
            f.write("")
        return 0, stats

    for image_dir in sorted(p for p in instances_root.iterdir() if p.is_dir()):
        meta_file = image_dir / 'metadata.json'
        if not meta_file.exists():
            continue
        with meta_file.open('r') as f:
            meta = json.load(f)

        for instance in meta.get('instances', []):
            mask_path = image_dir / instance['mask_file']
            manifest_entries.append({
                'dataset': spec.dataset_name,
                'split': spec.split,
                'image': meta.get('image'),
                'mask_path': str(mask_path.relative_to(spec.output_dir)),
                'source_image_path': meta.get('source_image_path'),
                'label_path': meta.get('label_path'),
                'label': int(instance['label']),
                'area': int(instance.get('area', 0)),
                'bbox': instance.get('bbox', []),
                'score': float(instance.get('score', 0.0)),
                'purity': float(instance.get('purity', 0.0)),
            })
            stats[instance['label']] += 1

    with manifest_path.open('w') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + '\n')

    return len(manifest_entries), stats


def process_split(
    spec: DatasetSplitSpec,
    predictor: SAM2ImagePredictor,
    *,
    overwrite: bool,
    min_area: int,
    min_overlap: float,
    iou_threshold: float = 0.5,
    sort_by_area: bool = True,
    restrict_to_bbox: bool = True,
    bbox_margin: float = 0.15,
):
    """Generate masks for one dataset split and record metadata."""

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    instances_root = spec.output_dir / 'instances'
    instances_root.mkdir(parents=True, exist_ok=True)

    label_files = sorted({p for ext in LABEL_EXTS for p in spec.label_dir.glob(f"*{ext}")})

    print("=" * 80)
    print(f"📂 Dataset: {spec.dataset_name} | Split: {spec.split}")
    print(f"   Images : {spec.image_dir}")
    print(f"   Labels : {spec.label_dir}")
    print(f"   Prompts: {spec.prompt_dir}")
    print(f"   Output : {spec.output_dir}")
    print("=" * 80)

    total_images = len(label_files)
    processed_images = 0
    skipped_existing = 0
    errors = []
    interim_stats = Counter()

    for label_file in tqdm(label_files, desc=f"Processing {spec.dataset_name}-{spec.split}"):
        image_name = label_file.stem
        image_file = find_image_file(spec.image_dir, image_name)
        if image_file is None:
            errors.append(f"Image missing for {image_name}")
            continue

        prompt_bboxes = load_bbox_prompts_for_image(spec.prompt_dir, image_name)
        if prompt_bboxes.size == 0:
            continue

        image_instance_dir = instances_root / image_name
        if image_instance_dir.exists() and not overwrite:
            skipped_existing += 1
            meta_file = image_instance_dir / 'metadata.json'
            if meta_file.exists():
                with meta_file.open('r') as f:
                    meta = json.load(f)
                for inst in meta.get('instances', []):
                    interim_stats[int(inst['label'])] += 1
            continue
        elif image_instance_dir.exists():
            shutil.rmtree(image_instance_dir)

        try:
            image = np.array(Image.open(image_file).convert('RGB'))
            gt_mask = np.array(Image.open(label_file)).astype(np.int32)
            instances = generate_sam2_instances(
                predictor,
                image,
                prompt_bboxes,
                gt_mask,
                min_area=min_area,
                min_overlap=min_overlap,
                iou_threshold=iou_threshold,
                sort_by_area=sort_by_area,
                restrict_to_bbox=restrict_to_bbox,
                bbox_margin=bbox_margin,
            )

            if not instances:
                continue

            image_instance_dir.mkdir(parents=True, exist_ok=True)
            per_image_meta = {
                'dataset': spec.dataset_name,
                'split': spec.split,
                'image': image_name,
                'source_image_path': str(image_file),
                'label_path': str(label_file),
                'instances': [],
            }

            for idx, inst in enumerate(instances):
                mask_img = (inst.mask.astype(np.uint8)) * 255
                mask_filename = f"mask_{idx:03d}.png"
                mask_path = image_instance_dir / mask_filename
                Image.fromarray(mask_img, mode='L').save(mask_path)

                per_image_meta['instances'].append({
                    'mask_file': mask_filename,
                    'label': inst.label,
                    'area': inst.area,
                    'bbox': inst.bbox,
                    'score': inst.score,
                    'purity': inst.purity,
                })
                interim_stats[inst.label] += 1

            with (image_instance_dir / 'metadata.json').open('w') as f:
                json.dump(per_image_meta, f, indent=2)

            processed_images += 1

        except Exception as err:  # pylint: disable=broad-except
            errors.append(f"{image_name}: {err}")
            if image_instance_dir.exists():
                shutil.rmtree(image_instance_dir)

    manifest_total, manifest_stats = rebuild_manifest(spec)

    print(f"\nFinished split {spec.dataset_name}-{spec.split}")
    print(f"  Images total    : {total_images}")
    print(f"  Images processed: {processed_images}")
    print(f"  Skipped existing: {skipped_existing}")
    print(f"  Instances saved : {manifest_total}")
    if errors:
        print(f"  Errors          : {len(errors)} (see log above)")
        for msg in errors[:5]:
            print(f"    - {msg}")

    top_classes = manifest_stats.most_common(10)
    if top_classes:
        print("\n📈 Top classes:")
        for class_id, count in top_classes:
            pct = 100.0 * count / max(manifest_total, 1)
            print(f"  Class {class_id:>2}: {count:6d} ({pct:5.2f}%)")
    else:
        print("\n⚠️  No valid instances generated for this split")

    return {
        'dataset': spec.dataset_name,
        'split': spec.split,
        'images': total_images,
        'processed': processed_images,
        'skipped': skipped_existing,
        'total_instances': manifest_total,
        'class_distribution': manifest_stats,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="../configs", config_name="configs", version_base=None)
def main(cfg: DictConfig):
    overwrite = bool(OmegaConf.select(cfg, 'precompute.overwrite', default=False))
    output_root = Path(OmegaConf.select(cfg, 'precompute.output_dir', default='./data/sam2_instances'))
    min_area = int(OmegaConf.select(cfg, 'precompute.min_area', default=100))
    min_overlap = float(OmegaConf.select(cfg, 'precompute.min_overlap', default=0.5))
    iou_threshold = float(OmegaConf.select(cfg, 'precompute.iou_threshold', default=0.5))
    dataset_filter = OmegaConf.select(cfg, 'precompute.dataset_filter', default=None)
    
    # Z-Order and BBox margin parameters
    sort_by_area = bool(OmegaConf.select(cfg, 'precompute.sort_by_area', default=True))
    restrict_to_bbox = bool(OmegaConf.select(cfg, 'precompute.restrict_to_bbox', default=True))
    bbox_margin = float(OmegaConf.select(cfg, 'precompute.bbox_margin', default=0.15))

    split_specs = discover_dataset_splits(cfg, output_root, dataset_filter=dataset_filter)
    if not split_specs:
        print("❌ No dataset splits matched the current configuration.")
        print("   Ensure cfg.dataset roots are correct and rsprompter_prompts exist.")
        return

    print(OmegaConf.to_yaml(cfg))
    if overwrite:
        print("\n⚠️  OVERWRITE MODE: existing per-image masks will be replaced.")
    else:
        print("\n✅ SKIP MODE: existing per-image outputs are reused.")
    
    print(f"\n🔧 Instance generation settings:")
    print(f"   sort_by_area     : {sort_by_area} (Z-order: large bboxes first)")
    print(f"   restrict_to_bbox : {restrict_to_bbox}")
    print(f"   bbox_margin      : {bbox_margin:.0%}")

    print("\n🔧 Loading SAM2...")
    print(f"  Checkpoint: {cfg.model.sam2_checkpoint}")
    print(f"  Config    : {cfg.model.sam2_config}")

    # Resolve config path (handle both absolute paths and relative names)
    sam2_config_path = cfg.model.sam2_config
    if not Path(sam2_config_path).is_absolute():
        # Try to resolve as relative path or search in SAM2 configs
        config_name, config_root, resolved_config = _resolve_sam2_config_path(sam2_config_path)
        sam2_config_path = str(resolved_config)
    else:
        # Already an absolute path
        resolved_config = Path(sam2_config_path)
        if not resolved_config.exists():
            raise FileNotFoundError(f"SAM2 config not found: {sam2_config_path}")
    
    print(f"  Resolved  : {resolved_config}")

    # Use direct config loading to avoid Hydra interpolation issues
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam2_model = build_sam2_from_config_path(
        config_path=str(resolved_config),
        ckpt_path=cfg.model.sam2_checkpoint,
        device=device,
    )

    predictor = SAM2ImagePredictor(sam2_model)
    print("✅ SAM2 ready\n")

    overall_stats = Counter()
    overall_instances = 0

    for spec in split_specs:
        summary = process_split(
            spec,
            predictor,
            overwrite=overwrite,
            min_area=min_area,
            min_overlap=min_overlap,
            iou_threshold=iou_threshold,
            sort_by_area=sort_by_area,
            restrict_to_bbox=restrict_to_bbox,
            bbox_margin=bbox_margin,
        )
        overall_stats += summary['class_distribution']
        overall_instances += summary['total_instances']

    print("\n" + "=" * 80)
    print("📊 Overall summary")
    print("=" * 80)
    print(f"Total instances: {overall_instances}")
    if overall_stats:
        for class_id, count in overall_stats.most_common(15):
            pct = 100.0 * count / max(overall_instances, 1)
            print(f"  Class {class_id:>2}: {count:6d} ({pct:5.2f}%)")
    else:
        print("No instances were generated across all splits.")

    print("\n🎉 SAM2 instance precomputation complete.")
    print("   Outputs stored under:", output_root)


if __name__ == "__main__":
    GlobalHydra.instance().clear()
    main()
