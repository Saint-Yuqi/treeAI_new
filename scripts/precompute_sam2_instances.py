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


def discover_dataset_splits(cfg: DictConfig, output_root: Path) -> List[DatasetSplitSpec]:
    """Build dataset split specs from cfg.dataset + rsprompter prompt cache."""

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


def generate_sam2_instances(
    predictor: SAM2ImagePredictor,
    image: np.ndarray,
    bboxes: np.ndarray,
    gt_mask: np.ndarray,
    min_area: int,
    min_overlap: float,
) -> List[InstancePrediction]:
    """Run SAM2 for the provided bbox prompts and filter valid instances."""

    if bboxes.size == 0:
        return []

    predictor.set_image(image)
    predictions: List[InstancePrediction] = []

    min_valid_pixels = max(int(min_area * min_overlap), 1)

    for bbox in bboxes:
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox[None, :],
            multimask_output=False,
        )

        mask = masks[0].astype(bool)
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
    overwrite = bool(cfg.get('overwrite', False))
    output_root = Path(OmegaConf.select(cfg, 'precompute.output_dir', default='./data/sam2_instances'))
    min_area = int(OmegaConf.select(cfg, 'precompute.min_area', default=100))
    min_overlap = float(OmegaConf.select(cfg, 'precompute.min_overlap', default=0.5))

    split_specs = discover_dataset_splits(cfg, output_root)
    if not split_specs:
        print("❌ No dataset splits matched the current configuration.")
        print("   Ensure cfg.dataset roots are correct and rsprompter_prompts exist.")
        return

    print(OmegaConf.to_yaml(cfg))
    if overwrite:
        print("\n⚠️  OVERWRITE MODE: existing per-image masks will be replaced.")
    else:
        print("\n✅ SKIP MODE: existing per-image outputs are reused.")

    print("\n🔧 Loading SAM2...")
    print(f"  Checkpoint: {cfg.model.sam2_checkpoint}")
    print(f"  Config    : {cfg.model.sam2_config}")

    config_name, config_root, resolved_config = _resolve_sam2_config_path(cfg.model.sam2_config)
    print(f"  Resolved  : {resolved_config}")

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_root), job_name="sam2_precompute", version_base=None):
        sam2_model = build_sam2(
            config_file=config_name,
            ckpt_path=cfg.model.sam2_checkpoint,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
    GlobalHydra.instance().clear()

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
