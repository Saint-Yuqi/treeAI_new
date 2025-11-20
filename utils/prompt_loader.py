#!/usr/bin/env python3
"""
Prompt loading helpers for SAM2 training/inference pipelines.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def load_bbox_prompts(
    preds_dir: str,
    *,
    min_score: float = 0.0,
    label_whitelist: Optional[Iterable[int]] = None,
    max_prompts: Optional[int] = None,
    sort_descending: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Load bounding-box prompts exported by RSPrompter.

    Parameters
    ----------
    preds_dir : str
        Directory containing JSON files produced by RSPrompter under
        ``third_party/RSPrompter/outputs/preds``.
    min_score : float, default 0.0
        Discard detections with confidence below this threshold.
    label_whitelist : Iterable[int], optional
        Keep only detections whose label appears in this collection.
        ``None`` keeps every label.
    max_prompts : int, optional
        Maximum number of prompts to retain per image after filtering.
    sort_descending : bool, default True
        Sort prompts by score before truncating ``max_prompts``.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping from image stem (e.g. ``000000000009``) to an array of
        shape ``(N, 4)`` storing XYXY bounding boxes in image coordinates.
    """
    preds_path = Path(preds_dir)
    if not preds_path.exists():
        raise FileNotFoundError(f"Prompt directory does not exist: {preds_path}")

    whitelist = set(label_whitelist) if label_whitelist is not None else None
    prompts: Dict[str, np.ndarray] = {}

    for json_path in sorted(preds_path.glob("*.json")):
        with json_path.open("r") as f:
            data = json.load(f)

        labels: Sequence[int] = data.get("labels", [])
        scores: Sequence[float] = data.get("scores", [])
        bboxes: Sequence[Sequence[float]] = data.get("bboxes", [])

        entries: List[Tuple[float, np.ndarray]] = []
        for label, score, bbox in zip(labels, scores, bboxes):
            if score is None or score < min_score:
                continue
            if whitelist is not None and label not in whitelist:
                continue
            if not isinstance(bbox, Sequence) or len(bbox) != 4:
                continue
            entries.append((float(score), np.asarray(bbox, dtype=np.float32)))

        if not entries:
            continue

        if sort_descending:
            entries.sort(key=lambda item: item[0], reverse=True)

        if max_prompts is not None:
            entries = entries[:max_prompts]

        prompts[json_path.stem] = np.stack([item[1] for item in entries], axis=0)

    return prompts


def get_prompts_for_image(
    prompts: Dict[str, np.ndarray],
    image_name: str,
) -> Optional[np.ndarray]:
    """
    Convenience accessor that handles missing entries.
    """
    key = Path(image_name).stem
    return prompts.get(key)


__all__ = ["load_bbox_prompts", "get_prompts_for_image"]
