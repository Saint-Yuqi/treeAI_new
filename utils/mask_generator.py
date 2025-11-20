#!/usr/bin/env python3
"""
Utility helpers for building SAM2 automatic mask generators with
TreeAI-friendly defaults.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator  # pyright: ignore[reportMissingImports]
from sam2.modeling.sam2_base import SAM2Base  # pyright: ignore[reportMissingImports]

# Default mask generator parameters tuned for TreeAI semantic training.
DEFAULT_AUTOMATIC_MASK_GENERATOR_CONFIG: Dict[str, Any] = {
    "points_per_side": 16,
    "points_per_batch": 128,
    "pred_iou_thresh": 0.45,
    "stability_score_thresh": 0.65,
    "stability_score_offset": 0.3,
    "crop_n_layers": 1,
    "box_nms_thresh": 0.5,
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 200.0,
    "use_m2m": True,
}


def build_automatic_mask_generator(
    model: SAM2Base,
    overrides: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> SAM2AutomaticMaskGenerator:
    """
    Build a SAM2AutomaticMaskGenerator pre-configured for TreeAI experiments.

    Parameters
    ----------
    model : SAM2Base
        A constructed SAM2 model (same object passed to SAM2AutomaticMaskGenerator).
    overrides : dict, optional
        Runtime overrides for the default configuration.
    **kwargs : Any
        Additional keyword arguments forwarded to SAM2AutomaticMaskGenerator. Values
        supplied here take precedence over both defaults and overrides.

    Returns
    -------
    SAM2AutomaticMaskGenerator
        Configured mask generator ready for use.
    """
    config: Dict[str, Any] = dict(DEFAULT_AUTOMATIC_MASK_GENERATOR_CONFIG)
    if overrides:
        config.update(overrides)
    if kwargs:
        config.update(kwargs)

    return SAM2AutomaticMaskGenerator(model=model, **config)


__all__ = [
    "build_automatic_mask_generator",
    "DEFAULT_AUTOMATIC_MASK_GENERATOR_CONFIG",
]
