"""
Build InputStrategy / FrameAggregationStrategy from config dictionary.

Input strategy (run_image_inference, run_multi_image_inference):
  input_strategy:
    type: "identity" | "multi_image" | "top_right_crop"

Aggregation (run_video_captioning, run_video_grounding):
  aggregation:
    type: "identity" | "multi" | "grid_no_resize" | "top_right_crop" | "image_strip"
"""

from __future__ import annotations

from typing import Any, Dict

from data.input_strategies.base import InputStrategy, FrameAggregationStrategy
from data.input_strategies.multi_image import (
    MultiImageStrategy,
    MultiImageAggregation,
    IdentityAggregation,
)
from data.input_strategies.image_merge import (
    GridMergeNoResizeAggregation,
    ImageStripAggregation,
)
from data.input_strategies.top_right_crop import (
    TopRightCropStrategy,
    TopRightCropAggregation,
)


def build_input_strategy(config: Dict[str, Any] | None) -> InputStrategy:
    """
    Build InputStrategy for image input (run_image_inference, run_multi_image_inference).
    Types: identity, multi_image, top_right_crop.
    """
    config = config or {}
    strategy_type = (config.get("type") or "identity").strip().lower()

    if strategy_type == "multi_image":
        return MultiImageStrategy()
    if strategy_type == "top_right_crop":
        return TopRightCropStrategy()
    if strategy_type == "identity":
        return IdentityAggregation()

    raise ValueError(
        f"Unknown input strategy type: {config.get('type')}. "
        "Expected 'identity', 'multi_image', or 'top_right_crop'."
    )


def build_aggregation_strategy(config: Dict[str, Any] | None) -> FrameAggregationStrategy:
    """
    Build FrameAggregationStrategy for video captioning/grounding.
    Types: identity, multi, grid_no_resize, top_right_crop, image_strip.
    """
    config = config or {}
    strategy_type = (config.get("type") or "multi").strip().lower()

    if strategy_type == "identity":
        return IdentityAggregation()
    if strategy_type == "multi":
        return MultiImageAggregation()
    if strategy_type == "grid_no_resize":
        return GridMergeNoResizeAggregation()
    if strategy_type == "top_right_crop":
        return TopRightCropAggregation()
    if strategy_type == "image_strip":
        return ImageStripAggregation()

    raise ValueError(
        f"Unknown aggregation type: {config.get('type')}. "
        "Expected 'identity', 'multi', 'grid_no_resize', 'top_right_crop', or 'image_strip'."
    )
