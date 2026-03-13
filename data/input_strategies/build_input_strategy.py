"""
Build InputStrategy / FrameAggregationStrategy from config dictionary.

Config format:
  input_strategy:
    type: "multi_image"   # "identity" | "multi_image" | "grid_merge" | "top_right_crop"

Legacy format (for video captioning):
  aggregation:
    type: "multi"   # "identity" | "multi" | "grid" | "grid_no_resize" | "top_right_crop"
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
    GridMergeStrategy,
    GridMergeAggregation,
    GridMergeNoResizeAggregation,
)
from data.input_strategies.top_right_crop import (
    TopRightCropStrategy,
    TopRightCropAggregation,
)


def build_input_strategy(config: Dict[str, Any] | None) -> InputStrategy:
    """
    Build InputStrategy from config dictionary.
    """
    config = config or {}
    strategy_type = (config.get("type") or "multi_image").strip().lower()

    if strategy_type == "multi_image":
        return MultiImageStrategy()
    if strategy_type == "grid_merge":
        return GridMergeStrategy()
    if strategy_type == "top_right_crop":
        return TopRightCropStrategy()

    if strategy_type == "identity":
        return IdentityAggregation()
    if strategy_type == "multi":
        return MultiImageAggregation()
    if strategy_type == "grid":
        return GridMergeAggregation()
    if strategy_type == "grid_no_resize":
        return GridMergeNoResizeAggregation()
    if strategy_type == "top_right_crop":
        return TopRightCropAggregation()

    raise ValueError(
        f"Unknown input strategy type: {config.get('type')}. "
        "Expected 'multi_image', 'grid_merge', 'top_right_crop', "
        "or legacy types: 'identity', 'multi', 'grid', 'grid_no_resize'."
    )


def build_aggregation_strategy(config: Dict[str, Any] | None) -> FrameAggregationStrategy:
    """
    Legacy function: build FrameAggregationStrategy from config.
    """
    strategy = build_input_strategy(config)

    if isinstance(strategy, FrameAggregationStrategy):
        return strategy

    class WrappedAggregation(FrameAggregationStrategy):
        def aggregate(self, images):
            return strategy.process(images)

    return WrappedAggregation()

