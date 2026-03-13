"""
Frame aggregation for video captioning/grounding.
Build FrameAggregationStrategy from config dictionary.
"""

from __future__ import annotations

from typing import Any, Dict

from data.input_strategies.build_input_strategy import (
    build_aggregation_strategy as _build_aggregation_strategy,
)


def build_aggregation_strategy(config: Dict[str, Any] | None):
    """Build FrameAggregationStrategy from config."""
    return _build_aggregation_strategy(config)
