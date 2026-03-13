"""
Multi-image input strategy.
Returns images unchanged as a list, suitable for multi-image model inputs.
"""

from __future__ import annotations

from typing import List

from PIL import Image

from data.input_strategies.base import InputStrategy, FrameAggregationStrategy


class MultiImageStrategy(InputStrategy):
    def process(self, images: List[Image.Image]) -> List[Image.Image]:
        return list(images)


class MultiImageAggregation(FrameAggregationStrategy):
    def aggregate(self, images: List[Image.Image]) -> List[Image.Image]:
        return list(images)

    def process(self, images: List[Image.Image]) -> List[Image.Image]:
        return self.aggregate(images)


class IdentityAggregation(FrameAggregationStrategy):
    def aggregate(self, images: List[Image.Image]) -> List[Image.Image]:
        return list(images)

    def process(self, images: List[Image.Image]) -> List[Image.Image]:
        return self.aggregate(images)

