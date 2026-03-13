"""
Top-right crop strategy.
Crops the top-right quadrant when dividing each image into 4 equal parts.
Suitable for inference on the upper-right region only.
"""

from __future__ import annotations

from typing import List, Union

from PIL import Image

from data.input_strategies.base import InputStrategy, FrameAggregationStrategy


def _crop_top_right(img: Image.Image) -> Image.Image:
    """Crop the top-right quadrant (1/4) of an image."""
    w, h = img.size
    left = w // 2
    top = 0
    right = w
    bottom = h // 2
    return img.crop((left, top, right, bottom))


class TopRightCropStrategy(InputStrategy):
    """Crops the top-right quadrant from each image."""

    def process(self, images: List[Image.Image]) -> List[Image.Image]:
        return [_crop_top_right(im) for im in images]


class TopRightCropAggregation(FrameAggregationStrategy):
    """Frame aggregation: crop top-right quadrant from each frame."""

    def aggregate(self, images: List[Image.Image]) -> List[Image.Image]:
        return [_crop_top_right(im) for im in images]

    def process(self, images: List[Image.Image]) -> List[Image.Image]:
        return self.aggregate(images)
