"""
Base classes for input strategies.
Input strategies define how multiple images/frames are processed before model input.
"""

from __future__ import annotations

from typing import List, Union
from abc import ABC, abstractmethod

from PIL import Image


class InputStrategy(ABC):
    """Takes a list of images and returns either a single merged image or a list of images."""

    @abstractmethod
    def process(self, images: List[Image.Image]) -> Union[Image.Image, List[Image.Image]]:
        raise NotImplementedError


class FrameAggregationStrategy(InputStrategy):
    """Legacy alias for InputStrategy (video captioning compatibility)."""

    def aggregate(self, images: List[Image.Image]) -> Union[Image.Image, List[Image.Image]]:
        return self.process(images)

    def process(self, images: List[Image.Image]) -> Union[Image.Image, List[Image.Image]]:
        return self.aggregate(images)
