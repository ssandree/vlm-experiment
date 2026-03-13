"""
Image merge strategy: merges multiple images into a single grid image.
"""

from __future__ import annotations

from typing import List

from PIL import Image

from data.input_strategies.base import InputStrategy, FrameAggregationStrategy


class GridMergeStrategy(InputStrategy):
    def process(self, images: List[Image.Image]) -> Image.Image:
        if len(images) != 4:
            raise ValueError(
                f"GridMergeStrategy requires exactly 4 images, got {len(images)}"
            )

        min_w = min(im.width for im in images)
        min_h = min(im.height for im in images)

        resized = []
        for im in images:
            r = im.resize((min_w, min_h), Image.Resampling.LANCZOS)
            if r.mode != "RGB":
                r = r.convert("RGB")
            resized.append(r)

        w, h = min_w * 2, min_h * 2
        out = Image.new("RGB", (w, h))
        out.paste(resized[0], (0, 0))
        out.paste(resized[1], (min_w, 0))
        out.paste(resized[2], (0, min_h))
        out.paste(resized[3], (min_w, min_h))
        return out


class GridMergeAggregation(FrameAggregationStrategy):
    def aggregate(self, images: List[Image.Image]) -> Image.Image:
        if len(images) != 4:
            raise ValueError(
                f"GridMergeAggregation requires exactly 4 images, got {len(images)}"
            )
        min_w = min(im.width for im in images)
        min_h = min(im.height for im in images)
        resized = []
        for im in images:
            r = im.resize((min_w, min_h), Image.Resampling.LANCZOS)
            if r.mode != "RGB":
                r = r.convert("RGB")
            resized.append(r)
        w, h = min_w * 2, min_h * 2
        out = Image.new("RGB", (w, h))
        out.paste(resized[0], (0, 0))
        out.paste(resized[1], (min_w, 0))
        out.paste(resized[2], (0, min_h))
        out.paste(resized[3], (min_w, min_h))
        return out

    def process(self, images: List[Image.Image]) -> Image.Image:
        return self.aggregate(images)


class GridMergeNoResizeAggregation(FrameAggregationStrategy):
    def aggregate(self, images: List[Image.Image]) -> Image.Image:
        if len(images) != 4:
            raise ValueError(
                f"GridMergeNoResizeAggregation requires exactly 4 images, got {len(images)}"
            )
        max_w = max(im.width for im in images)
        max_h = max(im.height for im in images)
        padded = []
        for im in images:
            if im.width == max_w and im.height == max_h:
                out = im.convert("RGB") if im.mode != "RGB" else im
            else:
                out = Image.new("RGB", (max_w, max_h), (0, 0, 0))
                out.paste(im, (0, 0))
            padded.append(out)
        w, h = max_w * 2, max_h * 2
        out = Image.new("RGB", (w, h))
        out.paste(padded[0], (0, 0))
        out.paste(padded[1], (max_w, 0))
        out.paste(padded[2], (0, max_h))
        out.paste(padded[3], (max_w, max_h))
        return out

    def process(self, images: List[Image.Image]) -> Image.Image:
        return self.aggregate(images)

