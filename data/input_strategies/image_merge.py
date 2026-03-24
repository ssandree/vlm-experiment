"""
Image merge strategy: merges multiple images into a single grid image (no resize).
"""

from __future__ import annotations

import sys
from typing import List, Union

from PIL import Image

from data.input_strategies.base import InputStrategy, FrameAggregationStrategy


def _make_one_grid_no_resize(images: List[Image.Image]) -> Image.Image:
    """Make a single 2x2 grid from exactly 4 images (no resize, pad to max size)."""
    assert len(images) == 4
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


class GridMergeNoResizeAggregation(FrameAggregationStrategy):
    """
    Frame aggregation: 4개씩 2x2 grid (리사이즈 없음).
    frame 개수가 4의 배수면 그리드 N개를 리스트로 반환; 4의 배수가 아니면 에러.
    """

    def aggregate(
        self, images: List[Image.Image]
    ) -> Union[Image.Image, List[Image.Image]]:
        n = len(images)
        if n == 0:
            raise ValueError("GridMergeNoResizeAggregation requires at least one image")
        if n % 4 != 0:
            print("frame 개수가 4의 배수가 아닙니다", file=sys.stderr)
            raise ValueError(
                f"GridMergeNoResizeAggregation requires frame count to be a multiple of 4, got {n}"
            )
        if n == 4:
            return _make_one_grid_no_resize(images)
        grids: List[Image.Image] = []
        for i in range(0, n, 4):
            chunk = images[i : i + 4]
            grids.append(_make_one_grid_no_resize(chunk))
        return grids

    def process(
        self, images: List[Image.Image]
    ) -> Union[Image.Image, List[Image.Image]]:
        return self.aggregate(images)


class ImageStripAggregation(FrameAggregationStrategy):
    """Frame aggregation: concatenate frames horizontally into one wide image (image strip)."""

    def aggregate(self, images: List[Image.Image]) -> Image.Image:
        if not images:
            raise ValueError("ImageStripAggregation requires at least one image")
        min_w = min(im.width for im in images)
        min_h = min(im.height for im in images)
        resized = []
        for im in images:
            r = im.resize((min_w, min_h), Image.Resampling.LANCZOS)
            if r.mode != "RGB":
                r = r.convert("RGB")
            resized.append(r)
        total_w = min_w * len(images)
        out = Image.new("RGB", (total_w, min_h))
        for i, im in enumerate(resized):
            out.paste(im, (i * min_w, 0))
        return out

    def process(self, images: List[Image.Image]) -> Image.Image:
        return self.aggregate(images)

