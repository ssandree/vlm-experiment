"""
FPS-based frame sampling: 초당 fps개 프레임으로 segment 구간에서 타임스탬프 추출.
비디오 길이만큼 추출 (cap 없음).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .sampling_strategy import FrameSamplingStrategy

Segment = Tuple[float, float]


class FpsFrameSampling(FrameSamplingStrategy):
    """초당 fps개 프레임. segment 길이에 따라 개수가 결정된다."""

    def __init__(self, fps: float):
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        self.fps = fps

    def sample(self, segment: Segment, clip_id: Optional[str] = None) -> List[float]:
        start, end = float(segment[0]), float(segment[1])
        duration = end - start
        if duration <= 0:
            raise ValueError(f"Invalid segment duration: start={start}, end={end}")

        n_by_fps = max(1, int(duration * self.fps))
        timestamps = [
            start + i / self.fps
            for i in range(n_by_fps)
            if start + i / self.fps < end - 1e-6
        ]
        if not timestamps:
            timestamps = [start + duration / 2]
        return timestamps
