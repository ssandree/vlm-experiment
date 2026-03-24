"""
Uniform frame sampling: 전체 비디오(segment)에서 일정한 시간 간격으로 N개 프레임.
- num_frames: 추출할 프레임 개수 (균등 간격).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

Segment = Tuple[float, float]


class UniformFrameSampling:
    """전체 비디오에서 일정한 시간 간격으로 N개의 프레임을 뽑는다."""

    def __init__(self, num_frames: int):
        if num_frames <= 0:
            raise ValueError(f"num_frames must be > 0, got {num_frames}")
        self.num_frames = num_frames

    def sample(self, segment: Segment, clip_id: Optional[str] = None) -> List[float]:
        return uniform_timestamps_for_segment(segment, self.num_frames)


def uniform_timestamps_for_segment(
    segment: Segment,
    num_frames: int,
) -> List[float]:
    """
    Segment 내에서 균등 간격으로 num_frames개 타임스탬프 반환.

    Args:
        segment: (start_sec, end_sec)
        num_frames: 목표 프레임 수 (일정한 시간 간격).
    """
    start, end = float(segment[0]), float(segment[1])
    duration = end - start
    if duration <= 0:
        raise ValueError(f"Invalid segment duration: start={start}, end={end}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be > 0, got {num_frames}")

    interval = duration / num_frames
    return [start + (i + 0.5) * interval for i in range(num_frames)]
