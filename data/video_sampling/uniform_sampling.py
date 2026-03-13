"""
Uniform frame sampling: timestamps at segment (start_sec, end_sec).
- num_frames: 고정 개수만큼 균등 간격
- fps: 설정 시 초당 프레임 수 기준으로 샘플링 (interval = 1/fps)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

Segment = Tuple[float, float]


class UniformFrameSampling:
    """균등 간격 또는 FPS 기반 프레임 샘플링."""

    def __init__(self, num_frames: int, fps: Optional[float] = None):
        if num_frames <= 0:
            raise ValueError(f"num_frames must be > 0, got {num_frames}")
        if fps is not None and fps <= 0:
            raise ValueError(f"fps must be > 0 when set, got {fps}")
        self.num_frames = num_frames
        self.fps = fps

    def sample(self, segment: Segment, clip_id: Optional[str] = None) -> List[float]:
        return uniform_timestamps_for_segment(
            segment, self.num_frames, fps=self.fps
        )


def uniform_timestamps_for_segment(
    segment: Segment,
    num_frames: int,
    fps: Optional[float] = None,
) -> List[float]:
    """
    Segment 내 타임스탬프 반환.

    Args:
        segment: (start_sec, end_sec)
        num_frames: 목표 프레임 수 (fps 미설정 시 균등 간격, fps 설정 시 상한)
        fps: 초당 프레임 수. 설정 시 interval=1/fps로 샘플링, num_frames로 cap
    """
    start, end = float(segment[0]), float(segment[1])
    duration = end - start
    if duration <= 0:
        raise ValueError(f"Invalid segment duration: start={start}, end={end}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be > 0, got {num_frames}")

    if fps is not None and fps > 0:
        # FPS 기반: start, start+1/fps, start+2/fps, ...
        n_by_fps = max(1, int(duration * fps))
        timestamps = [
            start + i / fps
            for i in range(n_by_fps)
            if start + i / fps < end - 1e-6
        ]
        if not timestamps:
            timestamps = [start + duration / 2]
        # num_frames로 cap (초과 시 균등 subsample)
        if len(timestamps) > num_frames:
            indices = [
                int(i * (len(timestamps) - 1) / (num_frames - 1))
                if num_frames > 1 else 0
                for i in range(num_frames)
            ]
            timestamps = [timestamps[i] for i in indices]
        return timestamps

    # 기존: num_frames개 균등 간격 (각 bin 중앙)
    interval = duration / num_frames
    return [start + (i + 0.5) * interval for i in range(num_frames)]
