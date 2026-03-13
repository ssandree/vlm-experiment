"""
Frame sampling strategies for segment-level video.
SegmentAwareSampling: 실험용 (anomaly segment 기반 샘플링).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

Segment = Tuple[float, float]


class FrameSamplingStrategy:
    def sample(self, segment: Segment, clip_id: Optional[str] = None) -> List[float]:
        raise NotImplementedError


class MiddleFrameSampling(FrameSamplingStrategy):
    def sample(self, segment: Segment, clip_id: Optional[str] = None) -> List[float]:
        start, end = float(segment[0]), float(segment[1])
        duration = end - start
        if duration <= 0:
            raise ValueError(f"Invalid segment duration: start={start}, end={end}")
        return [start + duration / 2.0]


class SegmentAwareSampling(FrameSamplingStrategy):
    """
    Segment-aware sampling: segment당 midpoint 1개, num_frames 맞춤.
    실험용 (UCFCrime 등 anomaly segment 기반).
    sampling1_perseg.get_video_timestamps_one_per_segment를 사용.
    """

    def __init__(
        self,
        annotation_map: Dict[str, Dict],
        num_frames: int = 4,
        fps: float = 30.0,
    ):
        if num_frames <= 0:
            raise ValueError(f"num_frames must be > 0, got {num_frames}")
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        self.annotation_map = dict(annotation_map)
        self.num_frames = num_frames
        self.fps = fps

    def _video_name_from_clip_id(self, clip_id: Optional[str]) -> str:
        if not clip_id:
            raise ValueError("SegmentAwareSampling requires clip_id")
        if "_" in clip_id:
            prefix, suffix = clip_id.rsplit("_", 1)
            if suffix.isdigit() and prefix in self.annotation_map:
                return prefix
        return clip_id

    def sample(self, segment: Segment, clip_id: Optional[str] = None) -> List[float]:
        from .sampling1_perseg import get_video_timestamps_one_per_segment

        video_name = self._video_name_from_clip_id(clip_id)
        if video_name not in self.annotation_map:
            raise KeyError(
                f"clip_id '{clip_id}' (video '{video_name}') not in segment_aware annotation."
            )
        data = self.annotation_map[video_name]
        duration = float(data.get("duration", 0.0))
        raw_ts = data.get("timestamps") or []
        timestamps: List[Tuple[float, float]] = []
        for seg in raw_ts:
            if isinstance(seg, (list, tuple)) and len(seg) >= 2:
                timestamps.append((float(seg[0]), float(seg[1])))

        return get_video_timestamps_one_per_segment(
            duration=duration,
            segments=timestamps,
            num_frames=self.num_frames,
            fps=self.fps,
        )
