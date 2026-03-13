"""
Manual frame sampling: clip_id별로 미리 정의된 타임스탬프 사용.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

Segment = Tuple[float, float]


class ManualFrameSampling:
    """clip_id별 수동 지정 타임스탬프로 프레임 샘플링."""

    def __init__(self, manual_frame_map: Dict[str, List[float]]):
        if not isinstance(manual_frame_map, dict):
            raise ValueError("manual_frame_map must be Dict[str, List[float]]")
        self.manual_frame_map = {k: list(v) for k, v in manual_frame_map.items()}

    def sample(self, segment: Segment, clip_id: Optional[str] = None) -> List[float]:
        if clip_id is None:
            raise ValueError("ManualFrameSampling requires clip_id")
        key = clip_id
        if key not in self.manual_frame_map and "_" in clip_id:
            prefix, suffix = clip_id.rsplit("_", 1)
            if suffix.isdigit() and prefix in self.manual_frame_map:
                key = prefix
        if key not in self.manual_frame_map:
            raise KeyError(f"clip_id '{clip_id}' not found in manual_frame_map.")
        start, end = float(segment[0]), float(segment[1])
        if end <= start:
            raise ValueError(f"Invalid segment duration: start={start}, end={end}")
        seconds = self.manual_frame_map[key]
        return [max(start, min(float(t), end)) for t in seconds]
