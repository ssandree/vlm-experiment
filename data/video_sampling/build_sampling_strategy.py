"""
Build FrameSamplingStrategy from config.
Supports: uniform, middle, manual, segment_aware (실험용).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .sampling_strategy import (
    FrameSamplingStrategy,
    MiddleFrameSampling,
    SegmentAwareSampling,
)
from .uniform_sampling import UniformFrameSampling
from .manual_sampling import ManualFrameSampling


def build_sampling_strategy(config: Dict[str, Any] | None) -> FrameSamplingStrategy:
    config = config or {}
    strategy_type = (config.get("type") or "uniform").strip().lower()
    num_frames = config.get("num_frames", 4)

    if strategy_type == "middle":
        return MiddleFrameSampling()
    if strategy_type == "uniform":
        fps = config.get("fps")
        fps = float(fps) if fps is not None else None
        return UniformFrameSampling(
            num_frames=int(num_frames),
            fps=fps,
        )
    if strategy_type == "manual":
        path = config.get("manual_frame_map_path")
        if not path:
            raise ValueError(
                "manual sampling requires manual_frame_map_path in config"
            )
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"manual_frame_map_path not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError("manual_frame_map JSON must be a dict")
        manual_frame_map: Dict[str, List[float]] = {
            k: [float(x) for x in v] for k, v in raw.items()
            if isinstance(v, (list, tuple))
        }
        return ManualFrameSampling(manual_frame_map=manual_frame_map)
    if strategy_type == "segment_aware":
        path = config.get("annotation_path")
        if not path:
            raise ValueError(
                "segment_aware sampling requires annotation_path in config"
            )
        path = Path(path)
        if not path.exists():
            path_json = path.with_suffix(".json") if path.suffix != ".json" else path
            if path_json.exists():
                path = path_json
            else:
                raise FileNotFoundError(
                    f"segment_aware annotation_path not found: {path}"
                )
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError(
                "segment_aware annotation must be a dict (video_name -> {duration, timestamps})"
            )
        annotation_map: Dict[str, Any] = {}
        for k, v in raw.items():
            if isinstance(v, dict) and ("timestamps" in v or "duration" in v):
                annotation_map[str(k)] = v
            else:
                annotation_map[str(k)] = v if isinstance(v, dict) else {}
        return SegmentAwareSampling(
            annotation_map=annotation_map,
            num_frames=int(config.get("num_frames", 4)),
            fps=float(config.get("fps", 30)),
        )
    raise ValueError(
        f"Unknown sampling type: {config.get('type')}. "
        "Expected 'uniform', 'middle', 'manual', or 'segment_aware'."
    )
