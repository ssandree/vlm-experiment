from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple, Union

from PIL import Image

from data.input_strategies.base import FrameAggregationStrategy
from data.utils.frame_decoding import FrameDecoder
from data.video_sampling.sampling_strategy import FrameSamplingStrategy
from run_inferences.video_common import save_aggregated_frames
from tasks.utils.stage_latency import StageLatencyProfiler

Timestamp = float
Segment = Tuple[float, float]


@dataclass(frozen=True)
class VideoSamplingArtifacts:
    """
    sampling -> decoding -> aggregation -> disk save 결과를 한 번에 담는다.
    """

    clip_id: str
    timestamps: List[Timestamp]  # sampler가 반환한 (기본) 타임스탬프
    decode_timestamps: List[Timestamp]  # decoder에 실제로 넣은 타임스탬프
    aggregated: Image.Image | list[Image.Image]
    frame_paths: List[str]  # 저장된 경로 (aggregated 기준)
    video_path: str
    segment: Segment


def _maybe_measure(
    profiler: StageLatencyProfiler | None, stage_name: str, fn
) -> Any:
    if profiler is None:
        return fn()
    return profiler.measure(stage_name, fn)


def sample_decode_aggregate_and_save(
    *,
    video_path: Path,
    segment: Segment,
    clip_id: str,
    sampler: FrameSamplingStrategy,
    decoder: FrameDecoder,
    aggregator: FrameAggregationStrategy,
    frames_dir: Path,
    profiler: StageLatencyProfiler | None = None,
    # captioning 쪽 코드가 과거에 segment 시작 시점을 빼서 decode에 넣던 흐름이 있어
    # 호환을 위해 옵션으로 분리한다.
    decode_timestamps_relative_to_segment_start: bool = False,
) -> VideoSamplingArtifacts:
    """
    video(또는 segment)로부터 샘플링 타임스탬프를 뽑고,
    해당 프레임을 디코드/aggregation한 뒤 JPEG로 저장한다.
    """

    def _sample():
        return sampler.sample(segment, clip_id=clip_id)

    timestamps = _maybe_measure(profiler, "sampling", _sample)
    timestamps = [float(t) for t in timestamps]

    if decode_timestamps_relative_to_segment_start:
        seg_start = float(segment[0])
        decode_timestamps = [max(0.0, float(t) - seg_start) for t in timestamps]
    else:
        decode_timestamps = timestamps

    def _decode():
        return decoder.decode(video_path, decode_timestamps)

    images = _maybe_measure(profiler, "decoding", _decode)

    def _aggregate():
        return aggregator.aggregate(images)

    aggregated = _maybe_measure(profiler, "aggregation", _aggregate)

    frame_paths = save_aggregated_frames(
        frames_dir=frames_dir,
        sample_id=clip_id,
        aggregated=aggregated,
    )

    return VideoSamplingArtifacts(
        clip_id=clip_id,
        timestamps=timestamps,
        decode_timestamps=decode_timestamps,
        aggregated=aggregated,
        frame_paths=frame_paths,
        video_path=str(video_path),
        segment=(float(segment[0]), float(segment[1])),
    )


def sample_decode_aggregate(
    *,
    video_path: Path,
    segment: Segment,
    clip_id: str,
    sampler: FrameSamplingStrategy,
    decoder: FrameDecoder,
    aggregator: FrameAggregationStrategy,
    profiler: StageLatencyProfiler | None = None,
    decode_timestamps_relative_to_segment_start: bool = False,
) -> tuple[
    List[Timestamp],
    List[Timestamp],
    Image.Image | list[Image.Image],
]:
    """
    disk save 없이 sampling -> decoding -> aggregation 까지만 수행한다.
    """

    def _sample():
        return sampler.sample(segment, clip_id=clip_id)

    timestamps = _maybe_measure(profiler, "sampling", _sample)
    timestamps = [float(t) for t in timestamps]

    if decode_timestamps_relative_to_segment_start:
        seg_start = float(segment[0])
        decode_timestamps = [max(0.0, float(t) - seg_start) for t in timestamps]
    else:
        decode_timestamps = timestamps

    def _decode():
        return decoder.decode(video_path, decode_timestamps)

    images = _maybe_measure(profiler, "decoding", _decode)

    def _aggregate():
        return aggregator.aggregate(images)

    aggregated = _maybe_measure(profiler, "aggregation", _aggregate)

    return timestamps, decode_timestamps, aggregated

