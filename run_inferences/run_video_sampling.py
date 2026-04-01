"""
Video sampling-only pipeline:

video(또는 segment)로부터 샘플링 타임스탬프를 뽑고,
디코드/aggregation한 뒤 JPEG로 저장한다.

모델 inference는 수행하지 않는다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from configs.config_resolver import ConfigResolver
from data.frame_aggregation.aggregation_strategy import build_aggregation_strategy
from data.loader.experiment_loaders import _get_video_duration
from data.loader.loader_factory import get_dataloader
from data.input_strategies.base import FrameAggregationStrategy
from data.utils.frame_decoding import DecordFrameDecoder
from data.video_sampling.build_sampling_strategy import build_sampling_strategy
from data.video_sampling.uniform_sampling import UniformFrameSampling
from pipelines.video_sampling_pipeline import sample_decode_aggregate_and_save
from run_inferences.video_common import pick_representative_image_paths
from tasks.utils.create_experiment_file import create_experiment_dir_and_metadata
from tasks.utils.json_utils import write_json_bundle
from tasks.utils.stage_latency import StageLatencyProfiler

Segment = Tuple[float, float]


def _load_annotation_json(ann_path: Path) -> dict:
    with open(ann_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return raw if isinstance(raw, dict) else {}


def run_video_sampling(experiment_path: str = "configs/experiment.yaml") -> Path:
    cfg = ConfigResolver(experiment_path)

    video_cfg = cfg.resolved_video_cfg.get("raw_video_cfg") or {}
    output_level = cfg.resolved_video_cfg.get("output_level", "segment")

    # 과거 코드 호환: captioning(segment)일 때 segment start를 빼서 decode에 넣던 흐름
    decode_relative_to_segment_start = cfg.resolved_video_cfg.get(
        "decode_relative_to_segment_start", False
    )

    sampling_cfg_fallback = cfg.resolved_video_cfg[
        "sampling_cfg_fallback_uniform"
    ]
    sampler = build_sampling_strategy(sampling_cfg_fallback)
    uniform_fallback = UniformFrameSampling(
        num_frames=max(1, int(sampling_cfg_fallback.get("num_frames", 4)))
    )

    aggregation_strategy: FrameAggregationStrategy = build_aggregation_strategy(
        cfg.resolved_video_cfg.get("aggregation_cfg") or {}
    )
    decoder = DecordFrameDecoder()

    loader = get_dataloader(dataset_cfg=cfg.resolved_dataset)

    prompt_cfg = cfg.prompt_cfg
    system_prompt = prompt_cfg.get("system_prompt", "")
    user_prompt = prompt_cfg.get("user_prompt", "")

    profiler = StageLatencyProfiler()

    exp_dir = create_experiment_dir_and_metadata(
        runtime_cfg=cfg.runtime_cfg,
        dataset_cfg=cfg.resolved_dataset,
        model_cfg=cfg.model_cfg,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        extra_meta={
            "task": "video_sampling",
            "video": {**video_cfg},
            "output_level": output_level,
        },
    )

    frames_dir = exp_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_paths_out: Dict[str, List[str]] = {}
    timestamps_out: Dict[str, List[float]] = {}
    decode_timestamps_out: Dict[str, List[float]] = {}
    video_paths_out: Dict[str, str] = {}
    segments_out: Dict[str, Any] = {}

    if output_level == "video":
        paths = cfg.resolved_dataset.get("paths", {})
        video_list = [Path(p) for p in paths.get("video_list", [])]
        if not video_list:
            raise ValueError(
                "video_sampling(output_level=video) requires dataset paths.video_list"
            )

        annotation: dict = {}
        ann_path = cfg.resolved_dataset.get("paths", {}).get("annotation")
        annotation_path = Path(ann_path) if ann_path is not None else None
        if annotation_path is not None:
            if annotation_path.exists():
                annotation = _load_annotation_json(annotation_path)
            elif annotation_path.with_suffix(".json").exists():
                annotation = _load_annotation_json(annotation_path.with_suffix(".json"))

        for video_path in video_list:
            video_path = Path(video_path)
            video_name = video_path.stem

            data = annotation.get(video_name) if isinstance(annotation.get(video_name), dict) else {}
            if data:
                duration = float(data.get("duration", 0.0)) if data.get("duration") is not None else 0.0
                raw_ts = data.get("timestamps") or []
                segments: List[Segment] = []
                for s in raw_ts:
                    if isinstance(s, (list, tuple)) and len(s) >= 2:
                        segments.append((float(s[0]), float(s[1])))
            else:
                duration = _get_video_duration(video_path)
                segments = []

            if duration <= 0:
                duration = 1.0

            segment: Segment = (0.0, duration)
            try:
                artifacts = sample_decode_aggregate_and_save(
                    video_path=video_path,
                    segment=segment,
                    clip_id=video_name,
                    sampler=sampler,
                    decoder=decoder,
                    aggregator=aggregation_strategy,
                    frames_dir=frames_dir,
                    profiler=profiler,
                    decode_timestamps_relative_to_segment_start=decode_relative_to_segment_start,
                )
            except KeyError:
                artifacts = sample_decode_aggregate_and_save(
                    video_path=video_path,
                    segment=segment,
                    clip_id=video_name,
                    sampler=uniform_fallback,
                    decoder=decoder,
                    aggregator=aggregation_strategy,
                    frames_dir=frames_dir,
                    profiler=profiler,
                    decode_timestamps_relative_to_segment_start=decode_relative_to_segment_start,
                )

            frame_paths_out[video_name] = artifacts.frame_paths
            timestamps_out[video_name] = artifacts.timestamps
            decode_timestamps_out[video_name] = artifacts.decode_timestamps
            video_paths_out[video_name] = str(video_path)
            segments_out[video_name] = [
                [float(a), float(b)] for a, b in segments
            ]

    else:
        # output_level == "segment"
        for batch_idx, batch in enumerate(loader):
            print(f"[Batch {batch_idx}]")

            clip_ids: List[str] = batch["clip_ids"]
            video_paths = batch["video_paths"]
            segments = batch["segments"]

            for clip_id in clip_ids:
                video_path = Path(video_paths[clip_id])
                segment = segments[clip_id]

                artifacts = sample_decode_aggregate_and_save(
                    video_path=video_path,
                    segment=(float(segment[0]), float(segment[1])),
                    clip_id=clip_id,
                    sampler=sampler,
                    decoder=decoder,
                    aggregator=aggregation_strategy,
                    frames_dir=frames_dir,
                    profiler=profiler,
                    decode_timestamps_relative_to_segment_start=decode_relative_to_segment_start,
                )

                frame_paths_out[clip_id] = artifacts.frame_paths
                timestamps_out[clip_id] = artifacts.timestamps
                decode_timestamps_out[clip_id] = artifacts.decode_timestamps
                video_paths_out[clip_id] = str(video_path)
                segments_out[clip_id] = [float(segment[0]), float(segment[1])]

    image_paths_out = pick_representative_image_paths(frame_paths_out)

    write_json_bundle(
        exp_dir,
        {
            "frame_paths.json": (frame_paths_out, False),
            "image_paths.json": image_paths_out,
            "video_paths.json": video_paths_out,
            "segments.json": segments_out,
            "timestamps.json": timestamps_out,
            "decode_timestamps.json": decode_timestamps_out,
            "latency.json": profiler.to_dict(),
        },
    )

    print(f"✔ Video sampling done. Saved to {exp_dir}")
    return exp_dir


def main():
    run_video_sampling("configs/experiment.yaml")


if __name__ == "__main__":
    main()

