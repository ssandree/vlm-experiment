"""
Video grounding inference entrypoint (segment-level, same mechanism as run_video_captioning).

이 리포는 Qwen3-VL-8B + UCFCrime subset(video) 전용 실험 환경입니다.
`CUDA_VISIBLE_DEVICES=6 python -m run_inferences.run_video_grounding` 으로 실행할 수 있습니다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Any

from PIL import Image

from configs.config_resolver import ConfigResolver
from models.model_factory import build_vlm
from data.loader.loader_factory import get_dataloader
from data.video_sampling.build_sampling_strategy import build_sampling_strategy
from data.frame_aggregation.aggregation_strategy import build_aggregation_strategy
from data.utils.frame_decoding import DecordFrameDecoder
from data.input_strategies.image_merge import GridMergeNoResizeAggregation
from run_inferences.video_common import (
    normalize_sampling_cfg,
    pick_representative_image_paths,
    save_aggregated_frames,
)
from tasks.grounding.grounding_task import GroundingTask
from tasks.utils.create_experiment_file import create_experiment_dir_and_metadata
from tasks.utils.json_utils import write_json_bundle
from tasks.utils.stage_latency import StageLatencyProfiler


def run_video_grounding(experiment_path: str = "configs/experiment.yaml") -> Path:
    """
    Run video grounding inference (sampling 모드).
    """
    cfg = ConfigResolver(experiment_path)
    video_cfg = cfg.exp_cfg.get("video") or cfg.resolved_dataset.get("video") or {}
    if video_cfg.get("input_mode") == "full":
        raise ValueError(
            "video.input_mode is 'full'. Use run_video_full for native video input."
        )

    vlm = build_vlm(cfg.model_cfg, cfg.runtime_cfg)
    loader = get_dataloader(dataset_cfg=cfg.resolved_dataset)

    prompt_cfg = cfg.prompt_cfg
    system_prompt = prompt_cfg["system_prompt"]
    user_prompt = prompt_cfg["user_prompt"]

    profiler = StageLatencyProfiler()

    exp_dir = create_experiment_dir_and_metadata(
        runtime_cfg=cfg.runtime_cfg,
        dataset_cfg=cfg.resolved_dataset,
        model_cfg=cfg.model_cfg,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        extra_meta=(
            {"task": "grounding", "video": video_cfg}
            if video_cfg
            else {"task": "grounding"}
        ),
    )

    task = GroundingTask()
    decoder = DecordFrameDecoder()

    sampling_cfg = dict(video_cfg.get("sampling") or {})
    num_frames = int(sampling_cfg.get("num_frames", 4))
    ann_path = cfg.resolved_dataset.get("paths", {}).get("annotation")
    annotation = None
    annotation_path = None
    if ann_path:
        ann_path = Path(ann_path)
        if ann_path.exists():
            annotation_path = ann_path
            with open(ann_path, "r", encoding="utf-8") as f:
                annotation = json.load(f)
        elif ann_path.with_suffix(".json").exists():
            annotation_path = ann_path.with_suffix(".json")
            with open(annotation_path, "r", encoding="utf-8") as f:
                annotation = json.load(f)

    sampling_cfg = normalize_sampling_cfg(
        sampling_cfg=sampling_cfg,
        datasets_root=Path(cfg.env_cfg["paths"]["datasets_root"]),
        annotation_path=annotation_path,
    )
    if annotation_path is None:
        sampling_cfg["type"] = "uniform"
        sampling_cfg["num_frames"] = num_frames
    sampler = build_sampling_strategy(sampling_cfg)

    aggregation_cfg = video_cfg.get("aggregation") or {}
    aggregator = build_aggregation_strategy(aggregation_cfg)

    all_predictions: dict[str, Any] = {}
    all_references: dict[str, dict[str, Any]] = {}
    video_paths_out: dict[str, str] = {}
    segments_out: dict[str, list[float]] = {}
    frame_paths_out: dict[str, list[str]] = {}

    frames_dir = exp_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, batch in enumerate(loader):
        print(f"[Batch {batch_idx}]")

        clip_ids: List[str] = batch["clip_ids"]
        video_paths = batch["video_paths"]
        segments = batch["segments"]
        sentences = batch["sentences"]

        video_to_clips: dict[str, list[str]] = {}
        video_to_path: dict[str, Path] = {}
        for cid in clip_ids:
            vpath = Path(video_paths[cid])
            vname = vpath.stem
            video_to_clips.setdefault(vname, []).append(cid)
            video_to_path[vname] = vpath

        video_aggregated: dict[str, Any] = {}
        video_saved_paths: dict[str, list[str]] = {}
        for video_name in video_to_clips:
            if annotation is not None and video_name not in annotation:
                continue
            if annotation is not None:
                data = annotation[video_name]
                if not isinstance(data, dict):
                    continue
                duration = float(data.get("duration", 0.0))
                if duration <= 0:
                    duration = 1.0
                segment = (0.0, duration)
            else:
                clip_id_for_video = video_to_clips[video_name][0]
                segment = segments[clip_id_for_video]
                duration = float(segment[1]) - float(segment[0])
                if duration <= 0:
                    duration = 1.0

            timestamps = profiler.measure(
                "sampling",
                lambda seg=segment, vn=video_name: sampler.sample(seg, clip_id=vn),
            )
            video_path = video_to_path[video_name]
            images = profiler.measure(
                "decoding",
                lambda: decoder.decode(video_path, timestamps),
            )
            aggregated = profiler.measure(
                "aggregation",
                lambda: aggregator.aggregate(images),
            )
            video_aggregated[video_name] = aggregated

            video_saved_paths[video_name] = save_aggregated_frames(
                frames_dir=frames_dir,
                sample_id=video_name,
                aggregated=aggregated,
            )

        for clip_id in clip_ids:
            segment = segments[clip_id]
            video_path = Path(video_paths[clip_id])
            video_name = video_path.stem
            phrase = sentences[clip_id]

            if video_name not in video_aggregated:
                continue
            aggregated = video_aggregated[video_name]
            paths = video_saved_paths[video_name]
            frame_paths_out[clip_id] = paths

            if isinstance(aggregated, Image.Image):
                sample_image = aggregated
            else:
                assert isinstance(aggregated, list) and aggregated, "aggregated must be Image or non-empty list"
                if len(aggregated) == 4:
                    sample_image = GridMergeNoResizeAggregation().aggregate(aggregated)
                else:
                    sample_image = aggregated[len(aggregated) // 2]

            sample = {
                "image": sample_image,
                "image_id": clip_id,
            }
            inputs = task.build_inputs(sample, prompt_cfg)
            result = profiler.measure(
                "inference",
                lambda: task.run_inference(
                    model=vlm,
                    inputs=inputs,
                    generation_cfg=cfg.model_cfg["generation"],
                ),
            )

            all_predictions[clip_id] = result.get("bbox")  # list of [x,y,w,h] per frame
            all_references[clip_id] = {"phrase": phrase}  # 참고용만, 평가는 전체 bbox 기준 아님
            video_paths_out[clip_id] = str(video_path)
            segments_out[clip_id] = [float(segment[0]), float(segment[1])]

    image_paths_out = pick_representative_image_paths(frame_paths_out)
    write_json_bundle(
        exp_dir,
        {
            "predictions.json": (all_predictions, False),
            "references.json": (all_references, False),
            "video_paths.json": video_paths_out,
            "image_paths.json": image_paths_out,
            "segments.json": segments_out,
            "frame_paths.json": frame_paths_out,
            "latency.json": profiler.to_dict(),
        },
    )

    print(f"✔ Video grounding inference done. Saved to {exp_dir}")
    return exp_dir


def main():
    run_video_grounding("configs/experiment.yaml")


if __name__ == "__main__":
    main()

