"""
Video captioning inference entrypoint (segment-level).

UCFCrime subset 등 video-native 로더 + sampling/aggregation 전략 + CaptioningTask 를
조합해 클립 단위 비디오 캡셔닝을 수행합니다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

from configs.config_resolver import ConfigResolver
from models.model_factory import build_vlm
from data.loader.loader_factory import get_dataloader
from data.video_sampling.build_sampling_strategy import build_sampling_strategy
from data.frame_aggregation.aggregation_strategy import build_aggregation_strategy
from data.loader.experiment_loaders import _get_video_duration
from data.utils.frame_decoding import DecordFrameDecoder
from run_inferences.video_common import (
    normalize_sampling_cfg,
    pick_representative_image_paths,
    save_aggregated_frames,
)
from tasks.captioning.caption_task import CaptioningTask
from tasks.utils.create_experiment_file import create_experiment_dir_and_metadata
from tasks.utils.json_utils import write_json_bundle
from tasks.utils.stage_latency import StageLatencyProfiler


def run_video_captioning(experiment_path: str = "configs/experiment.yaml") -> Path:
    """
    Run video captioning inference (sampling 모드).
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
        extra_meta={"video": video_cfg} if video_cfg else None,
    )

    task = CaptioningTask()
    decoder = DecordFrameDecoder()
    aggregation_cfg = video_cfg.get("aggregation") or {}
    aggregator = build_aggregation_strategy(aggregation_cfg)

    output_level = video_cfg.get("output_level", "segment")

    if output_level == "video":
        _run_video_level_captioning(
            cfg=cfg,
            video_cfg=video_cfg,
            exp_dir=exp_dir,
            vlm=vlm,
            task=task,
            decoder=decoder,
            aggregator=aggregator,
            prompt_cfg=prompt_cfg,
            profiler=profiler,
        )
        print(f"✔ Video-level captioning done. Saved to {exp_dir}")
        return exp_dir

    raw_sampling_cfg = dict(video_cfg.get("sampling") or {})
    root = Path(cfg.env_cfg["paths"]["datasets_root"])
    ann_path = cfg.resolved_dataset.get("paths", {}).get("annotation")
    sampling_cfg = normalize_sampling_cfg(
        raw_sampling_cfg,
        datasets_root=root,
        annotation_path=Path(ann_path) if ann_path is not None else None,
    )
    sampler = build_sampling_strategy(sampling_cfg)

    all_captions: dict[str, str] = {}
    all_references: dict[str, list[str]] = {}
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

        for clip_id in clip_ids:
            segment = segments[clip_id]
            video_path = Path(video_paths[clip_id])
            sentence = sentences[clip_id]

            timestamps = profiler.measure(
                "sampling",
                lambda: sampler.sample(segment, clip_id=clip_id),
            )

            images = profiler.measure(
                "decoding",
                lambda: decoder.decode(
                    video_path,
                    [max(0.0, t - float(segment[0])) for t in timestamps],
                ),
            )

            aggregated = profiler.measure(
                "aggregation",
                lambda: aggregator.aggregate(images),
            )

            frame_paths_out[clip_id] = save_aggregated_frames(
                frames_dir=frames_dir,
                sample_id=clip_id,
                aggregated=aggregated,
            )

            sample = {
                "image": aggregated,
                "image_id": clip_id,
                "references": [sentence],
            }
            inputs = task.build_inputs(sample, prompt_cfg)

            caption = profiler.measure(
                "inference",
                lambda: task.run_inference(
                    model=vlm,
                    inputs=inputs,
                    generation_cfg=cfg.model_cfg["generation"],
                ),
            )

            all_captions[clip_id] = caption
            all_references[clip_id] = [sentence]
            video_paths_out[clip_id] = str(video_path)
            segments_out[clip_id] = [float(segment[0]), float(segment[1])]

    image_paths_out = pick_representative_image_paths(frame_paths_out)

    write_json_bundle(
        exp_dir,
        {
            "captions.json": (all_captions, False),
            "reference_captions.json": (all_references, False),
            "video_paths.json": video_paths_out,
            "image_paths.json": image_paths_out,
            "segments.json": segments_out,
            "frame_paths.json": frame_paths_out,
            "latency.json": profiler.to_dict(),
        },
    )

    print(f"✔ Video captioning inference done. Saved to {exp_dir}")
    return exp_dir


def _run_video_level_captioning(
    cfg: "ConfigResolver",
    video_cfg: dict,
    exp_dir: Path,
    vlm: Any,
    task: CaptioningTask,
    decoder: "DecordFrameDecoder",
    aggregator: Any,
    prompt_cfg: dict,
    profiler: "StageLatencyProfiler",
) -> None:
    """Video 단위 캡셔닝: segment는 샘플링용 부가정보만 사용, video 하나당 caption 하나."""
    paths = cfg.resolved_dataset.get("paths", {})
    ann_path = paths.get("annotation")
    video_list = paths.get("video_list", [])
    if not video_list:
        raise ValueError(
            "output_level=video requires dataset paths.video_list"
        )

    annotation: dict = {}
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
        dict(video_cfg.get("sampling") or {}),
        datasets_root=Path(cfg.env_cfg["paths"]["datasets_root"]),
        annotation_path=annotation_path,
        fallback_uniform_when_segment_aware_without_annotation=True,
    )
    sampler = build_sampling_strategy(sampling_cfg)
    # Uniform fallback for videos not in annotation (when using segment_aware)
    from data.video_sampling.uniform_sampling import UniformFrameSampling
    uniform_fallback = UniformFrameSampling(num_frames=max(1, int(sampling_cfg.get("num_frames", 4))))

    all_captions: dict[str, str] = {}
    all_references: dict[str, list[str]] = {}
    video_paths_out: dict[str, str] = {}
    segments_out: dict[str, list] = {}
    frame_paths_out: dict[str, list[str]] = {}
    frames_dir = exp_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for video_path in video_list:
        video_path = Path(video_path)
        video_name = video_path.stem
        data = annotation.get(video_name) if isinstance(annotation.get(video_name), dict) else {}
        if data:
            duration = float(data.get("duration", 0.0))
            raw_ts = data.get("timestamps") or []
            segments = []
            for s in raw_ts:
                if isinstance(s, (list, tuple)) and len(s) >= 2:
                    segments.append((float(s[0]), float(s[1])))
            ref_sentences = data.get("sentences") or []
            if not isinstance(ref_sentences, list):
                ref_sentences = [str(ref_sentences)]
        else:
            duration = _get_video_duration(video_path)
            segments = []
            ref_sentences = []
        if duration <= 0:
            duration = 1.0

        segment = (0.0, duration)
        try:
            timestamps = profiler.measure(
                "sampling",
                lambda: sampler.sample(segment, clip_id=video_name),
            )
        except KeyError:
            # Video not in annotation (segment_aware) -> use uniform fallback
            timestamps = profiler.measure(
                "sampling",
                lambda: uniform_fallback.sample(segment),
            )

        images = profiler.measure(
            "decoding",
            lambda: decoder.decode(video_path, timestamps),
        )
        aggregated = profiler.measure(
            "aggregation",
            lambda: aggregator.aggregate(images),
        )

        frame_paths_out[video_name] = save_aggregated_frames(
            frames_dir=frames_dir,
            sample_id=video_name,
            aggregated=aggregated,
        )

        sample = {
            "image": aggregated,
            "image_id": video_name,
            "references": ref_sentences,
        }
        inputs = task.build_inputs(sample, prompt_cfg)
        caption = profiler.measure(
            "inference",
            lambda: task.run_inference(
                model=vlm,
                inputs=inputs,
                generation_cfg=cfg.model_cfg["generation"],
            ),
        )
        all_captions[video_name] = caption
        all_references[video_name] = ref_sentences
        video_paths_out[video_name] = str(video_path)
        segments_out[video_name] = [[float(a), float(b)] for a, b in segments]

    image_paths_out = pick_representative_image_paths(frame_paths_out)
    write_json_bundle(
        exp_dir,
        {
            "captions.json": (all_captions, False),
            "reference_captions.json": (all_references, False),
            "video_paths.json": video_paths_out,
            "image_paths.json": image_paths_out,
            "segments.json": segments_out,
            "frame_paths.json": frame_paths_out,
            "latency.json": profiler.to_dict(),
        },
    )


def main():
    run_video_captioning("configs/experiment.yaml")


if __name__ == "__main__":
    main()

