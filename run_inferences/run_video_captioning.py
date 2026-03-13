"""
Video captioning inference entrypoint (segment-level).

UCFCrime subset 등 video-native 로더 + sampling/aggregation 전략 + CaptioningTask 를
조합해 클립 단위 비디오 캡셔닝을 수행합니다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List

from PIL import Image

from configs.config_resolver import ConfigResolver
from models.model_factory import build_vlm
from data.loader.loader_factory import get_dataloader
from data.video_sampling.build_sampling_strategy import build_sampling_strategy
from data.video_sampling.sampling1_perseg import get_video_timestamps_one_per_segment
from data.frame_aggregation.aggregation_strategy import build_aggregation_strategy
from data.loader.experiment_loaders import _get_video_duration
from data.utils.frame_decoding import DecordFrameDecoder
from tasks.captioning.caption_task import CaptioningTask
from tasks.utils.create_experiment_file import create_experiment_dir_and_metadata
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

    sampling_cfg = dict(video_cfg.get("sampling") or {})
    root = Path(cfg.env_cfg["paths"]["datasets_root"])
    if sampling_cfg.get("type") == "manual" and sampling_cfg.get("manual_frame_map_path"):
        p = Path(sampling_cfg["manual_frame_map_path"])
        if not p.is_absolute():
            sampling_cfg["manual_frame_map_path"] = str(root / sampling_cfg["manual_frame_map_path"])
    if sampling_cfg.get("type") == "segment_aware":
        ann_path = cfg.resolved_dataset.get("paths", {}).get("annotation")
        if ann_path is not None:
            sampling_cfg["annotation_path"] = str(Path(ann_path))
    sampler = build_sampling_strategy(sampling_cfg)

    all_captions: dict[str, str] = {}
    all_references: dict[str, list[str]] = {}
    video_paths_out: dict[str, str] = {}
    segments_out: dict[str, list[float]] = {}
    frame_paths_out: dict[str, list[str]] = {}
    grid_image_paths_out: dict[str, str] = {}

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

            if isinstance(aggregated, Image.Image):
                grid_path = frames_dir / f"{clip_id}_grid.jpg"
                aggregated.save(grid_path, format="JPEG")
                grid_image_paths_out[clip_id] = str(grid_path.resolve())
                frame_paths_out[clip_id] = [grid_image_paths_out[clip_id]]
            else:
                clip_frame_paths: list[str] = []
                for idx, img in enumerate(aggregated):
                    frame_name = f"{clip_id}_f{idx:02d}.jpg"
                    frame_path = frames_dir / frame_name
                    img.save(frame_path, format="JPEG")
                    clip_frame_paths.append(str(frame_path.resolve()))
                frame_paths_out[clip_id] = clip_frame_paths

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

    image_paths_out: dict[str, str] = {}
    for clip_id, paths in frame_paths_out.items():
        if clip_id in grid_image_paths_out:
            image_paths_out[clip_id] = grid_image_paths_out[clip_id]
        else:
            rep_idx = len(paths) // 2
            image_paths_out[clip_id] = paths[rep_idx]

    with open(exp_dir / "captions.json", "w", encoding="utf-8") as f:
        json.dump(all_captions, f, indent=2, ensure_ascii=False)

    with open(exp_dir / "reference_captions.json", "w", encoding="utf-8") as f:
        json.dump(all_references, f, indent=2, ensure_ascii=False)

    with open(exp_dir / "video_paths.json", "w", encoding="utf-8") as f:
        json.dump(video_paths_out, f, indent=2)

    with open(exp_dir / "image_paths.json", "w", encoding="utf-8") as f:
        json.dump(image_paths_out, f, indent=2)

    with open(exp_dir / "segments.json", "w", encoding="utf-8") as f:
        json.dump(segments_out, f, indent=2)

    with open(exp_dir / "frame_paths.json", "w", encoding="utf-8") as f:
        json.dump(frame_paths_out, f, indent=2)

    with open(exp_dir / "latency.json", "w", encoding="utf-8") as f:
        json.dump(profiler.to_dict(), f, indent=2)

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
    if ann_path:
        ann_path = Path(ann_path)
        if not ann_path.exists():
            ann_path = ann_path.with_suffix(".json")
        if ann_path.exists():
            with open(ann_path, "r", encoding="utf-8") as f:
                annotation = json.load(f)

    sampling_cfg = video_cfg.get("sampling") or {}
    fps = float(sampling_cfg.get("fps", 30.0))

    all_captions: dict[str, str] = {}
    all_references: dict[str, list[str]] = {}
    video_paths_out: dict[str, str] = {}
    segments_out: dict[str, list] = {}
    frame_paths_out: dict[str, list[str]] = {}
    grid_image_paths_out: dict[str, str] = {}
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

        num_frames = int(sampling_cfg.get("num_frames") or 0)
        if num_frames <= 0:
            num_frames = max(1, int(duration * fps))

        timestamps = profiler.measure(
            "sampling",
            lambda: get_video_timestamps_one_per_segment(
                duration, segments, num_frames=num_frames, fps=fps
            ),
        )

        images = profiler.measure(
            "decoding",
            lambda: decoder.decode(video_path, timestamps),
        )
        aggregated = profiler.measure(
            "aggregation",
            lambda: aggregator.aggregate(images),
        )

        if isinstance(aggregated, Image.Image):
            grid_path = frames_dir / f"{video_name}_grid.jpg"
            aggregated.save(grid_path, format="JPEG")
            grid_image_paths_out[video_name] = str(grid_path.resolve())
            frame_paths_out[video_name] = [grid_image_paths_out[video_name]]
        else:
            clip_frame_paths = []
            for idx, img in enumerate(aggregated):
                frame_name = f"{video_name}_f{idx:02d}.jpg"
                frame_path = frames_dir / frame_name
                img.save(frame_path, format="JPEG")
                clip_frame_paths.append(str(frame_path.resolve()))
            frame_paths_out[video_name] = clip_frame_paths

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

    image_paths_out = {}
    for vid, paths in frame_paths_out.items():
        if vid in grid_image_paths_out:
            image_paths_out[vid] = grid_image_paths_out[vid]
        else:
            image_paths_out[vid] = paths[len(paths) // 2]

    with open(exp_dir / "captions.json", "w", encoding="utf-8") as f:
        json.dump(all_captions, f, indent=2, ensure_ascii=False)
    with open(exp_dir / "reference_captions.json", "w", encoding="utf-8") as f:
        json.dump(all_references, f, indent=2, ensure_ascii=False)
    with open(exp_dir / "video_paths.json", "w", encoding="utf-8") as f:
        json.dump(video_paths_out, f, indent=2)
    with open(exp_dir / "image_paths.json", "w", encoding="utf-8") as f:
        json.dump(image_paths_out, f, indent=2)
    with open(exp_dir / "segments.json", "w", encoding="utf-8") as f:
        json.dump(segments_out, f, indent=2)
    with open(exp_dir / "frame_paths.json", "w", encoding="utf-8") as f:
        json.dump(frame_paths_out, f, indent=2)
    with open(exp_dir / "latency.json", "w", encoding="utf-8") as f:
        json.dump(profiler.to_dict(), f, indent=2)


def main():
    run_video_captioning("configs/experiment.yaml")


if __name__ == "__main__":
    main()

