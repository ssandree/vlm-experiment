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
from data.video_sampling.sampling1_perseg import get_video_timestamps_one_per_segment
from data.video_sampling.uniform_sampling import uniform_timestamps_for_segment
from data.frame_aggregation.aggregation_strategy import build_aggregation_strategy
from data.utils.frame_decoding import DecordFrameDecoder
from data.input_strategies.image_merge import GridMergeNoResizeAggregation
from tasks.grounding.grounding_task import GroundingTask
from tasks.utils.create_experiment_file import create_experiment_dir_and_metadata
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
    fps = float(sampling_cfg.get("fps", 30.0))
    ann_path = cfg.resolved_dataset.get("paths", {}).get("annotation")
    annotation = None
    if ann_path:
        ann_path = Path(ann_path)
        if ann_path.exists():
            with open(ann_path, "r", encoding="utf-8") as f:
                annotation = json.load(f)
        elif ann_path.with_suffix(".json").exists():
            with open(ann_path.with_suffix(".json"), "r", encoding="utf-8") as f:
                annotation = json.load(f)

    aggregation_cfg = video_cfg.get("aggregation") or {}
    aggregator = build_aggregation_strategy(aggregation_cfg)

    all_predictions: dict[str, Any] = {}
    all_references: dict[str, dict[str, Any]] = {}
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
                raw_ts = data.get("timestamps") or []
                segs = []
                for s in raw_ts:
                    if isinstance(s, (list, tuple)) and len(s) >= 2:
                        segs.append((float(s[0]), float(s[1])))
                if duration <= 0:
                    duration = 1.0
                timestamps = profiler.measure(
                    "sampling",
                    lambda dur=duration, sg=segs: get_video_timestamps_one_per_segment(
                        dur, sg, num_frames=num_frames, fps=fps
                    ),
                )
            else:
                # video_only: segment from loader (0, duration)
                clip_id_for_video = video_to_clips[video_name][0]
                segment = segments[clip_id_for_video]
                timestamps = profiler.measure(
                    "sampling",
                    lambda seg=segment: uniform_timestamps_for_segment(seg, num_frames),
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

            grid_path = frames_dir / f"{video_name}_grid.jpg"
            if isinstance(aggregated, Image.Image):
                aggregated.save(grid_path, format="JPEG")
                _frame_paths = [str(grid_path.resolve())]
            else:
                _frame_paths = []
                for idx, img in enumerate(aggregated):
                    frame_name = f"{video_name}_f{idx:02d}.jpg"
                    frame_path = frames_dir / frame_name
                    img.save(frame_path, format="JPEG")
                    _frame_paths.append(str(frame_path.resolve()))
            video_saved_paths[video_name] = _frame_paths

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
            grid_image_paths_out[clip_id] = paths[0] if paths else ""

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

    image_paths_out: dict[str, str] = {}
    for clip_id, paths in frame_paths_out.items():
        if clip_id in grid_image_paths_out:
            image_paths_out[clip_id] = grid_image_paths_out[clip_id]
        else:
            rep_idx = len(paths) // 2
            image_paths_out[clip_id] = paths[rep_idx]

    with open(exp_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)

    with open(exp_dir / "references.json", "w", encoding="utf-8") as f:
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

    print(f"✔ Video grounding inference done. Saved to {exp_dir}")
    return exp_dir


def main():
    run_video_grounding("configs/experiment.yaml")


if __name__ == "__main__":
    main()

