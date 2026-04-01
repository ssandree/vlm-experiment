"""
Mixed video+image inference entrypoint.

- video.input_mode: sampling — 비디오에서 프레임 샘플링·aggregation 후 이미지들과 함께 multi-image 추론
- video.input_mode: full — 네이티브 비디오 통째로 + 정적 이미지(들)를 한 번에 모델에 전달 (Qwen3-VL)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from PIL import Image

from configs.config_resolver import ConfigResolver
from models.model_factory import build_vlm
from pipelines.run_model import normalize_assistant_output, run_model_multi_image
from tasks.utils.create_experiment_file import create_experiment_dir_and_metadata
from tasks.utils.json_utils import write_json_bundle
from tasks.utils.stage_latency import StageLatencyProfiler
from data.loader.experiment_loaders import _get_video_duration
from data.utils.frame_decoding import DecordFrameDecoder
from data.video_sampling.build_sampling_strategy import build_sampling_strategy
from data.frame_aggregation.aggregation_strategy import build_aggregation_strategy
from data.input_strategies.build_input_strategy import build_input_strategy
from pipelines.video_sampling_pipeline import sample_decode_aggregate


def _as_image_list(x: Image.Image | List[Image.Image]) -> List[Image.Image]:
    return x if isinstance(x, list) else [x]


def run_video_image_inference(experiment_path: str = "configs/experiment.yaml") -> Path:
    cfg = ConfigResolver(experiment_path)
    if cfg.resolved_dataset.get("mode") != "video_image_multi":
        raise ValueError(
            "run_video_image_inference requires dataset mode: video_image_multi. "
            f"Current: {cfg.resolved_dataset.get('mode')}"
        )

    paths = cfg.resolved_dataset.get("paths", {})
    video_list = [Path(p) for p in paths.get("video_list", [])]
    image_list = [Path(p) for p in paths.get("image_list", [])]
    if not video_list:
        raise ValueError("video_image_multi requires paths.video_list")
    if not image_list:
        raise ValueError("video_image_multi requires paths.image_list")

    for p in video_list:
        if not p.exists():
            raise FileNotFoundError(f"Video not found: {p}")
    for p in image_list:
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")

    vlm = build_vlm(cfg.model_cfg, cfg.runtime_cfg)
    prompt_cfg = cfg.prompt_cfg
    system_prompt = prompt_cfg.get("system_prompt", "")
    user_prompt = prompt_cfg.get("user_prompt", "")
    caption_prefix = ""
    if isinstance(prompt_cfg.get("baseline"), dict):
        caption_prefix = prompt_cfg["baseline"].get("prefix", "") or ""
    gen_cfg = cfg.model_cfg.get("generation") or {}

    video_cfg = cfg.resolved_video_cfg.get("raw_video_cfg") or {}
    input_mode = (video_cfg.get("input_mode") or "sampling").strip().lower()
    use_native_full_video = input_mode == "full"

    sampler = None
    aggregator = None
    decoder = None
    if not use_native_full_video:
        sampler = build_sampling_strategy(
            cfg.resolved_video_cfg["sampling_cfg_fallback_uniform"]
        )
        aggregator = build_aggregation_strategy(cfg.resolved_video_cfg.get("aggregation_cfg") or {})
        decoder = DecordFrameDecoder()

    image_cfg = cfg.exp_cfg.get("image") or {}
    strategy = build_input_strategy({"type": image_cfg.get("input_strategy", "identity")})
    still_images = [Image.open(p).convert("RGB") for p in image_list]
    processed_still = strategy.process(still_images)
    processed_still = processed_still if isinstance(processed_still, list) else [processed_still]

    exp_dir = create_experiment_dir_and_metadata(
        runtime_cfg=cfg.runtime_cfg,
        dataset_cfg=cfg.resolved_dataset,
        model_cfg=cfg.model_cfg,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        extra_meta={
            "task": cfg.task_name,
            "mode": "video_image_multi",
            "video": video_cfg,
            "image": image_cfg,
        },
    )
    frames_dir = exp_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    profiler = StageLatencyProfiler()
    all_predictions: Dict[str, str] = {}
    all_video_paths: Dict[str, str] = {}
    all_group_paths: Dict[str, Dict[str, List[str]]] = {}
    all_frame_paths: Dict[str, List[str]] = {}

    still_path_strs = [str(p) for p in image_list]

    for video_path in video_list:
        sample_id = video_path.stem

        if use_native_full_video:
            fps_full = float(video_cfg.get("fps", 1))

            def _run_full():
                return vlm.generate_video_with_images(
                    video_path=str(video_path.resolve()),
                    images=list(processed_still),
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    caption_prefix=caption_prefix,
                    gen_cfg=gen_cfg,
                    fps=fps_full,
                )

            raw = profiler.measure("inference", _run_full)
            outputs = {sample_id: normalize_assistant_output(raw or "")}

            frame_paths = []
            for idx, img in enumerate(processed_still):
                p = frames_dir / f"{sample_id}_ref{idx:02d}.jpg"
                img.save(p, format="JPEG")
                frame_paths.append(str(p.resolve()))
        else:
            segment = (0.0, float(_get_video_duration(video_path)))

            _, _, aggregated = sample_decode_aggregate(
                video_path=video_path,
                segment=segment,
                clip_id=sample_id,
                sampler=sampler,
                decoder=decoder,
                aggregator=aggregator,
                profiler=profiler,
                decode_timestamps_relative_to_segment_start=False,
            )
            video_images = _as_image_list(aggregated)

            mixed_images = video_images + processed_still
            outputs = run_model_multi_image(
                vlm=vlm,
                image_groups=[mixed_images],
                group_ids=[sample_id],
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                caption_prefix=caption_prefix,
                generation_cfg=gen_cfg,
                profiler=profiler,
            )

            frame_paths = []
            for idx, img in enumerate(mixed_images):
                p = frames_dir / f"{sample_id}_f{idx:02d}.jpg"
                img.save(p, format="JPEG")
                frame_paths.append(str(p.resolve()))

        all_predictions[sample_id] = outputs.get(sample_id, "")
        all_video_paths[sample_id] = str(video_path)
        all_group_paths[sample_id] = {
            "video_path": [str(video_path)],
            "image_paths": still_path_strs,
        }
        all_frame_paths[sample_id] = frame_paths

    write_json_bundle(
        exp_dir,
        {
            "predictions.json": (all_predictions, False),
            "video_paths.json": all_video_paths,
            "group_paths.json": all_group_paths,
            "frame_paths.json": all_frame_paths,
            "latency.json": profiler.to_dict(),
        },
    )

    print(f"✔ Video+Image mixed inference done. Saved to {exp_dir}")
    return exp_dir


def main():
    run_video_image_inference("configs/experiment.yaml")


if __name__ == "__main__":
    main()

