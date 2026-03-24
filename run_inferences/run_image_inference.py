"""
Image-only inference (annotation 불필요). prompt와 task 설정에 따라 수행.

`CUDA_VISIBLE_DEVICES=6 python -m run_inferences.run_image_inference` 로 실행.
experiment.yaml에서 dataset: image_only, task, prompt 설정.
image.input_strategy: identity | top_right_crop → 처리된 이미지를 exp_~/frames/ 에 저장.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from configs.config_resolver import ConfigResolver
from models.model_factory import build_vlm
from data.loader.loader_factory import get_dataloader
from data.input_strategies.build_input_strategy import build_input_strategy
from tasks.utils.create_experiment_file import create_experiment_dir_and_metadata
from tasks.utils.json_utils import write_json_bundle
from tasks.utils.stage_latency import StageLatencyProfiler
from pipelines.run_model import run_model


def run_image_inference(experiment_path: str = "configs/experiment.yaml") -> Path:
    """
    image_only 데이터셋으로 task 수행. annotation 없이 이미지 + prompt만 사용.
    image.input_strategy 적용 후 처리된 이미지를 exp_~/frames/ 에 저장.
    """
    cfg = ConfigResolver(experiment_path)

    if cfg.resolved_dataset.get("mode") != "image_only":
        raise ValueError(
            "run_image_inference requires dataset mode: image_only. "
            f"Current: {cfg.resolved_dataset.get('mode')}"
        )

    vlm = build_vlm(cfg.model_cfg, cfg.runtime_cfg)
    loader = get_dataloader(dataset_cfg=cfg.resolved_dataset)

    prompt_cfg = cfg.prompt_cfg
    system_prompt = prompt_cfg.get("system_prompt", "")
    user_prompt = prompt_cfg.get("user_prompt", "")
    caption_prefix = ""
    if isinstance(prompt_cfg.get("baseline"), dict):
        caption_prefix = prompt_cfg["baseline"].get("prefix", "") or ""
    task_name = cfg.task_name
    gen_cfg = cfg.model_cfg.get("generation") or {}

    image_cfg = cfg.exp_cfg.get("image") or {}
    strategy_cfg = {"type": image_cfg.get("input_strategy", "identity")}
    strategy = build_input_strategy(strategy_cfg)

    exp_dir = create_experiment_dir_and_metadata(
        runtime_cfg=cfg.runtime_cfg,
        dataset_cfg=cfg.resolved_dataset,
        model_cfg=cfg.model_cfg,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        extra_meta={"task": task_name, "image": image_cfg},
    )

    frames_dir = exp_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    profiler = StageLatencyProfiler()
    all_predictions: Dict[str, Any] = {}
    all_references: Dict[str, Any] = {}
    all_image_paths: Dict[str, str] = {}

    for batch_idx, batch in enumerate(loader):
        print(f"[Batch {batch_idx}]")

        images: List[Image.Image] = batch["images"]
        image_ids: List[str] = batch["image_ids"]
        image_paths: List[str] = batch["image_paths"]
        references: Dict[str, List] = batch["references"]

        processed = strategy.process(images)
        processed_list = processed if isinstance(processed, list) else [processed]

        for img, iid in zip(processed_list, image_ids):
            frame_path = frames_dir / f"{iid}.jpg"
            img.save(frame_path, format="JPEG")

        outputs = run_model(
            vlm=vlm,
            images=processed_list,
            image_ids=image_ids,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            caption_prefix=caption_prefix,
            generation_cfg=gen_cfg,
            profiler=profiler,
        )
        for iid, ipath in zip(image_ids, image_paths):
            all_predictions[iid] = outputs.get(iid, "")
            all_references[iid] = references.get(iid, [])
            all_image_paths[iid] = ipath

    write_json_bundle(
        exp_dir,
        {
            "predictions.json": (all_predictions, False),
            "references.json": (all_references, False),
            "image_paths.json": all_image_paths,
            "latency.json": profiler.to_dict(),
        },
    )

    print(f"✔ Image inference done. Saved to {exp_dir}")
    return exp_dir


def main():
    run_image_inference("configs/experiment.yaml")


if __name__ == "__main__":
    main()
