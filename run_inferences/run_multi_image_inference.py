"""
Multi-image inference: 여러 이미지를 한 번에 모델에 넣어 하나의 추론 결과 생성.

video sampling 없이, 이미지 그룹 단위로 multi-image input 수행.
`CUDA_VISIBLE_DEVICES=6 python -m run_inferences.run_multi_image_inference` 로 실행.

experiment.yaml에서 dataset: image_multi, task: captioning 설정.
configs/test_dataset/image_multi.yaml 에 image_groups 정의.
처리된 이미지는 exp_~/frames/ 에 저장.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

# torch/transformers 는 build_vlm() 호출 시에만 로드 (model_factory 지연 import).
print("[run_multi_image_inference] import: configs…", flush=True)
from configs.config_resolver import ConfigResolver

print("[run_multi_image_inference] import: model_factory (얇음·torch 아직 안 올림)", flush=True)
from models.model_factory import build_vlm

print("[run_multi_image_inference] import: loader / pipeline…", flush=True)
from data.loader.loader_factory import get_dataloader
from data.input_strategies.build_input_strategy import build_input_strategy
from pipelines.run_model import run_model_multi_image
from tasks.utils.create_experiment_file import create_experiment_dir_and_metadata
from tasks.utils.json_utils import write_json_bundle
from tasks.utils.stage_latency import StageLatencyProfiler


def run_multi_image_inference(
    experiment_path: str = "configs/experiment.yaml",
) -> Path:
    """
    image_multi 데이터셋으로 그룹 단위 multi-image 추론 수행.
    각 그룹(여러 이미지) → 하나의 추론 결과.
    """
    print(f"[run_multi_image_inference] ConfigResolver: {experiment_path}", flush=True)
    cfg = ConfigResolver(experiment_path)

    if cfg.resolved_dataset.get("mode") != "image_multi":
        raise ValueError(
            "run_multi_image_inference requires dataset mode: image_multi. "
            f"Current: {cfg.resolved_dataset.get('mode')}"
        )

    task_name = cfg.task_name
    if task_name not in ("captioning", "vqa"):
        raise ValueError(
            f"run_multi_image_inference supports captioning, vqa only. Got: {task_name}"
        )

    print(
        "[run_multi_image_inference] VLM 로드 시작 — torch/transformers 로딩 후 디스크→GPU. "
        "int8(bitsandbytes)는 주로 VRAM 절약·추론용이며, safetensors 전체 읽기는 fp16과 비슷하게 걸릴 수 있음",
        flush=True,
    )
    vlm = build_vlm(cfg.model_cfg, cfg.runtime_cfg)
    print("[run_multi_image_inference] 데이터로더 구성", flush=True)
    loader = get_dataloader(dataset_cfg=cfg.resolved_dataset)

    prompt_cfg = cfg.prompt_cfg
    system_prompt = prompt_cfg.get("system_prompt", "")
    user_prompt = prompt_cfg.get("user_prompt", "")
    caption_prefix = ""
    if isinstance(prompt_cfg.get("baseline"), dict):
        caption_prefix = prompt_cfg["baseline"].get("prefix", "") or ""
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
        extra_meta={"task": task_name, "mode": "multi_image", "image": image_cfg},
    )

    frames_dir = exp_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    profiler = StageLatencyProfiler()
    all_predictions: Dict[str, str] = {}
    all_references: Dict[str, List] = {}
    all_group_paths: Dict[str, List[str]] = {}

    for batch_idx, batch in enumerate(loader):
        print(f"[Batch {batch_idx}]")

        image_groups: List[List] = batch["image_groups"]
        group_ids: List[str] = batch["group_ids"]
        group_paths: List[List[str]] = batch["group_paths"]
        references: Dict[str, List] = batch["references"]

        processed_groups = [strategy.process(imgs) for imgs in image_groups]
        processed_groups = [
            p if isinstance(p, list) else [p] for p in processed_groups
        ]

        for group_imgs, gid in zip(processed_groups, group_ids):
            for idx, img in enumerate(group_imgs):
                frame_path = frames_dir / f"{gid}_f{idx:02d}.jpg"
                img.save(frame_path, format="JPEG")

        outputs = run_model_multi_image(
            vlm=vlm,
            image_groups=processed_groups,
            group_ids=group_ids,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            caption_prefix=caption_prefix,
            generation_cfg=gen_cfg,
            profiler=profiler,
        )

        for gid, gpaths in zip(group_ids, group_paths):
            all_predictions[gid] = outputs.get(gid, "")
            all_references[gid] = references.get(gid, [])
            all_group_paths[gid] = gpaths

    write_json_bundle(
        exp_dir,
        {
            "predictions.json": (all_predictions, False),
            "references.json": (all_references, False),
            "group_paths.json": all_group_paths,
            "latency.json": profiler.to_dict(),
        },
    )

    print(f"✔ Multi-image inference done. Saved to {exp_dir}")
    return exp_dir


def main():
    run_multi_image_inference("configs/experiment.yaml")


if __name__ == "__main__":
    main()
