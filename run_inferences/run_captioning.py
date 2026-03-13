"""
Captioning inference entrypoint (inference only).

`CUDA_VISIBLE_DEVICES=6 python -m run_inferences.run_captioning` 으로 실행할 수 있습니다.
"""

import json
import os
from pathlib import Path

from configs.config_resolver import ConfigResolver

from pipelines.run_model import run_model
from models.model_factory import build_vlm
from data.loader.loader_factory import get_dataloader
from tasks.utils.create_experiment_file import create_experiment_dir_and_metadata
from tasks.utils.stage_latency import StageLatencyProfiler


def normalize_image_id(image_id: str) -> str:
    """
    Dataset-independent image_id normalization.
    """
    return os.path.splitext(image_id)[0]


def run_captioning(experiment_path: str = "configs/experiment.yaml") -> Path:
    """
    Run captioning inference.
    """
    cfg = ConfigResolver(experiment_path)

    vlm = build_vlm(cfg.model_cfg, cfg.runtime_cfg)

    loader = get_dataloader(
        dataset_cfg=cfg.resolved_dataset
    )

    system_prompt = cfg.prompt_cfg["system_prompt"]
    user_prompt = cfg.prompt_cfg["user_prompt"]

    profiler = StageLatencyProfiler()

    exp_dir = create_experiment_dir_and_metadata(
        runtime_cfg=cfg.runtime_cfg,
        dataset_cfg=cfg.resolved_dataset,
        model_cfg=cfg.model_cfg,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    all_captions = {}
    all_references = {}
    all_image_paths = {}

    for batch_idx, batch in enumerate(loader):
        print(f"[Batch {batch_idx}]")

        caption_prefix = ""
        baseline_cfg = cfg.prompt_cfg.get("baseline")
        if isinstance(baseline_cfg, dict):
            caption_prefix = baseline_cfg.get("prefix", "")

        batch_captions = run_model(
            vlm=vlm,
            images=batch["images"],
            image_ids=batch["image_ids"],
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            caption_prefix=caption_prefix,
            generation_cfg=cfg.model_cfg["generation"],
        )

        for idx, raw_iid in enumerate(batch["image_ids"]):
            norm_iid = normalize_image_id(raw_iid)

            all_captions[norm_iid] = batch_captions[raw_iid]

            all_image_paths[norm_iid] = batch["image_paths"][idx]

            if raw_iid in batch["references"]:
                all_references[norm_iid] = batch["references"][raw_iid]
            else:
                raise KeyError(f"Reference not found for image_id: {raw_iid}")

    with open(exp_dir / "captions.json", "w", encoding="utf-8") as f:
        json.dump(all_captions, f, indent=2, ensure_ascii=False)

    with open(exp_dir / "reference_captions.json", "w", encoding="utf-8") as f:
        json.dump(all_references, f, indent=2, ensure_ascii=False)

    with open(exp_dir / "image_paths.json", "w", encoding="utf-8") as f:
        json.dump(all_image_paths, f, indent=2)

    with open(exp_dir / "latency.json", "w", encoding="utf-8") as f:
        json.dump(profiler.to_dict(), f, indent=2)

    print(f"✔ Captioning inference done. Saved to {exp_dir}")
    return exp_dir


def main():
    run_captioning("configs/experiment.yaml")


if __name__ == "__main__":
    main()

