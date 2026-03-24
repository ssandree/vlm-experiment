"""
Captioning inference entrypoint (inference only).

`CUDA_VISIBLE_DEVICES=6 python -m run_inferences.run_captioning` 으로 실행할 수 있습니다.
"""

from pathlib import Path

from configs.config_resolver import ConfigResolver
from utils.image_utils import normalize_image_id

from pipelines.run_model import run_model
from models.model_factory import build_vlm
from data.loader.loader_factory import get_dataloader
from tasks.utils.create_experiment_file import create_experiment_dir_and_metadata
from tasks.utils.json_utils import write_json_bundle
from tasks.utils.stage_latency import StageLatencyProfiler


def run_captioning(experiment_path: str = "configs/experiment.yaml") -> Path:
    """
    Run captioning inference.
    """
    cfg = ConfigResolver(experiment_path)

    # `run_captioning`은 "이미지 배치(images=...)"를 필요로 합니다.
    # 그런데 experiment.yaml에서 dataset 모드가 `video_only`이면 비디오 로더가
    # images 키를 제공하지 않으므로 KeyError가 발생합니다.
    # 이 경우에는 비디오 전용 엔트리포인트로 위임합니다.
    dataset_mode = cfg.resolved_dataset.get("mode")
    if dataset_mode == "video_only":
        from run_inferences.run_video_captioning import run_video_captioning

        return run_video_captioning(experiment_path)

    if dataset_mode != "image_only":
        raise ValueError(
            "run_captioning requires dataset mode: image_only. "
            f"Current: {dataset_mode}. "
            "For video, run_video_captioning; for image_multi, run_multi_image_inference."
        )

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
            profiler=profiler,
        )

        for idx, raw_iid in enumerate(batch["image_ids"]):
            norm_iid = normalize_image_id(raw_iid)

            all_captions[norm_iid] = batch_captions[raw_iid]

            all_image_paths[norm_iid] = batch["image_paths"][idx]

            if raw_iid in batch["references"]:
                all_references[norm_iid] = batch["references"][raw_iid]
            else:
                raise KeyError(f"Reference not found for image_id: {raw_iid}")

    write_json_bundle(
        exp_dir,
        {
            "captions.json": (all_captions, False),
            "reference_captions.json": (all_references, False),
            "image_paths.json": all_image_paths,
            "latency.json": profiler.to_dict(),
        },
    )

    print(f"✔ Captioning inference done. Saved to {exp_dir}")
    return exp_dir


def main():
    run_captioning("configs/experiment.yaml")


if __name__ == "__main__":
    main()

