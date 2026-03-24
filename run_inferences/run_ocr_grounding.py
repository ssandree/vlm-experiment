"""
OCR 기반 grounding inference.

1) image_only 데이터셋의 이미지를 불러오고
2) grounding task로 bbox를 예측한 뒤
3) 기존 captions.json(ocr 결과)와 묶어서 ocr_grounding.json으로 저장.

실행 예시:
CUDA_VISIBLE_DEVICES=0 python -m run_inferences.run_ocr_grounding \
  configs/experiment.yaml \
  outputs/exp_20260317_005110/captions.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from configs.config_resolver import ConfigResolver
from models.model_factory import build_vlm
from data.loader.loader_factory import get_dataloader
from tasks.grounding.grounding_task import GroundingTask
from tasks.utils.create_experiment_file import create_experiment_dir_and_metadata
from tasks.utils.json_utils import write_json_bundle
from tasks.utils.stage_latency import StageLatencyProfiler


def run_ocr_grounding(
    experiment_path: str = "configs/experiment.yaml",
    captions_path: str = "",
) -> Path:
    """
    image_only 데이터셋에 대해:
    - captions_path에서 OCR captions를 읽고
    - 같은 이미지들에 대해 grounding(bbox) 추론
    - caption + bbox를 ocr_grounding.json으로 저장.
    """
    cfg = ConfigResolver(experiment_path)

    if cfg.resolved_dataset.get("mode") != "image_only":
        raise ValueError(
            "run_ocr_grounding requires dataset mode: image_only. "
            f"Current: {cfg.resolved_dataset.get('mode')}"
        )

    if not captions_path:
        raise ValueError("captions_path must be provided (path to captions.json)")

    captions_path = Path(captions_path)
    if not captions_path.exists():
        raise FileNotFoundError(f"captions.json not found: {captions_path}")
    with open(captions_path, "r", encoding="utf-8") as f:
        ocr_captions: Dict[str, str] = json.load(f)

    vlm = build_vlm(cfg.model_cfg, cfg.runtime_cfg)
    loader = get_dataloader(dataset_cfg=cfg.resolved_dataset)

    prompt_cfg = cfg.prompt_cfg
    gen_cfg = cfg.model_cfg.get("generation") or {}

    profiler = StageLatencyProfiler()
    task = GroundingTask()

    exp_dir = create_experiment_dir_and_metadata(
        runtime_cfg=cfg.runtime_cfg,
        dataset_cfg=cfg.resolved_dataset,
        model_cfg=cfg.model_cfg,
        system_prompt=prompt_cfg.get("system_prompt", ""),
        user_prompt=prompt_cfg.get("user_prompt", ""),
        extra_meta={"task": "ocr_grounding"},
    )

    frames_dir = exp_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    out_results: Dict[str, Any] = {}
    image_paths_out: Dict[str, str] = {}

    for batch_idx, batch in enumerate(loader):
        print(f"[Batch {batch_idx}]")

        images: List[Image.Image] = batch["images"]
        image_ids: List[str] = batch["image_ids"]
        image_paths: List[str] = batch["image_paths"]

        for img, iid, ipath in zip(images, image_ids, image_paths):
            # 이미지 저장 (optional, grounding 시각화용)
            frame_path = frames_dir / f"{iid}.jpg"
            img.save(frame_path, format="JPEG")

            sample = {"image": img}
            inputs = task.build_inputs(sample, prompt_cfg)

            def _run():
                return task.run_inference(
                    model=vlm,
                    inputs=inputs,
                    generation_cfg=gen_cfg,
                )

            result = profiler.measure("inference", _run)
            bboxes = result.get("bbox", [])

            # OCR caption 매칭 (image_id 키 그대로 사용)
            caption = (
                ocr_captions.get(iid)
                or ocr_captions.get(str(iid))
                or ocr_captions.get(Path(iid).stem, "")
            )

            out_results[iid] = {
                "caption": caption,
                "bboxes": bboxes,
                "raw_output": result.get("raw_output", ""),
            }
            image_paths_out[iid] = ipath

    write_json_bundle(
        exp_dir,
        {
            "ocr_grounding.json": (out_results, False),
            "image_paths.json": image_paths_out,
            "latency.json": profiler.to_dict(),
        },
    )

    print(f"✔ OCR 기반 grounding done. Saved to {exp_dir}")
    return exp_dir


def main():
    import sys

    exp = "configs/experiment.yaml"
    caps = ""
    if len(sys.argv) >= 2:
        exp = sys.argv[1]
    if len(sys.argv) >= 3:
        caps = sys.argv[2]
    else:
        raise SystemExit(
            "Usage: python -m run_inferences.run_ocr_grounding "
            "configs/experiment.yaml outputs/exp_xxx/captions.json"
        )
    run_ocr_grounding(exp, caps)


if __name__ == "__main__":
    main()

