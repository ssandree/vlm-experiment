"""
YOLO 객체 검출 실행. image_only 데이터셋 또는 이미지 경로 리스트로 inference.

`CUDA_VISIBLE_DEVICES=0 python -m run_inferences.run_yolo_detection` 로 실행.
experiment.yaml에서 dataset: image_only 설정.
"""

from __future__ import annotations

import json
from pathlib import Path

from configs.config_resolver import ConfigResolver
from data.loader.loader_factory import get_dataloader
from detectors.yolo_detector import YOLODetector


def run_yolo_detection(
    experiment_path: str = "configs/experiment.yaml",
    weights_path: str | Path = None,
    conf_threshold: float = 0.25,
    output_dir: str | Path = None,
) -> Path:
    """
    image_only 데이터셋으로 YOLO 검출 수행.
    predictions.json: {image_id: [{"bbox": [x,y,w,h], "class": str, "conf": float}, ...]}
    """
    cfg = ConfigResolver(experiment_path)
    if cfg.resolved_dataset.get("mode") != "image_only":
        raise ValueError(
            "run_yolo_detection requires dataset mode: image_only. "
            f"Current: {cfg.resolved_dataset.get('mode')}"
        )

    detector = YOLODetector(
        weights_path=weights_path,
        conf_threshold=conf_threshold,
    )

    loader = get_dataloader(dataset_cfg=cfg.resolved_dataset)
    all_predictions: dict = {}
    all_image_paths: dict = {}

    for batch in loader:
        images = batch["images"]
        image_ids = batch["image_ids"]
        image_paths = batch["image_paths"]

        results = detector.detect(images)
        for iid, ipath, dets in zip(image_ids, image_paths, results):
            all_predictions[iid] = dets
            all_image_paths[iid] = ipath

    if output_dir is None:
        from datetime import datetime
        output_dir = Path("outputs") / f"yolo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)

    with open(output_dir / "image_paths.json", "w", encoding="utf-8") as f:
        json.dump(all_image_paths, f, indent=2)

    print(f"✔ YOLO detection done. Saved to {output_dir}")
    return output_dir


def main():
    run_yolo_detection("configs/experiment.yaml")


if __name__ == "__main__":
    main()
