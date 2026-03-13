"""
top_right_crop 적용 여부 검증 스크립트.
실행: python scripts/verify_top_right_crop.py

1. 이미지 로드 → top_right_crop 적용
2. 원본 vs crop 크기 비교 (crop은 w/2 x h/2 여야 함)
3. 모델에 실제 전달되는 이미지가 crop인지 확인
"""

import sys
from pathlib import Path

# 프로젝트 루트
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from configs.config_resolver import ConfigResolver
from data.loader.loader_factory import get_dataloader
from data.input_strategies.build_input_strategy import build_input_strategy


def main():
    cfg = ConfigResolver("configs/experiment.yaml")

    if cfg.resolved_dataset.get("mode") != "image_multi":
        print("⚠ dataset을 image_multi로 설정 후 실행하세요.")
        return

    image_cfg = cfg.exp_cfg.get("image") or {}
    strategy_type = image_cfg.get("input_strategy", "identity")
    print(f"[Config] image.input_strategy = {strategy_type!r}")

    strategy = build_input_strategy({"type": strategy_type})
    loader = get_dataloader(dataset_cfg=cfg.resolved_dataset)

    batch = next(iter(loader))
    image_groups = batch["image_groups"]
    group_ids = batch["group_ids"]

    print(f"\n[Batch] {len(image_groups)} groups, ids: {group_ids}")

    for imgs, gid in zip(image_groups, group_ids):
        orig_dims = [(im.width, im.height) for im in imgs]
        processed = strategy.process(imgs)
        proc_list = processed if isinstance(processed, list) else [processed]
        proc_dims = [(im.width, im.height) for im in proc_list]

        print(f"\n[Group {gid}]")
        for i, (o, p) in enumerate(zip(orig_dims, proc_dims)):
            expected_w = o[0] // 2
            expected_h = o[1] // 2
            is_cropped = p[0] == expected_w and p[1] == expected_h
            status = "✓ CROPPED" if is_cropped else "✗ NOT CROPPED (원본?)"
            print(f"  img[{i}]: orig {o} → processed {p}  {status}")
            if strategy_type == "top_right_crop" and not is_cropped:
                print(f"    (top_right_crop 시 기대값: {expected_w}x{expected_h})")


if __name__ == "__main__":
    main()
