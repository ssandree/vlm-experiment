"""
Video를 통째로 모델에 전달하는 inference (native video input).

sampling/decoding/aggregation 없이 비디오 파일 경로를 모델에 직접 전달.
configs/experiment.yaml에서 video.input_mode: full 설정 시 사용.

`CUDA_VISIBLE_DEVICES=0 python -m run_inferences.run_video_full`
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from configs.config_resolver import ConfigResolver
from models.model_factory import build_vlm
from data.loader.loader_factory import get_dataloader
from tasks.captioning.caption_task import CaptioningTask
from tasks.grounding.grounding_task import GroundingTask
from tasks.utils.create_experiment_file import create_experiment_dir_and_metadata


def _get_video_size(video_path: Path) -> tuple[int, int]:
    """Decord로 비디오 해상도 반환."""
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(str(video_path), ctx=cpu(0))
        f = vr[0]
        h, w = f.asnumpy().shape[:2]
        return int(w), int(h)
    except Exception:
        return 1920, 1080


def run_video_full(experiment_path: str = "configs/experiment.yaml") -> Path:
    """
    비디오를 통째로 모델에 전달. captioning 또는 grounding task 수행.
    """
    cfg = ConfigResolver(experiment_path)
    mode = cfg.resolved_dataset.get("mode")
    if mode not in ("video_only", "ucfcrime_subset"):
        raise ValueError(
            f"run_video_full requires dataset mode video_only or ucfcrime_subset. Got: {mode}"
        )

    vlm = build_vlm(cfg.model_cfg, cfg.runtime_cfg)
    loader = get_dataloader(dataset_cfg=cfg.resolved_dataset)

    prompt_cfg = cfg.prompt_cfg
    system_prompt = prompt_cfg.get("system_prompt", "")
    user_prompt = prompt_cfg.get("user_prompt", "")
    task_name = cfg.task_name
    gen_cfg = cfg.model_cfg.get("generation") or {}
    video_cfg = cfg.exp_cfg.get("video") or cfg.resolved_dataset.get("video") or {}
    if video_cfg.get("input_mode") != "full":
        raise ValueError(
            "run_video_full requires video.input_mode: full in configs/experiment.yaml. "
            "For sampling mode, use run_video_captioning or run_video_grounding."
        )
    fps = float(video_cfg.get("fps", 1))

    exp_dir = create_experiment_dir_and_metadata(
        runtime_cfg=cfg.runtime_cfg,
        dataset_cfg=cfg.resolved_dataset,
        model_cfg=cfg.model_cfg,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        extra_meta={"task": task_name, "video": {**video_cfg, "input_mode": "full"}},
    )

    all_outputs: dict[str, Any] = {}
    video_paths_out: dict[str, str] = {}

    for batch in loader:
        clip_ids = batch["clip_ids"]
        video_paths = batch["video_paths"]

        for clip_id in clip_ids:
            video_path = Path(video_paths[clip_id])
            print(f"[{clip_id}] Processing {video_path.name}...")

            if task_name == "captioning":
                caption = vlm.run_video(
                    video_path=str(video_path),
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    caption_prefix="",
                    gen_cfg=gen_cfg,
                    fps=fps,
                )
                all_outputs[clip_id] = caption
            elif task_name == "grounding":
                raw = vlm.run_video(
                    video_path=str(video_path),
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    caption_prefix="",
                    gen_cfg=gen_cfg,
                    fps=fps,
                )
                task = GroundingTask()
                bboxes = task._parse_bboxes_from_text(raw)
                w, h = _get_video_size(video_path)
                scaled = [task._scale_bbox_to_pixels(b, w, h) for b in bboxes]
                all_outputs[clip_id] = {"bbox": scaled, "raw_output": raw}
            else:
                out = vlm.run_video(
                    video_path=str(video_path),
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    caption_prefix="",
                    gen_cfg=gen_cfg,
                    fps=fps,
                )
                all_outputs[clip_id] = out

            video_paths_out[clip_id] = str(video_path)

    out_key = "captions" if task_name == "captioning" else "predictions"
    with open(exp_dir / f"{out_key}.json", "w", encoding="utf-8") as f:
        if task_name == "captioning":
            json.dump(all_outputs, f, indent=2, ensure_ascii=False)
        else:
            preds = {
                k: v.get("bbox", v) if isinstance(v, dict) else v
                for k, v in all_outputs.items()
            }
            json.dump(preds, f, indent=2, ensure_ascii=False)

    if task_name == "grounding":
        raw_out = {k: v.get("raw_output", "") for k, v in all_outputs.items() if isinstance(v, dict)}
        if raw_out:
            with open(exp_dir / "raw_outputs.json", "w", encoding="utf-8") as f:
                json.dump(raw_out, f, indent=2, ensure_ascii=False)

    with open(exp_dir / "video_paths.json", "w", encoding="utf-8") as f:
        json.dump(video_paths_out, f, indent=2)

    print(f"✔ Video full inference done. Saved to {exp_dir}")
    return exp_dir


def main():
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "configs/experiment.yaml"
    run_video_full(path)


if __name__ == "__main__":
    main()
