#!/usr/bin/env python3
"""
실험 설정(또는 직접 경로)과 동일한 프롬프트로 Processor만 로드해 프롬프트 토큰 수를 출력합니다.
모델 가중치는 로드하지 않습니다.

예:
  python scripts/count_prompt_tokens.py configs/experiment.yaml
  python scripts/count_prompt_tokens.py configs/experiment.yaml --mode image --image /path/to.jpg
  python scripts/count_prompt_tokens.py configs/experiment.yaml --mode video --video /path/to.mp4 --fps 1
  python scripts/count_prompt_tokens.py configs/experiment.yaml --mode multi --multi a.jpg,b.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 path 에 넣기
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image

from configs.config_resolver import ConfigResolver
from models.vlm_prompt_tokens import (
    caption_prefix_from_prompt_cfg,
    encode_llava_multimodal,
    encode_qwen_multi_image,
    encode_qwen_single_image,
    encode_qwen_video,
    load_llava_processor_only,
    load_qwen_processor_only,
    resolve_context_window_tokens,
)


def _load_images(paths: list[Path]) -> list[Image.Image]:
    out: list[Image.Image] = []
    for p in paths:
        out.append(Image.open(p).convert("RGB"))
    return out


def _auto_paths(cfg: ConfigResolver, mode: str) -> tuple[str, list[Path] | None, Path | None]:
    """mode 가 auto 일 때 데이터셋 설정에서 첫 입력 경로 결정."""
    ds = cfg.dataset_cfg
    paths = ds.get("paths") or {}
    if mode == "video":
        vl = paths.get("video_list") or []
        if not vl:
            raise SystemExit("auto/video: dataset paths.video_list 가 비었습니다. --video 로 지정하세요.")
        return "video", None, Path(vl[0])
    if mode == "multi":
        # image_multi 그룹 첫 번째
        ig = paths.get("image_groups") or paths.get("image_list")
        if ig and isinstance(ig[0], list):
            return "multi", list(ig[0]), None
        il = paths.get("image_list") or []
        if len(il) >= 2:
            return "multi", list(il[:2]), None
        raise SystemExit("auto/multi: image_groups 또는 image_list 가 2장 이상 없습니다. --multi 로 지정하세요.")
    # image
    il = paths.get("image_list") or []
    if not il:
        raise SystemExit("auto/image: dataset paths.image_list 가 비었습니다. --image 로 지정하세요.")
    return "image", [Path(il[0])], None


def main() -> None:
    ap = argparse.ArgumentParser(description="Processor-only 프롬프트 토큰 수 (실험 설정 정합)")
    ap.add_argument(
        "experiment",
        nargs="?",
        default="configs/experiment.yaml",
        help="experiment yaml (기본: configs/experiment.yaml)",
    )
    ap.add_argument("--mode", choices=("auto", "image", "multi", "video"), default="auto")
    ap.add_argument("--image", type=str, default="", help="단일 이미지 경로")
    ap.add_argument("--multi", type=str, default="", help="쉼표로 구분된 여러 이미지 경로")
    ap.add_argument("--video", type=str, default="", help="비디오 경로 (Qwen native video)")
    ap.add_argument("--fps", type=float, default=None, help="비디오 인코딩 시 fps (미지정 시 experiment video.fps)")
    args = ap.parse_args()

    cfg = ConfigResolver(args.experiment)
    model_name = cfg.model_cfg.get("name", "").lower()
    is_llava = "llava" in model_name
    is_qwen = "qwen" in model_name
    if not is_llava and not is_qwen:
        raise SystemExit(f"지원 모델: 이름에 qwen 또는 llava 포함 — got: {cfg.model_cfg.get('name')!r}")

    prompt_cfg = cfg.prompt_cfg
    system_prompt = prompt_cfg.get("system_prompt") or ""
    user_prompt = prompt_cfg.get("user_prompt") or ""
    caption_prefix = caption_prefix_from_prompt_cfg(prompt_cfg)

    gen_cfg = cfg.model_cfg.get("generation") or {}
    max_new = int(gen_cfg.get("max_new_tokens") or 0)

    mode = args.mode
    image_paths: list[Path] | None = None
    video_path: Path | None = None

    if args.video:
        mode = "video"
        video_path = Path(args.video).resolve()
    elif args.multi:
        mode = "multi"
        image_paths = [Path(x.strip()).resolve() for x in args.multi.split(",") if x.strip()]
    elif args.image:
        mode = "image"
        image_paths = [Path(args.image).resolve()]

    if mode == "auto":
        paths = cfg.dataset_cfg.get("paths") or {}
        ds_mode = str(cfg.dataset_cfg.get("mode", "") or "")
        prefer_video = ds_mode == "video_only" or bool(paths.get("video_list"))
        if prefer_video:
            try:
                mode, image_paths, video_path = _auto_paths(cfg, "video")
            except SystemExit:
                mode, image_paths, video_path = _auto_paths(cfg, "image")
        else:
            try:
                mode, image_paths, video_path = _auto_paths(cfg, "image")
            except SystemExit as e:
                try:
                    mode, image_paths, video_path = _auto_paths(cfg, "video")
                except SystemExit:
                    raise e

    model_dir = Path(cfg.model_cfg["resolved_root"])

    print("=== 입력 조건 ===")
    print(f"experiment:     {args.experiment}")
    print(f"model:          {cfg.model_cfg.get('name')}")
    print(f"prompt:         {cfg.prompt_name}")
    print(f"mode:           {mode}")
    if video_path is not None:
        fps = args.fps if args.fps is not None else float(cfg.exp_cfg.get("video", {}).get("fps", 1))
        print(f"video:          {video_path}")
        print(f"video fps:      {fps}")
    if image_paths:
        for i, p in enumerate(image_paths):
            print(f"image[{i}]:      {p}")
            try:
                im = Image.open(p)
                print(f"  size (WxH):   {im.size[0]} x {im.size[1]}")
            except OSError as e:
                print(f"  (열기 실패: {e})")
    print(f"caption_prefix len: {len(caption_prefix)} chars")
    print(f"user_prompt len:    {len(user_prompt)} chars")
    print()

    limits = None
    encode_report = None

    if is_qwen:
        processor = load_qwen_processor_only(cfg.model_cfg)
        limits = resolve_context_window_tokens(processor, model_dir=model_dir)

        if mode == "video":
            if video_path is None:
                raise SystemExit("--mode video 는 --video 경로가 필요합니다.")
            fps = args.fps if args.fps is not None else float(cfg.exp_cfg.get("video", {}).get("fps", 1))
            encode_report = encode_qwen_video(
                processor,
                video_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                caption_prefix=caption_prefix,
                fps=fps,
            )
        elif mode == "multi":
            if not image_paths:
                raise SystemExit("--mode multi 는 --multi 또는 데이터셋에 이미지 목록이 필요합니다.")
            images = _load_images(image_paths)
            full_user = caption_prefix + "\n" + user_prompt if caption_prefix else user_prompt
            encode_report = encode_qwen_multi_image(
                processor,
                images,
                system_prompt=system_prompt,
                user_text=full_user.strip(),
            )
        else:
            if not image_paths:
                raise SystemExit("--mode image 는 --image 또는 데이터셋 image_list 가 필요합니다.")
            images = _load_images(image_paths)
            encode_report = encode_qwen_single_image(
                processor,
                images[0],
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                caption_prefix=caption_prefix,
            )
    else:
        processor = load_llava_processor_only(cfg.model_cfg)
        limits = resolve_context_window_tokens(processor, model_dir=model_dir)
        if mode == "video":
            raise SystemExit("LLaVA 는 이 스크립트에서 native video 토큰 계산을 지원하지 않습니다. 프레임 이미지로 --multi 를 사용하세요.")
        if not image_paths:
            raise SystemExit("LLaVA: 이미지 경로가 필요합니다 (--image / --multi / 데이터셋).")
        images = _load_images(image_paths)
        full_user = caption_prefix + "\n" + user_prompt if caption_prefix else user_prompt
        encode_report = encode_llava_multimodal(
            processor,
            images,
            system_prompt=system_prompt,
            user_prompt=full_user.strip(),
        )

    prompt_tokens = int(encode_report.get("prompt_token_count") or 0)

    print("=== 계산된 토큰(프롬프트) ===")
    for k, v in sorted(encode_report.items()):
        print(f"  {k}: {v}")
    print()

    print("=== 컨텍스트 / 한도 (참고) ===")
    if limits:
        print(f"  tokenizer_model_max_length:      {limits.get('tokenizer_model_max_length')}")
        print(f"  config_max_position_embeddings: {limits.get('config_max_position_embeddings')}")
        print(f"  effective_context_tokens:       {limits.get('effective_context_tokens')}")
    print(f"  generation.max_new_tokens (cfg): {max_new}")
    eff = limits.get("effective_context_tokens") if limits else None
    if eff is not None and max_new > 0:
        remaining = eff - prompt_tokens - max_new
        print(f"  effective - prompt - max_new:    {remaining}")
        if remaining < 0:
            print("  ⚠ 경고: 프롬프트 + max_new_tokens 가 effective 컨텍스트를 넘길 수 있습니다.")
    elif eff is not None:
        print(f"  effective - prompt (생성 여유): {eff - prompt_tokens}")
    print()


if __name__ == "__main__":
    main()
