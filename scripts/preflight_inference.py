#!/usr/bin/env python3
"""
Qwen/LLaVA 추론 전 사전 점검(preflight):
1) processor-only 토큰/입력 크기 계산
2) 컨텍스트 여유와 KV cache 메모리 근사
3) (옵션) 짧은 dry-run generate로 실제 실행 가능성 확인

예시:
  python scripts/preflight_inference.py configs/experiment.yaml
  python scripts/preflight_inference.py configs/experiment.yaml --mode video --video /path/a.mp4 --fps 1
  python scripts/preflight_inference.py configs/experiment.yaml --dry-run --dry-max-new-tokens 4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image

from configs.config_resolver import ConfigResolver
from models.model_factory import build_vlm
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


def _human_bytes(n: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    v = float(n)
    while v >= 1024 and i < len(units) - 1:
        v /= 1024.0
        i += 1
    return f"{v:.2f} {units[i]}"


def _load_images(paths: list[Path]) -> list[Image.Image]:
    return [Image.open(p).convert("RGB") for p in paths]


def _first_or_none(v: Any) -> Any:
    if not v:
        return None
    return v[0]


def _auto_mode_from_dataset(cfg: ConfigResolver, mode: str) -> tuple[str, list[Path], Path | None]:
    ds = cfg.dataset_cfg
    paths = ds.get("paths") or {}
    image_paths: list[Path] = []
    video_path: Path | None = None

    if mode == "video":
        v = _first_or_none(paths.get("video_list"))
        if v is None:
            raise SystemExit("video 모드인데 dataset paths.video_list 가 비었습니다. --video 지정 필요.")
        video_path = Path(v)
        return "video", image_paths, video_path

    if mode == "multi":
        groups = paths.get("image_groups")
        if groups and isinstance(groups[0], list) and groups[0]:
            image_paths = [Path(x) for x in groups[0]]
            return "multi", image_paths, None
        il = paths.get("image_list") or []
        if len(il) >= 2:
            image_paths = [Path(il[0]), Path(il[1])]
            return "multi", image_paths, None
        raise SystemExit("multi 모드인데 이미지가 2장 미만입니다. --multi 지정 필요.")

    # image
    i = _first_or_none(paths.get("image_list"))
    if i is None:
        raise SystemExit("image 모드인데 dataset paths.image_list 가 비었습니다. --image 지정 필요.")
    image_paths = [Path(i)]
    return "image", image_paths, None


def _infer_hidden_and_layers(model_dir: Path) -> tuple[int | None, int | None]:
    try:
        import json

        cfg_path = model_dir / "config.json"
        if not cfg_path.exists():
            return None, None
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
        hidden = tc.get("hidden_size")
        layers = tc.get("num_hidden_layers")
        if hidden is None or layers is None:
            return None, None
        return int(hidden), int(layers)
    except Exception:
        return None, None


def _estimate_kv_cache_bytes(
    seq_len: int,
    max_new_tokens: int,
    hidden_size: int,
    num_layers: int,
    batch_size: int = 1,
    kv_dtype_bytes: int = 2,
) -> int:
    total_tokens = seq_len + max_new_tokens
    # 대략식(보수적): 2(K,V) * layers * tokens * hidden * dtype_bytes * batch
    return int(2 * num_layers * total_tokens * hidden_size * kv_dtype_bytes * batch_size)


def _cuda_mem_info() -> tuple[int, int] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free_b, total_b = torch.cuda.mem_get_info()
        return int(free_b), int(total_b)
    except Exception:
        return None


def _print_verdict(free_bytes: int | None, est_kv_bytes: int | None) -> None:
    if free_bytes is None or est_kv_bytes is None:
        print("실행 가능성 판정: 정보 부족 (CUDA/모델 config 확인 필요)")
        return
    # KV 외 추가 오버헤드(activations/allocator fragmentation) 보수 마진
    required = int(est_kv_bytes * 1.5)
    if required <= free_bytes * 0.6:
        level = "높음"
    elif required <= free_bytes * 0.9:
        level = "중간"
    else:
        level = "낮음"
    print(f"실행 가능성(근사): {level}")
    print(f"  free VRAM:        {_human_bytes(free_bytes)}")
    print(f"  est. KV x1.5:     {_human_bytes(required)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Preflight: 토큰/메모리 근사 + optional dry-run")
    ap.add_argument("experiment", nargs="?", default="configs/experiment.yaml")
    ap.add_argument("--mode", choices=("auto", "image", "multi", "video"), default="auto")
    ap.add_argument("--image", type=str, default="")
    ap.add_argument("--multi", type=str, default="")
    ap.add_argument("--video", type=str, default="")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true", help="짧은 generate 실행")
    ap.add_argument("--dry-max-new-tokens", type=int, default=4)
    args = ap.parse_args()

    cfg = ConfigResolver(args.experiment)
    model_name = (cfg.model_cfg.get("name") or "").lower()
    is_qwen = "qwen" in model_name
    is_llava = "llava" in model_name
    if not (is_qwen or is_llava):
        raise SystemExit(f"지원 모델 아님: {cfg.model_cfg.get('name')}")

    prompt_cfg = cfg.prompt_cfg
    system_prompt = prompt_cfg.get("system_prompt") or ""
    user_prompt = prompt_cfg.get("user_prompt") or ""
    caption_prefix = caption_prefix_from_prompt_cfg(prompt_cfg)
    gen_cfg = cfg.model_cfg.get("generation") or {}
    max_new = int(gen_cfg.get("max_new_tokens") or 0)
    model_dir = Path(cfg.model_cfg["resolved_root"])

    mode = args.mode
    image_paths: list[Path] = []
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
        ds_mode = str(cfg.dataset_cfg.get("mode") or "")
        prefer_video = ds_mode == "video_only" or bool(paths.get("video_list"))
        if prefer_video:
            try:
                mode, image_paths, video_path = _auto_mode_from_dataset(cfg, "video")
            except SystemExit:
                mode, image_paths, video_path = _auto_mode_from_dataset(cfg, "image")
        else:
            mode, image_paths, video_path = _auto_mode_from_dataset(cfg, "image")

    if is_qwen:
        processor = load_qwen_processor_only(cfg.model_cfg)
    else:
        processor = load_llava_processor_only(cfg.model_cfg)

    limits = resolve_context_window_tokens(processor, model_dir=model_dir)

    print("=== 입력 조건 ===")
    print(f"experiment:     {args.experiment}")
    print(f"model:          {cfg.model_cfg.get('name')}")
    print(f"prompt:         {cfg.prompt_name}")
    print(f"mode:           {mode}")
    if video_path is not None:
        fps = args.fps if args.fps is not None else float(cfg.exp_cfg.get("video", {}).get("fps", 1))
        print(f"video:          {video_path}")
        print(f"video fps:      {fps}")
    for i, p in enumerate(image_paths):
        print(f"image[{i}]:      {p}")
    print(f"caption_prefix len: {len(caption_prefix)} chars")
    print(f"user_prompt len:    {len(user_prompt)} chars")
    print()

    if is_qwen:
        if mode == "video":
            if video_path is None:
                raise SystemExit("--mode video 인데 --video/데이터셋 입력이 없습니다.")
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
            images = _load_images(image_paths)
            full_user = caption_prefix + "\n" + user_prompt if caption_prefix else user_prompt
            encode_report = encode_qwen_multi_image(
                processor,
                images,
                system_prompt=system_prompt,
                user_text=full_user.strip(),
            )
        else:
            images = _load_images(image_paths)
            encode_report = encode_qwen_single_image(
                processor,
                images[0],
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                caption_prefix=caption_prefix,
            )
    else:
        if mode == "video":
            raise SystemExit("LLaVA는 이 스크립트에서 native video preflight 미지원(멀티 이미지로 점검하세요).")
        images = _load_images(image_paths)
        full_user = caption_prefix + "\n" + user_prompt if caption_prefix else user_prompt
        encode_report = encode_llava_multimodal(
            processor,
            images,
            system_prompt=system_prompt,
            user_prompt=full_user.strip(),
        )

    prompt_tokens = int(encode_report.get("prompt_token_count") or 0)
    print("=== 계산된 토큰/입력 ===")
    for k, v in sorted(encode_report.items()):
        print(f"  {k}: {v}")
    print()

    print("=== 컨텍스트 여유 ===")
    effective = limits.get("effective_context_tokens") if limits else None
    print(f"  effective_context_tokens:       {effective}")
    print(f"  generation.max_new_tokens (cfg): {max_new}")
    if effective is not None:
        remain = int(effective) - prompt_tokens - max_new
        print(f"  effective - prompt - max_new:    {remain}")
        if remain < 0:
            print("  ⚠ 컨텍스트 초과 가능성 높음")
    print()

    print("=== 메모리 근사(KV cache) ===")
    hidden, layers = _infer_hidden_and_layers(model_dir)
    if hidden is not None and layers is not None:
        est_kv = _estimate_kv_cache_bytes(
            seq_len=prompt_tokens,
            max_new_tokens=max_new,
            hidden_size=hidden,
            num_layers=layers,
            batch_size=1,
            kv_dtype_bytes=2,  # fp16/bf16 가정
        )
        print(f"  hidden_size: {hidden}")
        print(f"  num_layers:  {layers}")
        print(f"  est_kv_cache_bytes: {_human_bytes(est_kv)}")
    else:
        est_kv = None
        print("  모델 config에서 hidden_size/num_layers 추출 실패 (근사 생략)")

    mem = _cuda_mem_info()
    if mem is not None:
        free_b, total_b = mem
        print(f"  cuda free/total: {_human_bytes(free_b)} / {_human_bytes(total_b)}")
    else:
        free_b = None
        print("  cuda mem info: unavailable")
    _print_verdict(free_b, est_kv)
    print()

    if not args.dry_run:
        return

    print("=== dry-run generate ===")
    vlm = build_vlm(cfg.model_cfg, cfg.runtime_cfg)
    dry_gen = dict(gen_cfg)
    dry_gen["max_new_tokens"] = max(1, int(args.dry_max_new_tokens))
    dry_gen["do_sample"] = False

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    t0 = time.perf_counter()
    if mode == "video":
        assert video_path is not None
        fps = args.fps if args.fps is not None else float(cfg.exp_cfg.get("video", {}).get("fps", 1))
        out = vlm.run_video(
            video_path=str(video_path),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            caption_prefix=caption_prefix,
            gen_cfg=dry_gen,
            fps=fps,
        )
    elif mode == "multi":
        images = _load_images(image_paths)
        out = vlm.run(
            task="captioning",
            image=images,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            caption_prefix=caption_prefix,
            gen_cfg=dry_gen,
        )
    else:
        images = _load_images(image_paths)
        out = vlm.run(
            task="captioning",
            image=images[0],
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            caption_prefix=caption_prefix,
            gen_cfg=dry_gen,
        )
    elapsed = time.perf_counter() - t0

    peak = None
    try:
        import torch

        if torch.cuda.is_available():
            peak = int(torch.cuda.max_memory_allocated())
    except Exception:
        pass

    print(f"  dry-run status: success")
    print(f"  elapsed:        {elapsed:.3f}s")
    print(f"  output chars:   {len(out or '')}")
    if peak is not None:
        print(f"  peak allocated: {_human_bytes(peak)}")


if __name__ == "__main__":
    main()

