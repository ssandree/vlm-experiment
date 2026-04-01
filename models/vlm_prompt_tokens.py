"""
VLM용: 가중치(모델) 없이 Processor만 로드해 추론과 동일한 방식으로 인코딩·토큰 길이 확인.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image


def load_qwen_processor_only(model_cfg: dict):
    """configs/model 의 model_cfg (resolved_root, vision 포함) 로 Qwen3-VL 계열 processor만 로드."""
    from transformers import AutoProcessor

    pretrained_path = str(model_cfg["resolved_root"])
    vision_cfg = model_cfg.get("vision", {}) or {}
    kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
    }
    min_pixels = vision_cfg.get("min_pixels")
    if min_pixels is not None:
        kwargs["min_pixels"] = int(min_pixels)
    max_pixels = vision_cfg.get("max_pixels")
    if max_pixels is not None:
        max_pixels = int(max_pixels)
        if min_pixels is not None and max_pixels < int(min_pixels):
            max_pixels = int(min_pixels)
        kwargs["max_pixels"] = max_pixels
    return AutoProcessor.from_pretrained(pretrained_path, **kwargs)


def load_llava_processor_only(model_cfg: dict):
    from transformers import LlavaProcessor

    return LlavaProcessor.from_pretrained(
        str(model_cfg["resolved_root"]),
        local_files_only=True,
    )


def max_position_embeddings_from_config(model_dir: Path) -> int | None:
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        return None
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    v = cfg.get("max_position_embeddings")
    if v is None:
        tc = cfg.get("text_config")
        if isinstance(tc, dict):
            v = tc.get("max_position_embeddings")
    return int(v) if v is not None else None


def _tokenizer_reported_max_length(tokenizer) -> int | None:
    m = getattr(tokenizer, "model_max_length", None)
    if m is None:
        return None
    try:
        mi = int(m)
    except (TypeError, ValueError):
        return None
    # transformers 기본 "무제한" 플레이스홀더
    if mi >= 1_000_000:
        return None
    return mi


def resolve_context_window_tokens(
    processor,
    model_dir: Path | None = None,
) -> dict[str, Any]:
    """학습 컨텍스트 상한(참고용). config.json 과 tokenizer.model_max_length 를 조합."""
    tok_max = _tokenizer_reported_max_length(processor.tokenizer)
    cfg_max = max_position_embeddings_from_config(model_dir) if model_dir else None
    if cfg_max is not None and tok_max is not None:
        effective = min(cfg_max, tok_max)
    elif cfg_max is not None:
        effective = cfg_max
    elif tok_max is not None:
        effective = tok_max
    else:
        effective = None
    return {
        "tokenizer_model_max_length": tok_max,
        "config_max_position_embeddings": cfg_max,
        "effective_context_tokens": effective,
    }


def _qwen_inputs_to_report(inputs: Any) -> dict[str, Any]:
    """CPU 배치에서 길이·그리드 요약 (토큰 출력용)."""
    report: dict[str, Any] = {}
    if inputs is None:
        return report
    iids = inputs.get("input_ids") if hasattr(inputs, "get") else None
    if iids is not None and hasattr(iids, "shape"):
        report["input_ids_shape"] = tuple(iids.shape)
        report["prompt_token_count"] = int(iids.shape[1])
    am = inputs.get("attention_mask") if hasattr(inputs, "get") else None
    if am is not None and hasattr(am, "shape"):
        report["attention_mask_shape"] = tuple(am.shape)
    for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        t = inputs.get(key) if hasattr(inputs, "get") else None
        if t is not None and hasattr(t, "shape"):
            report[f"{key}_shape"] = tuple(t.shape)
    ig = inputs.get("image_grid_thw") if hasattr(inputs, "get") else None
    if ig is not None and hasattr(ig, "flatten"):
        try:
            flat = ig.detach().cpu().flatten().tolist() if hasattr(ig, "detach") else list(ig.flatten())
            # 각 행 (t,h,w) 곱 → 이미지/클립 패치 수 합
            if flat and len(flat) % 3 == 0:
                idx = 0
                patch_sums = []
                while idx + 2 < len(flat):
                    t, h, w = flat[idx], flat[idx + 1], flat[idx + 2]
                    patch_sums.append(int(t) * int(h) * int(w))
                    idx += 3
                report["image_grid_thw_patch_counts"] = patch_sums
                report["image_grid_thw_patches_total"] = sum(patch_sums)
        except Exception:
            pass
    vg = inputs.get("video_grid_thw") if hasattr(inputs, "get") else None
    if vg is not None and hasattr(vg, "flatten"):
        try:
            flat = vg.detach().cpu().flatten().tolist() if hasattr(vg, "detach") else list(vg.flatten())
            if flat and len(flat) % 3 == 0:
                idx = 0
                patch_sums = []
                while idx + 2 < len(flat):
                    t, h, w = flat[idx], flat[idx + 1], flat[idx + 2]
                    patch_sums.append(int(t) * int(h) * int(w))
                    idx += 3
                report["video_grid_thw_patch_counts"] = patch_sums
                report["video_grid_thw_patches_total"] = sum(patch_sums)
        except Exception:
            pass
    return report


def encode_qwen_single_image(
    processor,
    image: Image.Image,
    system_prompt: str,
    user_prompt: str,
    caption_prefix: str = "",
) -> dict[str, Any]:
    """run_inference(단일 이미지)와 동일: apply_chat_template(tokenize=False) + processor(text, images)."""
    full_user_text = caption_prefix
    if user_prompt:
        full_user_text = caption_prefix + "\n" + user_prompt if caption_prefix else user_prompt

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": full_user_text},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    return _qwen_inputs_to_report(inputs)


def encode_qwen_multi_image(
    processor,
    images: list[Image.Image],
    system_prompt: str,
    user_text: str,
) -> dict[str, Any]:
    """run_inference_multi_image 와 동일한 템플릿 구조."""
    if not images:
        return {"error": "no images", "prompt_token_count": 0}
    content_for_template: list[dict[str, Any]] = []
    for _ in images:
        content_for_template.append({"type": "image"})
    content_for_template.append({"type": "text", "text": user_text})

    full_messages = [
        {"role": "system", "content": system_prompt or ""},
        {"role": "user", "content": content_for_template},
    ]

    text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=list(images),
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    return _qwen_inputs_to_report(inputs)


def encode_qwen_video(
    processor,
    video_path: str | Path,
    system_prompt: str,
    user_prompt: str,
    caption_prefix: str = "",
    fps: float = 1.0,
    prefer_decord: bool = True,
) -> dict[str, Any]:
    """run_inference_video_clean 과 동일: processor.apply_chat_template(tokenize=True, fps=...)."""
    video_path = str(Path(video_path).resolve())

    full_user_text = caption_prefix
    if user_prompt:
        full_user_text = caption_prefix + "\n" + user_prompt if caption_prefix else user_prompt

    messages = [
        {"role": "system", "content": system_prompt or ""},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": full_user_text},
            ],
        },
    ]

    def _apply():
        return processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            fps=fps,
        )

    # prefer_decord는 과거 버전 호환을 위한 시그니처 파라미터입니다.
    # 현재는 HuggingFace 공식 흐름만 사용합니다(몽키패치/보정 없음).
    _ = prefer_decord
    inputs = _apply()
    inputs.pop("token_type_ids", None)
    return _qwen_inputs_to_report(inputs)


def encode_llava_multimodal(
    processor,
    images: list[Image.Image],
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    """run_llava_inference 와 동일한 대화/프로세서 호출 (이미지 여러 장 지원)."""
    if not images:
        return {"error": "no images", "prompt_token_count": 0}

    cleaned_user_prompt = user_prompt.replace("<image>", "").strip()
    merged_prompt = f"{system_prompt}\n{cleaned_user_prompt}".strip()

    content = [{"type": "image"}] * len(images) + [{"type": "text", "text": merged_prompt}]
    conversation = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=prompt,
        images=images,
        return_tensors="pt",
    )
    return _qwen_inputs_to_report(inputs)


def caption_prefix_from_prompt_cfg(prompt_cfg: dict) -> str:
    baseline = prompt_cfg.get("baseline") or {}
    return (baseline.get("prefix") or "") if isinstance(baseline, dict) else ""
