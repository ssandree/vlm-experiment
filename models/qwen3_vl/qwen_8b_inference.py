# Qwen3-VL-8B inference (pipeline-safe)

import itertools
import logging
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from models.inference_input_log import print_model_input_shapes, print_model_output_tokens

logger = logging.getLogger(__name__)


class InferenceOOMError(RuntimeError):
    """Frame sweep에서 입력 크기/토큰 길이를 남기기 위한 OOM 래퍼 예외."""

    def __init__(self, message: str, stats: dict[str, Any]):
        super().__init__(message)
        self.stats = stats


def _maybe_move_inputs_to_model_device(model, inputs):
    """
    device_map="auto"처럼 sharded 로딩이면, inputs을 model.device(GPU0)로
    강제 이동하면 활성(activation)이 한 GPU로 몰려 OOM이 나기 쉽습니다.
    accelerate dispatch에서는 inputs을 CPU에 둬도 모듈 배치에 맞게 이동되는 편이라,
    여기서는 `device_map=None`(단일 GPU 적재)일 때만 명시적으로 이동합니다.
    """
    device_map_choice = getattr(model, "_vlm_device_map_choice", None)
    # DS/다른 래퍼에서는 model.device 속성이 없을 수 있어 파라미터 device로 안전하게 결정
    model_device = getattr(model, "device", None)
    if model_device is None:
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")

    if device_map_choice is None:
        return inputs.to(model_device)

    # LM 디바이스 추정: language head / embedding 파라미터 우선
    lm_device = model_device
    try:
        lm_device = model.model.embed_tokens.weight.device
    except Exception:
        pass

    # Vision 디바이스 추정:
    # 1) vision 모듈 파라미터 디바이스 우선
    # 2) 실패 시 hf_device_map에서 visual/vision key를 찾아 복원
    vision_device = lm_device
    try:
        vision_device = next(model.visual.parameters()).device
    except Exception:
        hf_map = getattr(model, "hf_device_map", None) or {}
        for k, v in hf_map.items():
            key = str(k).lower()
            if "visual" not in key and "vision" not in key:
                continue
            if isinstance(v, int):
                vision_device = torch.device(f"cuda:{v}")
                break
            if isinstance(v, str):
                s = v.strip()
                if s.startswith("cuda:"):
                    vision_device = torch.device(s)
                    break
                if s.isdigit():
                    vision_device = torch.device(f"cuda:{s}")
                    break

    # 1) 토큰 텐서는 LM 디바이스로 이동
    for k in ("input_ids", "attention_mask", "position_ids", "token_type_ids", "mm_token_type_ids"):
        v = inputs.get(k)
        if isinstance(v, torch.Tensor) and v.device != lm_device:
            inputs[k] = v.to(lm_device)

    # 2) Vision 텐서는 Vision 디바이스로 이동
    for k in ("pixel_values", "pixel_values_images", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
        v = inputs.get(k)
        if isinstance(v, torch.Tensor) and v.device != vision_device:
            inputs[k] = v.to(vision_device)

    return inputs


def _sanitize_generation_kwargs(gen_cfg: dict) -> dict:
    """
    Transformers는 `do_sample=False`일 때 `temperature/top_p/top_k` 같은
    sampling 관련 플래그를 "유효하지 않음" 경고로 처리할 수 있습니다.
    """
    gen_kwargs = dict(gen_cfg or {})
    if not bool(gen_kwargs.get("do_sample", False)):
        # Transformers 경고 억제:
        # do_sample=False인데 sampling 플래그가 남아 있으면
        # "may be ignored" 경고가 발생하므로 중립값으로 고정.
        gen_kwargs["temperature"] = 1.0
        gen_kwargs["top_p"] = 1.0
        gen_kwargs["top_k"] = 50
    return gen_kwargs


def _should_log_input_shapes() -> bool:
    return os.environ.get("VLM_DEBUG_INPUT_SHAPES", "").strip() == "1"


def _load_full_video_numpy_decord(video_path: str) -> tuple[np.ndarray, float] | None:
    """
    Transformers 기본 경로(torchvision→PyAV/swscale)가 EAGAIN 등으로 실패할 때 대비.
    성공 시 (T,H,W,3) uint8 과 컨테이너 FPS 반환.
    """
    try:
        from decord import VideoReader, cpu
    except ImportError:
        logger.debug("decord 미설치: native video는 파일 경로 + torchvision 경로로 처리합니다.")
        return None
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        n = int(len(vr))
        if n <= 0:
            return None
        native_fps = float(vr.get_avg_fps())
        if native_fps <= 0:
            native_fps = 24.0
        frames = vr.get_batch(list(range(n))).asnumpy()
        if frames.ndim != 4 or frames.shape[-1] != 3:
            return None
        return frames, native_fps
    except Exception as exc:
        logger.warning(
            "decord 디코딩 실패, 경로 기반 torchvision 디코딩으로 폴백합니다: %s",
            exc,
        )
        return None


def _build_native_video_payload(video_path: str) -> tuple[str | np.ndarray, list[dict[str, Any]] | None]:
    """decord 우선 로드; 실패 시 파일 경로 그대로(torchvision 경로)."""
    video_payload: str | np.ndarray = video_path
    video_metadata_kw: list[dict[str, Any]] | None = None
    backend = os.environ.get("VLM_NATIVE_VIDEO_BACKEND", "decord").strip().lower()
    if backend != "path":
        decord_out = _load_full_video_numpy_decord(video_path)
        if decord_out is not None:
            frames_np, native_fps = decord_out
            video_payload = frames_np
            t_n = int(frames_np.shape[0])
            video_metadata_kw = [
                {
                    "total_num_frames": t_n,
                    "fps": float(native_fps),
                    "duration": float(t_n / native_fps) if native_fps > 0 else 0.0,
                    "frames_indices": list(range(t_n)),
                }
            ]
            logger.info(
                "[native video] decord 로드: %d프레임, fps=%.3f (torchvision/PyAV 경로 생략)",
                t_n,
                native_fps,
            )
    return video_payload, video_metadata_kw


def _align_qwen3_video_grid_thw_for_rope(inputs, processor) -> None:
    mm_token_type_ids = inputs.get("mm_token_type_ids")
    attention_mask = inputs.get("attention_mask")
    video_grid_thw = inputs.get("video_grid_thw")
    pixel_nv = inputs.get("pixel_values_videos")
    if mm_token_type_ids is None or attention_mask is None or video_grid_thw is None:
        return
    n_groups = _count_mm_video_groups(mm_token_type_ids, attention_mask, 0)
    v_rows = int(video_grid_thw.shape[0])
    sum_t = int(video_grid_thw[:, 0].sum().item())
    v_proc = getattr(processor, "video_processor", None)
    merge = int(getattr(v_proc, "merge_size", 2)) if v_proc is not None else 2
    patch_from_grid = _patch_rows_from_video_grid(video_grid_thw, merge)
    pix_n = int(pixel_nv.shape[0]) if pixel_nv is not None else -1

    logger.info(
        "[VIDEO ALIGN] video_groups=%d video_grid_thw.rows=%d sum(T)=%d "
        "pixel_values_videos.shape[0]=%d patches_from_grid=%d merge=%d",
        n_groups,
        v_rows,
        sum_t,
        pix_n,
        patch_from_grid,
        merge,
    )

    if pixel_nv is not None and patch_from_grid != pix_n:
        logger.warning(
            "[VIDEO ALIGN] patches_from_grid=%d != pixel_values_videos.rows=%d",
            patch_from_grid,
            pix_n,
        )

    if n_groups > sum_t:
        raise RuntimeError(
            f"[VIDEO ALIGN] 비디오 mm 구간 수({n_groups})가 video_grid_thw의 "
            f"시간축 합 sum(T)={sum_t}보다 큽니다. 토큰화/그리드 불일치입니다."
        )

    if n_groups == sum_t and v_rows < n_groups:
        expanded = _expand_video_grid_thw_per_temporal_chunk(video_grid_thw)
        exp_patch = _patch_rows_from_video_grid(expanded, merge)
        if exp_patch != patch_from_grid:
            raise RuntimeError(
                f"[VIDEO ALIGN] grid 펼침 후 패치 합이 변했습니다 "
                f"({patch_from_grid} -> {exp_patch})."
            )
        inputs["video_grid_thw"] = expanded
        logger.info(
            "[VIDEO ALIGN] expanded video_grid_thw %s -> %s (RoPE 소비 횟수와 정합)",
            tuple(video_grid_thw.shape),
            tuple(expanded.shape),
        )

    elif n_groups > 0 and n_groups < v_rows:
        logger.warning(
            "[VIDEO ALIGN] video_groups=%d < video_grid_thw.rows=%d — "
            "pixel_values_videos와 짝을 맞추려면 그리드·픽셀을 함께 잘라야 해 "
            "자동 보정에서는 건드리지 않습니다.",
            n_groups,
            v_rows,
        )


def _generate_qwen3_after_chat_template(model, processor, inputs, gen_cfg: dict) -> str:
    input_length = inputs["input_ids"].shape[1]
    gen_kwargs = _sanitize_generation_kwargs(gen_cfg)
    if "pad_token_id" not in gen_kwargs:
        gen_kwargs["pad_token_id"] = (
            getattr(processor.tokenizer, "pad_token_id", None)
            or processor.tokenizer.eos_token_id
        )
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **gen_kwargs)
    print_model_output_tokens(
        generated_ids,
        input_length=input_length,
        tag="qwen_after_chat_template",
        batch=inputs,
    )
    generated_only = generated_ids[:, input_length:]
    if generated_only.shape[1] == 0:
        return ""
    decoded = processor.batch_decode(
        generated_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    from pipelines.run_model import normalize_assistant_output

    out = normalize_assistant_output(decoded)
    return out or (decoded or "").strip() or ""


def _count_mm_video_groups(
    mm_token_type_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_idx: int = 0,
) -> int:
    """get_rope_index()와 동일하게: attention_mask 내 연속 구간 중 mm 타입 2(비디오) 개수."""
    input_token_type = mm_token_type_ids[batch_idx]
    if attention_mask is not None:
        input_token_type = input_token_type[attention_mask[batch_idx].bool()]
    n = 0
    for key, _ in itertools.groupby(input_token_type.tolist()):
        if key == 2:
            n += 1
    return n


def _expand_video_grid_thw_per_temporal_chunk(video_grid_thw: torch.Tensor) -> torch.Tensor:
    """
    video_grid_thw 각 행 (T,H,W)를 T개의 (1,H,W) 행으로 펼침.
    패치 수 합: sum((T*H*W)//merge^2) 불변 → pixel_values_videos와 비전 분할 정합 유지.
    """
    rows: list[list[int]] = []
    for t, h, w in video_grid_thw.detach().cpu().tolist():
        ti, hi, wi = int(t), int(h), int(w)
        for _ in range(ti):
            rows.append([1, hi, wi])
    if not rows:
        return video_grid_thw
    return torch.tensor(
        rows,
        dtype=video_grid_thw.dtype,
        device=video_grid_thw.device,
    )


def _patch_rows_from_video_grid(
    video_grid_thw: torch.Tensor,
    spatial_merge: int,
) -> int:
    return int((video_grid_thw.prod(-1) // (spatial_merge**2)).sum().item())


def _assistant_prompt_token_ids(processor):
    """add_generation_prompt=True 시 추가되는 assistant 프롬프트의 토큰 ID 시퀀스."""
    for prompt_str in ("<|im_start|>assistant\n", "assistant\n"):
        ids = processor.tokenizer.encode(
            prompt_str, add_special_tokens=False, return_tensors=None
        )
        if ids:
            return ids
    return []


def _extract_assistant_from_generated_ids(processor, generated_ids, fallback_decode=""):
    """
    generated_ids에서 assistant 프롬프트 직후 구간만 잘라 디코딩.
    Vision 토큰 때문에 input_length로 자르면 안 되므로, 토큰 시퀀스 상에서
    assistant 프롬프트 마지막 등장 위치 다음부터만 디코딩.
    """
    if generated_ids is None or generated_ids.shape[1] == 0:
        return (fallback_decode or "").strip()
    prompt_ids = _assistant_prompt_token_ids(processor)
    if not prompt_ids:
        return ""
    seq = generated_ids[0].tolist()
    n = len(prompt_ids)
    start = None
    for i in range(len(seq) - n, -1, -1):
        if i >= 0 and seq[i : i + n] == prompt_ids:
            start = i + n
            break

    if start is None:
        return ""
    trimmed = generated_ids[:, start:]
    seq_trimmed = trimmed[0].tolist()
    while seq_trimmed and seq_trimmed[0] == 0:
        seq_trimmed.pop(0)
    while seq_trimmed and seq_trimmed[-1] == 0:
        seq_trimmed.pop()
    if not seq_trimmed:
        return (fallback_decode or "").strip()
    trimmed = torch.tensor([seq_trimmed], device=generated_ids.device, dtype=generated_ids.dtype)
    decoded = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    out = (decoded[0] or "").strip()
    for token in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        out = out.replace(token, "")
    return out.strip() or ""


def run_inference(
    model,
    processor,
    image: Image.Image,
    system_prompt: str,
    user_prompt: str,
    caption_prefix: str,
    gen_cfg: dict,
) -> str:
    full_user_text = caption_prefix
    if user_prompt:
        full_user_text = caption_prefix + "\n" + user_prompt

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
    inputs = _maybe_move_inputs_to_model_device(model, inputs)

    inputs.pop("token_type_ids", None)
    if _should_log_input_shapes():
        print_model_input_shapes(inputs, tag="qwen_single_image")

    gen_kwargs = _sanitize_generation_kwargs(gen_cfg)
    if "pad_token_id" not in gen_kwargs:
        gen_kwargs["pad_token_id"] = getattr(
            processor.tokenizer, "pad_token_id", None
        ) or processor.tokenizer.eos_token_id

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **gen_kwargs)
    print_model_output_tokens(
        generated_ids,
        input_length=inputs["input_ids"].shape[-1],
        tag="qwen_single_image",
        batch=inputs,
    )

    from pipelines.run_model import normalize_assistant_output

    out = _extract_assistant_from_generated_ids(
        processor,
        generated_ids,
        fallback_decode="",
    )

    if not out:
        caption_fallback = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        out = normalize_assistant_output(caption_fallback)

    return out or ""


def run_inference_multi_image(
    model,
    processor,
    messages: list,
    system_prompt: str,
    gen_cfg: dict,
) -> str:
    """
    Multi-image inference for MI-VQA style inputs.
    """
    if not messages or not messages[0].get("content"):
        logger.warning("run_inference_multi_image: empty messages or content")
        return ""

    content = messages[0]["content"]
    if not isinstance(content, list):
        logger.warning(
            "run_inference_multi_image: content is not list (type=%s)",
            type(content).__name__,
        )
        return ""

    content_for_template: list = []
    images_list: list[Image.Image] = []

    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "image" and "image" in item:
            content_for_template.append({"type": "image"})
            images_list.append(item["image"])
        elif item.get("type") == "text" and "text" in item:
            content_for_template.append({"type": "text", "text": item["text"]})

    if not images_list:
        logger.warning(
            "run_inference_multi_image: no images in content (len=%s). "
            "Check loader: sample['images'] must be non-empty.",
            len(content),
        )
        return ""

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
        images=images_list,
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    inputs = _maybe_move_inputs_to_model_device(model, inputs)
    inputs.pop("token_type_ids", None)
    if _should_log_input_shapes():
        print_model_input_shapes(inputs, tag="qwen_multi_image")

    gen_kwargs = _sanitize_generation_kwargs(gen_cfg)
    if "pad_token_id" not in gen_kwargs:
        gen_kwargs["pad_token_id"] = (
            getattr(processor.tokenizer, "pad_token_id", None)
            or processor.tokenizer.eos_token_id
        )

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **gen_kwargs)
    print_model_output_tokens(
        generated_ids,
        input_length=inputs["input_ids"].shape[-1],
        tag="qwen_multi_image",
        batch=inputs,
    )

    from pipelines.run_model import normalize_assistant_output
    out = _extract_assistant_from_generated_ids(
        processor,
        generated_ids,
        fallback_decode="",
    )
    out = normalize_assistant_output(out)
    if not out:
        caption_fallback = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        out = caption_fallback.strip()

    return out or ""


def run_inference_multi_image_with_stats(
    model,
    processor,
    images: list[Image.Image],
    system_prompt: str,
    user_prompt: str,
    caption_prefix: str,
    gen_cfg: dict,
) -> tuple[str, dict[str, Any]]:
    """
    Frame sweep / GPU OOM 실험용:
    - caption 생성
    - input_length, frame_count, tokens_per_frame 같은 입력 규모를 함께 반환

    반환값은 (caption_text, stats_dict)
    """
    # caption_prefix는 (baseline prefix가 있다면) user_prompt 앞에 붙이는 용도
    full_user_text = caption_prefix
    if user_prompt:
        full_user_text = caption_prefix + "\n" + user_prompt

    content_for_template: list[dict[str, Any]] = []
    for _img in images:
        content_for_template.append({"type": "image"})
    content_for_template.append({"type": "text", "text": full_user_text})

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
        images=images,
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    # input_length는 디바이스 이동 전에 계산 (OOM이 inputs.to(device)에서 발생해도 로깅 가능)
    input_length = int(inputs["input_ids"].shape[-1])
    frame_count = len(images)
    tokens_per_frame = (float(input_length) / float(frame_count)) if frame_count > 0 else None

    try:
        inputs = _maybe_move_inputs_to_model_device(model, inputs)
        inputs.pop("token_type_ids", None)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        msg = str(e)
        is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or (
            isinstance(e, RuntimeError) and "out of memory" in msg.lower()
        )
        if not is_oom:
            raise
        oom_stats = {
            "frame_count": frame_count,
            "input_length": input_length,
            "tokens_per_frame": tokens_per_frame,
        }
        raise InferenceOOMError(msg, oom_stats) from e

    gen_kwargs = _sanitize_generation_kwargs(gen_cfg)
    if "pad_token_id" not in gen_kwargs:
        gen_kwargs["pad_token_id"] = (
            getattr(processor.tokenizer, "pad_token_id", None)
            or processor.tokenizer.eos_token_id
        )

    # input_length 계산 이후 generate에서 OOM이 나도 input 규모(stats)는 남기기 위해 예외를 래핑합니다.
    try:
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, **gen_kwargs)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        msg = str(e)
        is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or (
            isinstance(e, RuntimeError) and "out of memory" in msg.lower()
        )
        if not is_oom:
            raise
        oom_stats = {
            "frame_count": frame_count,
            "input_length": input_length,
            "tokens_per_frame": tokens_per_frame,
        }
        raise InferenceOOMError(msg, oom_stats) from e

    from pipelines.run_model import normalize_assistant_output

    out = _extract_assistant_from_generated_ids(
        processor,
        generated_ids,
        fallback_decode="",
    )
    if not out:
        caption_fallback = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        out = normalize_assistant_output(caption_fallback)

    stats = {
        "frame_count": frame_count,
        "input_length": input_length,
        "tokens_per_frame": tokens_per_frame,
    }
    return out or "", stats

def run_inference_video_clean(
    model,
    processor,
    video_path: str | Path,
    system_prompt: str,
    user_prompt: str,
    caption_prefix: str,
    gen_cfg: dict,
    fps: int | float = 1,
) -> str:
    """
    비디오 파일을 통째로 모델에 전달 (native video input).
    기본값으로 decord로 먼저 디코딩해 numpy를 넘기며(PyAV/swscale EAGAIN 회피),
    실패 시에는 파일 경로를 두고 Transformers 기본(torchvision) 경로를 씁니다.
    경로만 쓰려면 환경변수 VLM_NATIVE_VIDEO_BACKEND=path.
    """
    video_path = str(Path(video_path).resolve())

    full_user_text = caption_prefix
    if user_prompt:
        full_user_text = caption_prefix + "\n" + user_prompt

    video_payload, video_metadata_kw = _build_native_video_payload(video_path)

    messages = [
        {"role": "system", "content": system_prompt or ""},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_payload},
                {"type": "text", "text": full_user_text},
            ],
        },
    ]

    template_kw: dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "fps": fps,
    }
    if video_metadata_kw is not None:
        template_kw["video_metadata"] = video_metadata_kw

    inputs = processor.apply_chat_template(messages, **template_kw)

    inputs.pop("token_type_ids", None)

    inputs = _maybe_move_inputs_to_model_device(model, inputs)
    if _should_log_input_shapes():
        print_model_input_shapes(inputs, tag="qwen_video")

    _align_qwen3_video_grid_thw_for_rope(inputs, processor)
    return _generate_qwen3_after_chat_template(model, processor, inputs, gen_cfg)


def run_inference_native_video_with_images(
    model,
    processor,
    video_path: str | Path,
    images: Sequence[Image.Image],
    system_prompt: str,
    user_prompt: str,
    caption_prefix: str,
    gen_cfg: dict,
    fps: int | float = 1,
    images_before_video: bool = True,
) -> str:
    """
    네이티브 비디오 1개 + PIL 이미지 여러 장을 한 번에 입력 (Qwen3-VL).
    images_before_video=True: 참조 이미지 → 비디오 → 텍스트 (image_video_prompt 등에 맞춤).
    """
    if not images:
        raise ValueError("run_inference_native_video_with_images: images가 비었습니다.")
    video_path = str(Path(video_path).resolve())
    full_user_text = caption_prefix
    if user_prompt:
        full_user_text = caption_prefix + "\n" + user_prompt

    video_payload, video_metadata_kw = _build_native_video_payload(video_path)

    user_content: list[dict[str, Any]] = []
    if images_before_video:
        for img in images:
            user_content.append({"type": "image", "image": img})
        user_content.append({"type": "video", "video": video_payload})
    else:
        user_content.append({"type": "video", "video": video_payload})
        for img in images:
            user_content.append({"type": "image", "image": img})
    user_content.append({"type": "text", "text": full_user_text})

    messages = [
        {"role": "system", "content": system_prompt or ""},
        {"role": "user", "content": user_content},
    ]

    template_kw: dict[str, Any] = {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "fps": fps,
    }
    if video_metadata_kw is not None:
        template_kw["video_metadata"] = video_metadata_kw

    inputs = processor.apply_chat_template(messages, **template_kw)
    inputs.pop("token_type_ids", None)
    inputs = _maybe_move_inputs_to_model_device(model, inputs)
    if _should_log_input_shapes():
        print_model_input_shapes(inputs, tag="qwen_video_plus_images")

    _align_qwen3_video_grid_thw_for_rope(inputs, processor)
    return _generate_qwen3_after_chat_template(model, processor, inputs, gen_cfg)


def run_inference_video(
    model,
    processor,
    video_path: str | Path,
    system_prompt: str,
    user_prompt: str,
    caption_prefix: str,
    gen_cfg: dict,
    fps: int | float = 1,
) -> str:
    """백워드 호환 래퍼. 실제 구현은 `run_inference_video_clean`."""
    return run_inference_video_clean(
        model=model,
        processor=processor,
        video_path=video_path,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        caption_prefix=caption_prefix,
        gen_cfg=gen_cfg,
        fps=fps,
    )

