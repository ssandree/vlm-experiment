# Qwen3-VL-8B inference (pipeline-safe)

import logging
from pathlib import Path

import torch
from PIL import Image

from models.inference_input_log import print_model_input_shapes

logger = logging.getLogger(__name__)


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
    ).to(model.device)

    inputs.pop("token_type_ids", None)
    print_model_input_shapes(inputs, tag="qwen_single_image")

    gen_kwargs = dict(gen_cfg)
    if "pad_token_id" not in gen_kwargs:
        gen_kwargs["pad_token_id"] = getattr(
            processor.tokenizer, "pad_token_id", None
        ) or processor.tokenizer.eos_token_id

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    from pipelines.run_model import normalize_assistant_output

    full_decode = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]
    out = _extract_assistant_from_generated_ids(
        processor,
        generated_ids,
        fallback_decode=full_decode,
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
    ).to(model.device)
    inputs.pop("token_type_ids", None)
    print_model_input_shapes(inputs, tag="qwen_multi_image")

    gen_kwargs = dict(gen_cfg)
    if "pad_token_id" not in gen_kwargs:
        gen_kwargs["pad_token_id"] = (
            getattr(processor.tokenizer, "pad_token_id", None)
            or processor.tokenizer.eos_token_id
        )

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    from pipelines.run_model import normalize_assistant_output
    full_decode = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0]
    out = normalize_assistant_output(full_decode)
    if not out:
        caption_fallback = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        out = caption_fallback.strip()

    return out or ""


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
    """
    비디오 파일을 통째로 모델에 전달 (native video input).
    sampling/decoding/aggregation 없이 processor가 비디오를 직접 처리.
    """
    video_path = str(Path(video_path).resolve())

    full_user_text = caption_prefix
    if user_prompt:
        full_user_text = caption_prefix + "\n" + user_prompt

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

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        fps=fps,
    )
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(model.device)
    print_model_input_shapes(inputs, tag="qwen_video")

    input_length = inputs["input_ids"].shape[1]

    gen_kwargs = dict(gen_cfg)
    if "pad_token_id" not in gen_kwargs:
        gen_kwargs["pad_token_id"] = (
            getattr(processor.tokenizer, "pad_token_id", None)
            or processor.tokenizer.eos_token_id
        )

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    # 비디오 입력 시 input_ids 길이로 생성된 토큰만 잘라 디코딩.
    # 전체 시퀀스를 디코딩하면 vision/video 토큰이 garbage로 출력됨.
    # (Hugging Face Qwen2-VL 공식 문서 권장 방식)
    generated_only = generated_ids[:, input_length:]
    if generated_only.shape[1] == 0:
        return ""
    decoded = processor.batch_decode(
        generated_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    out = (decoded[0] or "").strip()
    for token in ("<|im_end|>", "<|endoftext|>", "<|im_start|>", "<|file_sep|>"):
        out = out.replace(token, "")
    return out.strip() or ""

