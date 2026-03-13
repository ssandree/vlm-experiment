# pipelines/run_model.py

from typing import List, Dict, Union
from PIL import Image
from models.base_vlm import BaseVLM


def normalize_assistant_output(text: str) -> str:
    """Assistant 응답만 추출. Qwen 특수 토큰 제거."""
    if text is None:
        return ""
    if not isinstance(text, str):
        return str(text)

    for marker in ("<|im_start|>assistant\n", "<|im_start|>assistant\r\n", "<|im_start|>assistant"):
        if marker in text:
            text = text.rsplit(marker, 1)[-1]
            break
    else:
        if "assistant\n" in text:
            text = text.rsplit("assistant\n", 1)[-1]

    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0]

    for token in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
        text = text.replace(token, "")

    return text.strip() or ""


def run_model(
    vlm: BaseVLM,
    images: Union[List[Image.Image], List[List[Image.Image]]],
    image_ids: List[str],
    system_prompt: str,
    user_prompt: str,
    caption_prefix: str = "",
    generation_cfg: dict = None,
) -> Dict[str, str]:
    """
    이미지 1장당 추론 1회. 각 (image, image_id)에 대해 generate 호출.
    """
    if generation_cfg is None:
        generation_cfg = {}

    assert len(images) == len(image_ids)

    outputs: Dict[str, str] = {}

    for image, image_id in zip(images, image_ids):
        caption = vlm.generate(
            image=image,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            caption_prefix=caption_prefix,
            gen_cfg=generation_cfg,
        )
        outputs[image_id] = normalize_assistant_output(caption) if caption else ""

    return outputs


def run_model_multi_image(
    vlm: BaseVLM,
    image_groups: List[List[Image.Image]],
    group_ids: List[str],
    system_prompt: str,
    user_prompt: str,
    caption_prefix: str = "",
    generation_cfg: dict = None,
) -> Dict[str, str]:
    """
    그룹당 추론 1회. 여러 이미지를 한 번에 모델에 넣어 하나의 추론 결과 생성.
    image_groups: [[img1, img2, img3, img4], [img5, img6, ...], ...]
    """
    if generation_cfg is None:
        generation_cfg = {}

    assert len(image_groups) == len(group_ids)

    outputs: Dict[str, str] = {}

    for images, group_id in zip(image_groups, group_ids):
        caption = vlm.generate(
            image=images,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            caption_prefix=caption_prefix,
            gen_cfg=generation_cfg,
        )
        outputs[group_id] = normalize_assistant_output(caption) if caption else ""

    return outputs

