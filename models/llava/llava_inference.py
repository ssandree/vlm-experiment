# models/llava/llava_1.5_7b_inference.py

import torch
from typing import List, Union
from PIL import Image

from models.inference_input_log import print_model_input_shapes


def run_llava_inference(
    model,
    processor,
    image: Union[Image.Image, List[Image.Image]],
    system_prompt: str,
    user_prompt: str,
    gen_cfg: dict,
) -> str:
    """
    LLaVA-1.5 inference (adapter-compatible)

    Supports single image or multiple images. The number of <image> placeholders
    in the prompt MUST match the number of images (processor expands each
    placeholder to ~576 tokens; vision encoder outputs that many features per image).
    """

    # Normalize to list (single image -> [image])
    images = [image] if isinstance(image, Image.Image) else list(image)
    if not images:
        raise ValueError("At least one image is required")

    # Remove <image> token if present (model-agnostic prompts may contain it)
    cleaned_user_prompt = user_prompt.replace("<image>", "").strip()
    merged_prompt = f"{system_prompt}\n{cleaned_user_prompt}".strip()

    # LLaVA chat format: N images -> N {"type": "image"} placeholders.
    # Processor replaces each <image> with num_image_tokens (576) placeholders.
    # Vision encoder outputs len(images) * 576 features. Must match.
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
    ).to(model.device)

    print_model_input_shapes(inputs, tag="llava")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_cfg.get("max_new_tokens", 64),
            do_sample=gen_cfg.get("do_sample", False),
            temperature=gen_cfg.get("temperature", 0.0),  # ⭐ evaluation 안정성
            use_cache=True,
        )

    # prompt 길이 이후만 디코드
    output_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    caption = processor.tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
    )

    return caption.strip()
