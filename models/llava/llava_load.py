# models/llava/llava_load.py

import torch
from transformers import BitsAndBytesConfig, LlavaProcessor, LlavaForConditionalGeneration


def load_llava(model_cfg: dict, runtime_cfg: dict):
    model_path = model_cfg["resolved_root"]
    print(f"[Model Load] Using LOCAL model: {model_path}")

    precision_cfg = model_cfg.get("precision", {})
    _ = runtime_cfg  # reserved for future runtime overrides

    # -------------------------
    # precision
    # -------------------------
    dtype_str = precision_cfg.get("dtype", "float16").lower()
    use_int8 = dtype_str == "int8"

    torch_dtype = torch.float16
    quantization_config = None

    if use_int8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        print("[Model Load] INT8 enabled (LLaVA)")

    # -------------------------
    # processor (config의 vision.image_size는 LlavaProcessor에서 별도 인자 미지원)
    # -------------------------
    processor = LlavaProcessor.from_pretrained(
        model_path,
        local_files_only=True,
    )

    # -------------------------
    # model
    # -------------------------
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch_dtype,
        device_map="auto",
        quantization_config=quantization_config,
    )

    model.eval()
    return model, processor
