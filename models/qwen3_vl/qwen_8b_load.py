import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import BitsAndBytesConfig


def load_qwen_model(model_cfg: dict, runtime_cfg: dict):
    """
    Load Qwen3-VL-8B model using unified (flat) model_cfg schema.
    """

    precision_cfg = model_cfg.get("precision", {})
    precision_runtime = runtime_cfg.get("precision", {})
    runtime_device = runtime_cfg.get("device", {})

    pretrained_path = model_cfg["resolved_root"]
    print(f"[Model Load] Using LOCAL model: {pretrained_path}")

    dtype_str = precision_runtime.get(
        "dtype", precision_cfg.get("dtype", "float16")
    )

    use_int8 = dtype_str == "int8"

    torch_dtype = torch.float16

    quantization_config = None
    if use_int8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_enable_fp32_cpu_offload=False,
        )
        print("[Model Load] INT8 enabled (bitsandbytes)")

    model = AutoModelForImageTextToText.from_pretrained(
        pretrained_path,
        torch_dtype=torch_dtype,
        device_map=runtime_device.get("device_map", "auto"),
        quantization_config=quantization_config,
        trust_remote_code=True,
        local_files_only=True,
    )

    vision_cfg = model_cfg.get("vision", {})
    image_size = int(vision_cfg.get("image_size", 336))
    min_pixels = image_size * image_size
    max_pixels = max(min_pixels, 2560 * 2560)
    processor = AutoProcessor.from_pretrained(
        pretrained_path,
        trust_remote_code=True,
        local_files_only=True,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    model.eval()
    return model, processor

