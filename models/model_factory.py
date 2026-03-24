from models.qwen3_vl.qwen3_8b_adapter import Qwen3VLM


def build_vlm(model_cfg: dict, runtime_cfg: dict):
    """
    Build VLM instance from model_cfg.

    지원 모델: Qwen3-VL-8B, LLaVA-1.5-7B
    """
    name = model_cfg["name"].lower()

    if "qwen" in name:
        return Qwen3VLM(model_cfg, runtime_cfg)

    if "llava" in name:
        from models.llava.llava_adapter import LLaVA15VLM
        return LLaVA15VLM(model_cfg, runtime_cfg)

    raise ValueError(f"Unsupported model in this repo: {model_cfg['name']}")

