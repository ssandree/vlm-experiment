from models.qwen3_vl.qwen3_8b_adapter import Qwen3VLM


def build_vlm(model_cfg: dict, runtime_cfg: dict):
    """
    Build VLM instance from model_cfg.

    이 실험 리포에서는 Qwen3-VL-8B만 지원합니다.
    """
    name = model_cfg["name"].lower()

    if "qwen" in name:
        return Qwen3VLM(model_cfg, runtime_cfg)

    raise ValueError(f"Unsupported model in this repo: {model_cfg['name']}")

