def build_vlm(model_cfg: dict, runtime_cfg: dict):
    """
    Build VLM instance from model_cfg.

    지원 모델: Qwen3-VL-8B, LLaVA-1.5-7B

    Qwen/LLaVA 는 import 시 torch·transformers 를 끌어오므로 여기서만 지연 import 한다.
    (모듈 최상단에서 Qwen3VLM 을 import 하면 run_* 진입 전 수 분이 말 없이 걸림)
    """
    name = model_cfg["name"].lower()

    if "qwen" in name:
        from models.qwen3_vl.qwen3_8b_adapter import Qwen3VLM

        return Qwen3VLM(model_cfg, runtime_cfg)

    if "llava" in name:
        from models.llava.llava_adapter import LLaVA15VLM
        return LLaVA15VLM(model_cfg, runtime_cfg)

    raise ValueError(f"Unsupported model in this repo: {model_cfg['name']}")

