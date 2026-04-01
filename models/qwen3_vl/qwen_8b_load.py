import gc
import os
import torch
import torch.distributed as dist
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import BitsAndBytesConfig

def _resolve_device_map(runtime_cfg: dict) -> tuple[str | None, dict[str, str] | None]:
    """
    - 기본(기존 동작): `device_map=None` 이면 단일 GPU로 적재 후 `model.to("cuda")` 사용
    - 멀티 GPU: 가시 GPU 수가 2장 이상이면 기본적으로 `device_map="auto"` 사용

    runtime_cfg 옵션(선택):
    - device_map: None | "auto" | ...
    - max_memory_gb: 각 GPU에 대한 GiB 상한(예: 20 -> "20GiB")
    """
    # runtime override 우선
    runtime_device_map = runtime_cfg.get("device_map", None)
    if runtime_device_map is not None:
        if runtime_device_map == "auto":
            max_memory_gb = runtime_cfg.get("max_memory_gb")
            if max_memory_gb is None:
                return "auto", None
            num_gpus = torch.cuda.device_count()
            if num_gpus <= 0:
                return "auto", None
            return (
                "auto",
                {str(i): f"{int(max_memory_gb)}GiB" for i in range(num_gpus)},
            )
        return str(runtime_device_map), None

    num_gpus = torch.cuda.device_count()
    use_multi_gpu = bool(runtime_cfg.get("use_multi_gpu", True))
    if use_multi_gpu and num_gpus >= 2:
        # CUDA_VISIBLE_DEVICES를 통해 "가시 GPU"만 device_map에 반영됨
        max_memory_gb = runtime_cfg.get("max_memory_gb")
        if max_memory_gb is not None:
            return (
                "auto",
                {str(i): f"{int(max_memory_gb)}GiB" for i in range(num_gpus)},
            )
        return "auto", None

    return None, None

def _resolve_torch_dtype(dtype_str: str) -> torch.dtype:
    key = (dtype_str or "float16").lower()
    if key in ("float16", "fp16", "half"):
        return torch.float16
    if key in ("bfloat16", "bf16"):
        return torch.bfloat16
    if key in ("float32", "fp32"):
        return torch.float32
    # int8 양자화 시 compute dtype은 fp16로 유지
    if key == "int8":
        return torch.float16
    raise ValueError(f"Unsupported precision.dtype: {dtype_str}")


def load_qwen_model(model_cfg: dict, runtime_cfg: dict):
    """
    Load Qwen3-VL-8B model using unified (flat) model_cfg schema.
    """

    precision_cfg = model_cfg.get("precision", {})

    pretrained_path = model_cfg["resolved_root"]
    print(f"[Model Load] Using LOCAL model: {pretrained_path}")

    dtype_str = precision_cfg.get("dtype", "float16")
    if isinstance(dtype_str, str):
        dtype_str = dtype_str.lower()

    use_int8 = dtype_str == "int8"
    torch_dtype = _resolve_torch_dtype(dtype_str)

    quantization_config = None
    if use_int8:
        int8_threshold = float(precision_cfg.get("llm_int8_threshold", 6.0))
        int8_cpu_offload = bool(
            precision_cfg.get("llm_int8_enable_fp32_cpu_offload", True)
        )
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=int8_threshold,
            llm_int8_enable_fp32_cpu_offload=False,
        )
        print(
            "[Model Load] INT8 enabled (bitsandbytes, "
            f"threshold={int8_threshold}, cpu_offload=False)"
        )

    # ------------------------------------------------------------
    # 기존 HF device_map 기반 path (기존 동작 유지)
    # ------------------------------------------------------------
    device_map, max_memory = _resolve_device_map(runtime_cfg)
    num_gpus = torch.cuda.device_count()
    print(
        f"[Model Load] device_map={device_map!r} "
        f"(visible_gpus={num_gpus}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r})"
    )

    model = AutoModelForImageTextToText.from_pretrained(
        pretrained_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=quantization_config,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    hf_device_map = getattr(model, "hf_device_map", None)

    # `device_map="auto"`인 경우, vision/visual 쪽이 GPU0에 몰리면 activation이 GPU0에서 터질 수 있습니다.
    # 로딩 후 hf_device_map을 보고 vision/visual 모듈이 주로 cuda:0에 있으면 다른 GPU로 강제 이동합니다.
    balance_vision_device = bool(runtime_cfg.get("balance_vision_device", True))
    if balance_vision_device and device_map == "auto" and hf_device_map and num_gpus >= 2:
        vision_keys = [
            k
            for k in hf_device_map.keys()
            if ("visual" in k.lower() or "vision" in k.lower())
        ]
        if vision_keys:
            # hf_device_map 값은 int(예: 0) 또는 "cuda:0" 형태가 섞일 수 있음
            def _norm_to_int_device(v) -> int | None:
                if isinstance(v, int):
                    return v
                if isinstance(v, str):
                    s = v.strip()
                    if s.isdigit():
                        return int(s)
                    if s.startswith("cuda:"):
                        try:
                            return int(s.split(":", 1)[1])
                        except Exception:
                            return None
                return None

            vision_dev_idxs = []
            for k in vision_keys:
                idx = _norm_to_int_device(hf_device_map[k])
                if idx is not None:
                    vision_dev_idxs.append(idx)

            if vision_dev_idxs:
                # 가장 많이 배정된 디바이스가 GPU0(=OOM 지점)일 확률이 높음
                src_idx = max(set(vision_dev_idxs), key=vision_dev_idxs.count)

                dst_idx = runtime_cfg.get("vision_target_device_idx", None)
                if dst_idx is None:
                    # src와 다른 첫 후보
                    candidates = [i for i in range(num_gpus) if i != int(src_idx)]
                    if not candidates:
                        dst_idx = src_idx
                    else:
                        dst_idx = candidates[0]
                else:
                    dst_idx = int(dst_idx)
                    if dst_idx == int(src_idx):
                        # 같은 디바이스면 다른 후보 선택
                        candidates = [i for i in range(num_gpus) if i != int(src_idx)]
                        dst_idx = candidates[0] if candidates else src_idx

                # vision/visual 관련 키들만 재배치 (hf_device_map의 값 타입은 대개 int)
                new_device_map = dict(hf_device_map)
                for k in vision_keys:
                    new_device_map[k] = int(dst_idx)

                if int(dst_idx) != int(src_idx):
                    print(
                        f"[Model Load] Rebalancing vision modules: "
                        f"src={src_idx}, dst={dst_idx} (vision_keys={len(vision_keys)})"
                    )
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                    model = AutoModelForImageTextToText.from_pretrained(
                        pretrained_path,
                        torch_dtype=torch_dtype,
                        device_map=new_device_map,
                        max_memory=max_memory,
                        quantization_config=quantization_config,
                        trust_remote_code=True,
                        local_files_only=True,
                        low_cpu_mem_usage=True,
                    )

    # device_map="auto" / sharded이면 이미 여러 GPU에 분산 배치될 수 있어 `.to("cuda")`를 호출하면 안 됩니다.
    if device_map is None:
        model = model.to("cuda")

    # 추론 코드에서 inputs를 model.device로 강제 이동하지 않도록 플래그 저장
    setattr(model, "_vlm_device_map_choice", device_map)

    vision_cfg = model_cfg.get("vision", {}) or {}
    processor_kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
    }
    min_pixels = vision_cfg.get("min_pixels")
    if min_pixels is not None:
        processor_kwargs["min_pixels"] = int(min_pixels)
    max_pixels = vision_cfg.get("max_pixels")
    if max_pixels is not None:
        max_pixels = int(max_pixels)
        if min_pixels is not None and max_pixels < int(min_pixels):
            max_pixels = int(min_pixels)
        processor_kwargs["max_pixels"] = max_pixels

    processor = AutoProcessor.from_pretrained(
        pretrained_path,
        **processor_kwargs,
    )

    model.eval()
    return model, processor

