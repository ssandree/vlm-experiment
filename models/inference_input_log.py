"""Processor 출력(배치)의 텐서 shape 로깅 — 추론 직전 토큰/비전 입력 확인용."""

from __future__ import annotations

from typing import Any, Mapping


def _get_item(batch: Any, key: str) -> Any:
    if isinstance(batch, Mapping):
        return batch.get(key)
    return getattr(batch, key, None)


def _shape_str(x: Any) -> str:
    if x is None:
        return "None"
    if hasattr(x, "shape"):
        return str(tuple(x.shape))
    return f"type={type(x).__name__}"


def print_model_input_shapes(batch: Any, tag: str = "inference") -> None:
    """
    image/video + prompt 를 processor 에 넣은 직후 배치의 shape 를 출력한다.
    input_ids 가 있으면 사용자가 요청한 형태로 shape 를 명시적으로 출력한다.
    """
    prefix = f"[VLM inputs:{tag}]"
    if batch is None:
        print(f"{prefix} batch is None")
        return

    input_ids = _get_item(batch, "input_ids")
    if input_ids is not None:
        # 요청 예: print(inputs["input_ids"].shape)
        print(f"{prefix} inputs['input_ids'].shape = {tuple(input_ids.shape)}")
    else:
        print(f"{prefix} inputs['input_ids'] missing")

    extra_keys = (
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
    )
    parts = []
    for k in extra_keys:
        v = _get_item(batch, k)
        if v is not None:
            parts.append(f"{k}={_shape_str(v)}")
    if parts:
        print(f"{prefix} " + ", ".join(parts))
