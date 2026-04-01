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


def print_model_output_tokens(
    generated_ids: Any,
    input_length: int | None = None,
    tag: str = "inference",
    max_ids_to_print: int = 64,
    batch: Any = None,
) -> None:
    """
    model.generate() 결과에서 최종 토큰 정보를 출력한다.
    - input_length 가 주어지면 prompt 이후(generated-only) 토큰 기준으로 출력
    - 토큰 id는 길 때 앞/뒤 일부만 출력
    """
    prefix = f"[VLM output:{tag}]"
    if generated_ids is None or not hasattr(generated_ids, "shape"):
        print(f"{prefix} generated_ids missing")
        return
    if len(generated_ids.shape) != 2 or generated_ids.shape[0] < 1:
        print(f"{prefix} generated_ids.shape = {_shape_str(generated_ids)}")
        return

    seq = generated_ids[0]
    if input_length is None:
        final_ids = seq
    else:
        final_ids = seq[input_length:]

    if hasattr(final_ids, "detach"):
        ids = final_ids.detach().cpu().tolist()
    else:
        ids = list(final_ids)

    total = len(ids)
    frame_count = None
    grid_summary = "N/A"
    if batch is not None:
        video_grid = _get_item(batch, "video_grid_thw")
        image_grid = _get_item(batch, "image_grid_thw")
        grid = video_grid if video_grid is not None else image_grid
        if grid is not None and hasattr(grid, "shape") and len(getattr(grid, "shape", ())) == 2:
            try:
                grid_cpu = grid.detach().cpu()
                rows = int(grid_cpu.shape[0])
                t_sum = int(grid_cpu[:, 0].sum().item())
                h_min = int(grid_cpu[:, 1].min().item())
                h_max = int(grid_cpu[:, 1].max().item())
                w_min = int(grid_cpu[:, 2].min().item())
                w_max = int(grid_cpu[:, 2].max().item())
                frame_count = t_sum
                modal = "video" if video_grid is not None else "image"
                grid_summary = (
                    f"{modal}:rows={rows},t_sum={t_sum},"
                    f"h=[{h_min},{h_max}],w=[{w_min},{w_max}]"
                )
            except Exception:
                grid_summary = _shape_str(grid)

    tokens_per_frame = None
    if frame_count and frame_count > 0 and input_length is not None:
        tokens_per_frame = float(input_length) / float(frame_count)
    print(
        f"{prefix} total_output_tokens={total}, "
        f"input_length={input_length if input_length is not None else 'N/A'}, "
        f"generated_ids_shape={tuple(generated_ids.shape)}"
    )
    if frame_count is not None:
        if tokens_per_frame is not None:
            print(
                f"{prefix} frame_count={frame_count}, grid_thw={grid_summary}, "
                f"tokens_per_frame={tokens_per_frame:.2f}"
            )
        else:
            print(f"{prefix} frame_count={frame_count}, grid_thw={grid_summary}")
    if total <= max_ids_to_print:
        print(f"{prefix} token_ids={ids}")
        return

    head_n = max_ids_to_print // 2
    tail_n = max_ids_to_print - head_n
    head_ids = ids[:head_n]
    tail_ids = ids[-tail_n:]
    print(
        f"{prefix} token_ids(head {head_n})={head_ids} ... "
        f"token_ids(tail {tail_n})={tail_ids}"
    )
