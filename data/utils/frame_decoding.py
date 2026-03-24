"""
Frame decoding utilities for video processing.

Decord 기반 FrameDecoder 구현.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image

from typing import Protocol, Any


class FrameDecoder(Protocol):
    """
    교체 가능한 비디오 디코더 인터페이스.
    """

    def decode(self, video_path: Path, timestamps: List[float]) -> List[Any]:  # pragma: no cover - interface
        ...


class DecordFrameDecoder(FrameDecoder):
    def __init__(self, ctx_device: int = 0):
        try:
            from decord import VideoReader, cpu, bridge  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "DecordFrameDecoder 를 사용하려면 'decord' 패키지가 필요합니다. "
                "예: pip install decord"
            ) from e
        self._VideoReader = VideoReader
        self._ctx = cpu(ctx_device)
        bridge.set_bridge("native")

    def decode(self, video_path: Path, timestamps: List[float]) -> List[Image.Image]:
        vr = self._VideoReader(str(video_path), ctx=self._ctx)
        num_frames = len(vr)
        fps = float(vr.get_avg_fps()) if vr.get_avg_fps() > 0 else 30.0

        images: List[Image.Image] = []
        for t in timestamps:
            frame_idx = int(round(t * fps))
            frame_idx = max(0, min(frame_idx, num_frames - 1))
            frame = vr[frame_idx]
            frame_np = frame.asnumpy()
            if frame_np.dtype != "uint8":
                if frame_np.dtype.kind == "f":
                    if frame_np.max() <= 1.0:
                        frame_np = (frame_np * 255.0).clip(0, 255).astype("uint8")
                    else:
                        frame_np = frame_np.clip(0, 255).astype("uint8")
                else:
                    frame_np = frame_np.astype("uint8")
            pil_img = Image.fromarray(frame_np)
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            images.append(pil_img)

        return images

