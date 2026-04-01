# models/llava/llava_1.5_7b_adapter.py

from typing import List, Union
from PIL import Image
from models.base_vlm import BaseVLM
from models.llava.llava_load import load_llava
from models.llava.llava_inference import run_llava_inference
from pathlib import Path
from data.utils.frame_decoding import DecordFrameDecoder


class LLaVA15VLM(BaseVLM):
    def __init__(self, model_cfg, runtime_cfg):
        self.model, self.processor = load_llava(
            model_cfg, runtime_cfg
        )
        self.model.eval()

    def generate(
        self,
        image: Union[Image.Image, List[Image.Image]],
        system_prompt: str,
        user_prompt: str,
        caption_prefix: str = "",
        gen_cfg: dict = None,
    ) -> str:
        if gen_cfg is None:
            gen_cfg = {}
        # vlm_experiment 파이프라인: caption_prefix 지원
        full_user = (caption_prefix + "\n" + user_prompt).strip() if caption_prefix else user_prompt
        return run_llava_inference(
            model=self.model,
            processor=self.processor,
            image=image,
            system_prompt=system_prompt,
            user_prompt=full_user,
            gen_cfg=gen_cfg,
        )

    def generate_video(
        self,
        video_path: str,
        system_prompt: str,
        user_prompt: str,
        caption_prefix: str,
        gen_cfg: dict,
        fps: int | float = 1,
    ) -> str:
        """
        LLaVA-1.5는 native video 토큰을 지원하지 않으므로,
        지정한 fps로 비디오를 프레임 시퀀스로 디코드하여 다중 이미지 입력으로 추론합니다.
        """
        # 1) 비디오 메타 정보로 타임스탬프 생성
        try:
            from decord import VideoReader, cpu  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "generate_video requires 'decord'. Please install: pip install decord"
            ) from e

        vr = VideoReader(str(Path(video_path).resolve()), ctx=cpu(0))
        avg_fps = float(vr.get_avg_fps()) if vr.get_avg_fps() > 0 else 30.0
        num_frames = len(vr)
        duration_sec = num_frames / avg_fps if avg_fps > 0 else max(1.0, num_frames / 30.0)

        # fps 기준 균등 샘플 타임스탬프 (0 포함, duration 초과하지 않도록 클램프)
        if fps <= 0:
            fps = 1
        num_samples = max(1, int(duration_sec * float(fps)) + 1)
        timestamps = [min(i / float(fps), max(0.0, duration_sec - 1e-6)) for i in range(num_samples)]

        # 2) 프레임 디코딩 (PIL.Image 리스트)
        decoder = DecordFrameDecoder()
        images = decoder.decode(Path(video_path), timestamps)
        if not images:
            return ""

        # 3) caption_prefix + user_prompt 결합 후, 다중 이미지 경로로 추론 실행
        full_user = caption_prefix.strip()
        if user_prompt:
            full_user = (caption_prefix + "\n" + user_prompt).strip() if caption_prefix else user_prompt

        return run_llava_inference(
            model=self.model,
            processor=self.processor,
            image=images,
            system_prompt=system_prompt or "",
            user_prompt=full_user,
            gen_cfg=gen_cfg or {},
        )

    def generate_video_with_images(
        self,
        video_path: str,
        images: List[Image.Image],
        system_prompt: str,
        user_prompt: str,
        caption_prefix: str,
        gen_cfg: dict,
        fps: int | float = 1,
    ) -> str:
        raise NotImplementedError(
            "LLaVA는 Qwen3-VL과 같은 네이티브 비디오+이미지 동시 입력을 지원하지 않습니다. "
            "video.input_mode: sampling 을 사용하거나 Qwen 모델을 선택하세요."
        )
