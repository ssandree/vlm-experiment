# models/qwen3_vl/qwen3_8b_adapter.py

from typing import Any, List, Sequence
from PIL import Image

from models.base_vlm import BaseVLM
from models.qwen3_vl.qwen_8b_load import load_qwen_model
from models.qwen3_vl.qwen_8b_inference import (
    run_inference,
    run_inference_multi_image,
    run_inference_native_video_with_images,
    run_inference_video_clean,
)


class Qwen3VLM(BaseVLM):
    def __init__(self, model_cfg, runtime_cfg):
        self.model, self.processor = load_qwen_model(
            model_cfg, runtime_cfg
        )
        self.model.eval()

    def generate(
        self,
        image: Image.Image | Sequence[Image.Image],
        system_prompt: str,
        user_prompt: str,
        caption_prefix: str,
        gen_cfg: dict,
    ) -> str:
        """
        Captioning / grounding / VQA 공통 단일 진입점.
        단일 이미지 및 multi-image 모두 지원.
        """
        full_user_text = caption_prefix
        if user_prompt:
            full_user_text = caption_prefix + "\n" + user_prompt

        if isinstance(image, (list, tuple)):
            content: List[dict[str, Any]] = []
            for img in image:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": full_user_text})
            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
            return run_inference_multi_image(
                model=self.model,
                processor=self.processor,
                messages=messages,
                system_prompt=system_prompt,
                gen_cfg=gen_cfg,
            )

        return run_inference(
            model=self.model,
            processor=self.processor,
            image=image,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            caption_prefix=caption_prefix,
            gen_cfg=gen_cfg,
        )

    def generate_multi_image(
        self,
        messages: list,
        system_prompt: str,
        gen_cfg: dict,
    ) -> str:
        return run_inference_multi_image(
            model=self.model,
            processor=self.processor,
            messages=messages,
            system_prompt=system_prompt,
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
        """비디오 파일을 통째로 모델에 전달 (native video input)."""
        return run_inference_video_clean(
            model=self.model,
            processor=self.processor,
            video_path=video_path,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            caption_prefix=caption_prefix,
            gen_cfg=gen_cfg,
            fps=fps,
        )

    def generate_video_with_images(
        self,
        video_path: str,
        images: list[Image.Image],
        system_prompt: str,
        user_prompt: str,
        caption_prefix: str,
        gen_cfg: dict,
        fps: int | float = 1,
    ) -> str:
        """네이티브 비디오 + 참조 이미지(들)를 한 프롬프트에 전달."""
        return run_inference_native_video_with_images(
            model=self.model,
            processor=self.processor,
            video_path=video_path,
            images=images,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            caption_prefix=caption_prefix,
            gen_cfg=gen_cfg,
            fps=fps,
            images_before_video=True,
        )

