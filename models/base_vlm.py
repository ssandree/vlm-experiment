import logging
from abc import ABC, abstractmethod
from PIL import Image

logger = logging.getLogger(__name__)


class BaseVLM(ABC):
    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        caption_prefix: str,
        gen_cfg: dict,
    ) -> str:
        """Return caption text only"""
        ...

    def run(self, task: str, **kwargs):
        """
        Unified interface for task-based inference.
        """
        if task == "captioning":
            return self.run_captioning(**kwargs)
        elif task == "grounding":
            return self.run_grounding(**kwargs)
        elif task == "vqa":
            return self.run_vqa(**kwargs)
        elif task == "mi_vqa":
            return self.run_mivqa(**kwargs)
        else:
            raise NotImplementedError(f"Task '{task}' not implemented")

    def run_captioning(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        caption_prefix: str,
        gen_cfg: dict,
    ) -> str:
        return self.generate(
            image=image,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            caption_prefix=caption_prefix,
            gen_cfg=gen_cfg,
        )

    def run_grounding(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        phrase: str = None,
        gen_cfg: dict = None,
    ) -> str:
        return self.generate(
            image=image,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            caption_prefix="",
            gen_cfg=gen_cfg or {},
        )

    def run_vqa(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        gen_cfg: dict = None,
    ) -> str:
        return self.generate(
            image=image,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            caption_prefix="",
            gen_cfg=gen_cfg or {},
        )

    def run_mivqa(
        self,
        messages: list,
        system_prompt: str,
        gen_cfg: dict = None,
    ) -> str:
        return self.generate_multi_image(
            messages=messages,
            system_prompt=system_prompt or "",
            gen_cfg=gen_cfg or {},
        )

    def run_video(
        self,
        video_path: str,
        system_prompt: str,
        user_prompt: str,
        caption_prefix: str,
        gen_cfg: dict,
        fps: int | float = 1,
    ) -> str:
        """비디오 파일을 통째로 모델에 전달 (native video input)."""
        return self.generate_video(
            video_path=video_path,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            caption_prefix=caption_prefix,
            gen_cfg=gen_cfg,
            fps=fps,
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
        """Override: native video inference. 기본 구현 없음."""
        raise NotImplementedError("This model does not support native video input")

    def generate_multi_image(
        self,
        messages: list,
        system_prompt: str,
        gen_cfg: dict,
    ) -> str:
        """
        Multi-image inference fallback: use first image and concat texts.
        """
        if not messages or not messages[0].get("content"):
            logger.warning("[MI_VQA_DEBUG] generate_multi_image: empty messages or content")
            return ""
        content = messages[0]["content"]
        first_image = None
        text_parts = []
        num_images = 0
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "image" and "image" in item:
                    num_images += 1
                    if first_image is None:
                        first_image = item["image"]
                elif item.get("type") == "text" and "text" in item:
                    text_parts.append(item["text"])
        user_prompt = "\n".join(text_parts) if text_parts else ""
        if first_image is None:
            logger.warning("[MI_VQA_DEBUG] generate_multi_image: no image in content")
            return ""
        logger.info(
            "[MI_VQA_DEBUG] generate_multi_image fallback: num_images=%s user_prompt_len=%s",
            num_images,
            len(user_prompt),
        )
        return self.generate(
            image=first_image,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            caption_prefix="",
            gen_cfg=gen_cfg,
        )

