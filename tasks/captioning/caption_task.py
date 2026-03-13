import os
from typing import Dict, Any, List

from tasks.base_task import BaseTask
from pipelines.run_model import normalize_assistant_output


def normalize_image_id(image_id: str) -> str:
    """
    Dataset-independent image_id normalization.
    - Flickr30k: "4567734402.jpg" -> "4567734402"
    - COCO: "391895" -> "391895"
    - ImageNet: "n01440764_18.JPEG" -> "n01440764_18"
    """
    return os.path.splitext(image_id)[0]


class CaptioningTask(BaseTask):
    """Captioning task implementation (inference-only in this repo)."""

    @property
    def task_name(self) -> str:
        return "captioning"

    def build_inputs(self, sample: Dict[str, Any], prompt_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build model inputs from a dataset sample.
        """
        caption_prefix = ""
        if "baseline" in prompt_cfg and "prefix" in prompt_cfg["baseline"]:
            caption_prefix = prompt_cfg["baseline"]["prefix"]

        image = sample.get("image")
        if image is None:
            image = sample.get("images")

        return {
            "image": image,
            "system_prompt": prompt_cfg.get("system_prompt", ""),
            "user_prompt": prompt_cfg.get("user_prompt", ""),
            "caption_prefix": caption_prefix,
        }

    def run_inference(
        self,
        model: Any,
        inputs: Dict[str, Any],
        generation_cfg: Dict[str, Any],
    ) -> str:
        """
        Run model inference for a single sample.
        """
        caption = model.run(
            task="captioning",
            image=inputs["image"],
            system_prompt=inputs["system_prompt"],
            user_prompt=inputs["user_prompt"],
            caption_prefix=inputs["caption_prefix"],
            gen_cfg=generation_cfg,
        )
        return normalize_assistant_output(caption)

    def evaluate(
        self,
        predictions: Dict[str, str],
        references: Dict[str, List[str]],
        image_ids: List[str],
        image_paths: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Evaluation is not wired in this experiment repo.
        """
        return {
            "num_samples": len(image_ids),
            "details": {},
        }

