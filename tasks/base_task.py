from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseTask(ABC):
    """Base class for all tasks (captioning, grounding, VQA, etc.)."""

    @property
    @abstractmethod
    def task_name(self) -> str:
        ...

    @abstractmethod
    def build_inputs(self, sample: Dict[str, Any], prompt_cfg: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @abstractmethod
    def run_inference(self, model: Any, inputs: Dict[str, Any], generation_cfg: Dict[str, Any]) -> Any:  # type: ignore[name-defined]
        ...

    @abstractmethod
    def evaluate(self, predictions: Dict[str, Any], references: Dict[str, Any], image_ids: List[str]) -> Dict[str, Any]:
        ...

