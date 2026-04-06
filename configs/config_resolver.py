import os
from pathlib import Path

from utils.load_yaml import load_yaml

_CONFIG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CONFIG_DIR.parent


def _resolve_paths_yaml_path() -> Path:
    """
    우선순위: 환경변수 VLM_PATHS_FILE → configs/paths.local.yaml(존재 시) → configs/paths.yaml
    상대 경로는 저장소 루트 기준으로 해석합니다.
    """
    env = os.environ.get("VLM_PATHS_FILE", "").strip()
    if env:
        p = Path(env)
        return p if p.is_absolute() else (_REPO_ROOT / p).resolve()
    local = _CONFIG_DIR / "paths.local.yaml"
    if local.is_file():
        return local
    return _CONFIG_DIR / "paths.yaml"


class ConfigResolver:
    def __init__(self, experiment_path: str):
        self.exp_cfg = load_yaml(experiment_path)

        paths_file = _resolve_paths_yaml_path()
        if not paths_file.is_file():
            raise FileNotFoundError(
                f"Paths config not found: {paths_file}. "
                "Set VLM_PATHS_FILE or add configs/paths.local.yaml (see paths.local.example.yaml)."
            )
        self.paths_cfg = load_yaml(str(paths_file))
        self.model_name = self.exp_cfg["model"]
        dataset_val = self.exp_cfg["dataset"]
        if isinstance(dataset_val, dict):
            self.dataset_name = (
                dataset_val.get("name") or dataset_val.get("mode") or dataset_val.get("dataset")
            )
        else:
            self.dataset_name = dataset_val
        if not self.dataset_name:
            raise ValueError("experiment.yaml: dataset.name is required")
        self.task_name = self.exp_cfg.get("task", "captioning")  # Default to captioning for backward compatibility
        prompt_val = self.exp_cfg["prompt"]
        if isinstance(prompt_val, dict):
            self.prompt_name = prompt_val.get("name") or prompt_val.get("prompt")
        else:
            self.prompt_name = prompt_val
        if not self.prompt_name:
            raise ValueError("experiment.yaml: prompt.name is required")

        self.model_cfg = load_yaml(f"configs/model/{self.model_name}.yaml")
        self.dataset_cfg = self._load_dataset_cfg(self.dataset_name)
        self.prompt_cfg = load_yaml(f"configs/prompt/{self.prompt_name}.yaml")
        # 실험 제어 옵션은 `configs/experiment.yaml`의 `runtime:` 섹션을 통해서만 반영합니다.
        self.runtime_cfg = dict(self.exp_cfg.get("runtime") or {})

        self._resolve_paths()
        self._merge_experiment_dataset_cfg()  # ⭐ 추가
        self._resolve_video_cfg()  # ⭐ 추가

    def _normalize_sampling_cfg(
        self,
        sampling_cfg: dict,
        *,
        datasets_root: Path,
        annotation_path: Path | None,
        fallback_uniform_when_segment_aware_without_annotation: bool,
    ) -> dict:
        """
        run_inferences.video_common.normalize_sampling_cfg 와 동일한 의미론을
        config 레벨에서 미리 정규화합니다.
        """
        cfg = dict(sampling_cfg or {})

        # manual sampling: datasets_root 기준으로 경로 해석
        if cfg.get("type") == "manual" and cfg.get("manual_frame_map_path"):
            p = Path(cfg["manual_frame_map_path"])
            if not p.is_absolute():
                cfg["manual_frame_map_path"] = str(datasets_root / p)

        strategy_type = (cfg.get("type") or "").strip().lower()

        # segment_aware sampling: annotation_path 주입 또는 fallback
        if strategy_type == "segment_aware":
            if annotation_path is not None:
                cfg["annotation_path"] = str(annotation_path)
            elif fallback_uniform_when_segment_aware_without_annotation:
                cfg["type"] = "uniform"
                cfg["num_frames"] = int(cfg.get("num_frames", 4))

        return cfg

    def _resolve_video_cfg(self) -> None:
        """
        video / prompt / dataset / sampling 간의 관계를 엔트리포인트에서 흩어지지 않게,
        ConfigResolver에서 한 번에 정리합니다.
        """
        raw_video_cfg = self.exp_cfg.get("video") or self.resolved_dataset.get("video") or {}

        datasets_root = Path(self.paths_cfg["paths"]["datasets_root"])

        ann_path = self.resolved_dataset.get("paths", {}).get("annotation")
        annotation_path = Path(ann_path) if ann_path is not None else None

        sampling_raw = dict(raw_video_cfg.get("sampling") or {})
        sampling_cfg_strict = self._normalize_sampling_cfg(
            sampling_raw,
            datasets_root=datasets_root,
            annotation_path=annotation_path,
            fallback_uniform_when_segment_aware_without_annotation=False,
        )
        sampling_cfg_fallback_uniform = self._normalize_sampling_cfg(
            sampling_raw,
            datasets_root=datasets_root,
            annotation_path=annotation_path,
            fallback_uniform_when_segment_aware_without_annotation=True,
        )

        output_level = raw_video_cfg.get("output_level", "segment")
        input_mode = raw_video_cfg.get("input_mode", "full")
        aggregation_cfg = raw_video_cfg.get("aggregation") or {}

        # 과거 구현 호환:
        # - captioning + segment output_level: segment_start 기준으로 decode timestamp 오프셋 필요
        decode_relative_to_segment_start = (
            self.task_name == "captioning" and output_level == "segment"
        )

        # grounding/video_image_multi는 annotation 없을 수 있으므로 fallback_uniform 정책을 기본값으로 제공
        sampling_cfg_grounding = (
            sampling_cfg_strict
            if annotation_path is not None
            else sampling_cfg_fallback_uniform
        )
        self.resolved_video_cfg = {
            "raw_video_cfg": raw_video_cfg,
            "input_mode": input_mode,
            "output_level": output_level,
            "aggregation_cfg": aggregation_cfg,
            "sampling_cfg_strict": sampling_cfg_strict,
            "sampling_cfg_fallback_uniform": sampling_cfg_fallback_uniform,
            "sampling_cfg_grounding": sampling_cfg_grounding,
            "decode_relative_to_segment_start": decode_relative_to_segment_start,
            "fps": raw_video_cfg.get("fps", 1),
        }

    def _load_dataset_cfg(self, dataset_name: str) -> dict:
        """저장소 루트 기준 configs/test_dataset/{name}.yaml"""
        local_path = _REPO_ROOT / "configs/test_dataset" / f"{dataset_name}.yaml"
        if local_path.is_file():
            return load_yaml(str(local_path))

        raise FileNotFoundError(
            f"Dataset config for '{dataset_name}' not found: {local_path}"
        )

    def _resolve_paths(self):
        root_model = Path(self.paths_cfg["paths"]["model_root"])
        root_data = Path(self.paths_cfg["paths"]["datasets_root"])

        self.model_cfg["resolved_root"] = root_model / self.model_cfg["root"]

        resolved = {}
        for k, v in self.dataset_cfg.get("paths", {}).items():
            if isinstance(v, list):
                # image_groups: [[path, path, ...], [path, path, ...]]
                if v and isinstance(v[0], list):
                    resolved[k] = [
                        [root_data / x for x in group] for group in v
                    ]
                else:
                    resolved[k] = [root_data / x for x in v]
            else:
                resolved[k] = root_data / v

        self.dataset_cfg["paths"] = resolved

    def _merge_experiment_dataset_cfg(self):
        runtime_ds = dict(self.exp_cfg.get("runtime") or {})

        self.resolved_dataset = {
            **self.dataset_cfg,
            **runtime_ds,
        }

