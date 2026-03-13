from pathlib import Path
from utils.load_yaml import load_yaml


class ConfigResolver:
    def __init__(self, experiment_path: str):
        self.exp_cfg = load_yaml(experiment_path)

        self.env = self.exp_cfg["env"]
        self.model_name = self.exp_cfg["model"]
        self.dataset_name = self.exp_cfg["dataset"]
        self.task_name = self.exp_cfg.get("task", "captioning")  # Default to captioning for backward compatibility
        self.prompt_name = self.exp_cfg["prompt"]

        self.env_cfg = load_yaml(f"configs/env/{self.env}.yaml")
        self.model_cfg = load_yaml(f"configs/model/{self.model_name}.yaml")
        self.dataset_cfg = self._load_dataset_cfg(self.dataset_name)
        self.prompt_cfg = load_yaml(f"configs/prompt/{self.prompt_name}.yaml")
        self.runtime_cfg = load_yaml("configs/runtime.yaml")

        self._resolve_paths()
        self._merge_experiment_dataset_cfg()  # ⭐ 추가

    def _load_dataset_cfg(self, dataset_name: str) -> dict:
        """
        test_dataset config를 우선 현재 프로젝트의 configs/test_dataset 에서 찾고,
        없으면 /home/vailab02/vlm_experiment/configs/test_dataset/ 에서 찾는다.
        """
        local_path = Path("configs/test_dataset") / f"{dataset_name}.yaml"
        if local_path.exists():
            return load_yaml(str(local_path))

        exp_path = (
            Path("/home/vailab02/vlm_experiment/configs/test_dataset")
            / f"{dataset_name}.yaml"
        )
        if exp_path.exists():
            return load_yaml(str(exp_path))

        raise FileNotFoundError(
            f"Dataset config for '{dataset_name}' not found in "
            f"'configs/test_dataset/' or "
            f"'/home/vailab02/vlm_experiment/configs/test_dataset/'."
        )

    def _resolve_paths(self):
        root_model = Path(self.env_cfg["paths"]["model_root"])
        root_data = Path(self.env_cfg["paths"]["datasets_root"])

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
        runtime_ds = self.exp_cfg["runtime"]

        self.resolved_dataset = {
            **self.dataset_cfg,
            **runtime_ds,
        }

