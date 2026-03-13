from pathlib import Path
from datetime import datetime
import json


def create_experiment_dir_and_metadata(
    runtime_cfg: dict,
    dataset_cfg: dict,
    model_cfg: dict,
    system_prompt: str,
    user_prompt: str,
    extra_meta: dict | None = None,
) -> Path:
    """
    Create experiment directory and experiment.json.
    Returns experiment directory path.
    """

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    experiment_id = f"exp_{timestamp}"

    outputs_root = Path("outputs/")
    outputs_root.mkdir(parents=True, exist_ok=True)
    exp_dir = outputs_root / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    experiment_meta = {
        "experiment_id": experiment_id,
        "timestamp": now.isoformat(),
        "dataset": {
            "name": dataset_cfg["mode"],
            "batch_size": dataset_cfg["batch_size"],
            "num_samples": dataset_cfg["num_samples"],
        },
        "model": {
            "name": model_cfg["name"],
            "precision": model_cfg.get("precision"),
            "generation": model_cfg.get("generation"),
        },
        "prompt": {
            "system": system_prompt,
            "user": user_prompt,
        },
        "runtime": runtime_cfg,
    }
    if extra_meta:
        experiment_meta.update(extra_meta)

    with open(exp_dir / "experiment.json", "w", encoding="utf-8") as f:
        json.dump(experiment_meta, f, indent=2, ensure_ascii=False)

    return exp_dir

