from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


def normalize_sampling_cfg(
    sampling_cfg: dict[str, Any],
    datasets_root: Path,
    annotation_path: Path | None = None,
    fallback_uniform_when_segment_aware_without_annotation: bool = False,
) -> dict[str, Any]:
    cfg = dict(sampling_cfg or {})

    if cfg.get("type") == "manual" and cfg.get("manual_frame_map_path"):
        p = Path(cfg["manual_frame_map_path"])
        if not p.is_absolute():
            cfg["manual_frame_map_path"] = str(datasets_root / p)

    strategy_type = (cfg.get("type") or "").strip().lower()
    if strategy_type == "segment_aware":
        if annotation_path is not None:
            cfg["annotation_path"] = str(annotation_path)
        elif fallback_uniform_when_segment_aware_without_annotation:
            cfg["type"] = "uniform"
            cfg["num_frames"] = int(cfg.get("num_frames", 4))

    return cfg


def save_aggregated_frames(
    frames_dir: Path,
    sample_id: str,
    aggregated: Image.Image | list[Image.Image],
) -> list[str]:
    if isinstance(aggregated, Image.Image):
        grid_path = frames_dir / f"{sample_id}_grid.jpg"
        aggregated.save(grid_path, format="JPEG")
        return [str(grid_path.resolve())]

    out_paths: list[str] = []
    for idx, img in enumerate(aggregated):
        frame_name = f"{sample_id}_f{idx:02d}.jpg"
        frame_path = frames_dir / frame_name
        img.save(frame_path, format="JPEG")
        out_paths.append(str(frame_path.resolve()))
    return out_paths


def pick_representative_image_paths(
    frame_paths_out: dict[str, list[str]],
) -> dict[str, str]:
    image_paths_out: dict[str, str] = {}
    for sample_id, paths in frame_paths_out.items():
        if not paths:
            image_paths_out[sample_id] = ""
            continue
        if len(paths) == 1:
            image_paths_out[sample_id] = paths[0]
            continue
        image_paths_out[sample_id] = paths[len(paths) // 2]
    return image_paths_out

