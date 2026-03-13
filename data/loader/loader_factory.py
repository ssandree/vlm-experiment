from data.loader.experiment_loaders import (
    UCFCrimeSubsetEvalLoader,
    anomaly_cctv_loader,
    image_only_loader,
    image_multi_loader,
    video_only_loader,
)


def get_dataloader(dataset_cfg: dict):
    """
    Minimal dataloader factory for this experiment repo.
    """
    mode = dataset_cfg["mode"]
    p = dataset_cfg["paths"]

    if mode == "ucfcrime_subset":
        return UCFCrimeSubsetEvalLoader(
            video_list=p["video_list"],
            annotation_path=p["annotation"],
            batch_size=dataset_cfg["batch_size"],
            num_samples=dataset_cfg.get("num_samples"),
            shuffle_seed=dataset_cfg["shuffle_seed"],
        )

    if mode == "anomaly_cctv":
        return anomaly_cctv_loader(
            batch_size=dataset_cfg["batch_size"],
            num_samples=dataset_cfg.get("num_samples"),
            shuffle_seed=dataset_cfg["shuffle_seed"],
        )

    if mode == "image_only":
        return image_only_loader(
            image_list=p["image_list"],
            batch_size=dataset_cfg.get("batch_size", 1),
            num_samples=dataset_cfg.get("num_samples"),
            shuffle_seed=dataset_cfg.get("shuffle_seed", 42),
        )

    if mode == "image_multi":
        return image_multi_loader(
            image_groups=p["image_groups"],
            num_samples=dataset_cfg.get("num_samples"),
            shuffle_seed=dataset_cfg.get("shuffle_seed", 42),
        )

    if mode == "video_only":
        return video_only_loader(
            video_list=p["video_list"],
            batch_size=dataset_cfg.get("batch_size", 1),
            num_samples=dataset_cfg.get("num_samples"),
            shuffle_seed=dataset_cfg.get("shuffle_seed", 42),
        )

    raise ValueError(f"Unsupported dataset mode in this repo: {mode}")

