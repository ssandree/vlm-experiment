# vlm_experiment/data/loader/experiment_loaders.py
# 벤치마크와 무관한 실험용 데이터 로더: anomaly_cctv, ucfcrime_subset

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import random
import json
import math

from PIL import Image


class UCFCrimeSubsetEvalLoader:
    """
    UCFCrime 비디오를 video_list로 지정한 만큼만 사용하는 evaluation loader.

    run_video_captioning 파이프라인과 동일한 배치 형식:
    clip_ids, video_paths, segments, sentences.
    """

    def __init__(
        self,
        video_list: List[str] | List[Path],
        annotation_path: str | Path,
        batch_size: int = 1,
        num_samples: Optional[int] = None,
        shuffle_seed: int = 42,
    ):
        self.video_paths = [Path(p) for p in video_list]
        self.ann_path = Path(annotation_path)

        self.batch_size = _int_or_none(batch_size) or 1
        self.num_samples = _int_or_none(num_samples)
        self.shuffle_seed = _int_or_none(shuffle_seed) or 42

        for p in self.video_paths:
            if not p.exists():
                raise FileNotFoundError(f"UCFCrime subset video not found: {p}")

        ann_file = self.ann_path if self.ann_path.exists() else self.ann_path.with_suffix(".json")
        if not ann_file.exists():
            raise FileNotFoundError(
                f"UCFCrime annotation not found: {self.ann_path} (nor {ann_file})"
            )
        with open(ann_file, "r", encoding="utf-8") as f:
            raw_ann = json.load(f)

        self._samples: List[Dict[str, Any]] = []
        for video_path in self.video_paths:
            video_name = video_path.stem
            data = raw_ann.get(video_name)
            if not isinstance(data, dict):
                continue
            timestamps = data.get("timestamps") or []
            sentences = data.get("sentences") or []
            if len(timestamps) != len(sentences):
                continue
            for i, (seg, sent) in enumerate(zip(timestamps, sentences)):
                if not isinstance(seg, (list, tuple)) or len(seg) != 2:
                    continue
                start, end = float(seg[0]), float(seg[1])
                clip_id = f"{video_name}_{i}"
                self._samples.append({
                    "clip_id": clip_id,
                    "video_path": video_path,
                    "segment": (start, end),
                    "sentence": str(sent).strip(),
                })

        if not self._samples:
            raise RuntimeError(
                "UCFCrimeSubsetEvalLoader: no valid clips for the given videos."
            )

        indices = list(range(len(self._samples)))
        if self.num_samples is not None and self.num_samples < len(indices):
            random.seed(self.shuffle_seed)
            random.shuffle(indices)
            indices = sorted(indices[: self.num_samples])
        self._sample_indices = indices
        self._cursor = 0

    def __len__(self) -> int:
        n = len(self._sample_indices)
        return math.ceil(n / self.batch_size) if n else 0

    def __iter__(self):
        self._cursor = 0
        return self

    def __next__(self) -> Dict[str, Any]:
        n = len(self._sample_indices)
        if self._cursor >= n:
            raise StopIteration

        batch_end = min(self._cursor + self.batch_size, n)
        batch_indices = self._sample_indices[self._cursor:batch_end]
        self._cursor = batch_end

        return _build_video_batch_from_samples(self._samples, batch_indices)


def _int_or_none(x):
    """YAML에서 'None' 문자열 또는 실제 None일 때 정수 변환."""
    if x is None or (isinstance(x, str) and x.strip().lower() == "none"):
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _get_video_duration(video_path: Path) -> float:
    """Decord로 비디오 길이(초) 반환."""
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(str(video_path), ctx=cpu(0))
        n = len(vr)
        fps = float(vr.get_avg_fps()) if vr.get_avg_fps() and vr.get_avg_fps() > 0 else 30.0
        return n / fps if n > 0 else 1.0
    except Exception as e:
        raise RuntimeError(f"Failed to get duration for {video_path}: {e}") from e


def _build_video_batch_from_samples(
    samples: List[Dict[str, Any]],
    sample_indices: List[int],
) -> Dict[str, Any]:
    clip_ids: List[str] = []
    video_paths: Dict[str, Path] = {}
    segments: Dict[str, Tuple[float, float]] = {}
    sentences: Dict[str, str] = {}

    for idx in sample_indices:
        s = samples[idx]
        cid = s["clip_id"]
        clip_ids.append(cid)
        video_paths[cid] = s["video_path"]
        segments[cid] = s["segment"]
        sentences[cid] = s["sentence"]

    return {
        "clip_ids": clip_ids,
        "video_paths": video_paths,
        "segments": segments,
        "sentences": sentences,
    }


def video_only_loader(
    video_list: List[str] | List[Path],
    batch_size: int = 1,
    num_samples: Optional[int] = None,
    shuffle_seed: Optional[int] = 42,
) -> "VideoOnlyLoader":
    """
    annotation 없이 비디오 경로만 사용. prompt에 따라 task 수행.
    video_list는 resolved path (Path 객체 또는 str) 리스트.
    """
    return VideoOnlyLoader(
        video_paths=[Path(p) for p in video_list],
        batch_size=_int_or_none(batch_size) or 1,
        num_samples=_int_or_none(num_samples),
        shuffle_seed=_int_or_none(shuffle_seed) or 42,
    )


class VideoOnlyLoader:
    """
    비디오 경로 리스트만 사용. annotation 불필요.
    출력: clip_ids, video_paths, segments, sentences (run_video_captioning/grounding 호환).
    """

    def __init__(
        self,
        video_paths: List[Path],
        batch_size: int = 1,
        num_samples: Optional[int] = None,
        shuffle_seed: Optional[int] = 42,
    ):
        self.video_paths = list(video_paths)
        self.batch_size = batch_size or 1
        self.num_samples = _int_or_none(num_samples) if num_samples is not None else None
        self.shuffle_seed = shuffle_seed or 42

        for p in self.video_paths:
            if not p.exists():
                raise FileNotFoundError(f"Video not found: {p}")

        self._samples: List[Dict[str, Any]] = []
        for video_path in self.video_paths:
            clip_id = video_path.stem or f"video_{len(self._samples)}"
            self._samples.append({
                "clip_id": clip_id,
                "video_path": video_path,
                "segment": None,
                "sentence": "",
            })

        indices = list(range(len(self._samples)))
        if self.num_samples is not None and self.num_samples < len(indices):
            random.seed(self.shuffle_seed)
            random.shuffle(indices)
            indices = sorted(indices[: self.num_samples])
        self._indices = indices
        self._cursor = 0

    def __len__(self) -> int:
        n = len(self._indices)
        return math.ceil(n / self.batch_size) if n else 0

    def __iter__(self):
        self._cursor = 0
        return self

    def __next__(self) -> Dict[str, Any]:
        n = len(self._indices)
        if self._cursor >= n:
            raise StopIteration

        end = min(self._cursor + self.batch_size, n)
        batch_idx = self._indices[self._cursor:end]
        self._cursor = end

        for idx in batch_idx:
            s = self._samples[idx]
            if s["segment"] is None:
                duration = _get_video_duration(s["video_path"])
                s["segment"] = (0.0, duration)

        return _build_video_batch_from_samples(self._samples, batch_idx)


def image_only_loader(
    image_list: List[str] | List[Path],
    batch_size: int = 1,
    num_samples: Optional[int] = None,
    shuffle_seed: Optional[int] = 42,
) -> "ImageOnlyLoader":
    """
    annotation 없이 이미지 경로만 사용. prompt에 따라 task 수행.
    image_list는 resolved path (Path 객체 또는 str) 리스트.
    """
    return ImageOnlyLoader(
        image_paths=[Path(p) for p in image_list],
        batch_size=_int_or_none(batch_size) or 1,
        num_samples=_int_or_none(num_samples),
        shuffle_seed=_int_or_none(shuffle_seed) or 42,
    )


class ImageOnlyLoader:
    """
    이미지 경로 리스트만 사용. annotation 불필요.
    출력: images, image_ids, image_paths, references (빈 리스트).
    """

    def __init__(
        self,
        image_paths: List[Path],
        batch_size: int = 1,
        num_samples: Optional[int] = None,
        shuffle_seed: Optional[int] = 42,
    ):
        self.image_paths = list(image_paths)
        self.batch_size = batch_size or 1
        self.num_samples = _int_or_none(num_samples) if num_samples is not None else None
        self.shuffle_seed = shuffle_seed or 42

        for p in self.image_paths:
            if not p.exists():
                raise FileNotFoundError(f"Image not found: {p}")

        indices = list(range(len(self.image_paths)))
        n_samples = self.num_samples
        if n_samples is not None and isinstance(n_samples, int) and n_samples < len(indices):
            random.seed(self.shuffle_seed)
            random.shuffle(indices)
            indices = sorted(indices[:n_samples])
        self._indices = indices
        self._cursor = 0

    def __len__(self) -> int:
        n = len(self._indices)
        return math.ceil(n / self.batch_size) if n else 0

    def __iter__(self):
        self._cursor = 0
        return self

    def __next__(self) -> Dict[str, Any]:
        n = len(self._indices)
        if self._cursor >= n:
            raise StopIteration

        end = min(self._cursor + self.batch_size, n)
        batch_idx = self._indices[self._cursor:end]
        self._cursor = end

        images: List[Image.Image] = []
        image_ids: List[str] = []
        image_paths: List[str] = []
        references: Dict[str, List] = {}

        for idx in batch_idx:
            p = self.image_paths[idx]
            img = Image.open(p).convert("RGB")
            images.append(img)
            iid = p.stem or f"image_{idx}"
            image_ids.append(iid)
            image_paths.append(str(p))
            references[iid] = []

        return {
            "images": images,
            "image_ids": image_ids,
            "image_paths": image_paths,
            "references": references,
        }


def image_multi_loader(
    image_groups: List[List[str]] | List[List[Path]],
    num_samples: Optional[int] = None,
    shuffle_seed: Optional[int] = 42,
) -> "ImageMultiLoader":
    """
    여러 이미지를 그룹 단위로 묶어서 로드. 그룹당 하나의 추론 결과 생성.
    image_groups: [[path1, path2, path3, path4], [path5, path6, ...], ...]
    """
    groups = [[Path(p) for p in group] for group in image_groups]
    return ImageMultiLoader(
        image_groups=groups,
        num_samples=_int_or_none(num_samples),
        shuffle_seed=_int_or_none(shuffle_seed) or 42,
    )


class ImageMultiLoader:
    """
    이미지 그룹 단위 로더. 각 그룹 = 여러 이미지 = 하나의 추론 결과.
    출력: image_groups, group_ids, group_paths, references
    """

    def __init__(
        self,
        image_groups: List[List[Path]],
        num_samples: Optional[int] = None,
        shuffle_seed: int = 42,
    ):
        self.image_groups = list(image_groups)
        self.num_samples = num_samples
        self.shuffle_seed = shuffle_seed

        for group in self.image_groups:
            for p in group:
                if not p.exists():
                    raise FileNotFoundError(f"Image not found: {p}")

        indices = list(range(len(self.image_groups)))
        if self.num_samples is not None and self.num_samples < len(indices):
            random.seed(self.shuffle_seed)
            random.shuffle(indices)
            indices = sorted(indices[: self.num_samples])
        self._indices = indices
        self._cursor = 0

    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self):
        self._cursor = 0
        return self

    def __next__(self) -> Dict[str, Any]:
        if self._cursor >= len(self._indices):
            raise StopIteration

        idx = self._indices[self._cursor]
        self._cursor += 1

        paths = self.image_groups[idx]
        images: List[Image.Image] = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            images.append(img)

        group_id = paths[0].stem if paths else f"group_{idx}"
        group_paths = [str(p) for p in paths]

        return {
            "image_groups": [images],
            "group_ids": [group_id],
            "group_paths": [group_paths],
            "references": {group_id: []},
        }


def anomaly_cctv_loader(
    batch_size: int = 1,
    num_samples: Optional[int] = None,
    shuffle_seed: int = 42,
) -> Any:
    """
    단일 anomaly CCTV 이미지만 로드하는 간단한 image loader.
    출력 포맷: images, image_ids, image_paths, references (COCO/Flickr 호환).
    """

    class _SingleImageLoader:
        def __init__(self):
            self.batch_size = batch_size
            self.image_path = Path(
                "/data1/vailab02_dir/Classification_DB/anomaly_CCTV/"
                "1-3_cam01_fight04_place02_night_spring_capture.png"
            )
            if not self.image_path.exists():
                raise FileNotFoundError(self.image_path)

            self.samples = [
                {"image_id": "anomaly_cctv_0", "image_path": self.image_path}
            ]
            self._cursor = 0

        def __len__(self) -> int:
            return math.ceil(len(self.samples) / self.batch_size)

        def __iter__(self):
            self._cursor = 0
            return self

        def __next__(self) -> Dict[str, Any]:
            if self._cursor >= len(self.samples):
                raise StopIteration

            batch_samples = self.samples[
                self._cursor : self._cursor + self.batch_size
            ]
            self._cursor += self.batch_size

            images: List[Image.Image] = []
            image_ids: List[str] = []
            image_paths: List[str] = []
            references: Dict[str, List[str]] = {}

            for s in batch_samples:
                img = Image.open(s["image_path"]).convert("RGB")
                images.append(img)
                image_ids.append(s["image_id"])
                image_paths.append(str(s["image_path"]))
                references[s["image_id"]] = []

            return {
                "images": images,
                "image_ids": image_ids,
                "image_paths": image_paths,
                "references": references,
            }

    return _SingleImageLoader()
