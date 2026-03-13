"""
YOLO 객체 검출기. /data1/vailab02_dir/vlm_models/yolo_pretrained 가중치 사용.

입력: 1개 또는 여러 개의 image (PIL.Image 또는 경로)
출력: bbox [x, y, w, h] (픽셀) + object class
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

from PIL import Image

DEFAULT_WEIGHTS_DIR = Path("/data1/vailab02_dir/vlm_models/yolo_pretrained")
DEFAULT_MODEL = "yolo11m.pt"


class YOLODetector:
    """
    YOLO 기반 객체 검출. 입력 1장 또는 다중 이미지, 출력 bbox + class.
    """

    def __init__(
        self,
        weights_path: Union[str, Path] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        verbose: bool = False,
    ):
        """
        Args:
            weights_path: 가중치 경로. None이면 DEFAULT_WEIGHTS_DIR/yolo11m.pt 사용
            conf_threshold: confidence 임계값
            iou_threshold: NMS IoU 임계값
            verbose: YOLO inference 시 verbose 출력 여부
        """
        from ultralytics import YOLO

        if weights_path is None:
            weights_path = DEFAULT_WEIGHTS_DIR / DEFAULT_MODEL
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

        self.model = YOLO(str(weights_path))
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.verbose = verbose

    def detect(
        self,
        images: Union[
            Image.Image,
            str,
            Path,
            List[Image.Image],
            List[str],
            List[Path],
        ],
    ) -> List[List[dict]]:
        """
        이미지(들)에 대해 객체 검출 수행.

        Args:
            images: 단일 이미지(PIL/경로) 또는 이미지 리스트

        Returns:
            이미지별 검출 결과 리스트. 각 요소는
            [{"bbox": [x, y, w, h], "class": str, "conf": float}, ...]
        """
        single = not isinstance(images, (list, tuple))
        if single:
            images = [images]

        # PIL Image → numpy (BGR for OpenCV compatibility in YOLO)
        import numpy as np

        sources = []
        for img in images:
            if isinstance(img, (str, Path)):
                sources.append(str(img))
            elif isinstance(img, Image.Image):
                arr = np.array(img)
                if img.mode == "RGBA":
                    arr = arr[:, :, :3]  # RGB
                sources.append(arr)
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

        results = self.model.predict(
            source=sources,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=self.verbose,
        )

        out: List[List[dict]] = []
        for r in results:
            dets = []
            if r.boxes is not None:
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    w = float(x2 - x1)
                    h = float(y2 - y1)
                    cls_name = self.model.names[int(cls_ids[i])]
                    dets.append({
                        "bbox": [float(x1), float(y1), w, h],
                        "class": cls_name,
                        "conf": float(confs[i]),
                    })
            out.append(dets)

        if single:
            return out[0]
        return out

    @property
    def class_names(self) -> dict:
        """class_id -> class_name 매핑"""
        return self.model.names
