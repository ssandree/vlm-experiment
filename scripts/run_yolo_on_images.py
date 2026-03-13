#!/usr/bin/env python3
"""
이미지 경로를 인자로 받아 YOLO 검출 수행. config 불필요.

Usage:
  python scripts/run_yolo_on_images.py image1.jpg image2.png
  python scripts/run_yolo_on_images.py /path/to/images/*.jpg
"""

import argparse
import json
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from detectors.yolo_detector import YOLODetector


def main():
    parser = argparse.ArgumentParser(description="YOLO object detection on images")
    parser.add_argument(
        "images",
        nargs="+",
        help="Image path(s) or glob pattern",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to YOLO weights (default: yolo_pretrained/yolo11m.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output JSON path (default: print to stdout)",
    )
    args = parser.parse_args()

    detector = YOLODetector(
        weights_path=args.weights,
        conf_threshold=args.conf,
    )

    # 경로 확장 (glob)
    paths = []
    for p in args.images:
        path = Path(p)
        if "*" in str(p):
            paths.extend(sorted(path.parent.glob(path.name)))
        elif path.exists():
            paths.append(path)
        else:
            print(f"[WARN] Not found: {p}", file=sys.stderr)

    if not paths:
        print("No valid image paths.", file=sys.stderr)
        sys.exit(1)

    results = detector.detect([str(p) for p in paths])
    output = {
        str(p.name): dets for p, dets in zip(paths, results)
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Saved to {args.output}")
    else:
        print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
