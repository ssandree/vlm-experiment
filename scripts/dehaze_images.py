#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _dark_channel(img_bgr: np.ndarray, patch_size: int) -> np.ndarray:
    min_rgb = np.min(img_bgr, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    return cv2.erode(min_rgb, kernel)


def _estimate_atmospheric_light(img_bgr: np.ndarray, dark: np.ndarray) -> np.ndarray:
    h, w = dark.shape
    n_pixels = h * w
    top_k = max(n_pixels // 1000, 1)  # 상위 0.1%
    indices = np.argpartition(dark.reshape(-1), -top_k)[-top_k:]
    flat_img = img_bgr.reshape(-1, 3)
    a = np.max(flat_img[indices], axis=0)
    return a


def _estimate_transmission(
    img_bgr: np.ndarray,
    atmosphere: np.ndarray,
    omega: float,
    patch_size: int,
) -> np.ndarray:
    normed = img_bgr / (atmosphere.reshape(1, 1, 3) + 1e-6)
    dark_normed = _dark_channel(normed, patch_size)
    transmission = 1.0 - omega * dark_normed
    return np.clip(transmission, 0.05, 1.0)


def dehaze_bgr(
    image_bgr: np.ndarray,
    omega: float = 0.72,
    t0: float = 0.42,
    patch_size: int = 31,
    blend: float = 0.35,
) -> np.ndarray:
    img = image_bgr.astype(np.float32) / 255.0

    dark = _dark_channel(img, patch_size)
    atmosphere = _estimate_atmospheric_light(img, dark)
    transmission = _estimate_transmission(img, atmosphere, omega, patch_size)
    transmission = np.maximum(transmission, t0)

    j = np.empty_like(img)
    for c in range(3):
        j[:, :, c] = (img[:, :, c] - atmosphere[c]) / transmission + atmosphere[c]
    j = np.clip(j, 0.0, 1.0)
    dehazed = (j * 255.0).astype(np.uint8)

    # 과한 색변형 방지를 위해 원본과 부드럽게 블렌딩
    alpha = float(np.clip(blend, 0.0, 1.0))
    out = cv2.addWeighted(image_bgr, 1.0 - alpha, dehazed, alpha, 0.0)
    return out


def _iter_images(input_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch dehaze images with dark-channel prior.")
    parser.add_argument("--input-dirs", nargs="+", required=True, help="One or more input directories")
    parser.add_argument("--output-dir", required=True, help="Directory to save dehazed images")
    parser.add_argument("--suffix", default="_dehaze", help="Output filename suffix")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for input_dir_raw in args.input_dirs:
        input_dir = Path(input_dir_raw).resolve()
        if not input_dir.exists():
            print(f"[SKIP] not found: {input_dir}")
            continue
        if not input_dir.is_dir():
            print(f"[SKIP] not a directory: {input_dir}")
            continue

        for img_path in _iter_images(input_dir):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[WARN] failed to read: {img_path}")
                continue

            out = dehaze_bgr(img)
            # 사용자 요구: 저장 파일명은 무조건 pizza_fight로 시작
            stem = img_path.stem
            if not stem.startswith("pizza_fight"):
                stem = f"pizza_fight_{stem}"
            out_name = f"{stem}{args.suffix}{img_path.suffix.lower()}"
            out_path = output_dir / out_name
            ok = cv2.imwrite(str(out_path), out)
            if not ok:
                print(f"[WARN] failed to write: {out_path}")
                continue
            total += 1

    print(f"[DONE] saved {total} images to {output_dir}")


if __name__ == "__main__":
    main()
