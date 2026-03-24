"""
Image-related utilities.
"""

import os


def normalize_image_id(image_id: str) -> str:
    """
    Dataset-independent image_id normalization.
    - Flickr30k: "4567734402.jpg" -> "4567734402"
    - COCO: "391895" -> "391895"
    - ImageNet: "n01440764_18.JPEG" -> "n01440764_18"
    """
    return os.path.splitext(image_id)[0]
