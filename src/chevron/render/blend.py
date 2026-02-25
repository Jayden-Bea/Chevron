from __future__ import annotations

import cv2
import numpy as np


def feather_mask(mask: np.ndarray, ksize: int = 31) -> np.ndarray:
    mask_f = mask.astype(np.float32)
    blurred = cv2.GaussianBlur(mask_f, (ksize, ksize), 0)
    mx = float(blurred.max()) if blurred.size else 1.0
    if mx <= 0:
        return blurred
    return blurred / mx


def blend_layers(layers: list[np.ndarray], masks: list[np.ndarray]) -> np.ndarray:
    if not layers:
        raise ValueError("No layers to blend")
    h, w = layers[0].shape[:2]
    acc = np.zeros((h, w, 3), dtype=np.float32)
    weight = np.zeros((h, w, 1), dtype=np.float32)
    for layer, mask in zip(layers, masks):
        m = feather_mask(mask)[..., None]
        acc += layer.astype(np.float32) * m
        weight += m
    weight = np.clip(weight, 1e-6, None)
    out = acc / weight
    return np.clip(out, 0, 255).astype(np.uint8)
