from __future__ import annotations

import cv2
import numpy as np


def compute_homography(image_points: list[list[float]], field_points: list[list[float]]) -> np.ndarray:
    if len(image_points) < 4 or len(field_points) < 4:
        raise ValueError("Need at least 4 correspondences")
    src = np.array(image_points, dtype=np.float32)
    dst = np.array(field_points, dtype=np.float32)
    h, _ = cv2.findHomography(src, dst, cv2.RANSAC)
    if h is None:
        raise RuntimeError("Homography estimation failed")
    return h


def apply_homography(h: np.ndarray, uv: np.ndarray) -> np.ndarray:
    pts = uv.reshape(-1, 1, 2).astype(np.float32)
    out = cv2.perspectiveTransform(pts, h)
    return out.reshape(-1, 2)
