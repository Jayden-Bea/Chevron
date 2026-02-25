from __future__ import annotations

import cv2
import numpy as np


def warp_to_canvas(frame, h: np.ndarray, canvas_size: tuple[int, int]):
    width, height = canvas_size
    return cv2.warpPerspective(frame, h, (width, height))
