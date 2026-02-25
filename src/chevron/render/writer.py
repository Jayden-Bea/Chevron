from __future__ import annotations

from pathlib import Path

import cv2


class VideoWriter:
    def __init__(self, out_path: str | Path, fps: float, size: tuple[int, int]):
        self.path = str(out_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self.path, fourcc, fps, size)

    def write(self, frame):
        self._writer.write(frame)

    def close(self):
        self._writer.release()
