from __future__ import annotations

from pathlib import Path

import cv2


def export_layout_preview(video_path: str, crops: dict[str, list[int]], out_path: str | Path, frame_idx: int = 0) -> None:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read frame for layout preview")

    colors = {"top": (0, 255, 0), "bottom_left": (255, 0, 0), "bottom_right": (0, 0, 255)}
    for name, rect in crops.items():
        x, y, w, h = rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors.get(name, (255, 255, 0)), 2)
        cv2.putText(frame, name, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(out_path), frame)
