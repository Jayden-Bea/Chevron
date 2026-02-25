from __future__ import annotations

from pathlib import Path

import cv2


def extract_view_frames(video_path: str, crops: dict[str, list[int]], out_dir: str | Path, frame_idx: int = 0) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read frame")

    paths = {}
    for name, (x, y, w, h) in crops.items():
        crop = frame[y : y + h, x : x + w]
        p = out / f"{name}_calib_frame.png"
        cv2.imwrite(str(p), crop)
        paths[name] = str(p)
    return paths
