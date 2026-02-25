from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..split.layout import get_layout
from ..utils.hashing import stable_hash
from ..utils.io import ensure_dir, write_json
from .blend import blend_layers
from .warp import warp_to_canvas
from .writer import VideoWriter


def render_matches(video_path: str, segments: list[dict], calib: dict, cfg: dict, out_dir: str | Path) -> list[str]:
    out = ensure_dir(out_dir)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    layout = get_layout(cfg, width, height)

    canvas = calib["canvas"]
    size = (int(canvas["width_px"]), int(canvas["height_px"]))
    hs = {k: np.array(v, dtype=np.float32) for k, v in calib["homographies"].items()}
    outputs = []

    for i, seg in enumerate(segments, start=1):
        match_dir = ensure_dir(out / f"match_{i:03d}")
        writer = VideoWriter(match_dir / "topdown.mp4", fps=fps, size=size)
        start_idx = int(seg["start_time_s"] * fps)
        end_idx = int(seg["end_time_s"] * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        frame_meta = []
        for frame_idx in range(start_idx, end_idx):
            ok, frame = cap.read()
            if not ok:
                break
            layers = []
            masks = []
            for name, (x, y, w, h) in layout.items():
                crop = frame[y : y + h, x : x + w]
                warped = warp_to_canvas(crop, hs[name], size)
                mask = (warped.sum(axis=2) > 0).astype(np.float32)
                layers.append(warped)
                masks.append(mask)
            stitched = blend_layers(layers, masks)
            writer.write(stitched)
            frame_meta.append({"frame_idx": frame_idx - start_idx, "t_video_s": frame_idx / fps, "t_match_s": None})
        writer.close()

        meta = {
            "src_vod_start_s": seg["start_time_s"],
            "src_vod_end_s": seg["end_time_s"],
            "fps": fps,
            "frame_count": len(frame_meta),
            "frames": frame_meta,
            "config_hash": stable_hash(cfg),
            "calib_version": calib.get("version", "v1"),
        }
        write_json(match_dir / "match_meta.json", meta)
        outputs.append(str(match_dir / "topdown.mp4"))

    cap.release()
    return outputs
