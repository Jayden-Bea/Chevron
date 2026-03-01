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


def render_matches(
    video_path: str,
    segments: list[dict],
    calib: dict,
    cfg: dict,
    out_dir: str | Path,
    output_fps: float | None = None,
    progress_interval_s: float = 5.0,
    progress_callback=None,
    skip_existing: bool = True,
) -> list[str]:
    out = ensure_dir(out_dir)
    cap = cv2.VideoCapture(video_path)
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if output_fps and output_fps > 0:
        frame_step = max(1, int(round(source_fps / float(output_fps))))
    else:
        frame_step = 1
    fps = source_fps / frame_step
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    layout = get_layout(cfg, width, height)

    canvas = calib["canvas"]
    size = (int(canvas["width_px"]), int(canvas["height_px"]))
    hs = {k: np.array(v, dtype=np.float32) for k, v in calib["homographies"].items()}
    outputs = []
    safe_interval_s = max(0.1, float(progress_interval_s))

    view_order = ["top", "bottom_left", "bottom_right"]

    for i, seg in enumerate(segments, start=1):
        match_dir = ensure_dir(out / f"match_{i:03d}")
        view_output_paths = {view: match_dir / f"topdown_{view}.mp4" for view in view_order}
        output_path = match_dir / "topdown.mp4"
        meta_path = match_dir / "match_meta.json"
        start_idx = int(seg["start_time_s"] * source_fps)
        end_idx = int(seg["end_time_s"] * source_fps)
        total_match_frames = max((end_idx - start_idx + frame_step - 1) // frame_step, 0)
        frame_log_interval = max(1, int(round(safe_interval_s * fps)))

        expected_outputs = [output_path, *view_output_paths.values()]
        has_all_outputs = all(path.exists() and path.stat().st_size > 0 for path in expected_outputs)
        if skip_existing and has_all_outputs and meta_path.exists():
            outputs.append(str(output_path))
            if progress_callback:
                progress_callback(
                    {
                        "event": "match_skipped",
                        "match_index": i,
                        "total_matches": len(segments),
                        "output": str(output_path),
                        "reason": "already_rendered",
                    }
                )
            continue

        writer = VideoWriter(output_path, fps=fps, size=size)
        view_writers = {view: VideoWriter(path, fps=fps, size=size) for view, path in view_output_paths.items()}
        blank_frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        if progress_callback:
            progress_callback(
                {
                    "event": "match_start",
                    "match_index": i,
                    "total_matches": len(segments),
                    "start_frame": start_idx,
                    "end_frame": end_idx,
                    "total_frames": total_match_frames,
                    "start_time_s": float(seg["start_time_s"]),
                    "end_time_s": float(seg["end_time_s"]),
                }
            )

        frame_meta = []
        output_frame_idx = 0
        for frame_idx in range(start_idx, end_idx, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break
            layers = []
            masks = []
            warped_views = {}
            for name in view_order:
                if name not in layout or name not in hs:
                    continue
                x, y, w, h = layout[name]
                crop = frame[y : y + h, x : x + w]
                warped = warp_to_canvas(crop, hs[name], size)
                warped_views[name] = warped
                mask = (warped.sum(axis=2) > 0).astype(np.float32)
                layers.append(warped)
                masks.append(mask)

            for name, view_writer in view_writers.items():
                view_writer.write(warped_views.get(name, blank_frame))

            if not layers:
                continue
            stitched = blend_layers(layers, masks)
            writer.write(stitched)
            frame_meta.append({"frame_idx": output_frame_idx, "t_video_s": frame_idx / source_fps, "t_match_s": None})

            if progress_callback and (output_frame_idx % frame_log_interval == 0):
                progress_callback(
                    {
                        "event": "match_progress",
                        "match_index": i,
                        "total_matches": len(segments),
                        "match_frame_idx": output_frame_idx,
                        "match_total_frames": total_match_frames,
                        "video_frame_idx": frame_idx,
                        "video_time_s": round(frame_idx / source_fps, 3),
                    }
                )
            output_frame_idx += 1
        writer.close()
        for view_writer in view_writers.values():
            view_writer.close()

        meta = {
            "src_vod_start_s": seg["start_time_s"],
            "src_vod_end_s": seg["end_time_s"],
            "fps": fps,
            "source_fps": source_fps,
            "frame_count": len(frame_meta),
            "frames": frame_meta,
            "config_hash": stable_hash(cfg),
            "calib_version": calib.get("version", "v1"),
            "outputs": {
                "combined": output_path.name,
                **{view: path.name for view, path in view_output_paths.items()},
            },
        }
        write_json(meta_path, meta)
        outputs.append(str(output_path))
        if progress_callback:
            progress_callback(
                {
                    "event": "match_complete",
                    "match_index": i,
                    "total_matches": len(segments),
                    "written_frames": len(frame_meta),
                    "output": str(output_path),
                }
            )

    cap.release()
    return outputs
