from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

import cv2
import numpy as np

from ..utils.io import ensure_dir, write_json


@dataclass
class DetectionSettings:
    threshold: float = 0.58
    scale_min: float = 0.75
    scale_max: float = 1.25
    scale_steps: int = 12
    nms_iou_threshold: float = 0.25
    max_detections_per_frame: int = 300
    max_candidates_per_scale: int = 400


def _iter_videos(video_dir: Path) -> list[Path]:
    return sorted(p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4")


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0
    union = float(aw * ah + bw * bh - inter)
    return inter / union if union > 0 else 0.0


def _nms(detections: list[tuple[int, int, int, int, float]], iou_threshold: float, max_keep: int) -> list[tuple[int, int, int, int, float]]:
    ordered = sorted(detections, key=lambda d: d[4], reverse=True)
    kept: list[tuple[int, int, int, int, float]] = []
    for det in ordered:
        box = (det[0], det[1], det[2], det[3])
        if any(_iou(box, (k[0], k[1], k[2], k[3])) > iou_threshold for k in kept):
            continue
        kept.append(det)
        if len(kept) >= max_keep:
            break
    return kept


def _find_template_matches(gray_frame: np.ndarray, gray_template: np.ndarray, settings: DetectionSettings) -> list[tuple[int, int, int, int, float]]:
    frame_h, frame_w = gray_frame.shape[:2]
    tpl_h, tpl_w = gray_template.shape[:2]
    scales = np.linspace(settings.scale_min, settings.scale_max, settings.scale_steps)

    candidates: list[tuple[int, int, int, int, float]] = []
    for scale in scales:
        scaled_w = max(1, int(round(tpl_w * float(scale))))
        scaled_h = max(1, int(round(tpl_h * float(scale))))
        if scaled_w > frame_w or scaled_h > frame_h:
            continue

        scaled_template = gray_template if (scaled_w == tpl_w and scaled_h == tpl_h) else cv2.resize(gray_template, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        result = cv2.matchTemplate(gray_frame, scaled_template, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(result >= float(settings.threshold))
        if ys.size == 0:
            continue

        scores = result[ys, xs]
        if scores.size > settings.max_candidates_per_scale:
            top_idx = np.argpartition(scores, -settings.max_candidates_per_scale)[-settings.max_candidates_per_scale :]
            ys, xs, scores = ys[top_idx], xs[top_idx], scores[top_idx]

        for x, y, score in zip(xs.tolist(), ys.tolist(), scores.tolist()):
            candidates.append((int(x), int(y), int(scaled_w), int(scaled_h), float(score)))

    return _nms(candidates, iou_threshold=float(settings.nms_iou_threshold), max_keep=int(settings.max_detections_per_frame))


def _render_heatmap(occupancy_map: np.ndarray) -> np.ndarray:
    if np.max(occupancy_map) <= 0:
        return np.zeros((*occupancy_map.shape, 3), dtype=np.uint8)
    normalized = cv2.normalize(occupancy_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)


def _mp4_to_gif(src_mp4: Path, out_gif: Path) -> None:
    cmd = ["ffmpeg", "-y", "-i", str(src_mp4), "-vf", "fps=10,scale=iw:-1:flags=lanczos", str(out_gif)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to write gif {out_gif}: {exc.stderr.strip()}") from exc


def _open_tmp_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, float(fps)), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open temporary writer: {path}")
    return writer


def _write_gif_from_frames(frames: list[np.ndarray], out_gif: Path, fps: float) -> None:
    if not frames:
        return
    tmp_mp4 = out_gif.with_suffix(".tmp.mp4")
    h, w = frames[0].shape[:2]
    writer = _open_tmp_writer(tmp_mp4, fps=fps, width=w, height=h)
    for frame in frames:
        writer.write(frame)
    writer.release()
    _mp4_to_gif(tmp_mp4, out_gif)
    tmp_mp4.unlink(missing_ok=True)


def detect_fuel(video_dir: str | Path, reference_image: str | Path, out_dir: str | Path, settings: DetectionSettings | None = None, show_preview: bool = False, combine: bool = False) -> dict:
    cfg = settings or DetectionSettings()
    src_dir = Path(video_dir)
    if not src_dir.exists() or not src_dir.is_dir():
        raise FileNotFoundError(f"Video directory does not exist: {src_dir}")
    ref_path = Path(reference_image)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference image not found: {ref_path}")

    videos = _iter_videos(src_dir)
    if not videos:
        raise ValueError(f"No mp4 files found in directory: {src_dir}")

    out = ensure_dir(out_dir)
    reference = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    if reference is None:
        raise ValueError(f"Failed to decode reference image: {ref_path}")
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    per_match: list[dict] = []
    combined_occ: np.ndarray | None = None
    combined_frames_sum: list[np.ndarray] = []
    combined_frames_count: list[int] = []
    combine_shape: tuple[int, int] | None = None
    combine_skipped_videos: list[str] = []
    all_counts: list[dict] = []
    total_frames = 0
    total_hits = 0

    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        ok, probe = cap.read()
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if not ok or probe is None:
            cap.release()
            continue
        h, w = probe.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        stem = video_path.stem
        map_png = out / f"{stem}_map.png"
        map_gif = out / f"{stem}_map.gif"
        tmp_timeline_mp4 = out / f"{stem}_timeline.tmp.mp4"

        occupancy = np.zeros((h, w), dtype=np.float32)
        timeline_writer = _open_tmp_writer(tmp_timeline_mp4, fps=fps, width=w, height=h)
        counts: list[int] = []
        match_hits = 0
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = _find_template_matches(gray, reference_gray, cfg)

            timeline = np.zeros_like(frame)
            for x, y, bw, bh, score in detections:
                occupancy[y : y + bh, x : x + bw] += 1.0
                center = (x + bw // 2, y + bh // 2)
                cv2.circle(timeline, center, max(2, min(w, h) // 80), (0, 255, 255), -1)
                if show_preview:
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 1)
                    cv2.putText(frame, f"{score:.2f}", (x, max(16, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

            count = len(detections)
            counts.append(count)
            all_counts.append({"video": str(video_path), "frame_idx": frame_idx, "fuel_count": count})
            match_hits += count
            cv2.putText(timeline, f"fuel_count={count}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
            timeline_writer.write(timeline)

            if show_preview:
                cv2.putText(frame, f"fuel_count={count}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("chevron-detect-preview", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    show_preview = False
                    cv2.destroyAllWindows()

            frame_idx += 1

        cap.release()
        timeline_writer.release()
        if show_preview:
            cv2.destroyAllWindows()

        cv2.imwrite(str(map_png), _render_heatmap(occupancy))
        _mp4_to_gif(tmp_timeline_mp4, map_gif)

        if combine:
            if combine_shape is None:
                combine_shape = (h, w)
                combined_occ = np.zeros_like(occupancy)
            if combine_shape == (h, w) and combined_occ is not None:
                combined_occ += occupancy
                timeline_cap = cv2.VideoCapture(str(tmp_timeline_mp4))
                frame_i = 0
                while timeline_cap.isOpened():
                    ok_t, fr = timeline_cap.read()
                    if not ok_t:
                        break
                    fr32 = fr.astype(np.float32)
                    if frame_i >= len(combined_frames_sum):
                        combined_frames_sum.append(fr32)
                        combined_frames_count.append(1)
                    else:
                        combined_frames_sum[frame_i] += fr32
                        combined_frames_count[frame_i] += 1
                    frame_i += 1
                timeline_cap.release()
            else:
                combine_skipped_videos.append(str(video_path))

        tmp_timeline_mp4.unlink(missing_ok=True)

        total_frames += frame_idx
        total_hits += match_hits
        per_match.append(
            {
                "input": str(video_path),
                "outputs": {"map_png": str(map_png), "map_gif": str(map_gif)},
                "frames": frame_idx,
                "detections": match_hits,
                "detections_per_frame_avg": (match_hits / frame_idx) if frame_idx else 0.0,
                "detections_per_frame_max": max(counts) if counts else 0,
            }
        )

    combined_outputs = {}
    if combine and per_match and combined_occ is not None:
        combined_png = out / "match_combined_map.png"
        combined_gif = out / "match_combined_map.gif"

        included_for_combine = max(1, len(per_match) - len(combine_skipped_videos))
        avg_occ = combined_occ / float(included_for_combine)
        cv2.imwrite(str(combined_png), _render_heatmap(avg_occ))

        avg_frames = [np.clip(s / max(1, c), 0, 255).astype(np.uint8) for s, c in zip(combined_frames_sum, combined_frames_count)]
        _write_gif_from_frames(avg_frames, combined_gif, fps=10.0)
        combined_outputs = {"map_png": str(combined_png), "map_gif": str(combined_gif)}

    counts_path = out / "tracked_counts.json"
    write_json(counts_path, {"rows": all_counts, "total_frames": total_frames})

    summary = {
        "video_dir": str(src_dir),
        "reference_image": str(ref_path),
        "per_match": per_match,
        "combined_outputs": combined_outputs,
        "combine_skipped_videos": combine_skipped_videos,
        "outputs": {"counts_json": str(counts_path)},
        "settings": {
            "threshold": cfg.threshold,
            "scale_min": cfg.scale_min,
            "scale_max": cfg.scale_max,
            "scale_steps": cfg.scale_steps,
            "nms_iou_threshold": cfg.nms_iou_threshold,
            "max_detections_per_frame": cfg.max_detections_per_frame,
            "max_candidates_per_scale": cfg.max_candidates_per_scale,
        },
        "total_frames": total_frames,
        "total_hits": total_hits,
        "detections_per_frame_avg": (total_hits / total_frames) if total_frames else 0.0,
    }
    write_json(out / "detect_summary.json", summary)
    return summary
