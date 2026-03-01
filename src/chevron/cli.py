from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import subprocess
import sys
from pathlib import Path
from time import monotonic, time


from .utils.config import load_config
from .utils.ffmpeg import crop_video
from .utils.io import ensure_dir, read_json, write_json


_LOG_MONOTONIC_START = monotonic()


def _log(message: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    elapsed_s = monotonic() - _LOG_MONOTONIC_START
    print(f"[{ts} +{elapsed_s:8.2f}s] {message}", flush=True)


def _format_segment_progress(details: dict) -> str:
    def _fmt_score(value) -> str:
        if isinstance(value, (int, float)):
            return f"{float(value):.3f}"
        return "n/a"

    total_frames = details.get("total_frames", 0) or 0
    frame_idx = details.get("frame_idx", 0)
    if total_frames > 0:
        pct = max(0.0, min(100.0, (frame_idx / total_frames) * 100.0))
        progress = f"{frame_idx}/{total_frames} ({pct:.1f}%)"
    else:
        progress = f"{frame_idx} frames"

    start_score = _fmt_score(details.get("start_score"))
    stop_score = _fmt_score(details.get("stop_score"))
    return (
        "[chevron] run: segment progress "
        f"t={details.get('time_s', 0.0):.1f}s "
        f"frame={progress} "
        f"state={details.get('state')} "
        f"scores(start={start_score}, stop={stop_score}) "
        f"hits(start={details.get('start_hit')}, stop={details.get('stop_hit')}) "
        f"segments={details.get('segments_found', 0)}"
    )


def _format_render_progress(details: dict) -> str:
    event = details.get("event")
    match_index = details.get("match_index", 0)
    total_matches = details.get("total_matches", 0)

    if event == "match_start":
        return (
            "[chevron] run: render start "
            f"match={match_index}/{total_matches} "
            f"video_s={details.get('start_time_s', 0.0):.1f}->{details.get('end_time_s', 0.0):.1f} "
            f"frames={details.get('total_frames', 0)}"
        )

    if event == "match_progress":
        frame_idx = details.get("match_frame_idx", 0)
        match_total = details.get("match_total_frames", 0) or 0
        if match_total > 0:
            pct = max(0.0, min(100.0, (frame_idx / match_total) * 100.0))
            frame_text = f"{frame_idx}/{match_total} ({pct:.1f}%)"
        else:
            frame_text = str(frame_idx)
        return (
            "[chevron] run: render progress "
            f"match={match_index}/{total_matches} "
            f"frame={frame_text} "
            f"video_t={details.get('video_time_s', 0.0):.1f}s"
        )

    if event == "match_complete":
        return (
            "[chevron] run: render complete "
            f"match={match_index}/{total_matches} "
            f"written_frames={details.get('written_frames', 0)} "
            f"output={details.get('output')}"
        )

    if event == "match_skipped":
        return (
            "[chevron] run: render skipped "
            f"match={match_index}/{total_matches} "
            f"reason={details.get('reason')} "
            f"output={details.get('output')}"
        )

    return f"[chevron] run: render event={event} details={details}"


def _export_raw_matches(
    video_path: str | Path,
    segments: list[dict],
    out_dir: str | Path,
    output_fps: float | None = None,
) -> list[str]:
    out = ensure_dir(out_dir)
    outputs: list[str] = []

    for i, seg in enumerate(segments, start=1):
        start_s = float(seg.get("start_time_s", 0.0))
        end_s = float(seg.get("end_time_s", 0.0))
        if end_s <= start_s:
            continue

        clip_path = out / f"match_{i:03d}.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_s:.3f}",
            "-to",
            f"{end_s:.3f}",
            "-i",
            str(video_path),
            "-an",
        ]
        if output_fps and output_fps > 0:
            cmd.extend(["-vf", f"fps={float(output_fps):.3f}"])
        cmd.extend([
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(clip_path),
        ])
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to export raw match clip {clip_path}: {exc.stderr.strip()}") from exc

        outputs.append(str(clip_path))

    write_json(
        out / "matches_raw.json",
        {
            "source_video": str(video_path),
            "outputs": outputs,
            "segment_count": len(segments),
        },
    )
    return outputs


def _resolve_output_fps(cfg: dict, cli_fps: int) -> int:
    configured = (cfg.get("processing") or {}).get("output_fps")
    if configured is None:
        return int(cli_fps)
    value = int(configured)
    if value <= 0:
        raise ValueError("processing.output_fps must be > 0")
    return value


def _probe_video_duration_s(video_path: str | Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video for duration probing: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    cap.release()
    if fps <= 0.0 or frame_count <= 0.0:
        raise RuntimeError(f"Unable to determine fps/frame count for clip: {video_path}")
    return frame_count / fps


def cmd_detect(args):
    from .detect import DetectionSettings, detect_fuel

    detect_cfg = {}
    if args.config:
        cfg = load_config(args.config)
        detect_cfg = cfg.get("detect", {}) or {}

    def _opt(name: str, cli_value):
        return detect_cfg.get(name, cli_value)

    _log("[chevron] detect: starting")
    settings = DetectionSettings(
        threshold=float(_opt("threshold", args.threshold)),
        scale_min=float(_opt("scale_min", args.scale_min)),
        scale_max=float(_opt("scale_max", args.scale_max)),
        scale_steps=int(_opt("scale_steps", args.scale_steps)),
        nms_iou_threshold=float(_opt("nms_iou_threshold", args.nms_iou_threshold)),
        max_detections_per_frame=int(_opt("max_detections_per_frame", args.max_detections_per_frame)),
        max_candidates_per_scale=int(_opt("max_candidates_per_scale", args.max_candidates_per_scale)),
    )
    summary = detect_fuel(
        video_dir=args.video_dir,
        reference_image=args.reference,
        out_dir=args.out,
        settings=settings,
        show_preview=bool(args.preview),
        combine=bool(args.combine),
    )
    _log(f"[chevron] detect: outputs -> {summary.get('outputs', {})}")
    if summary.get("combined_outputs"):
        _log(f"[chevron] detect: combined -> {summary['combined_outputs']}")
    _log("[chevron] detect: complete")



def cmd_ingest(args):
    from .ingest import ingest

    _log("[chevron] ingest: starting")
    cookie_header = os.getenv("CHEVRON_YOUTUBE_COOKIE")
    cookies_from_browser = os.getenv("CHEVRON_YOUTUBE_BROWSER")
    cookies_file = os.getenv("CHEVRON_YOUTUBE_COOKIES_FILE")
    meta = ingest(
        url=args.url,
        video=args.video,
        out_dir=args.out,
        fps=args.fps,
        logger=_log,
        youtube_cookie_header=args.youtube_cookie or cookie_header,
        youtube_cookies_from_browser=args.youtube_cookies_from_browser or cookies_from_browser,
        youtube_cookies_file=args.youtube_cookies_file or cookies_file,
        start_at=args.start_at,
        end_at=args.end_at,
    )

    if args.select_capture_area:
        from .ui.capture_area_selector import select_capture_area

        proxy = meta.get("proxy")
        if not proxy:
            raise RuntimeError("Ingest did not produce a proxy video for capture-area selection")

        capture_area_out = Path(args.capture_area_out) if args.capture_area_out else Path(args.out) / "capture_area.json"
        _log(f"[chevron] ingest: opening capture-area selector -> {proxy}")
        capture_area = select_capture_area(proxy, capture_area_out)
        _log(f"[chevron] ingest: capture area saved -> {capture_area_out} ({capture_area})")

    _log("[chevron] ingest: complete")



def _write_run_status(path: Path, stage: str, details: dict | None = None) -> None:
    payload = {"stage": stage, "updated_unix_s": time()}
    if details:
        payload["details"] = details
    write_json(path, payload)


def _load_existing_ingest_meta(workdir: Path) -> dict | None:
    ingest_meta_path = workdir / "ingest_meta.json"
    if not ingest_meta_path.exists():
        return None
    meta = read_json(ingest_meta_path)
    proxy = Path(meta.get("proxy", ""))
    if not proxy.exists():
        return None
    return meta


def _resolve_pipeline_video(proxy: str | Path, workdir: Path) -> str:
    proxy_path = Path(proxy)
    capture_area_path = workdir / "capture_area.json"
    if not capture_area_path.exists():
        return str(proxy_path)

    capture_area = read_json(capture_area_path)
    required = ("x", "y", "w", "h")
    if not all(k in capture_area for k in required):
        raise RuntimeError(f"Invalid capture area json at {capture_area_path}: expected keys {required}")

    x = int(capture_area["x"])
    y = int(capture_area["y"])
    w = int(capture_area["w"])
    h = int(capture_area["h"])
    if w <= 0 or h <= 0:
        raise RuntimeError(f"Invalid capture area dimensions in {capture_area_path}: w={w}, h={h}")

    cropped_proxy = workdir / "proxy_cropped.mp4"
    crop_video(proxy_path, cropped_proxy, x=x, y=y, w=w, h=h)
    return str(cropped_proxy)


def _load_existing_calibration(calib_path: Path) -> dict | None:
    if not calib_path.exists():
        return None
    return read_json(calib_path)

def cmd_segment(args):
    from .segment.detector import detect_segments

    cfg = load_config(args.config)
    _log("[chevron] segment: starting")

    def _on_progress(details: dict) -> None:
        _log(_format_segment_progress(details))

    detect_segments(
        args.video,
        cfg,
        args.out,
        debug_dir=Path(args.out).with_suffix("").as_posix() + "_debug",
        progress_interval_s=cfg.get("monitoring", {}).get("segment_progress_interval_s", 10.0),
        progress_callback=_on_progress,
    )
    _log("[chevron] segment: complete")


def cmd_split(args):
    import cv2
    from .split.layout import get_layout
    from .split.tools import export_layout_preview

    cfg = load_config(args.config)
    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Unable to read video")
    h, w = frame.shape[:2]
    layout = get_layout(cfg, w, h)
    out = ensure_dir(args.out)
    write_json(out / "crops.json", layout)
    export_layout_preview(args.video, layout, out / "layout_preview.png")
    _log("[chevron] split: complete")


def cmd_calibrate(args):
    from .calib.homography import compute_homography
    from .calib.manual_click import extract_view_frames

    cfg = load_config(args.config)
    crops = cfg["split"]["crops"]
    out = ensure_dir(args.out)
    _log(f"[chevron] calibrate: extracting view frames -> {out / 'calib_frames'}")
    extract_view_frames(args.video, crops, out / "calib_frames")

    correspondences = getattr(args, "correspondences", None) or cfg["calibration"]["correspondences"]

    homographies = {}
    for cam_name, pairs in correspondences.items():
        _log(f"[chevron] calibrate: computing homography for view={cam_name} points={len(pairs.get('image_points', []))}")
        h = compute_homography(pairs["image_points"], pairs["field_points"])
        homographies[cam_name] = h.tolist()

    field = cfg["field"]
    px = field["px_per_unit"]
    calib = {
        "version": "v1",
        "field": field,
        "canvas": {
            "width_px": int(field["width_units"] * px),
            "height_px": int(field["height_units"] * px),
        },
        "homographies": homographies,
    }
    write_json(out / "calib.json", calib)
    _log("[chevron] calibrate: complete")


def cmd_render(args):
    from .render.pipeline import render_matches

    seg = read_json(args.segments)["segments"]
    calib = read_json(args.calib)
    cfg = load_config(args.config)
    output_fps = _resolve_output_fps(cfg, args.fps)

    raw_out = Path(args.out) / "matches_raw"
    _log(f"[chevron] render: exporting raw matches -> {raw_out}")
    raw_outputs = _export_raw_matches(args.video, seg, raw_out, output_fps=output_fps)
    _log(f"[chevron] render: raw export complete clips={len(raw_outputs)}")

    def _on_render_progress(details: dict) -> None:
        _log(_format_render_progress(details))

    render_matches(
        args.video,
        seg,
        calib,
        cfg,
        args.out,
        output_fps=output_fps,
        progress_interval_s=cfg.get("monitoring", {}).get("render_progress_interval_s", 5.0),
        progress_callback=_on_render_progress,
    )
    _log("[chevron] render: complete")




def cmd_verify(args):
    from .ui.verify_gui import run_local_verify

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_local_verify(args.video, args.config, out_json=out, frame_idx=args.frame)
    _log(f"[chevron] verify: saved correspondences -> {out}")

def cmd_run(args):
    from .ingest import ingest

    work = ensure_dir(args.out)
    cfg = load_config(args.config)
    output_fps = _resolve_output_fps(cfg, args.fps)
    workdir = ensure_dir(work / "workdir")
    run_status_path = workdir / "run_status.json"

    _log(f"[chevron] run: writing status to {run_status_path}")
    _write_run_status(run_status_path, "ingest_starting", {"resume": bool(args.resume)})
    ingest_meta = _load_existing_ingest_meta(workdir) if args.resume else None
    if ingest_meta is None:
        _log("[chevron] run: ingest stage")
        cookie_header = os.getenv("CHEVRON_YOUTUBE_COOKIE")
        cookies_from_browser = os.getenv("CHEVRON_YOUTUBE_BROWSER")
        cookies_file = os.getenv("CHEVRON_YOUTUBE_COOKIES_FILE")
        ingest_meta = ingest(
            url=args.url,
            video=args.video,
            out_dir=workdir,
            fps=output_fps,
            logger=_log,
            youtube_cookie_header=args.youtube_cookie or cookie_header,
            youtube_cookies_from_browser=args.youtube_cookies_from_browser or cookies_from_browser,
            youtube_cookies_file=args.youtube_cookies_file or cookies_file,
            start_at=args.start_at,
            end_at=args.end_at,
        )
        _write_run_status(run_status_path, "ingest_complete", {"proxy": ingest_meta.get("proxy")})
    else:
        _log("[chevron] run: reusing previous ingest output")
        _write_run_status(run_status_path, "ingest_reused", {"proxy": ingest_meta.get("proxy")})

    proxy = ingest_meta["proxy"]
    pipeline_video = _resolve_pipeline_video(proxy, workdir)
    segments_path = workdir / "segments.json"
    _log("[chevron] run: segment stage")
    from .segment.detector import detect_segments

    _write_run_status(run_status_path, "segment_starting", {"video": pipeline_video})

    def _on_segment_progress(details: dict) -> None:
        _log(_format_segment_progress(details))
        _write_run_status(run_status_path, "segment_running", details)

    detect_segments(
        pipeline_video,
        cfg,
        segments_path,
        debug_dir=workdir / "segment_debug",
        progress_interval_s=cfg.get("monitoring", {}).get("segment_progress_interval_s", 10.0),
        progress_callback=_on_segment_progress,
    )
    segment_payload = read_json(segments_path)
    segments = segment_payload.get("segments", [])
    _write_run_status(run_status_path, "segment_complete", {"segments": str(segments_path)})

    _log("[chevron] run: raw match export stage")
    raw_out = work / "matches_raw"
    _write_run_status(run_status_path, "raw_export_starting", {"matches_raw_out": str(raw_out)})
    raw_outputs = _export_raw_matches(proxy, segments, raw_out, output_fps=output_fps)
    _write_run_status(run_status_path, "raw_export_complete", {"clips": len(raw_outputs), "matches_raw_out": str(raw_out)})

    proxy_path = Path(proxy)
    if proxy_path.exists():
        proxy_path.unlink()
        _log(f"[chevron] run: removed ingest proxy after raw export -> {proxy_path}")
    _write_run_status(run_status_path, "ingest_proxy_deleted", {"proxy": str(proxy_path)})

    _log("[chevron] run: render stage")
    from .render.pipeline import render_matches

    total_clips = len(raw_outputs)
    for clip_index, clip_path in enumerate(raw_outputs, start=1):
        clip = Path(clip_path)
        clip_duration_s = _probe_video_duration_s(clip)
        if clip_duration_s <= 0.0:
            raise RuntimeError(f"Raw match clip has non-positive duration: {clip}")

        clip_workdir = ensure_dir(workdir / f"match_{clip_index:03d}")
        clip_verify_out = clip_workdir / "verify_correspondences.json"
        clip_calib_out = ensure_dir(clip_workdir / "calib")
        clip_calib_path = clip_calib_out / "calib.json"
        clip_render_out = ensure_dir(work / "matches" / f"match_{clip_index:03d}")

        _log(f"[chevron] run: verify stage clip={clip_index}/{total_clips}")
        _write_run_status(
            run_status_path,
            "verify_starting",
            {
                "clip_index": clip_index,
                "clip_total": total_clips,
                "video": str(clip),
                "config": args.config,
                "frame_idx": 0,
            },
        )
        verify_namespace = argparse.Namespace(
            video=str(clip),
            config=args.config,
            out=str(clip_verify_out),
            frame=0,
        )
        cmd_verify(verify_namespace)
        verified = read_json(clip_verify_out)
        correspondences = verified.get("correspondences", {})
        _write_run_status(
            run_status_path,
            "verify_complete",
            {"clip_index": clip_index, "clip_total": total_clips, "video": str(clip), "config": args.config, "out": str(clip_verify_out)},
        )

        _log(f"[chevron] run: calibrate stage clip={clip_index}/{total_clips}")
        _write_run_status(
            run_status_path,
            "calibrate_starting",
            {"clip_index": clip_index, "clip_total": total_clips, "out": str(clip_calib_out)},
        )
        namespace = argparse.Namespace(video=str(clip), config=args.config, out=clip_calib_out, correspondences=correspondences)
        cmd_calibrate(namespace)
        calib = read_json(clip_calib_path)
        _write_run_status(
            run_status_path,
            "calibrate_complete",
            {"clip_index": clip_index, "clip_total": total_clips, "calib": str(clip_calib_path)},
        )

        _write_run_status(
            run_status_path,
            "render_starting",
            {"clip_index": clip_index, "clip_total": total_clips, "matches_out": str(clip_render_out), "video": str(clip)},
        )

        def _on_render_progress(details: dict, *, _clip_index: int = clip_index) -> None:
            details_with_clip = {**details, "clip_index": _clip_index, "clip_total": total_clips}
            _log(_format_render_progress(details_with_clip))
            _write_run_status(run_status_path, "render_running", details_with_clip)

        render_matches(
            str(clip),
            [{"start_time_s": 0.0, "end_time_s": clip_duration_s}],
            calib,
            cfg,
            clip_render_out,
            output_fps=output_fps,
            progress_interval_s=cfg.get("monitoring", {}).get("render_progress_interval_s", 5.0),
            progress_callback=_on_render_progress,
        )

    _write_run_status(run_status_path, "render_complete")
    _log("[chevron] run: complete")


def build_parser():
    p = argparse.ArgumentParser(prog="chevron")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest")
    s.add_argument("--url")
    s.add_argument("--video")
    s.add_argument("--out", required=True)
    s.add_argument("--fps", type=int, default=30)
    s.add_argument("--youtube-cookie", required=False, help="YouTube Cookie value (or pasted request headers containing Cookie:) from a logged-in browser session")
    s.add_argument("--youtube-cookies-from-browser", required=False, help="Use yt-dlp browser cookies directly (examples: chrome, edge, firefox, chrome:profile) for minimal setup")
    s.add_argument("--youtube-cookies-file", required=False, help="Path to a Netscape-format cookies.txt export for robust YouTube auth fallback")
    s.add_argument("--start-at", required=False, help="Optional YouTube ingest trim start timestamp in HH:MM:SS (used with --end-at)")
    s.add_argument("--end-at", required=False, help="Optional YouTube ingest trim end timestamp in HH:MM:SS (used with --start-at)")
    s.add_argument("--select-capture-area", action="store_true")
    s.add_argument("--capture-area-out", required=False)
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("segment")
    s.add_argument("--video", required=True)
    s.add_argument("--config", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_segment)

    s = sub.add_parser("split")
    s.add_argument("--video", required=True)
    s.add_argument("--segments", required=False)
    s.add_argument("--config", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_split)

    s = sub.add_parser("calibrate")
    s.add_argument("--video", required=True)
    s.add_argument("--config", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("render")
    s.add_argument("--video", required=True)
    s.add_argument("--segments", required=True)
    s.add_argument("--calib", required=True)
    s.add_argument("--config", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--fps", type=int, default=30)
    s.set_defaults(func=cmd_render)


    s = sub.add_parser("verify")
    s.add_argument("--video", required=True)
    s.add_argument("--config", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--frame", type=int, default=0)
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("run")
    s.add_argument("--url")
    s.add_argument("--video")
    s.add_argument("--config", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--fps", type=int, default=30)
    s.add_argument("--youtube-cookie", required=False, help="YouTube Cookie value (or pasted request headers containing Cookie:) from a logged-in browser session")
    s.add_argument("--youtube-cookies-from-browser", required=False, help="Use yt-dlp browser cookies directly (examples: chrome, edge, firefox, chrome:profile) for minimal setup")
    s.add_argument("--youtube-cookies-file", required=False, help="Path to a Netscape-format cookies.txt export for robust YouTube auth fallback")
    s.add_argument("--start-at", required=False, help="Optional YouTube ingest trim start timestamp in HH:MM:SS (used with --end-at)")
    s.add_argument("--end-at", required=False, help="Optional YouTube ingest trim end timestamp in HH:MM:SS (used with --start-at)")
    s.add_argument("--resume", dest="resume", action="store_true")
    s.add_argument("--no-resume", dest="resume", action="store_false")
    s.add_argument("--verify-port", type=int, default=8501)
    s.add_argument("--verify-host", default="127.0.0.1")
    s.add_argument("--verify-browser", dest="verify_browser", action="store_true")
    s.add_argument("--no-verify-browser", dest="verify_browser", action="store_false")
    s.set_defaults(resume=True)
    s.set_defaults(verify_browser=True)
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("detect")
    s.add_argument("--video-dir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--reference", required=True)
    s.add_argument("--config", required=False)
    s.add_argument("--threshold", type=float, default=0.58)
    s.add_argument("--scale-min", type=float, default=0.75)
    s.add_argument("--scale-max", type=float, default=1.25)
    s.add_argument("--scale-steps", type=int, default=12)
    s.add_argument("--nms-iou-threshold", type=float, default=0.25)
    s.add_argument("--max-detections-per-frame", type=int, default=300)
    s.add_argument("--max-candidates-per-scale", type=int, default=400)
    s.add_argument("--combine", action="store_true")
    s.add_argument("--preview", action="store_true")
    s.set_defaults(func=cmd_detect)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(f"[chevron] error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
