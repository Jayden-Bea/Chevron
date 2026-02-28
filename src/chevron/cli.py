from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from time import time

from .utils.config import load_config
from .utils.io import ensure_dir, read_json, write_json


def _log(message: str) -> None:
    print(message, flush=True)


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



def cmd_ingest(args):
    from .ingest import ingest

    _log("[chevron] ingest: starting")
    ingest(url=args.url, video=args.video, out_dir=args.out, fps=args.fps)
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
        ingest_meta = ingest(url=args.url, video=args.video, out_dir=workdir, fps=output_fps)
        _write_run_status(run_status_path, "ingest_complete", {"proxy": ingest_meta.get("proxy")})
    else:
        _log("[chevron] run: reusing previous ingest output")
        _write_run_status(run_status_path, "ingest_reused", {"proxy": ingest_meta.get("proxy")})

    proxy = ingest_meta["proxy"]
    segments_path = workdir / "segments.json"
    _log("[chevron] run: segment stage")
    from .segment.detector import detect_segments

    _write_run_status(run_status_path, "segment_starting", {"video": proxy})

    def _on_segment_progress(details: dict) -> None:
        _log(_format_segment_progress(details))
        _write_run_status(run_status_path, "segment_running", details)

    detect_segments(
        proxy,
        cfg,
        segments_path,
        debug_dir=workdir / "segment_debug",
        progress_interval_s=cfg.get("monitoring", {}).get("segment_progress_interval_s", 10.0),
        progress_callback=_on_segment_progress,
    )
    _write_run_status(run_status_path, "segment_complete", {"segments": str(segments_path)})

    calib_out = workdir / "calib"
    calib_path = calib_out / "calib.json"
    verify_out = workdir / "verify_correspondences.json"

    _log("[chevron] run: verify stage")
    _write_run_status(run_status_path, "verify_starting", {"video": proxy, "config": args.config})
    verify_namespace = argparse.Namespace(
        video=proxy,
        config=args.config,
        out=str(verify_out),
        frame=0,
    )
    cmd_verify(verify_namespace)
    verified = read_json(verify_out)
    correspondences = verified.get("correspondences", {})
    _write_run_status(run_status_path, "verify_complete", {"video": proxy, "config": args.config, "out": str(verify_out)})

    _log("[chevron] run: calibrate stage")
    _write_run_status(run_status_path, "calibrate_starting", {"out": str(calib_out)})
    namespace = argparse.Namespace(video=proxy, config=args.config, out=calib_out, correspondences=correspondences)
    cmd_calibrate(namespace)
    calib = read_json(calib_path)
    _write_run_status(run_status_path, "calibrate_complete", {"calib": str(calib_path)})

    seg = read_json(segments_path)["segments"]
    _log("[chevron] run: raw match export stage")
    raw_out = work / "matches_raw"
    _write_run_status(run_status_path, "raw_export_starting", {"matches_raw_out": str(raw_out)})
    raw_outputs = _export_raw_matches(proxy, seg, raw_out, output_fps=output_fps)
    _write_run_status(run_status_path, "raw_export_complete", {"clips": len(raw_outputs), "matches_raw_out": str(raw_out)})

    _log("[chevron] run: render stage")
    from .render.pipeline import render_matches

    _write_run_status(run_status_path, "render_starting", {"matches_out": str(work / "matches")})
    def _on_render_progress(details: dict) -> None:
        _log(_format_render_progress(details))
        _write_run_status(run_status_path, "render_running", details)

    render_matches(
        proxy,
        seg,
        calib,
        cfg,
        work / "matches",
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
    s.add_argument("--resume", dest="resume", action="store_true")
    s.add_argument("--no-resume", dest="resume", action="store_false")
    s.set_defaults(resume=True)
    s.set_defaults(func=cmd_run)
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
