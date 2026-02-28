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
    total_frames = details.get("total_frames", 0) or 0
    frame_idx = details.get("frame_idx", 0)
    if total_frames > 0:
        pct = max(0.0, min(100.0, (frame_idx / total_frames) * 100.0))
        progress = f"{frame_idx}/{total_frames} ({pct:.1f}%)"
    else:
        progress = f"{frame_idx} frames"

    return (
        "[chevron] run: segment progress "
        f"t={details.get('time_s', 0.0):.1f}s "
        f"frame={progress} "
        f"state={details.get('state')} "
        f"scores(start={details.get('start_score', 0.0):.3f}, stop={details.get('stop_score', 0.0):.3f}) "
        f"hits(start={details.get('start_hit')}, stop={details.get('stop_hit')}) "
        f"segments={details.get('segments_found', 0)}"
    )


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
    extract_view_frames(args.video, crops, out / "calib_frames")

    homographies = {}
    for cam_name, pairs in cfg["calibration"]["correspondences"].items():
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
    render_matches(args.video, seg, calib, cfg, args.out)
    _log("[chevron] render: complete")




def cmd_verify(args):
    app_path = Path(__file__).resolve().parent / "ui" / "verify_app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
        "--server.headless",
        "false" if args.browser else "true",
        "--",
        "--video",
        args.video,
        "--config",
        args.config,
    ]
    if args.calib:
        cmd.extend(["--calib", args.calib])
    if args.frame is not None:
        cmd.extend(["--frame", str(args.frame)])
    subprocess.run(cmd, check=True)

def cmd_run(args):
    from .ingest import ingest

    work = ensure_dir(args.out)
    cfg = load_config(args.config)
    workdir = ensure_dir(work / "workdir")
    run_status_path = workdir / "run_status.json"

    _log(f"[chevron] run: writing status to {run_status_path}")
    _write_run_status(run_status_path, "ingest_starting", {"resume": bool(args.resume)})
    ingest_meta = _load_existing_ingest_meta(workdir) if args.resume else None
    if ingest_meta is None:
        _log("[chevron] run: ingest stage")
        ingest_meta = ingest(url=args.url, video=args.video, out_dir=workdir, fps=args.fps)
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
    _log("[chevron] run: calibrate stage")
    _write_run_status(run_status_path, "calibrate_starting", {"out": str(calib_out)})
    namespace = argparse.Namespace(video=proxy, config=args.config, out=calib_out)
    cmd_calibrate(namespace)
    calib_path = calib_out / "calib.json"
    _write_run_status(run_status_path, "calibrate_complete", {"calib": str(calib_path)})

    seg = read_json(segments_path)["segments"]
    calib = read_json(calib_path)
    _log("[chevron] run: render stage")
    from .render.pipeline import render_matches

    _write_run_status(run_status_path, "render_starting", {"matches_out": str(work / "matches")})
    render_matches(proxy, seg, calib, cfg, work / "matches")
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
    s.set_defaults(func=cmd_render)


    s = sub.add_parser("verify")
    s.add_argument("--video", required=True)
    s.add_argument("--config", required=True)
    s.add_argument("--calib")
    s.add_argument("--frame", type=int, default=0)
    s.add_argument("--port", type=int, default=8501)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--browser", dest="browser", action="store_true")
    s.add_argument("--no-browser", dest="browser", action="store_false")
    s.set_defaults(browser=True)
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
