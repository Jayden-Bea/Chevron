from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .calib.homography import compute_homography
from .calib.manual_click import extract_view_frames
from .ingest import ingest
from .render.pipeline import render_matches
from .segment.detector import detect_segments
from .split.layout import get_layout
from .split.tools import export_layout_preview
from .utils.config import load_config
from .utils.io import ensure_dir, read_json, write_json


def cmd_ingest(args):
    ingest(url=args.url, video=args.video, out_dir=args.out, fps=args.fps)


def cmd_segment(args):
    cfg = load_config(args.config)
    detect_segments(args.video, cfg, args.out, debug_dir=Path(args.out).with_suffix("").as_posix() + "_debug")


def cmd_split(args):
    import cv2

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


def cmd_calibrate(args):
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


def cmd_render(args):
    seg = read_json(args.segments)["segments"]
    calib = read_json(args.calib)
    cfg = load_config(args.config)
    render_matches(args.video, seg, calib, cfg, args.out)




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
    work = ensure_dir(args.out)
    cfg = load_config(args.config)
    ingest_meta = ingest(url=args.url, video=args.video, out_dir=work / "workdir", fps=args.fps)
    proxy = ingest_meta["proxy"]
    segments_path = work / "workdir" / "segments.json"
    detect_segments(proxy, cfg, segments_path, debug_dir=work / "workdir" / "segment_debug")

    calib_out = work / "workdir" / "calib"
    namespace = argparse.Namespace(video=proxy, config=args.config, out=calib_out)
    cmd_calibrate(namespace)
    calib_path = calib_out / "calib.json"

    seg = read_json(segments_path)["segments"]
    calib = read_json(calib_path)
    render_matches(proxy, seg, calib, cfg, work / "matches")


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
    s.set_defaults(func=cmd_run)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
