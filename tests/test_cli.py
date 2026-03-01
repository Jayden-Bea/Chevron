from pathlib import Path
import re

from chevron.cli import (
    _export_raw_matches,
    _format_render_progress,
    _format_segment_progress,
    _log,
    _load_existing_calibration,
    _resolve_output_fps,
    _resolve_pipeline_video,
    build_parser,
    cmd_detect,
    cmd_ingest,
    cmd_run,
)


def test_format_segment_progress_formats_numeric_scores():
    message = _format_segment_progress(
        {
            "frame_idx": 50,
            "total_frames": 100,
            "time_s": 5.0,
            "state": "IN_MATCH",
            "start_score": 0.81234,
            "stop_score": 0.12345,
            "start_hit": True,
            "stop_hit": False,
            "segments_found": 1,
        }
    )

    assert "scores(start=0.812, stop=0.123)" in message


def test_format_segment_progress_handles_missing_scores():
    message = _format_segment_progress(
        {
            "frame_idx": 0,
            "total_frames": 10,
            "time_s": 0.0,
            "state": "AUDIO_CUE",
            "start_score": None,
            "stop_score": None,
            "start_hit": True,
            "stop_hit": None,
            "segments_found": 2,
        }
    )

    assert "scores(start=n/a, stop=n/a)" in message


def test_format_render_progress_for_match_updates():
    start = _format_render_progress(
        {
            "event": "match_start",
            "match_index": 2,
            "total_matches": 4,
            "start_time_s": 30.0,
            "end_time_s": 180.0,
            "total_frames": 4500,
        }
    )
    progress = _format_render_progress(
        {
            "event": "match_progress",
            "match_index": 2,
            "total_matches": 4,
            "match_frame_idx": 2250,
            "match_total_frames": 4500,
            "video_time_s": 105.0,
        }
    )
    complete = _format_render_progress(
        {
            "event": "match_complete",
            "match_index": 2,
            "total_matches": 4,
            "written_frames": 4495,
            "output": "out/match_002/topdown.mp4",
        }
    )

    assert "render start" in start
    assert "match=2/4" in start
    assert "render progress" in progress
    assert "2250/4500" in progress
    assert "render complete" in complete
    assert "written_frames=4495" in complete


def test_export_raw_matches_writes_expected_clips(monkeypatch, tmp_path):
    calls = []

    class Result:
        returncode = 0

    def fake_run(cmd, check, capture_output, text):
        calls.append(cmd)
        out_path = Path(cmd[-1])
        out_path.write_bytes(b"clip")
        return Result()

    monkeypatch.setattr("subprocess.run", fake_run)

    outputs = _export_raw_matches(
        "video.mp4",
        [
            {"start_time_s": 1.0, "end_time_s": 3.5},
            {"start_time_s": 5.0, "end_time_s": 5.0},
            {"start_time_s": 8.25, "end_time_s": 10.0},
        ],
        tmp_path / "matches_raw",
    )

    assert len(calls) == 2
    assert len(outputs) == 2
    assert outputs[0].endswith("match_001.mp4")
    assert outputs[1].endswith("match_003.mp4")
    assert (tmp_path / "matches_raw" / "matches_raw.json").exists()


def test_format_render_progress_for_skipped_match():
    message = _format_render_progress(
        {
            "event": "match_skipped",
            "match_index": 1,
            "total_matches": 3,
            "reason": "already_rendered",
            "output": "out/match_001/topdown.mp4",
        }
    )

    assert "render skipped" in message
    assert "reason=already_rendered" in message


def test_load_existing_calibration_returns_none_for_missing_file(tmp_path):
    assert _load_existing_calibration(tmp_path / "missing_calib.json") is None


def test_load_existing_calibration_reads_existing_file(tmp_path):
    calib_path = tmp_path / "calib.json"
    calib_path.write_text('{"version": "v1"}', encoding="utf-8")

    calib = _load_existing_calibration(calib_path)

    assert calib == {"version": "v1"}


def test_verify_parser_requires_output_path():
    parser = build_parser()

    args = parser.parse_args([
        "verify",
        "--video",
        "proxy.mp4",
        "--config",
        "configs/example_config.yml",
        "--out",
        "out/verify_correspondences.json",
    ])

    assert args.out.endswith("verify_correspondences.json")




def test_ingest_parser_accepts_capture_area_selection_flags():
    parser = build_parser()

    args = parser.parse_args([
        "ingest",
        "--video",
        "workdir/proxy.mp4",
        "--out",
        "workdir",
        "--select-capture-area",
        "--capture-area-out",
        "workdir/custom_capture_area.json",
    ])

    assert args.select_capture_area is True
    assert args.capture_area_out == "workdir/custom_capture_area.json"


def test_cmd_ingest_runs_capture_area_selector_when_enabled(monkeypatch, tmp_path):
    import argparse
    import sys
    import types

    captured = {}

    def fake_ingest(url, video, out_dir, fps):
        proxy = tmp_path / "proxy.mp4"
        proxy.write_bytes(b"fake")
        return {"proxy": str(proxy)}

    def fake_select_capture_area(video_path, out_json):
        captured["video_path"] = video_path
        captured["out_json"] = str(out_json)
        return {"x": 10, "y": 20, "w": 100, "h": 80, "frame_idx": 0}

    monkeypatch.setitem(sys.modules, "chevron.ingest", types.SimpleNamespace(ingest=fake_ingest))
    monkeypatch.setitem(
        sys.modules,
        "chevron.ui.capture_area_selector",
        types.SimpleNamespace(select_capture_area=fake_select_capture_area),
    )

    args = argparse.Namespace(
        url=None,
        video="input.mp4",
        out=str(tmp_path),
        fps=10,
        select_capture_area=True,
        capture_area_out=None,
    )

    cmd_ingest(args)

    assert captured["video_path"].endswith("proxy.mp4")
    assert captured["out_json"].endswith("capture_area.json")



def test_resolve_pipeline_video_uses_proxy_when_no_capture_area(tmp_path):
    proxy = tmp_path / "proxy.mp4"
    proxy.write_bytes(b"proxy")

    resolved = _resolve_pipeline_video(str(proxy), tmp_path)

    assert resolved == str(proxy)


def test_resolve_pipeline_video_crops_when_capture_area_exists(monkeypatch, tmp_path):
    proxy = tmp_path / "proxy.mp4"
    proxy.write_bytes(b"proxy")
    (tmp_path / "capture_area.json").write_text('{"x": 11, "y": 22, "w": 333, "h": 444}', encoding="utf-8")

    captured = {}

    def fake_crop_video(in_path, out_path, x, y, w, h):
        captured["in_path"] = str(in_path)
        captured["out_path"] = str(out_path)
        captured["crop"] = (x, y, w, h)

    monkeypatch.setattr("chevron.cli.crop_video", fake_crop_video)

    resolved = _resolve_pipeline_video(str(proxy), tmp_path)

    assert captured["in_path"].endswith("proxy.mp4")
    assert captured["out_path"].endswith("proxy_cropped.mp4")
    assert captured["crop"] == (11, 22, 333, 444)
    assert resolved.endswith("proxy_cropped.mp4")


def test_cmd_run_uses_full_proxy_for_raw_export_and_cropped_video_for_pipeline(monkeypatch, tmp_path):
    import argparse
    import sys
    import types

    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text("""field:
  width_units: 54
  height_units: 27
  px_per_unit: 10
""", encoding="utf-8")

    out_dir = tmp_path / "out"
    workdir = out_dir / "workdir"
    workdir.mkdir(parents=True)

    proxy = workdir / "proxy.mp4"
    proxy.write_bytes(b"proxy")
    (workdir / "capture_area.json").write_text('{"x": 1, "y": 2, "w": 30, "h": 40}', encoding="utf-8")

    calls = {}

    def fake_ingest(url, video, out_dir, fps):
        return {"proxy": str(proxy)}

    def fake_detect_segments(video, cfg, out, debug_dir, progress_interval_s, progress_callback):
        calls["segment_video"] = str(video)
        Path(out).write_text('{"segments": [{"start_time_s": 0.0, "end_time_s": 1.0}], "fps": 30}', encoding="utf-8")

    def fake_cmd_verify(ns):
        Path(ns.out).write_text('{"correspondences": {}}', encoding="utf-8")

    def fake_cmd_calibrate(ns):
        Path(ns.out).mkdir(parents=True, exist_ok=True)
        (Path(ns.out) / "calib.json").write_text('{"version": "v1"}', encoding="utf-8")

    def fake_export_raw(video_path, segments, out_dir, output_fps=None):
        calls["raw_video"] = str(video_path)
        return []

    def fake_render_matches(video, seg, calib, cfg, out, output_fps, progress_interval_s, progress_callback):
        calls["render_video"] = str(video)

    def fake_crop_video(in_path, out_path, x, y, w, h):
        Path(out_path).write_bytes(b"cropped")

    monkeypatch.setitem(sys.modules, "chevron.ingest", types.SimpleNamespace(ingest=fake_ingest))
    monkeypatch.setitem(sys.modules, "chevron.segment.detector", types.SimpleNamespace(detect_segments=fake_detect_segments))
    monkeypatch.setitem(sys.modules, "chevron.render.pipeline", types.SimpleNamespace(render_matches=fake_render_matches))
    monkeypatch.setattr("chevron.cli.cmd_verify", fake_cmd_verify)
    monkeypatch.setattr("chevron.cli.cmd_calibrate", fake_cmd_calibrate)
    monkeypatch.setattr("chevron.cli._export_raw_matches", fake_export_raw)
    monkeypatch.setattr("chevron.cli.crop_video", fake_crop_video)

    args = argparse.Namespace(
        url=None,
        video="input.mp4",
        config=str(cfg_path),
        out=str(out_dir),
        fps=30,
        resume=False,
        verify_port=8501,
        verify_host="127.0.0.1",
        verify_browser=False,
    )

    cmd_run(args)

    assert calls["raw_video"].endswith("proxy.mp4")
    assert calls["segment_video"].endswith("proxy_cropped.mp4")
    assert calls["render_video"].endswith("proxy_cropped.mp4")

def test_run_parser_defaults_resume_enabled():
    parser = build_parser()

    args = parser.parse_args([
        "run",
        "--video",
        "proxy.mp4",
        "--config",
        "configs/example_config.yml",
        "--out",
        "out_dir",
    ])

    assert args.resume is True


def test_resolve_output_fps_uses_config_when_available():
    fps = _resolve_output_fps({"processing": {"output_fps": 10}}, cli_fps=30)

    assert fps == 10


def test_resolve_output_fps_falls_back_to_cli_default():
    fps = _resolve_output_fps({}, cli_fps=30)

    assert fps == 30


def test_log_includes_absolute_and_relative_timestamps(capsys):
    _log("[chevron] test message")

    captured = capsys.readouterr().out.strip()
    assert "[chevron] test message" in captured
    assert re.search(r"^\[\d{4}-\d{2}-\d{2}T.*Z \+\s*\d+\.\d{2}s\]", captured)


def test_detect_parser_accepts_detection_options():
    parser = build_parser()

    args = parser.parse_args([
        "detect",
        "--video-dir",
        "inputs/matches",
        "--out",
        "out_dir/detect",
        "--reference",
        "fixtures/fuel.png",
        "--threshold",
        "0.6",
        "--scale-min",
        "0.8",
        "--scale-max",
        "1.4",
        "--scale-steps",
        "20",
        "--nms-iou-threshold",
        "0.2",
        "--max-detections-per-frame",
        "500",
        "--max-candidates-per-scale",
        "600",
        "--config",
        "configs/example_config.yml",
        "--combine",
        "--preview",
    ])

    assert args.video_dir == "inputs/matches"
    assert args.out == "out_dir/detect"
    assert args.reference == "fixtures/fuel.png"
    assert args.threshold == 0.6
    assert args.scale_min == 0.8
    assert args.scale_max == 1.4
    assert args.scale_steps == 20
    assert args.nms_iou_threshold == 0.2
    assert args.max_detections_per_frame == 500
    assert args.max_candidates_per_scale == 600
    assert args.config == "configs/example_config.yml"
    assert args.combine is True
    assert args.preview is True


def test_cmd_detect_invokes_detection_pipeline(tmp_path):
    import sys
    import types

    captured = {}

    class FakeSettings:
        def __init__(
            self,
            threshold,
            scale_min,
            scale_max,
            scale_steps,
            nms_iou_threshold,
            max_detections_per_frame,
            max_candidates_per_scale,
        ):
            self.threshold = threshold
            self.scale_min = scale_min
            self.scale_max = scale_max
            self.scale_steps = scale_steps
            self.nms_iou_threshold = nms_iou_threshold
            self.max_detections_per_frame = max_detections_per_frame
            self.max_candidates_per_scale = max_candidates_per_scale

    def fake_detect_fuel(video_dir, reference_image, out_dir, settings, show_preview, combine):
        captured["video_dir"] = video_dir
        captured["reference_image"] = reference_image
        captured["out_dir"] = out_dir
        captured["settings"] = settings
        captured["show_preview"] = show_preview
        captured["combine"] = combine
        return {"outputs": {"counts_json": "counts.json"}, "combined_outputs": {"map_png": "combined.png"}}

    fake_module = types.SimpleNamespace(DetectionSettings=FakeSettings, detect_fuel=fake_detect_fuel)
    sys.modules["chevron.detect"] = fake_module

    args = type(
        "Args",
        (),
        {
            "video_dir": str(tmp_path / "videos"),
            "reference": str(tmp_path / "reference.png"),
            "out": str(tmp_path / "out"),
            "threshold": 0.58,
            "scale_min": 0.75,
            "scale_max": 1.25,
            "scale_steps": 12,
            "nms_iou_threshold": 0.22,
            "max_detections_per_frame": 333,
            "max_candidates_per_scale": 444,
            "preview": True,
            "combine": True,
            "config": None,
        },
    )()

    cmd_detect(args)

    assert captured["video_dir"] == str(tmp_path / "videos")
    assert captured["reference_image"] == str(tmp_path / "reference.png")
    assert captured["out_dir"] == str(tmp_path / "out")
    assert captured["show_preview"] is True
    assert captured["combine"] is True
    assert captured["settings"].threshold == 0.58
    assert captured["settings"].scale_min == 0.75
    assert captured["settings"].scale_max == 1.25
    assert captured["settings"].scale_steps == 12
    assert captured["settings"].nms_iou_threshold == 0.22
    assert captured["settings"].max_detections_per_frame == 333
    assert captured["settings"].max_candidates_per_scale == 444


def test_cmd_detect_uses_config_overrides(monkeypatch, tmp_path):
    captured = {}

    class FakeSettings:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    def fake_detect_fuel(video_dir, reference_image, out_dir, settings, show_preview, combine):
        captured["settings"] = settings
        captured["combine"] = combine
        return {"outputs": {}}

    monkeypatch.setattr("chevron.cli.load_config", lambda _p: {"detect": {"threshold": 0.9, "scale_steps": 3}})

    import sys, types
    fake_module = types.SimpleNamespace(DetectionSettings=FakeSettings, detect_fuel=fake_detect_fuel)
    sys.modules["chevron.detect"] = fake_module

    args = type("Args", (), {
        "video_dir": str(tmp_path / "videos"),
        "reference": str(tmp_path / "reference.png"),
        "out": str(tmp_path / "out"),
        "threshold": 0.58,
        "scale_min": 0.75,
        "scale_max": 1.25,
        "scale_steps": 12,
        "nms_iou_threshold": 0.22,
        "max_detections_per_frame": 333,
        "max_candidates_per_scale": 444,
        "preview": False,
        "combine": False,
        "config": "cfg.yml",
    })()

    cmd_detect(args)

    assert captured["settings"].threshold == 0.9
    assert captured["settings"].scale_steps == 3
    assert captured["settings"].scale_min == 0.75
    assert captured["combine"] is False
