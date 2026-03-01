from pathlib import Path
import re

from chevron.cli import (
    _export_raw_matches,
    _format_render_progress,
    _format_segment_progress,
    _log,
    _load_existing_calibration,
    _resolve_output_fps,
    build_parser,
    cmd_detect,
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
