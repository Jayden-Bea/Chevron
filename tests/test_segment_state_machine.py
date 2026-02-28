import numpy as np

from chevron.segment.detector import (
    SegmentStateMachine,
    _build_template_scale_factors,
    _find_audio_cue_times,
    _resolve_search_rect,
    _validate_match_inputs,
)


def test_state_machine_emits_segment_after_debounce():
    sm = SegmentStateMachine(min_start_s=0.2, min_stop_s=0.2, min_match_s=0.5, max_match_s=10)
    t = 0.0
    dt = 0.1

    for _ in range(3):
        state, seg = sm.update(t, dt, start_hit=True, stop_hit=False)
        t += dt
    assert state == "IN_MATCH"
    assert seg is None

    for _ in range(7):
        state, seg = sm.update(t, dt, start_hit=False, stop_hit=False)
        t += dt

    emitted = None
    for _ in range(3):
        state, seg = sm.update(t, dt, start_hit=False, stop_hit=True)
        emitted = emitted or seg
        t += dt

    assert emitted is not None
    start, end = emitted
    assert end > start


def test_validate_match_inputs_allows_roi_smaller_than_template():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    template = np.zeros((30, 40, 3), dtype=np.uint8)

    _validate_match_inputs(frame, [0, 0, 20, 20], template, "start")


def test_validate_match_inputs_rejects_out_of_bounds_roi():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    template = np.zeros((10, 10, 3), dtype=np.uint8)

    try:
        _validate_match_inputs(frame, [95, 95, 10, 10], template, "stop")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "outside frame bounds" in str(exc)


def test_resolve_search_rect_defaults_to_full_frame():
    frame = np.zeros((80, 120, 3), dtype=np.uint8)

    rect = _resolve_search_rect(frame, None, "start")

    assert rect == [0, 0, 120, 80]


def test_resolve_search_rect_rejects_bad_shape():
    frame = np.zeros((80, 120, 3), dtype=np.uint8)

    try:
        _resolve_search_rect(frame, [0, 0, 120], "start")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "must be [x, y, w, h]" in str(exc)


def test_build_template_scale_factors_handles_zero_tolerance():
    assert _build_template_scale_factors(0.0) == [1.0]


def test_build_template_scale_factors_builds_symmetric_range():
    factors = _build_template_scale_factors(1.0, step_pct=0.5)

    assert factors == [0.99, 0.995, 1.0, 1.005, 1.01]


def test_fit_template_to_image_downscales_when_needed():
    from chevron.segment.detector import _fit_template_to_image

    template = np.zeros((64, 343, 3), dtype=np.uint8)
    fitted = _fit_template_to_image(template, img_h=80, img_w=200)

    h, w = fitted.shape[:2]
    assert h <= 80
    assert w <= 200


def test_detect_segments_reports_progress_callback(tmp_path):
    import cv2
    from chevron.segment.detector import detect_segments

    video_path = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (32, 32),
    )
    for _ in range(12):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()

    start_tpl_path = tmp_path / "start.png"
    stop_tpl_path = tmp_path / "stop.png"
    cv2.imwrite(str(start_tpl_path), np.zeros((8, 8, 3), dtype=np.uint8))
    cv2.imwrite(str(stop_tpl_path), np.zeros((8, 8, 3), dtype=np.uint8))

    cfg = {
        "templates": {"start": str(start_tpl_path), "stop": str(stop_tpl_path)},
        "thresholds": {"start": 2.0, "stop": 2.0},
        "debounce": {"min_start_s": 5.0, "min_stop_s": 5.0, "min_match_s": 10.0, "max_match_s": 20.0},
    }

    progress = []
    out_json = tmp_path / "segments.json"
    segments = detect_segments(
        video_path,
        cfg,
        out_json,
        progress_interval_s=0.25,
        progress_callback=progress.append,
    )

    assert segments == []
    assert progress
    assert progress[0]["frame_idx"] == 0
    assert progress[0]["total_frames"] >= 1
    assert "start_score" in progress[0]
    assert "stop_score" in progress[0]


def test_find_audio_cue_times_finds_embedded_cues_with_noise():
    sample_rate_hz = 16000
    cue_t = np.linspace(0, 0.5, int(sample_rate_hz * 0.5), endpoint=False)
    cue = (0.6 * np.sin(2 * np.pi * 880 * cue_t)).astype(np.float32)

    program = np.random.default_rng(0).normal(0, 0.04, size=int(sample_rate_hz * 6.0)).astype(np.float32)
    first = int(1.0 * sample_rate_hz)
    second = int(4.0 * sample_rate_hz)
    program[first : first + cue.size] += cue
    program[second : second + cue.size] += cue

    hits = _find_audio_cue_times(
        program_audio=program,
        cue_audio=cue,
        sample_rate_hz=sample_rate_hz,
        threshold=0.30,
        min_separation_s=2.0,
    )

    assert len(hits) == 2
    assert abs(hits[0][0] - 1.0) < 0.3
    assert abs(hits[1][0] - 4.0) < 0.3


def test_find_audio_cue_times_respects_minimum_separation():
    sample_rate_hz = 16000
    cue_t = np.linspace(0, 0.4, int(sample_rate_hz * 0.4), endpoint=False)
    cue = (0.7 * np.sin(2 * np.pi * 1200 * cue_t)).astype(np.float32)

    program = np.zeros(int(sample_rate_hz * 3.0), dtype=np.float32)
    start_a = int(0.8 * sample_rate_hz)
    start_b = int(1.2 * sample_rate_hz)
    program[start_a : start_a + cue.size] += cue
    program[start_b : start_b + cue.size] += cue

    hits = _find_audio_cue_times(
        program_audio=program,
        cue_audio=cue,
        sample_rate_hz=sample_rate_hz,
        threshold=0.30,
        min_separation_s=1.0,
    )

    assert len(hits) == 1
