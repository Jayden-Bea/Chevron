import numpy as np

from chevron.segment.detector import (
    SegmentStateMachine,
    _build_template_scale_factors,
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
