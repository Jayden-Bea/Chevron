from chevron.segment.detector import SegmentStateMachine


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
