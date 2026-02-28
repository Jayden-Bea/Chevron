from chevron.cli import _format_segment_progress


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
