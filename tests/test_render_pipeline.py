import cv2
import numpy as np

from chevron.render.pipeline import render_matches


def test_render_matches_emits_progress_events(tmp_path):
    video_path = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 32))
    for _ in range(12):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:, :, 1] = 120
        writer.write(frame)
    writer.release()

    segments = [{"start_time_s": 0.0, "end_time_s": 1.0}]
    calib = {
        "version": "v1",
        "canvas": {"width_px": 16, "height_px": 16},
        "homographies": {
            "top": [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
            "bottom_left": [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
            "bottom_right": [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
        },
    }
    cfg = {
        "split": {
            "crops": {
                "top": [0, 0, 16, 16],
                "bottom_left": [0, 16, 16, 16],
                "bottom_right": [16, 16, 16, 16],
            }
        }
    }

    events = []
    outputs = render_matches(
        str(video_path),
        segments,
        calib,
        cfg,
        tmp_path / "matches",
        progress_interval_s=0.1,
        progress_callback=events.append,
    )

    assert len(outputs) == 1
    assert any(evt.get("event") == "match_start" for evt in events)
    assert any(evt.get("event") == "match_progress" for evt in events)
    assert any(evt.get("event") == "match_complete" for evt in events)


def test_render_matches_skips_existing_output(tmp_path):
    video_path = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 32))
    for _ in range(6):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()

    matches_dir = tmp_path / "matches"
    existing_dir = matches_dir / "match_001"
    existing_dir.mkdir(parents=True)
    (existing_dir / "topdown.mp4").write_bytes(b"already there")
    (existing_dir / "match_meta.json").write_text("{}", encoding="utf-8")

    segments = [{"start_time_s": 0.0, "end_time_s": 0.5}]
    calib = {
        "version": "v1",
        "canvas": {"width_px": 16, "height_px": 16},
        "homographies": {
            "top": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "bottom_left": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "bottom_right": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        },
    }
    cfg = {
        "split": {
            "crops": {
                "top": [0, 0, 16, 16],
                "bottom_left": [0, 16, 16, 16],
                "bottom_right": [16, 16, 16, 16],
            }
        }
    }

    events = []
    outputs = render_matches(
        str(video_path),
        segments,
        calib,
        cfg,
        matches_dir,
        progress_callback=events.append,
    )

    assert outputs == [str(existing_dir / "topdown.mp4")]
    assert any(evt.get("event") == "match_skipped" for evt in events)
    assert not any(evt.get("event") == "match_start" for evt in events)
