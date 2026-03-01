import cv2
import numpy as np

from chevron.detect.pipeline import DetectionSettings, detect_fuel


def _write_test_video(path, frame_count=8):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (80, 64))
    for idx in range(frame_count):
        frame = np.zeros((64, 80, 3), dtype=np.uint8)
        x1 = 8 + (idx * 2)
        x2 = 44 + (idx // 2)
        y = 22
        cv2.rectangle(frame, (x1, y), (x1 + 10, y + 10), (255, 255, 255), -1)
        cv2.rectangle(frame, (x2, y), (x2 + 10, y + 10), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def test_detect_fuel_generates_match_named_outputs_and_counts(tmp_path):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()

    _write_test_video(video_dir / "match_002.mp4")

    reference = np.full((10, 10, 3), 255, dtype=np.uint8)
    ref_path = tmp_path / "reference.png"
    cv2.imwrite(str(ref_path), reference)

    out_dir = tmp_path / "detect_out"
    summary = detect_fuel(
        video_dir=video_dir,
        reference_image=ref_path,
        out_dir=out_dir,
        settings=DetectionSettings(
            threshold=0.55,
            scale_min=1.0,
            scale_max=1.0,
            scale_steps=1,
            nms_iou_threshold=0.2,
            max_detections_per_frame=30,
            max_candidates_per_scale=50,
        ),
    )

    assert (out_dir / "match_002_map.png").exists()
    assert (out_dir / "match_002_map.gif").exists()
    assert (out_dir / "tracked_counts.json").exists()
    assert summary["per_match"][0]["outputs"]["map_png"].endswith("match_002_map.png")
    assert summary["per_match"][0]["outputs"]["map_gif"].endswith("match_002_map.gif")
    assert summary["total_hits"] > 0
    assert summary["total_frames"] == 8
    assert summary["per_match"][0]["detections_per_frame_max"] >= 2


def test_detect_fuel_combine_generates_combined_outputs(tmp_path):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()

    _write_test_video(video_dir / "match_001.mp4", frame_count=6)
    _write_test_video(video_dir / "match_003.mp4", frame_count=6)

    reference = np.full((10, 10, 3), 255, dtype=np.uint8)
    ref_path = tmp_path / "reference.png"
    cv2.imwrite(str(ref_path), reference)

    out_dir = tmp_path / "detect_out"
    summary = detect_fuel(
        video_dir=video_dir,
        reference_image=ref_path,
        out_dir=out_dir,
        combine=True,
        settings=DetectionSettings(
            threshold=0.55,
            scale_min=1.0,
            scale_max=1.0,
            scale_steps=1,
            nms_iou_threshold=0.2,
            max_detections_per_frame=30,
            max_candidates_per_scale=50,
        ),
    )

    assert (out_dir / "match_combined_map.png").exists()
    assert (out_dir / "match_combined_map.gif").exists()
    assert summary["combined_outputs"]["map_png"].endswith("match_combined_map.png")
    assert summary["combined_outputs"]["map_gif"].endswith("match_combined_map.gif")


def test_detect_fuel_raises_when_no_videos(tmp_path):
    video_dir = tmp_path / "empty"
    video_dir.mkdir()
    ref = np.full((8, 8, 3), 255, dtype=np.uint8)
    ref_path = tmp_path / "reference.png"
    cv2.imwrite(str(ref_path), ref)

    try:
        detect_fuel(video_dir=video_dir, reference_image=ref_path, out_dir=tmp_path / "out")
    except ValueError as exc:
        assert "No mp4 files found" in str(exc)
    else:
        raise AssertionError("Expected ValueError when video directory is empty")


def test_detect_fuel_mixed_resolution_without_combine_does_not_fail(tmp_path):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()

    _write_test_video(video_dir / "match_001.mp4", frame_count=4)

    writer = cv2.VideoWriter(str(video_dir / "match_002.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (96, 72))
    for _ in range(4):
        frame = np.zeros((72, 96, 3), dtype=np.uint8)
        cv2.rectangle(frame, (10, 20), (20, 30), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()

    reference = np.full((10, 10, 3), 255, dtype=np.uint8)
    ref_path = tmp_path / "reference.png"
    cv2.imwrite(str(ref_path), reference)

    summary = detect_fuel(video_dir=video_dir, reference_image=ref_path, out_dir=tmp_path / "out", combine=False)

    assert len(summary["per_match"]) == 2


def test_detect_fuel_mixed_resolution_with_combine_skips_incompatible(tmp_path):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()

    _write_test_video(video_dir / "match_001.mp4", frame_count=4)

    writer = cv2.VideoWriter(str(video_dir / "match_002.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (96, 72))
    for _ in range(4):
        frame = np.zeros((72, 96, 3), dtype=np.uint8)
        cv2.rectangle(frame, (10, 20), (20, 30), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()

    reference = np.full((10, 10, 3), 255, dtype=np.uint8)
    ref_path = tmp_path / "reference.png"
    cv2.imwrite(str(ref_path), reference)

    summary = detect_fuel(video_dir=video_dir, reference_image=ref_path, out_dir=tmp_path / "out", combine=True)

    assert (tmp_path / "out" / "match_combined_map.png").exists()
    assert str(video_dir / "match_002.mp4") in summary["combine_skipped_videos"]
