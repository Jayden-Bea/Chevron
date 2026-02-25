import numpy as np

from chevron.ui.verify_app import (
    compute_reprojection_metrics,
    draw_crops,
    draw_polygon,
    draw_points,
    draw_rois,
    resolve_selected_image,
)


def test_overlay_helpers_keep_shape_and_change_pixels():
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    rois = {"start": [5, 5, 30, 20], "stop": [50, 8, 25, 18]}
    crops = {"top": [0, 0, 120, 40], "bottom_left": [0, 40, 60, 40], "bottom_right": [60, 40, 60, 40]}
    pts = [[10, 10], [40, 12], [45, 30], [12, 28], [25, 20]]

    out = draw_rois(frame, rois)
    out = draw_crops(out, crops)
    out = draw_points(out, pts, label="top")
    out = draw_polygon(out, pts)

    assert out.shape == frame.shape
    assert np.count_nonzero(out != frame) > 0


def test_reprojection_metrics_synthetic_identity_mapping():
    correspondences = {
        "top": {
            "image_points": [[0, 0], [1, 0], [1, 1], [0, 1], [0.2, 0.8]],
            "field_points": [[0, 0], [2, 0], [2, 2], [0, 2], [0.4, 1.6]],
        }
    }
    homographies = {
        "top": np.array(
            [
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    }

    metrics = compute_reprojection_metrics(correspondences, homographies)

    assert len(metrics) == 1
    assert metrics[0]["view"] == "top"
    assert metrics[0]["avg_px"] < 1e-5
    assert metrics[0]["p95_px"] < 1e-5


def test_reprojection_metrics_handles_mismatched_point_counts():
    correspondences = {
        "top": {
            "image_points": [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]],
            "field_points": [[0, 0], [2, 0], [2, 2], [0, 2]],
        }
    }
    homographies = {
        "top": np.array(
            [
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    }

    metrics = compute_reprojection_metrics(correspondences, homographies)

    assert len(metrics) == 1
    assert metrics[0]["view"] == "top"
    assert metrics[0]["count"] == 4
    assert metrics[0]["avg_px"] < 1e-5



def test_resolve_selected_image_handles_missing_crop():
    broadcast = np.zeros((20, 30, 3), dtype=np.uint8)
    crop_displays = {"top": np.ones((10, 10, 3), dtype=np.uint8)}
    warped = {}

    selected = resolve_selected_image("bottom_right crop", broadcast, crop_displays, warped, None)

    assert selected is None
