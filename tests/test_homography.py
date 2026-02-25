import numpy as np

from chevron.calib.homography import apply_homography, compute_homography


def test_homography_maps_square_to_scaled_square():
    image_pts = [[0, 0], [10, 0], [10, 10], [0, 10]]
    field_pts = [[0, 0], [20, 0], [20, 20], [0, 20]]
    h = compute_homography(image_pts, field_pts)
    out = apply_homography(h, np.array([[5, 5]], dtype=np.float32))
    assert np.allclose(out[0], [10, 10], atol=1e-2)
