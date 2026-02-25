import numpy as np

from chevron.render.blend import blend_layers


def test_blend_layers_weighted_combination():
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    b = np.zeros((4, 4, 3), dtype=np.uint8)
    a[..., 0] = 255
    b[..., 2] = 255
    ma = np.ones((4, 4), dtype=np.float32)
    mb = np.ones((4, 4), dtype=np.float32)
    out = blend_layers([a, b], [ma, mb])
    assert out.shape == a.shape
    assert out[..., 0].mean() > 0
    assert out[..., 2].mean() > 0
