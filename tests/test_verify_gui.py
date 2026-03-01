import pytest

pytest.importorskip("cv2", exc_type=ImportError)

import numpy as np

from chevron.ui.verify_gui import run_local_verify


class _FakeCapture:
    def __init__(self, frame):
        self._frame = frame

    def set(self, *_args, **_kwargs):
        return True

    def read(self):
        return True, self._frame.copy()

    def release(self):
        return None


def _stub_cv2_gui(monkeypatch, keys):
    key_iter = iter(keys)

    monkeypatch.setattr("cv2.namedWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr("cv2.setMouseCallback", lambda *args, **kwargs: None)
    monkeypatch.setattr("cv2.imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr("cv2.destroyWindow", lambda *args, **kwargs: None)
    monkeypatch.setattr("cv2.destroyAllWindows", lambda *args, **kwargs: None)
    monkeypatch.setattr("cv2.waitKey", lambda *_args, **_kwargs: next(key_iter, ord("n")))


def test_run_local_verify_includes_precalibrated_views_in_sequential_order(monkeypatch, tmp_path, capsys):
    frame = np.zeros((90, 120, 3), dtype=np.uint8)
    out_json = tmp_path / "verify_correspondences.json"
    out_json.write_text(
        """
        {
          "correspondences": {
            "top": {
              "image_points": [[0, 0], [10, 0], [10, 10], [0, 10]],
              "field_points": [[0, 0], [20, 0], [20, 20], [0, 20]]
            }
          }
        }
        """,
        encoding="utf-8",
    )

    monkeypatch.setattr("cv2.VideoCapture", lambda *_args, **_kwargs: _FakeCapture(frame))
    monkeypatch.setattr(
        "chevron.ui.verify_gui.load_config",
        lambda *_args, **_kwargs: {"field": {"width_units": 10, "height_units": 5, "px_per_unit": 10}},
    )
    monkeypatch.setattr(
        "chevron.ui.verify_gui.get_layout",
        lambda *_args, **_kwargs: {
            "top": [0, 0, 120, 30],
            "bottom_left": [0, 30, 60, 60],
            "bottom_right": [60, 30, 60, 60],
        },
    )
    _stub_cv2_gui(monkeypatch, [ord("q"), ord("q"), ord("q")])

    result = run_local_verify("video.mp4", "config.yml", out_json, frame_idx=7)

    stdout = capsys.readouterr().out
    assert "sequential verify order -> top, bottom_right, bottom_left" in stdout
    assert "calibrating view -> top" in stdout
    assert len(result["top"]["image_points"]) == 4


def test_run_local_verify_q_advances_instead_of_quitting(monkeypatch, tmp_path, capsys):
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    out_json = tmp_path / "verify_correspondences.json"

    monkeypatch.setattr("cv2.VideoCapture", lambda *_args, **_kwargs: _FakeCapture(frame))
    monkeypatch.setattr(
        "chevron.ui.verify_gui.load_config",
        lambda *_args, **_kwargs: {"field": {"width_units": 10, "height_units": 5, "px_per_unit": 10}},
    )
    monkeypatch.setattr(
        "chevron.ui.verify_gui.get_layout",
        lambda *_args, **_kwargs: {
            "top": [0, 0, 120, 20],
            "bottom_left": [0, 20, 60, 60],
            "bottom_right": [60, 20, 60, 60],
        },
    )
    _stub_cv2_gui(monkeypatch, [ord("q"), ord("q"), ord("q")])

    run_local_verify("video.mp4", "config.yml", out_json, frame_idx=0)

    stdout = capsys.readouterr().out
    assert "calibrating view -> top" in stdout
    assert "calibrating view -> bottom_right" in stdout
    assert "calibrating view -> bottom_left" in stdout


def test_run_local_verify_supports_skip_shortcut(monkeypatch, tmp_path, capsys):
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    out_json = tmp_path / "verify_correspondences.json"
    out_json.write_text(
        """
        {
          "correspondences": {
            "top": {
              "image_points": [[0, 0], [10, 0], [10, 10], [0, 10]],
              "field_points": [[0, 0], [20, 0], [20, 20], [0, 20]]
            }
          }
        }
        """,
        encoding="utf-8",
    )

    monkeypatch.setattr("cv2.VideoCapture", lambda *_args, **_kwargs: _FakeCapture(frame))
    monkeypatch.setattr(
        "chevron.ui.verify_gui.load_config",
        lambda *_args, **_kwargs: {"field": {"width_units": 10, "height_units": 5, "px_per_unit": 10}},
    )
    monkeypatch.setattr(
        "chevron.ui.verify_gui.get_layout",
        lambda *_args, **_kwargs: {
            "top": [0, 0, 120, 20],
            "bottom_left": [0, 20, 60, 60],
            "bottom_right": [60, 20, 60, 60],
        },
    )
    _stub_cv2_gui(monkeypatch, [ord("s"), ord("q"), ord("q")])

    result = run_local_verify("video.mp4", "config.yml", out_json, frame_idx=0)

    stdout = capsys.readouterr().out
    assert "keys -> n/space(next view), s(skip view)" in stdout
    assert "skipping view -> top" in stdout
    assert len(result["top"]["image_points"]) == 4
