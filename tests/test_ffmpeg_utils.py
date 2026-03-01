import subprocess

import pytest

from chevron.utils.ffmpeg import run_ffmpeg


def test_run_ffmpeg_uses_quiet_logging_flags(monkeypatch):
    called = {}

    def fake_run(cmd, check, text, capture_output):
        called["cmd"] = cmd
        return None

    monkeypatch.setattr("subprocess.run", fake_run)

    run_ffmpeg(["-i", "input.mp4", "out.mp4"])

    assert called["cmd"][:5] == ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]


def test_run_ffmpeg_raises_runtime_error_with_stderr(monkeypatch):
    def fake_run(cmd, check, text, capture_output):
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd, stderr="Late SEI failure")

    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="Late SEI failure"):
        run_ffmpeg(["-i", "input.mp4", "out.mp4"])
