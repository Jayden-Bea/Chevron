from pathlib import Path
import subprocess

import pytest

from chevron.ingest import _download_youtube, _is_retryable_ytdlp_error


def test_is_retryable_ytdlp_error_matches_expected_messages():
    assert _is_retryable_ytdlp_error("ERROR: 403 Forbidden")
    assert _is_retryable_ytdlp_error("HTTP Error 403: Forbidden")
    assert _is_retryable_ytdlp_error("youtube client outdated")
    assert not _is_retryable_ytdlp_error("network timeout")


def test_download_youtube_retries_across_clients(monkeypatch, tmp_path: Path):
    calls: list[list[str]] = []

    def fake_run(cmd, check, text, capture_output):
        calls.append(cmd)
        if len(calls) < 3:
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=cmd,
                output="",
                stderr="ERROR: 403 Forbidden",
            )

        out_file = tmp_path / "abc123.mp4"
        out_file.write_bytes(b"video")
        return None

    monkeypatch.setattr("subprocess.run", fake_run)

    downloaded = _download_youtube("https://youtube.com/watch?v=abc123", tmp_path)

    assert downloaded.name == "abc123.mp4"
    assert len(calls) == 3
    assert "--extractor-args" not in calls[0]
    assert any("youtube:player_client=android" in part for part in calls[1])
    assert any("youtube:player_client=web" in part for part in calls[2])


def test_download_youtube_raises_on_non_retryable_error(monkeypatch, tmp_path: Path):
    def fake_run(cmd, check, text, capture_output):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=cmd,
            output="",
            stderr="ERROR: unavailable",
        )

    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        _download_youtube("https://youtube.com/watch?v=abc123", tmp_path)


def test_download_youtube_raises_runtime_error_after_exhausting_retryable_clients(monkeypatch, tmp_path: Path):
    calls: list[list[str]] = []

    def fake_run(cmd, check, text, capture_output):
        calls.append(cmd)
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=cmd,
            output="",
            stderr="ERROR: YouTube client outdated",
        )

    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="trying multiple YouTube clients"):
        _download_youtube("https://youtube.com/watch?v=abc123", tmp_path)

    assert len(calls) == 5


def test_download_youtube_raises_runtime_error_when_ytdlp_missing(monkeypatch, tmp_path: Path):
    def fake_run(cmd, check, text, capture_output):
        raise FileNotFoundError("yt-dlp")

    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="yt-dlp is required"):
        _download_youtube("https://youtube.com/watch?v=abc123", tmp_path)
