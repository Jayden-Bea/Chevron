from pathlib import Path
import subprocess

import pytest

from chevron.ingest import _download_youtube, _is_retryable_ytdlp_error, _youtube_download_strategies


def test_is_retryable_ytdlp_error_matches_expected_messages():
    assert _is_retryable_ytdlp_error("ERROR: 403 Forbidden")
    assert _is_retryable_ytdlp_error("HTTP Error 403: Forbidden")
    assert _is_retryable_ytdlp_error("youtube client outdated")
    assert _is_retryable_ytdlp_error("HTTP Error 429: Too Many Requests")
    assert _is_retryable_ytdlp_error("Sign in to confirm you're not a bot")
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

    download_result = _download_youtube("https://youtube.com/watch?v=abc123", tmp_path)

    assert download_result.source_path.name == "abc123.mp4"
    assert download_result.successful_strategy == "android_creator"
    assert len(calls) == 3
    assert "--extractor-args" not in calls[0]
    assert any("youtube:player_client=android" in part for part in calls[1])
    assert any("youtube:player_client=android_creator" in part for part in calls[2])
    assert download_result.attempts[0]["strategy"] == "default"
    assert download_result.attempts[-1]["status"] == "success"


def test_download_youtube_raises_on_non_retryable_error(monkeypatch, tmp_path: Path):
    def fake_run(cmd, check, text, capture_output):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=cmd,
            output="",
            stderr="ERROR: unavailable",
        )

    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="exhausting all configured YouTube connection strategies"):
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

    with pytest.raises(RuntimeError, match="Last error"):
        _download_youtube("https://youtube.com/watch?v=abc123", tmp_path)

    assert len(calls) == len(_youtube_download_strategies())


def test_download_youtube_strategies_are_exhaustive_enough_for_fallbacks():
    strategies = _youtube_download_strategies()
    names = [strategy["name"] for strategy in strategies]

    assert names[0] == "default"
    assert "android" in names
    assert "web" in names
    assert "ios" in names
    assert "tv_embedded" in names
    assert "android_ipv4" in names


def test_download_youtube_raises_runtime_error_when_ytdlp_missing(monkeypatch, tmp_path: Path):
    def fake_run(cmd, check, text, capture_output):
        raise FileNotFoundError("yt-dlp")

    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="yt-dlp is required"):
        _download_youtube("https://youtube.com/watch?v=abc123", tmp_path)
