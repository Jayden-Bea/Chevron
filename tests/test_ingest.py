from pathlib import Path
import subprocess

import pytest

from chevron.ingest import (
    _download_youtube,
    _is_retryable_ytdlp_error,
    _normalize_youtube_cookie_header,
    _youtube_download_strategies,
)


def test_is_retryable_ytdlp_error_matches_expected_messages():
    assert _is_retryable_ytdlp_error("ERROR: 403 Forbidden")
    assert _is_retryable_ytdlp_error("HTTP Error 403: Forbidden")
    assert _is_retryable_ytdlp_error("youtube client outdated")
    assert _is_retryable_ytdlp_error("HTTP Error 429: Too Many Requests")
    assert _is_retryable_ytdlp_error("Sign in to confirm you're not a bot")
    assert _is_retryable_ytdlp_error("Sign in to confirm you’re not a bot")
    assert not _is_retryable_ytdlp_error("network timeout")


def test_download_youtube_retries_across_clients(monkeypatch, tmp_path: Path):
    calls: list[list[str]] = []
    logs: list[str] = []

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

    monkeypatch.setattr("chevron.ingest._resolve_youtube_title", lambda _: "Sample title")
    monkeypatch.setattr("subprocess.run", fake_run)

    download_result = _download_youtube("https://youtube.com/watch?v=abc123", tmp_path, logger=logs.append)

    assert download_result.source_path.name == "abc123.mp4"
    assert download_result.successful_strategy == "android_creator"
    assert download_result.successful_strategy_args == ["--extractor-args", "youtube:player_client=android_creator"]
    assert len(calls) == 3
    assert "--extractor-args" not in calls[0]
    assert any("youtube:player_client=android" in part for part in calls[1])
    assert any("youtube:player_client=android_creator" in part for part in calls[2])
    assert download_result.attempts[0]["strategy"] == "default"
    assert download_result.attempts[-1]["status"] == "success"
    assert "Attempting to ingest youtube video Sample title with the following settings:\n[]" in logs[0]
    assert "Ingest \x1b[31mfailed\x1b[0m." in logs[1]
    assert "Ingest \x1b[32msucceeded\x1b[0m." in logs[-2]
    assert "Saving device settings as: ['--extractor-args', 'youtube:player_client=android_creator']" in logs[-1]


def test_download_youtube_raises_on_non_retryable_error(monkeypatch, tmp_path: Path):
    def fake_run(cmd, check, text, capture_output):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=cmd,
            output="",
            stderr="ERROR: unavailable",
        )

    monkeypatch.setattr("chevron.ingest._resolve_youtube_title", lambda _: "Sample title")
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

    monkeypatch.setattr("chevron.ingest._resolve_youtube_title", lambda _: "Sample title")
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
    assert "web_relaxed_network" in names


def test_download_youtube_strategies_include_browser_cookie_fallbacks(monkeypatch):
    monkeypatch.setattr("chevron.ingest.which", lambda command: "/usr/bin/fake" if command in {"firefox", "chromium"} else None)

    strategies = _youtube_download_strategies()
    names = [strategy["name"] for strategy in strategies]

    assert "web_firefox_cookies" in names
    assert "web_chrome_cookies" in names


def test_download_youtube_raises_runtime_error_when_ytdlp_missing(monkeypatch, tmp_path: Path):
    def fake_run(cmd, check, text, capture_output):
        raise FileNotFoundError("yt-dlp")

    monkeypatch.setattr("chevron.ingest._resolve_youtube_title", lambda _: "Sample title")
    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="yt-dlp is required"):
        _download_youtube("https://youtube.com/watch?v=abc123", tmp_path)


def test_download_youtube_logs_auth_guidance_after_auth_related_failures(monkeypatch, tmp_path: Path):
    logs: list[str] = []

    def fake_run(cmd, check, text, capture_output):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=cmd,
            output="",
            stderr="Sign in to confirm you're not a bot",
        )

    monkeypatch.setattr("chevron.ingest._resolve_youtube_title", lambda _: "Sample title")
    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="Last error"):
        _download_youtube("https://youtube.com/watch?v=abc123", tmp_path, logger=logs.append)

    assert any(
        "Provide --youtube-cookies-file, --youtube-cookies-from-browser, or --youtube-cookie" in entry
        for entry in logs
    )


def test_download_youtube_does_not_log_auth_guidance_on_non_auth_failures(monkeypatch, tmp_path: Path):
    logs: list[str] = []

    def fake_run(cmd, check, text, capture_output):
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd, output="", stderr="ERROR: unavailable")

    monkeypatch.setattr("chevron.ingest._resolve_youtube_title", lambda _: "Sample title")
    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="Last error"):
        _download_youtube("https://youtube.com/watch?v=abc123", tmp_path, logger=logs.append)

    assert not any("Provide --youtube-cookies-file" in entry for entry in logs)


def test_download_youtube_uses_provided_cookie_header_first(monkeypatch, tmp_path: Path):
    calls: list[list[str]] = []

    def fake_run(cmd, check, text, capture_output):
        calls.append(cmd)
        out_file = tmp_path / "cookie_first.mp4"
        out_file.write_bytes(b"video")
        return None

    monkeypatch.setattr("chevron.ingest._resolve_youtube_title", lambda _: "Sample title")
    monkeypatch.setattr("subprocess.run", fake_run)

    result = _download_youtube(
        "https://youtube.com/watch?v=abc123",
        tmp_path,
        youtube_cookie_header="SID=fake",
    )

    assert result.successful_strategy == "manual_cookie_header"
    assert result.attempts[0]["strategy"] == "manual_cookie_header"
    assert "--add-header" in calls[0]


def test_download_youtube_falls_back_to_automatic_strategies_when_provided_cookie_fails(monkeypatch, tmp_path: Path):
    calls = {"count": 0}

    def fake_run(cmd, check, text, capture_output):
        calls["count"] += 1
        if "--add-header" in cmd:
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd, output="", stderr="cookie failed")
        if calls["count"] == 2:
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd, output="", stderr="403 Forbidden")

        out_file = tmp_path / "auto_after_cookie.mp4"
        out_file.write_bytes(b"video")
        return None

    monkeypatch.setattr("chevron.ingest._resolve_youtube_title", lambda _: "Sample title")
    monkeypatch.setattr("subprocess.run", fake_run)

    result = _download_youtube(
        "https://youtube.com/watch?v=abc123",
        tmp_path,
        youtube_cookie_header="SID=fake",
    )

    assert result.successful_strategy == "android"
    assert result.attempts[0]["strategy"] == "manual_cookie_header"
    assert result.attempts[0]["status"] == "failed"


def test_normalize_youtube_cookie_header_accepts_plain_cookie_value():
    normalized = _normalize_youtube_cookie_header("SID=abc; HSID=def")

    assert normalized == "SID=abc; HSID=def"


def test_normalize_youtube_cookie_header_accepts_cookie_prefix():
    normalized = _normalize_youtube_cookie_header("Cookie: SID=abc; HSID=def")

    assert normalized == "SID=abc; HSID=def"


def test_normalize_youtube_cookie_header_extracts_cookie_line_from_header_block():
    normalized = _normalize_youtube_cookie_header(
        "Accept: text/html\nUser-Agent: test\nCookie: SID=abc; HSID=def\nReferer: https://youtube.com"
    )

    assert normalized == "SID=abc; HSID=def"


def test_normalize_youtube_cookie_header_rejects_header_blob_without_cookie():
    normalized = _normalize_youtube_cookie_header("Accept: text/html\nUser-Agent: test")

    assert normalized is None


def test_download_youtube_prefers_user_browser_cookies_strategy(monkeypatch, tmp_path: Path):
    calls: list[list[str]] = []

    def fake_run(cmd, check, text, capture_output):
        calls.append(cmd)
        out_file = tmp_path / "browser_cookie.mp4"
        out_file.write_bytes(b"video")
        return None

    monkeypatch.setattr("chevron.ingest._resolve_youtube_title", lambda _: "Sample title")
    monkeypatch.setattr("subprocess.run", fake_run)

    result = _download_youtube(
        "https://youtube.com/watch?v=abc123",
        tmp_path,
        youtube_cookies_from_browser="chrome",
    )

    assert result.successful_strategy == "user_browser_cookies_chrome"
    assert result.successful_strategy_args == [
        "--extractor-args",
        "youtube:player_client=web",
        "--cookies-from-browser",
        "chrome",
    ]
    assert "--cookies-from-browser" in calls[0]


def test_download_youtube_falls_back_after_user_browser_cookies_failure(monkeypatch, tmp_path: Path):
    calls = {"count": 0}

    def fake_run(cmd, check, text, capture_output):
        calls["count"] += 1
        if "--cookies-from-browser" in cmd:
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd, output="", stderr="browser cookie error")
        if calls["count"] == 2:
            out_file = tmp_path / "fallback.mp4"
            out_file.write_bytes(b"video")
            return None
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd, output="", stderr="403 Forbidden")

    monkeypatch.setattr("chevron.ingest._resolve_youtube_title", lambda _: "Sample title")
    monkeypatch.setattr("subprocess.run", fake_run)

    result = _download_youtube(
        "https://youtube.com/watch?v=abc123",
        tmp_path,
        youtube_cookies_from_browser="edge",
    )

    assert result.successful_strategy == "default"
    assert result.attempts[0]["strategy"] == "user_browser_cookies_edge"
    assert result.attempts[0]["status"] == "failed"


def test_download_youtube_prefers_user_cookies_file_strategy(monkeypatch, tmp_path: Path):
    calls: list[list[str]] = []
    cookies_file = tmp_path / "cookies.txt"
    cookies_file.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")

    def fake_run(cmd, check, text, capture_output):
        calls.append(cmd)
        out_file = tmp_path / "cookies_file.mp4"
        out_file.write_bytes(b"video")
        return None

    monkeypatch.setattr("chevron.ingest._resolve_youtube_title", lambda _: "Sample title")
    monkeypatch.setattr("subprocess.run", fake_run)

    result = _download_youtube(
        "https://youtube.com/watch?v=abc123",
        tmp_path,
        youtube_cookies_file=cookies_file,
    )

    assert result.successful_strategy == "user_cookies_file"
    assert result.successful_strategy_args == [
        "--extractor-args",
        "youtube:player_client=web",
        "--cookies",
        str(cookies_file),
    ]
    assert "--cookies" in calls[0]


def test_download_youtube_raises_for_missing_user_cookies_file(tmp_path: Path):
    with pytest.raises(RuntimeError, match="does not exist"):
        _download_youtube(
            "https://youtube.com/watch?v=abc123",
            tmp_path,
            youtube_cookies_file=tmp_path / "missing_cookies.txt",
        )


def test_download_youtube_accepts_browser_profile_spec(monkeypatch, tmp_path: Path):
    calls: list[list[str]] = []

    def fake_run(cmd, check, text, capture_output):
        calls.append(cmd)
        out_file = tmp_path / "browser_profile.mp4"
        out_file.write_bytes(b"video")
        return None

    monkeypatch.setattr("chevron.ingest._resolve_youtube_title", lambda _: "Sample title")
    monkeypatch.setattr("subprocess.run", fake_run)

    result = _download_youtube(
        "https://youtube.com/watch?v=abc123",
        tmp_path,
        youtube_cookies_from_browser="chrome:Default",
    )

    assert result.successful_strategy == "user_browser_cookies_chrome"
    assert "chrome:Default" in calls[0]
