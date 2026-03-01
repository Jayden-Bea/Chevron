from pathlib import Path
import subprocess

import pytest

from chevron.ingest import (
    YouTubeDownloadResult,
    _download_youtube,
    _ensure_yt_dlp_minimum_version,
    _is_retryable_ytdlp_error,
    _normalize_youtube_cookie_header,
    _parse_yt_dlp_version,
    _YTDLP_PREFERRED_FORMAT_ARGS,
    _shuffle_youtube_strategies,
    _youtube_download_strategies,
    ingest,
)


@pytest.fixture(autouse=True)
def mock_minimum_ytdlp_version(monkeypatch):
    monkeypatch.setattr(
        "chevron.ingest._ensure_yt_dlp_minimum_version",
        lambda minimum_version="2026.02.21": ("2026.02.21", (2026, 2, 21)),
    )


def test_parse_yt_dlp_version_handles_stable_and_none():
    assert _parse_yt_dlp_version("2026.02.21") == (2026, 2, 21)
    assert _parse_yt_dlp_version("stable@2026.03.01 from yt-dlp/yt-dlp") == (2026, 3, 1)
    assert _parse_yt_dlp_version("nightly") is None


def test_ensure_yt_dlp_minimum_version_rejects_older_build(monkeypatch):
    monkeypatch.undo()

    class Completed:
        stdout = "2025.12.31\n"

    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: Completed())

    with pytest.raises(RuntimeError, match="require >= 2026.02.21"):
        _ensure_yt_dlp_minimum_version("2026.02.21")



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
    assert len(calls) == 3
    assert "--extractor-args" not in calls[0]
    assert calls[0][3:3 + len(_YTDLP_PREFERRED_FORMAT_ARGS)] == _YTDLP_PREFERRED_FORMAT_ARGS
    assert download_result.attempts[0]["strategy"] == "default"
    assert download_result.successful_strategy == download_result.attempts[-1]["strategy"]
    assert download_result.attempts[-1]["status"] == "success"
    assert "Using yt-dlp version 2026.02.21" in logs[0]
    assert any("Resolved video metadata; yt-dlp will now attempt to fetch media bytes." in entry for entry in logs)
    assert any("yt-dlp accepted the request and is now downloading the media stream." in entry for entry in logs)
    assert any("Attempting YouTube ingest strategy=default video='Sample title' args=[]" in entry for entry in logs)
    assert any("Ingest \x1b[31mfailed\x1b[0m strategy=default returncode=1." in entry for entry in logs)
    assert "Ingest \x1b[32msucceeded\x1b[0m." in logs[-2]
    assert "Saving device settings as:" in logs[-1]


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
    assert "default_modern_ua" in names
    assert "android" in names
    assert "android_modern_ua" in names
    assert "web" in names
    assert "ios" in names
    assert "tv_embedded" in names
    assert "android_ipv4" in names
    assert "web_relaxed_network" in names




def test_shuffle_youtube_strategies_is_deterministic_per_url():
    strategies = _youtube_download_strategies()
    order_a = [entry["name"] for entry in _shuffle_youtube_strategies("https://youtube.com/watch?v=aaa", strategies)]
    order_b = [entry["name"] for entry in _shuffle_youtube_strategies("https://youtube.com/watch?v=aaa", strategies)]
    order_c = [entry["name"] for entry in _shuffle_youtube_strategies("https://youtube.com/watch?v=bbb", strategies)]

    assert order_a == order_b
    assert order_a[0] == "default"
    assert order_a != order_c

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
    assert calls[0][3:3 + len(_YTDLP_PREFERRED_FORMAT_ARGS)] == _YTDLP_PREFERRED_FORMAT_ARGS
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

    assert result.successful_strategy != "manual_cookie_header"
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


def test_download_youtube_uses_explicit_output_path(monkeypatch, tmp_path: Path):
    output_path = tmp_path / "proxy.mp4"
    calls: list[list[str]] = []

    def fake_run(cmd, check, text, capture_output):
        calls.append(cmd)
        output_path.write_bytes(b"video")
        return None

    monkeypatch.setattr("chevron.ingest._resolve_youtube_title", lambda _: "Sample title")
    monkeypatch.setattr("subprocess.run", fake_run)

    result = _download_youtube(
        "https://youtube.com/watch?v=abc123",
        tmp_path,
        output_path=output_path,
    )

    assert calls[0][2] == str(output_path)
    assert result.source_path == output_path


def test_download_youtube_uses_explicit_output_path_outside_cache_dir(monkeypatch, tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    output_path = tmp_path / "workdir" / "proxy.mp4"
    output_path.parent.mkdir()

    def fake_run(cmd, check, text, capture_output):
        output_path.write_bytes(b"video")
        return None

    monkeypatch.setattr("chevron.ingest._resolve_youtube_title", lambda _: "Sample title")
    monkeypatch.setattr("subprocess.run", fake_run)

    result = _download_youtube(
        "https://youtube.com/watch?v=abc123",
        cache_dir,
        output_path=output_path,
    )

    assert result.source_path == output_path


def test_download_youtube_raises_when_success_has_no_output(monkeypatch, tmp_path: Path):
    def fake_run(cmd, check, text, capture_output):
        return None

    monkeypatch.setattr("chevron.ingest._resolve_youtube_title", lambda _: "Sample title")
    monkeypatch.setattr("subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="produced no output files"):
        _download_youtube(
            "https://youtube.com/watch?v=abc123",
            tmp_path,
        )


def test_ingest_url_uses_downloaded_mp4_as_proxy_without_normalization(monkeypatch, tmp_path: Path):
    out_dir = tmp_path / "workdir"
    proxy_path = out_dir / "proxy.mp4"

    monkeypatch.setattr(
        "chevron.ingest._download_youtube",
        lambda *args, **kwargs: YouTubeDownloadResult(
            source_path=proxy_path,
            successful_strategy="default",
            successful_strategy_args=[],
            attempts=[{"strategy": "default", "status": "success", "returncode": 0}],
        ),
    )

    def fail_normalize(*args, **kwargs):
        raise AssertionError("normalize_video should not run for URL ingest")

    monkeypatch.setattr("chevron.ingest.normalize_video", fail_normalize)

    meta = ingest(url="https://youtube.com/watch?v=abc123", video=None, out_dir=out_dir, fps=10)

    assert meta["source"].endswith("proxy.mp4")
    assert meta["proxy"].endswith("proxy.mp4")


def test_ingest_local_mp4_copies_to_proxy_without_normalization(monkeypatch, tmp_path: Path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source")
    out_dir = tmp_path / "workdir"

    def fail_normalize(*args, **kwargs):
        raise AssertionError("normalize_video should not run for local MP4 ingest")

    monkeypatch.setattr("chevron.ingest.normalize_video", fail_normalize)

    meta = ingest(url=None, video=str(source), out_dir=out_dir, fps=12)

    proxy = Path(meta["proxy"])
    assert proxy.exists()
    assert proxy.read_bytes() == b"source"
    assert meta["source"].endswith("source/input.mp4")
    assert meta["proxy"].endswith("proxy.mp4")


def test_ingest_local_non_mp4_still_normalizes_to_proxy(monkeypatch, tmp_path: Path):
    source = tmp_path / "input.mkv"
    source.write_bytes(b"source")
    out_dir = tmp_path / "workdir"

    called = {}

    def fake_normalize(in_path, out_path, fps):
        called["in_path"] = Path(in_path)
        called["out_path"] = Path(out_path)
        called["fps"] = fps

    monkeypatch.setattr("chevron.ingest.normalize_video", fake_normalize)

    meta = ingest(url=None, video=str(source), out_dir=out_dir, fps=12)

    assert called["in_path"].name == "input.mkv"
    assert called["out_path"].name == "proxy.mp4"
    assert called["fps"] == 12
    assert meta["source"].endswith("source/input.mkv")
    assert meta["proxy"].endswith("proxy.mp4")
