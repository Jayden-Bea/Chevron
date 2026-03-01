from __future__ import annotations

import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path
from shutil import which

from .utils.ffmpeg import normalize_video
from .utils.io import ensure_dir, write_json


_RED = "\x1b[31m"
_GREEN = "\x1b[32m"
_RESET = "\x1b[0m"


def _colored(text: str, color: str) -> str:
    return f"{color}{text}{_RESET}"


@dataclass
class YouTubeDownloadResult:
    source_path: Path
    successful_strategy: str
    successful_strategy_args: list[str]
    attempts: list[dict[str, str | int]]


def _is_retryable_ytdlp_error(message: str) -> bool:
    lowered = message.lower().replace("’", "'")
    is_403 = "403" in lowered and "forbidden" in lowered
    is_429 = "429" in lowered and "too many requests" in lowered
    is_outdated_client = "youtube client outdated" in lowered
    is_bot_challenge = "sign in to confirm" in lowered and "not a bot" in lowered
    return is_403 or is_429 or is_outdated_client or is_bot_challenge


def _resolve_youtube_title(url: str) -> str:
    cmd = ["yt-dlp", "--skip-download", "--print", "%(title)s", url]
    try:
        completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return url

    lines = [line.strip() for line in (completed.stdout or "").splitlines() if line.strip()]
    return lines[-1] if lines else url


def _youtube_download_strategies() -> list[dict[str, str | list[str]]]:
    # Ordered from least invasive to most aggressive compatibility workarounds.
    strategies: list[dict[str, str | list[str]]] = [
        {"name": "default", "args": []},
        {"name": "android", "args": ["--extractor-args", "youtube:player_client=android"]},
        {"name": "android_creator", "args": ["--extractor-args", "youtube:player_client=android_creator"]},
        {"name": "android_music", "args": ["--extractor-args", "youtube:player_client=android_music"]},
        {"name": "android_vr", "args": ["--extractor-args", "youtube:player_client=android_vr"]},
        {"name": "web", "args": ["--extractor-args", "youtube:player_client=web"]},
        {"name": "web_creator", "args": ["--extractor-args", "youtube:player_client=web_creator"]},
        {"name": "web_embedded", "args": ["--extractor-args", "youtube:player_client=web_embedded"]},
        {"name": "web_music", "args": ["--extractor-args", "youtube:player_client=web_music"]},
        {"name": "mweb", "args": ["--extractor-args", "youtube:player_client=mweb"]},
        {"name": "ios", "args": ["--extractor-args", "youtube:player_client=ios"]},
        {"name": "tv", "args": ["--extractor-args", "youtube:player_client=tv"]},
        {"name": "tv_embedded", "args": ["--extractor-args", "youtube:player_client=tv_embedded"]},
    ]

    network_profiles = [
        ("ipv4", ["--force-ipv4"]),
        ("relaxed_network", ["--geo-bypass", "--socket-timeout", "15", "--retries", "10"]),
    ]

    for profile_name, profile_args in network_profiles:
        strategies.extend(
            [
                {
                    "name": f"android_{profile_name}",
                    "args": ["--extractor-args", "youtube:player_client=android", *profile_args],
                },
                {
                    "name": f"web_{profile_name}",
                    "args": ["--extractor-args", "youtube:player_client=web", *profile_args],
                },
                {
                    "name": f"ios_{profile_name}",
                    "args": ["--extractor-args", "youtube:player_client=ios", *profile_args],
                },
            ]
        )

    if which("firefox"):
        strategies.append(
            {
                "name": "web_firefox_cookies",
                "args": [
                    "--extractor-args",
                    "youtube:player_client=web",
                    "--cookies-from-browser",
                    "firefox",
                ],
            }
        )

    if which("google-chrome") or which("chromium"):
        strategies.append(
            {
                "name": "web_chrome_cookies",
                "args": [
                    "--extractor-args",
                    "youtube:player_client=web",
                    "--cookies-from-browser",
                    "chrome",
                ],
            }
        )

    return strategies


def _should_offer_manual_auth_fallback(attempts: list[dict[str, str | int]]) -> bool:
    for attempt in attempts:
        error = str(attempt.get("error", "")).lower().replace("’", "'")
        if "sign in" in error or "not a bot" in error or "403" in error or "forbidden" in error:
            return True
    return False


def _normalize_youtube_cookie_header(raw_value: str | None) -> str | None:
    if not raw_value:
        return None

    value = raw_value.strip()
    if not value:
        return None

    if "\n" in value:
        for line in value.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("cookie:"):
                candidate = stripped.split(":", 1)[1].strip()
                return candidate or None

    if value.lower().startswith("cookie:"):
        value = value.split(":", 1)[1].strip()

    # Heuristic: treat obvious full header blocks pasted on one line as invalid cookie values.
    looks_like_header_blob = bool(re.search(r"\b(accept|user-agent|sec-ch-ua|referer):", value, re.IGNORECASE))
    if looks_like_header_blob and "=" not in value:
        return None

    return value or None




def _prompt_for_youtube_cookie_header(logger: Callable[[str], None] | None = None) -> str | None:
    if not sys.stdin.isatty():
        return None

    if logger:
        logger(
            "All automatic YouTube download strategies failed. "
            "As a final fallback, provide your YouTube Cookie header from a logged-in browser session. "
            "If DevTools shows 'Provisional headers are shown' and no Cookie row, use DevTools -> Application/Storage -> Cookies -> https://www.youtube.com and copy key=value pairs, "
            "or copy request headers and paste the full text so Chevron can extract the Cookie line."
        )

    cookie_header = getpass("Paste YouTube Cookie value (or a full header block) and press Enter (leave blank to skip): ")
    return _normalize_youtube_cookie_header(cookie_header)


def _attempt_manual_cookie_download(
    url: str,
    output_tmpl: str,
    cache_dir: Path,
    cookie_header: str,
) -> Path:
    cmd = [
        "yt-dlp",
        "-o",
        output_tmpl,
        "--extractor-args",
        "youtube:player_client=web",
        "--add-header",
        f"Cookie: {cookie_header}",
        "--retries",
        "10",
        "--socket-timeout",
        "15",
        url,
    ]
    subprocess.run(cmd, check=True, text=True, capture_output=True)
    return max(cache_dir.glob("*"), key=lambda p: p.stat().st_mtime)


def _download_youtube(
    url: str,
    cache_dir: Path,
    logger: Callable[[str], None] | None = None,
    youtube_cookie_header: str | None = None,
    youtube_cookies_from_browser: str | None = None,
    youtube_cookies_file: str | Path | None = None,
) -> YouTubeDownloadResult:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_tmpl = str(cache_dir / "%(id)s.%(ext)s")

    strategies = _youtube_download_strategies()
    if youtube_cookies_from_browser:
        browser_spec = str(youtube_cookies_from_browser).strip()
        if browser_spec:
            browser_name = browser_spec.split(":", 1)[0].split("+", 1)[0].strip().lower() or "browser"
            strategies = [
                {
                    "name": f"user_browser_cookies_{browser_name}",
                    "args": [
                        "--extractor-args",
                        "youtube:player_client=web",
                        "--cookies-from-browser",
                        browser_spec,
                    ],
                },
                *strategies,
            ]

    if youtube_cookies_file:
        cookies_path = Path(youtube_cookies_file).expanduser()
        if not cookies_path.exists() or not cookies_path.is_file():
            raise RuntimeError(
                "The provided YouTube cookies file does not exist or is not a file: "
                f"{cookies_path}"
            )
        strategies = [
            {
                "name": "user_cookies_file",
                "args": [
                    "--extractor-args",
                    "youtube:player_client=web",
                    "--cookies",
                    str(cookies_path),
                ],
            },
            *strategies,
        ]
    attempts: list[dict[str, str | int]] = []
    video_title = _resolve_youtube_title(url)

    normalized_cookie_header = _normalize_youtube_cookie_header(youtube_cookie_header)

    if normalized_cookie_header:
        if logger:
            logger(
                "Attempting YouTube ingest with user-provided Cookie header "
                "(from --youtube-cookie / CHEVRON_YOUTUBE_COOKIE)."
            )
        try:
            latest = _attempt_manual_cookie_download(url, output_tmpl, cache_dir, normalized_cookie_header)
            attempts.append({"strategy": "manual_cookie_header", "status": "success", "returncode": 0})
            if logger:
                logger(f"Ingest {_colored('succeeded', _GREEN)} using provided Cookie header.")
            return YouTubeDownloadResult(
                source_path=latest,
                successful_strategy="manual_cookie_header",
                successful_strategy_args=["--add-header", "Cookie: [REDACTED]"],
                attempts=attempts,
            )
        except subprocess.CalledProcessError as err:
            message = "\n".join([err.stdout or "", err.stderr or ""])
            attempts.append(
                {
                    "strategy": "manual_cookie_header",
                    "status": "failed",
                    "returncode": int(err.returncode),
                    "error": message.strip() or "unknown yt-dlp failure",
                }
            )
            if logger:
                logger("Provided Cookie header failed; retrying automatic YouTube strategies.")

    elif youtube_cookie_header and logger:
        logger("Provided --youtube-cookie value did not contain a usable Cookie header; retrying automatic YouTube strategies.")

    for strategy in strategies:
        cmd = ["yt-dlp", "-o", output_tmpl]
        cmd.extend(strategy["args"])
        cmd.append(url)

        if logger:
            logger(
                "Attempting to ingest youtube video "
                f"{video_title} with the following settings:\n{strategy['args'] or '[]'}"
            )

        try:
            subprocess.run(cmd, check=True, text=True, capture_output=True)
            latest = max(cache_dir.glob("*"), key=lambda p: p.stat().st_mtime)
            attempts.append({"strategy": strategy["name"], "status": "success", "returncode": 0})
            if logger:
                logger(f"Ingest {_colored('succeeded', _GREEN)}.")
                logger(f"Saving device settings as: {strategy['args'] or '[]'}")
            return YouTubeDownloadResult(
                source_path=latest,
                successful_strategy=strategy["name"],
                successful_strategy_args=list(strategy["args"]),
                attempts=attempts,
            )
        except FileNotFoundError as err:
            raise RuntimeError(
                "yt-dlp is required for URL ingestion but was not found on PATH. "
                "Install yt-dlp and retry, or use --video with a local file."
            ) from err
        except subprocess.CalledProcessError as err:
            message = "\n".join([err.stdout or "", err.stderr or ""])
            attempts.append(
                {
                    "strategy": strategy["name"],
                    "status": "retryable" if _is_retryable_ytdlp_error(message) else "failed",
                    "returncode": int(err.returncode),
                    "error": message.strip() or "unknown yt-dlp failure",
                }
            )
            if logger:
                logger(f"Ingest {_colored('failed', _RED)}.")
            continue

    if _should_offer_manual_auth_fallback(attempts):
        cookie_header = _prompt_for_youtube_cookie_header(logger=logger)
        if cookie_header:
            try:
                latest = _attempt_manual_cookie_download(url, output_tmpl, cache_dir, cookie_header)
                attempts.append({"strategy": "manual_cookie_header", "status": "success", "returncode": 0})
                if logger:
                    logger(f"Ingest {_colored('succeeded', _GREEN)} using manual Cookie header fallback.")
                return YouTubeDownloadResult(
                    source_path=latest,
                    successful_strategy="manual_cookie_header",
                    successful_strategy_args=["--add-header", "Cookie: [REDACTED]"],
                    attempts=attempts,
                )
            except FileNotFoundError as err:
                raise RuntimeError(
                    "yt-dlp is required for URL ingestion but was not found on PATH. "
                    "Install yt-dlp and retry, or use --video with a local file."
                ) from err
            except subprocess.CalledProcessError as err:
                message = "\n".join([err.stdout or "", err.stderr or ""])
                attempts.append(
                    {
                        "strategy": "manual_cookie_header",
                        "status": "failed",
                        "returncode": int(err.returncode),
                        "error": message.strip() or "unknown yt-dlp failure",
                    }
                )
                if logger:
                    logger(f"Manual auth fallback {_colored('failed', _RED)}.")

    last_error = attempts[-1].get("error", "unknown yt-dlp failure") if attempts else "unknown yt-dlp failure"
    raise RuntimeError(
        "yt-dlp could not download the URL after exhausting all configured YouTube connection strategies. "
        f"Last error: {last_error}"
    )


def ingest(
    url: str | None,
    video: str | None,
    out_dir: str | Path,
    fps: int = 30,
    logger: Callable[[str], None] | None = None,
    youtube_cookie_header: str | None = None,
    youtube_cookies_from_browser: str | None = None,
    youtube_cookies_file: str | Path | None = None,
) -> dict:
    out = ensure_dir(out_dir)
    source_dir = ensure_dir(out / "source")
    proxy_path = out / "proxy.mp4"

    if url:
        download_result = _download_youtube(
            url,
            source_dir,
            logger=logger,
            youtube_cookie_header=youtube_cookie_header,
            youtube_cookies_from_browser=youtube_cookies_from_browser,
            youtube_cookies_file=youtube_cookies_file,
        )
        src = download_result.source_path
    elif video:
        src = Path(video)
        download_result = None
    else:
        raise ValueError("One of url or video must be provided")

    source_copy = source_dir / src.name
    if src != source_copy:
        shutil.copy2(src, source_copy)

    normalize_video(source_copy, proxy_path, fps=fps)

    meta = {
        "source": str(source_copy),
        "proxy": str(proxy_path),
        "fps": fps,
    }
    if download_result:
        meta["youtube_download"] = {
            "successful_strategy": download_result.successful_strategy,
            "successful_strategy_args": download_result.successful_strategy_args,
            "attempts": download_result.attempts,
        }
    write_json(out / "ingest_meta.json", meta)
    return meta
