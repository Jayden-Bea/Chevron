from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from shutil import which

from .utils.ffmpeg import normalize_video
from .utils.io import ensure_dir, write_json


_RED = "\x1b[31m"
_GREEN = "\x1b[32m"
_RESET = "\x1b[0m"
_DEFAULT_YOUTUBE_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"

_YTDLP_MIN_VERSION = "2026.02.21"
_YTDLP_PREFERRED_FORMAT_ARGS = [
    "-f",
    "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/best",
    "--merge-output-format",
    "mp4",
]


def _parse_yt_dlp_version(version_text: str) -> tuple[int, int, int] | None:
    match = re.search(r"(\d{4})\.(\d{2})\.(\d{2})", version_text)
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def _ensure_yt_dlp_minimum_version(minimum_version: str = _YTDLP_MIN_VERSION) -> tuple[str, tuple[int, int, int]]:
    cmd = ["yt-dlp", "--version"]
    try:
        completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except FileNotFoundError as err:
        raise RuntimeError(
            "yt-dlp is required for URL ingestion but was not found on PATH. "
            "Install yt-dlp and retry, or use --video with a local file."
        ) from err
    except subprocess.CalledProcessError as err:
        stderr = (err.stderr or "").strip()
        raise RuntimeError(f"Failed to determine yt-dlp version: {stderr or err}") from err

    resolved = (completed.stdout or "").strip().splitlines()
    resolved_version = resolved[-1].strip() if resolved else "unknown"
    resolved_tuple = _parse_yt_dlp_version(resolved_version)
    minimum_tuple = _parse_yt_dlp_version(minimum_version)

    if resolved_tuple is None or minimum_tuple is None:
        raise RuntimeError(
            "Unable to parse yt-dlp version output. "
            f"Expected YYYY.MM.DD, got '{resolved_version}'."
        )

    if resolved_tuple < minimum_tuple:
        raise RuntimeError(
            "Chevron requires a newer yt-dlp build for resilient YouTube ingestion. "
            f"Found {resolved_version}, require >= {minimum_version}. "
            "Please update with `pip install -U yt-dlp` and retry."
        )

    return resolved_version, resolved_tuple


def _truncate_for_log(text: str, max_chars: int = 400) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "... [truncated]"


def _format_ytdlp_error_for_log(err: subprocess.CalledProcessError) -> str:
    stdout = (err.stdout or "").strip()
    stderr = (err.stderr or "").strip()
    parts = []
    if stderr:
        parts.append(f"stderr={_truncate_for_log(stderr)}")
    if stdout:
        parts.append(f"stdout={_truncate_for_log(stdout)}")
    if not parts:
        parts.append("no stdout/stderr captured")
    return "; ".join(parts)


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




def _with_user_agent(name: str, args: list[str]) -> dict[str, str | list[str]]:
    return {"name": name, "args": [*args, "--user-agent", _DEFAULT_YOUTUBE_USER_AGENT]}


def _youtube_download_strategies() -> list[dict[str, str | list[str]]]:
    # Ordered from least invasive to most aggressive compatibility workarounds.
    strategies: list[dict[str, str | list[str]]] = [
        {"name": "default", "args": []},
        _with_user_agent("default_modern_ua", []),
        {"name": "android", "args": ["--extractor-args", "youtube:player_client=android"]},
        _with_user_agent("android_modern_ua", ["--extractor-args", "youtube:player_client=android"]),
        {"name": "android_creator", "args": ["--extractor-args", "youtube:player_client=android_creator"]},
        {"name": "android_music", "args": ["--extractor-args", "youtube:player_client=android_music"]},
        {"name": "android_vr", "args": ["--extractor-args", "youtube:player_client=android_vr"]},
        {"name": "web", "args": ["--extractor-args", "youtube:player_client=web"]},
        _with_user_agent("web_modern_ua", ["--extractor-args", "youtube:player_client=web"]),
        {"name": "web_creator", "args": ["--extractor-args", "youtube:player_client=web_creator"]},
        {"name": "web_embedded", "args": ["--extractor-args", "youtube:player_client=web_embedded"]},
        {"name": "web_music", "args": ["--extractor-args", "youtube:player_client=web_music"]},
        {"name": "mweb", "args": ["--extractor-args", "youtube:player_client=mweb"]},
        {"name": "ios", "args": ["--extractor-args", "youtube:player_client=ios"]},
        _with_user_agent("ios_modern_ua", ["--extractor-args", "youtube:player_client=ios"]),
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




def _shuffle_youtube_strategies(url: str, strategies: list[dict[str, str | list[str]]]) -> list[dict[str, str | list[str]]]:
    if len(strategies) <= 1:
        return list(strategies)

    # Keep the plain default first, but shuffle the remaining fallback strategies per-URL.
    # This avoids a single static fingerprint while remaining deterministic for a given URL.
    first = strategies[0]
    remaining = strategies[1:]

    def _sort_key(strategy: dict[str, str | list[str]]) -> str:
        name = str(strategy.get("name", ""))
        payload = f"{url}|{name}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    return [first, *sorted(remaining, key=_sort_key)]


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
        *_YTDLP_PREFERRED_FORMAT_ARGS,
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

    strategies = _shuffle_youtube_strategies(url, _youtube_download_strategies())
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
    resolved_ytdlp_version, _ = _ensure_yt_dlp_minimum_version()
    if logger:
        logger(
            "Using yt-dlp version "
            f"{resolved_ytdlp_version} (minimum required: {_YTDLP_MIN_VERSION})."
        )
        logger(
            "Resolved video metadata; yt-dlp will now attempt to fetch media bytes. "
            "If this step takes a while, the download is still in progress."
        )

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
                logger(
                    "Provided Cookie header failed; retrying automatic YouTube strategies. "
                    f"returncode={err.returncode} details={_format_ytdlp_error_for_log(err)}"
                )

    elif youtube_cookie_header and logger:
        logger("Provided --youtube-cookie value did not contain a usable Cookie header; retrying automatic YouTube strategies.")

    for strategy in strategies:
        cmd = ["yt-dlp", "-o", output_tmpl, *_YTDLP_PREFERRED_FORMAT_ARGS]
        cmd.extend(strategy["args"])
        cmd.append(url)

        if logger:
            logger(
                "Attempting YouTube ingest "
                f"strategy={strategy['name']} "
                f"video={video_title!r} "
                f"args={strategy['args'] or '[]'}"
            )
            logger("yt-dlp accepted the request and is now downloading the media stream.")

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
                logger(
                    f"Ingest {_colored('failed', _RED)} "
                    f"strategy={strategy['name']} returncode={err.returncode}."
                )
                logger(f"yt-dlp failure details: {_format_ytdlp_error_for_log(err)}")
            continue

    if _should_offer_manual_auth_fallback(attempts) and logger:
        logger(
            "Authentication-related YouTube failures were detected. "
            "Provide --youtube-cookies-file, --youtube-cookies-from-browser, or --youtube-cookie to retry with authenticated access."
        )

    last_error = attempts[-1].get("error", "unknown yt-dlp failure") if attempts else "unknown yt-dlp failure"
    raise RuntimeError(
        "yt-dlp could not download the URL after exhausting all configured YouTube connection strategies. "
        f"Last error: {last_error}\n\n"
        "Troubleshooting tips:\n"
        "- Update yt-dlp: `pip install -U yt-dlp`\n"
        "- Try probing formats: `yt-dlp -F <url>`\n"
        "- Retry with authenticated access (`--youtube-cookies-file` or `--youtube-cookies-from-browser`)\n"
        "- Chevron already shuffles YouTube client/user-agent strategies automatically; if all fail, retry later or use a local video file"
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
