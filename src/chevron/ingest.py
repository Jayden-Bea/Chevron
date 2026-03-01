from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .utils.ffmpeg import normalize_video
from .utils.io import ensure_dir, write_json


@dataclass
class YouTubeDownloadResult:
    source_path: Path
    successful_strategy: str
    attempts: list[dict[str, str | int]]


def _is_retryable_ytdlp_error(message: str) -> bool:
    lowered = message.lower()
    is_403 = "403" in lowered and "forbidden" in lowered
    is_429 = "429" in lowered and "too many requests" in lowered
    is_outdated_client = "youtube client outdated" in lowered
    is_bot_challenge = "sign in to confirm" in lowered and "not a bot" in lowered
    return is_403 or is_429 or is_outdated_client or is_bot_challenge


def _youtube_download_strategies() -> list[dict[str, str | list[str]]]:
    # Ordered from least invasive to most aggressive compatibility workarounds.
    return [
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
        {
            "name": "android_ipv4",
            "args": ["--extractor-args", "youtube:player_client=android", "--force-ipv4"],
        },
        {
            "name": "web_ipv4",
            "args": ["--extractor-args", "youtube:player_client=web", "--force-ipv4"],
        },
        {
            "name": "ios_ipv4",
            "args": ["--extractor-args", "youtube:player_client=ios", "--force-ipv4"],
        },
    ]


def _download_youtube(url: str, cache_dir: Path) -> YouTubeDownloadResult:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_tmpl = str(cache_dir / "%(id)s.%(ext)s")

    strategies = _youtube_download_strategies()
    attempts: list[dict[str, str | int]] = []

    for strategy in strategies:
        cmd = ["yt-dlp", "-o", output_tmpl]
        cmd.extend(strategy["args"])
        cmd.append(url)

        try:
            subprocess.run(cmd, check=True, text=True, capture_output=True)
            latest = max(cache_dir.glob("*"), key=lambda p: p.stat().st_mtime)
            attempts.append({"strategy": strategy["name"], "status": "success", "returncode": 0})
            return YouTubeDownloadResult(
                source_path=latest,
                successful_strategy=strategy["name"],
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

    last_error = attempts[-1].get("error", "unknown yt-dlp failure") if attempts else "unknown yt-dlp failure"
    raise RuntimeError(
        "yt-dlp could not download the URL after exhausting all configured YouTube connection strategies. "
        f"Last error: {last_error}"
    )


def ingest(url: str | None, video: str | None, out_dir: str | Path, fps: int = 30) -> dict:
    out = ensure_dir(out_dir)
    source_dir = ensure_dir(out / "source")
    proxy_path = out / "proxy.mp4"

    if url:
        download_result = _download_youtube(url, source_dir)
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
            "attempts": download_result.attempts,
        }
    write_json(out / "ingest_meta.json", meta)
    return meta
