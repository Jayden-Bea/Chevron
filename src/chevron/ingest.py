from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .utils.ffmpeg import normalize_video
from .utils.io import ensure_dir, write_json


def _is_retryable_ytdlp_error(message: str) -> bool:
    lowered = message.lower()
    is_403 = "403" in lowered and "forbidden" in lowered
    is_outdated_client = "youtube client outdated" in lowered
    return is_403 or is_outdated_client


def _download_youtube(url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_tmpl = str(cache_dir / "%(id)s.%(ext)s")

    client_settings = [
        None,
        "android",
        "web",
        "ios",
        "tv_embedded",
    ]

    for idx, client in enumerate(client_settings):
        cmd = ["yt-dlp", "-o", output_tmpl]
        if client is not None:
            cmd.extend(["--extractor-args", f"youtube:player_client={client}"])
        cmd.append(url)

        try:
            subprocess.run(cmd, check=True, text=True, capture_output=True)
            break
        except FileNotFoundError as err:
            raise RuntimeError(
                "yt-dlp is required for URL ingestion but was not found on PATH. "
                "Install yt-dlp and retry, or use --video with a local file."
            ) from err
        except subprocess.CalledProcessError as err:
            message = "\n".join([err.stdout or "", err.stderr or ""])
            is_last_client = idx == len(client_settings) - 1
            if _is_retryable_ytdlp_error(message) and not is_last_client:
                continue
            if _is_retryable_ytdlp_error(message) and is_last_client:
                raise RuntimeError(
                    "yt-dlp could not download the URL after trying multiple YouTube clients"
                ) from err
            raise

    latest = max(cache_dir.glob("*"), key=lambda p: p.stat().st_mtime)
    return latest


def ingest(url: str | None, video: str | None, out_dir: str | Path, fps: int = 30) -> dict:
    out = ensure_dir(out_dir)
    source_dir = ensure_dir(out / "source")
    proxy_path = out / "proxy.mp4"

    if url:
        src = _download_youtube(url, source_dir)
    elif video:
        src = Path(video)
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
    write_json(out / "ingest_meta.json", meta)
    return meta
