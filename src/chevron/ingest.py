from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .utils.ffmpeg import normalize_video
from .utils.io import ensure_dir, write_json


def _download_youtube(url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_tmpl = str(cache_dir / "%(id)s.%(ext)s")
    subprocess.run(["yt-dlp", "-o", output_tmpl, url], check=True)
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
