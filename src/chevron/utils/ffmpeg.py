from __future__ import annotations

import subprocess
from pathlib import Path


def run_ffmpeg(args: list[str]) -> None:
    cmd = ["ffmpeg", "-y", *args]
    subprocess.run(cmd, check=True)


def normalize_video(in_path: str | Path, out_path: str | Path, fps: int = 30) -> None:
    run_ffmpeg([
        "-i",
        str(in_path),
        "-vf",
        f"fps={fps}",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ])
