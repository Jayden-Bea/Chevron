from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile
from typing import Callable
import wave

import cv2
import numpy as np

from ..utils.io import ensure_dir, write_json


@dataclass
class SegmentStateMachine:
    min_start_s: float = 0.5
    min_stop_s: float = 0.5
    min_match_s: float = 60.0
    max_match_s: float = 220.0

    state: str = "IDLE"
    start_streak_s: float = 0.0
    stop_streak_s: float = 0.0
    current_start_s: float | None = None

    def update(self, t_s: float, dt_s: float, start_hit: bool, stop_hit: bool) -> tuple[str, tuple[float, float] | None]:
        segment = None
        if self.state == "IDLE":
            self.start_streak_s = self.start_streak_s + dt_s if start_hit else 0.0
            if self.start_streak_s >= self.min_start_s:
                self.state = "IN_MATCH"
                self.current_start_s = t_s
                self.stop_streak_s = 0.0
        elif self.state == "IN_MATCH":
            self.stop_streak_s = self.stop_streak_s + dt_s if stop_hit else 0.0
            duration = t_s - (self.current_start_s or t_s)
            if self.stop_streak_s >= self.min_stop_s and duration >= self.min_match_s:
                segment = ((self.current_start_s or 0.0), t_s)
                self.state = "COOLDOWN"
                self.start_streak_s = 0.0
            elif duration >= self.max_match_s:
                segment = ((self.current_start_s or 0.0), t_s)
                self.state = "COOLDOWN"
        elif self.state == "COOLDOWN":
            if not start_hit and not stop_hit:
                self.state = "IDLE"
        return self.state, segment


def _roi(frame, rect):
    x, y, w, h = rect
    return frame[y : y + h, x : x + w]


def _resolve_search_rect(frame, rect, template_name: str) -> list[int]:
    frame_h, frame_w = frame.shape[:2]
    if rect is None:
        return [0, 0, frame_w, frame_h]
    if len(rect) != 4:
        raise ValueError(f"ROI '{template_name}' must be [x, y, w, h], got {rect}")
    return [int(v) for v in rect]


def _validate_match_inputs(frame, rect, template, template_name: str) -> None:
    if template is None:
        raise ValueError(f"Template image for '{template_name}' could not be loaded")

    x, y, w, h = map(int, rect)
    if w <= 0 or h <= 0:
        raise ValueError(f"ROI '{template_name}' must have positive width/height, got {rect}")

    frame_h, frame_w = frame.shape[:2]
    if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
        raise ValueError(
            f"ROI '{template_name}' {rect} is outside frame bounds {(frame_w, frame_h)}"
        )


def _build_template_scale_factors(tolerance_pct: float, step_pct: float = 0.5) -> list[float]:
    tolerance_pct = max(0.0, float(tolerance_pct))
    step_pct = max(0.1, float(step_pct))
    if tolerance_pct <= 0.0:
        return [1.0]

    factors: set[float] = {1.0}
    delta = step_pct
    while delta <= tolerance_pct + 1e-9:
        factors.add(max(0.1, 1.0 - delta / 100.0))
        factors.add(1.0 + delta / 100.0)
        delta += step_pct
    return sorted(factors)




def _fit_template_to_image(template, img_h: int, img_w: int):
    tpl_h, tpl_w = template.shape[:2]
    if tpl_h <= img_h and tpl_w <= img_w:
        return template

    fit_scale = min(img_w / max(tpl_w, 1), img_h / max(tpl_h, 1))
    fit_scale = max(0.01, fit_scale)
    resized = cv2.resize(template, dsize=None, fx=fit_scale, fy=fit_scale, interpolation=cv2.INTER_AREA)

    # guard against rounding edge-cases where one dimension is still too large by 1px
    res_h, res_w = resized.shape[:2]
    if res_h > img_h or res_w > img_w:
        final_w = max(1, min(img_w, res_w))
        final_h = max(1, min(img_h, res_h))
        resized = cv2.resize(resized, (final_w, final_h), interpolation=cv2.INTER_AREA)
    return resized

def _template_score(img, template, scale_factors: list[float] | None = None):
    factors = scale_factors or [1.0]
    best = -1.0
    img_h, img_w = img.shape[:2]

    for factor in factors:
        if abs(factor - 1.0) < 1e-9:
            tpl = template
        else:
            tpl = cv2.resize(template, dsize=None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)

        tpl = _fit_template_to_image(tpl, img_h, img_w)
        tpl_h, tpl_w = tpl.shape[:2]
        if tpl_h < 1 or tpl_w < 1:
            continue

        out = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(out)
        if max_val > best:
            best = float(max_val)

    return best


def _extract_audio_samples(path: str | Path, sample_rate_hz: int) -> tuple[np.ndarray, int]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = Path(tmp.name)
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate_hz),
            "-f",
            "wav",
            str(tmp_wav),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg audio extract failed for {path}: {proc.stderr.strip()}")

        with wave.open(str(tmp_wav), "rb") as wav_file:
            sr = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()
            pcm = wav_file.readframes(n_frames)

        if sample_width != 2:
            raise ValueError(f"Unsupported WAV sample width {sample_width * 8} bits for {path}")

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)
        return audio, sr
    finally:
        tmp_wav.unlink(missing_ok=True)


def _spectrogram(signal: np.ndarray, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    sig = np.asarray(signal, dtype=np.float32)
    if sig.size < n_fft:
        sig = np.pad(sig, (0, n_fft - sig.size))

    win = np.hanning(n_fft).astype(np.float32)
    frames: list[np.ndarray] = []
    for start in range(0, sig.size - n_fft + 1, hop):
        frame = sig[start : start + n_fft] * win
        spec = np.abs(np.fft.rfft(frame))
        frames.append(np.log1p(spec))

    if not frames:
        return np.zeros((1, 1), dtype=np.float32)
    out = np.stack(frames, axis=1).astype(np.float32)
    out -= out.mean(axis=0, keepdims=True)
    std = out.std(axis=0, keepdims=True) + 1e-6
    return out / std


def _find_audio_cue_times(
    program_audio: np.ndarray,
    cue_audio: np.ndarray,
    sample_rate_hz: int,
    threshold: float,
    min_separation_s: float,
) -> list[tuple[float, float]]:
    prog_spec = _spectrogram(program_audio)
    cue_spec = _spectrogram(cue_audio)
    if prog_spec.shape[1] < cue_spec.shape[1]:
        return []

    scores = cv2.matchTemplate(prog_spec, cue_spec, cv2.TM_CCOEFF_NORMED).reshape(-1)
    hop = 512
    min_sep_frames = max(1, int((min_separation_s * sample_rate_hz) / hop))

    candidates = np.where(scores >= threshold)[0]
    if candidates.size == 0:
        return []

    order = sorted(candidates.tolist(), key=lambda idx: float(scores[idx]), reverse=True)
    chosen: list[int] = []
    for idx in order:
        if any(abs(idx - selected) < min_sep_frames for selected in chosen):
            continue
        chosen.append(idx)

    chosen.sort()
    return [(idx * hop / sample_rate_hz, float(scores[idx])) for idx in chosen]


def _detect_segments_audio_cue(
    video_path: str | Path,
    cfg: dict,
    total_frames: int,
    fps: float,
    progress_callback: Callable[[dict], None] | None,
) -> list[dict]:
    audio_cfg = cfg.get("audio_cue", {})
    cue_path = audio_cfg.get("template")
    if not cue_path:
        raise ValueError("audio_cue.template is required when segment.mode=audio_cue")

    sample_rate_hz = int(audio_cfg.get("sample_rate_hz", 16000))
    threshold = float(audio_cfg.get("threshold", 0.45))
    min_separation_s = float(audio_cfg.get("min_separation_s", 60.0))
    match_length_s = float(cfg.get("segment", {}).get("match_length_s", 150.0))

    program_audio, program_sr = _extract_audio_samples(video_path, sample_rate_hz=sample_rate_hz)
    cue_audio, cue_sr = _extract_audio_samples(cue_path, sample_rate_hz=sample_rate_hz)
    if program_sr != cue_sr:
        raise ValueError(f"Audio sample-rate mismatch: video={program_sr}, cue={cue_sr}")

    cue_hits = _find_audio_cue_times(
        program_audio=program_audio,
        cue_audio=cue_audio,
        sample_rate_hz=program_sr,
        threshold=threshold,
        min_separation_s=min_separation_s,
    )

    duration_s = total_frames / fps if fps > 0 else 0.0
    segments: list[dict] = []
    for t_start, score in cue_hits:
        t_end = min(t_start + match_length_s, duration_s)
        if t_end <= t_start:
            continue
        segments.append(
            {
                "start_time_s": round(t_start, 3),
                "end_time_s": round(t_end, 3),
                "confidence": round(float(score), 3),
                "debug_frame_paths": [],
            }
        )

    if progress_callback:
        progress_callback(
            {
                "frame_idx": 0,
                "time_s": 0.0,
                "fps": round(fps, 3),
                "total_frames": total_frames,
                "state": "AUDIO_CUE",
                "start_score": None,
                "stop_score": None,
                "start_hit": bool(cue_hits),
                "stop_hit": None,
                "segments_found": len(segments),
            }
        )
    return segments


def detect_segments(
    video_path: str | Path,
    cfg: dict,
    out_json: str | Path,
    debug_dir: str | Path | None = None,
    progress_interval_s: float = 10.0,
    progress_callback: Callable[[dict], None] | None = None,
) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt_s = 1.0 / fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    mode = cfg.get("segment", {}).get("mode", "template")
    start_tpl = None
    stop_tpl = None
    rois = cfg.get("rois", {})
    start_roi_cfg = rois.get("start")
    stop_roi_cfg = rois.get("stop")
    threshold_start = cfg.get("thresholds", {}).get("start", 0.8)
    threshold_stop = cfg.get("thresholds", {}).get("stop", 0.8)
    matching = cfg.get("matching", {})
    scale_tolerance_pct = float(matching.get("template_scale_tolerance_pct", 0.0))
    scale_step_pct = float(matching.get("template_scale_step_pct", 0.5))
    scale_factors = _build_template_scale_factors(scale_tolerance_pct, scale_step_pct)

    machine = SegmentStateMachine(**cfg.get("debounce", {}))
    segments: list[dict] = []
    t_s = 0.0
    frame_idx = 0
    dbg = ensure_dir(debug_dir) if debug_dir else None
    last_progress_log_s = -float(progress_interval_s)

    start_roi: list[int] | None = None
    stop_roi: list[int] | None = None

    if mode == "audio_cue":
        cap.release()
        segments = _detect_segments_audio_cue(
            video_path=video_path,
            cfg=cfg,
            total_frames=total_frames,
            fps=fps,
            progress_callback=progress_callback,
        )
        write_json(out_json, {"segments": segments, "fps": fps})
        return segments

    start_tpl = cv2.imread(cfg["templates"]["start"], cv2.IMREAD_COLOR)
    stop_tpl = cv2.imread(cfg["templates"]["stop"], cv2.IMREAD_COLOR)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx == 0:
            start_roi = _resolve_search_rect(frame, start_roi_cfg, "start")
            stop_roi = _resolve_search_rect(frame, stop_roi_cfg, "stop")
            _validate_match_inputs(frame, start_roi, start_tpl, "start")
            _validate_match_inputs(frame, stop_roi, stop_tpl, "stop")
        s_score = _template_score(_roi(frame, start_roi), start_tpl, scale_factors=scale_factors)
        e_score = _template_score(_roi(frame, stop_roi), stop_tpl, scale_factors=scale_factors)
        start_hit = s_score >= threshold_start
        stop_hit = e_score >= threshold_stop
        state, seg = machine.update(t_s=t_s, dt_s=dt_s, start_hit=start_hit, stop_hit=stop_hit)

        if progress_callback and t_s - last_progress_log_s >= progress_interval_s:
            progress_callback(
                {
                    "frame_idx": frame_idx,
                    "time_s": round(t_s, 3),
                    "fps": round(fps, 3),
                    "total_frames": total_frames,
                    "state": state,
                    "start_score": round(float(s_score), 4),
                    "stop_score": round(float(e_score), 4),
                    "start_hit": start_hit,
                    "stop_hit": stop_hit,
                    "segments_found": len(segments),
                }
            )
            last_progress_log_s = t_s
        if dbg and frame_idx % int(fps * 5) == 0:
            cv2.imwrite(str(dbg / f"score_{frame_idx:06d}.jpg"), frame)
        if seg:
            segments.append(
                {
                    "start_time_s": round(seg[0], 3),
                    "end_time_s": round(seg[1], 3),
                    "confidence": round((threshold_start + threshold_stop) / 2.0, 3),
                    "debug_frame_paths": [],
                }
            )
        t_s += dt_s
        frame_idx += 1

    cap.release()
    write_json(out_json, {"segments": segments, "fps": fps})
    return segments
