from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

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


def _template_score(img, template):
    out = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(out)
    return float(max_val)


def detect_segments(video_path: str | Path, cfg: dict, out_json: str | Path, debug_dir: str | Path | None = None) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt_s = 1.0 / fps

    start_tpl = cv2.imread(cfg["templates"]["start"], cv2.IMREAD_COLOR)
    stop_tpl = cv2.imread(cfg["templates"]["stop"], cv2.IMREAD_COLOR)
    start_roi = cfg["rois"]["start"]
    stop_roi = cfg["rois"]["stop"]
    threshold_start = cfg.get("thresholds", {}).get("start", 0.8)
    threshold_stop = cfg.get("thresholds", {}).get("stop", 0.8)

    machine = SegmentStateMachine(**cfg.get("debounce", {}))
    segments: list[dict] = []
    t_s = 0.0
    frame_idx = 0
    dbg = ensure_dir(debug_dir) if debug_dir else None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        s_score = _template_score(_roi(frame, start_roi), start_tpl)
        e_score = _template_score(_roi(frame, stop_roi), stop_tpl)
        _, seg = machine.update(t_s=t_s, dt_s=dt_s, start_hit=s_score >= threshold_start, stop_hit=e_score >= threshold_stop)
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
