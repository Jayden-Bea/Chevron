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


def detect_segments(video_path: str | Path, cfg: dict, out_json: str | Path, debug_dir: str | Path | None = None) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt_s = 1.0 / fps

    start_tpl = cv2.imread(cfg["templates"]["start"], cv2.IMREAD_COLOR)
    stop_tpl = cv2.imread(cfg["templates"]["stop"], cv2.IMREAD_COLOR)
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

    start_roi: list[int] | None = None
    stop_roi: list[int] | None = None

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
