from __future__ import annotations

import argparse
import copy
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from chevron.calib.homography import apply_homography, compute_homography
from chevron.render.blend import blend_layers
from chevron.render.warp import warp_to_canvas
from chevron.split.layout import get_layout
from chevron.utils.config import load_config
from chevron.utils.io import read_json

VIEW_NAMES = ["top", "bottom_left", "bottom_right"]


class FrameReader:
    def __init__(self, video_path: str | Path, cache_size: int = 10):
        self.video_path = str(video_path)
        self.cache_size = cache_size
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def read_frame(self, frame_idx: int) -> np.ndarray:
        idx = int(np.clip(frame_idx, 0, max(self.frame_count - 1, 0)))
        if idx in self._cache:
            frame = self._cache.pop(idx)
            self._cache[idx] = frame
            return frame.copy()

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Unable to read frame {idx}")

        self._cache[idx] = frame.copy()
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return frame

    def close(self) -> None:
        self.cap.release()


def draw_rois(frame_bgr: np.ndarray, rois_dict: dict[str, list[int]]) -> np.ndarray:
    out = frame_bgr.copy()
    for label, rect in (rois_dict or {}).items():
        x, y, w, h = map(int, rect)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(out, label, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return out


def draw_crops(frame_bgr: np.ndarray, crops_dict: dict[str, list[int]]) -> np.ndarray:
    out = frame_bgr.copy()
    color_map = {
        "top": (255, 100, 0),
        "bottom_left": (80, 220, 80),
        "bottom_right": (180, 80, 255),
    }
    for label, rect in (crops_dict or {}).items():
        x, y, w, h = map(int, rect)
        color = color_map.get(label, (255, 255, 255))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, label, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def draw_points(img: np.ndarray, pts: list[list[float]] | np.ndarray, label: str | None = None) -> np.ndarray:
    out = img.copy()
    arr = np.array(pts, dtype=np.float32).reshape(-1, 2) if len(pts) else np.empty((0, 2), dtype=np.float32)
    for i, (x, y) in enumerate(arr):
        cv2.circle(out, (int(round(x)), int(round(y))), 4, (0, 255, 255), -1)
        point_label = f"{label}:{i}" if label else str(i)
        cv2.putText(out, point_label, (int(round(x)) + 5, int(round(y)) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return out


def draw_polygon(img: np.ndarray, pts4: list[list[float]] | np.ndarray) -> np.ndarray:
    out = img.copy()
    arr = np.array(pts4, dtype=np.float32).reshape(-1, 2)
    if arr.shape[0] < 4:
        return out
    quad = arr[:4].astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [quad], isClosed=True, color=(255, 0, 255), thickness=2)
    return out


def clone_correspondences(correspondences: dict[str, Any]) -> dict[str, dict[str, list[list[float]]]]:
    out: dict[str, dict[str, list[list[float]]]] = {}
    for view in VIEW_NAMES:
        pairs = correspondences.get(view, {}) if isinstance(correspondences.get(view, {}), dict) else {}
        img_pts = pairs.get("image_points", [])
        fld_pts = pairs.get("field_points", [])
        out[view] = {
            "image_points": [[float(x), float(y)] for x, y in img_pts],
            "field_points": [[float(x), float(y)] for x, y in fld_pts],
        }
    return out


def compute_homographies(
    cfg: dict[str, Any],
    calib_data: dict[str, Any] | None,
    correspondences: dict[str, dict[str, list[list[float]]]] | None = None,
    prefer_correspondences: bool = False,
) -> dict[str, np.ndarray]:
    hs: dict[str, np.ndarray] = {}
    if calib_data and calib_data.get("homographies") and not prefer_correspondences:
        hs.update({k: np.array(v, dtype=np.float32) for k, v in calib_data["homographies"].items()})

    corr = correspondences if correspondences is not None else (cfg.get("calibration") or {}).get("correspondences") or {}
    for view, pairs in corr.items():
        if view in hs:
            continue
        image_points = pairs.get("image_points") if isinstance(pairs, dict) else None
        field_points = pairs.get("field_points") if isinstance(pairs, dict) else None
        if not image_points or not field_points:
            continue
        if min(len(image_points), len(field_points)) < 4:
            continue
        try:
            hs[view] = compute_homography(image_points, field_points)
        except (ValueError, RuntimeError):
            continue

    if calib_data and calib_data.get("homographies") and prefer_correspondences:
        for view, h in calib_data["homographies"].items():
            if view not in hs:
                hs[view] = np.array(h, dtype=np.float32)

    return hs




def resolve_field_settings(cfg: dict[str, Any], calib_data: dict[str, Any] | None) -> dict[str, float]:
    field_cfg = cfg.get("field") or {}
    width_units = float(field_cfg.get("width_units", 16.46))
    height_units = float(field_cfg.get("height_units", 8.23))
    px_per_unit = float(field_cfg.get("px_per_unit", 120.0))

    if calib_data and calib_data.get("canvas"):
        canvas = calib_data["canvas"]
        canvas_w = float(canvas.get("width_px", width_units * px_per_unit))
        canvas_h = float(canvas.get("height_px", height_units * px_per_unit))
        if width_units > 0:
            px_per_unit = canvas_w / width_units
        elif height_units > 0:
            px_per_unit = canvas_h / height_units

    return {
        "width_units": width_units,
        "height_units": height_units,
        "px_per_unit": px_per_unit,
    }

def compute_canvas_size(cfg: dict[str, Any], calib_data: dict[str, Any] | None, field_override: dict[str, float] | None = None) -> tuple[int, int]:
    field = field_override or resolve_field_settings(cfg, calib_data)
    return int(field["width_units"] * field["px_per_unit"]), int(field["height_units"] * field["px_per_unit"])


def compute_reprojection_metrics(correspondences: dict[str, dict[str, list[list[float]]]], homographies: dict[str, np.ndarray]) -> list[dict[str, float | str]]:
    metrics: list[dict[str, float | str]] = []
    for view in VIEW_NAMES:
        pairs = correspondences.get(view)
        h = homographies.get(view)
        if not pairs or h is None:
            continue
        src = np.array(pairs.get("image_points", []), dtype=np.float32)
        dst = np.array(pairs.get("field_points", []), dtype=np.float32)
        n = min(src.shape[0], dst.shape[0])
        if n == 0:
            continue
        src = src[:n]
        dst = dst[:n]
        pred = apply_homography(h, src)
        errs = np.linalg.norm(pred - dst, axis=1)
        metrics.append(
            {
                "view": view,
                "count": int(errs.size),
                "avg_px": float(np.mean(errs)),
                "median_px": float(np.median(errs)),
                "p95_px": float(np.percentile(errs, 95)),
            }
        )
    return metrics


def extract_crops(frame: np.ndarray, layout: dict[str, list[int]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for view, (x, y, w, h) in layout.items():
        out[view] = frame[y : y + h, x : x + w].copy()
    return out


def build_composite(warped: dict[str, np.ndarray]) -> np.ndarray | None:
    layers = []
    masks = []
    for view in VIEW_NAMES:
        img = warped.get(view)
        if img is None:
            continue
        layers.append(img)
        masks.append((img.sum(axis=2) > 0).astype(np.float32))
    if not layers:
        return None
    return blend_layers(layers, masks)


def resolve_selected_image(
    selected_view: str,
    display_broadcast: np.ndarray,
    crop_displays: dict[str, np.ndarray],
    warped: dict[str, np.ndarray],
    stitched: np.ndarray | None,
) -> np.ndarray | None:
    if selected_view.endswith("crop"):
        return crop_displays.get(selected_view.replace(" crop", ""))
    if selected_view.startswith("warped"):
        return warped.get(selected_view.replace("warped ", ""))
    if selected_view == "stitched":
        return stitched
    return display_broadcast


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--calib")
    p.add_argument("--frame", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import streamlit as st

    args = parse_args(argv)
    cfg = load_config(args.config)
    calib_data = read_json(args.calib) if args.calib else None

    if "reader" not in st.session_state:
        st.session_state["reader"] = FrameReader(args.video, cache_size=10)
    reader: FrameReader = st.session_state["reader"]

    layout = get_layout(cfg, reader.width, reader.height)
    rois = cfg.get("rois", {})

    correspondences = ((cfg.get("calibration") or {}).get("correspondences") or {}).copy()
    if calib_data and calib_data.get("correspondences"):
        correspondences.update(calib_data["correspondences"])

    session_corr_key = f"editable_corr::{Path(args.config).resolve()}::{Path(args.video).resolve()}::{Path(args.calib).resolve() if args.calib else 'none'}"
    if session_corr_key not in st.session_state:
        st.session_state[session_corr_key] = clone_correspondences(correspondences)
    editable_correspondences = st.session_state[session_corr_key]

    st.title("Chevron Config Verifier")
    st.caption("Tip: use the ingest proxy video for fast, smooth scrubbing.")
    st.info(
        """
**How to pick calibration points (before rendering):**
- **Image points** are pixel coordinates on each camera crop (top / bottom_left / bottom_right). Pick points on the *field plane* (tape intersections, corners, marked vertices) and avoid robots/occlusions.
- **Field points** are the matching coordinates on the top-down field canvas (same real-world locations as the image points).
- Add at least **4 well-spread point pairs per view**, keeping the order aligned between image and field lists.
- After adjusting points, run calibration to generate `calib.json`, then continue rendering.
        """
    )

    st.sidebar.header("Controls")
    frame_idx = st.sidebar.slider("Frame", min_value=0, max_value=max(reader.frame_count - 1, 0), value=int(np.clip(args.frame, 0, max(reader.frame_count - 1, 0))))
    sec = st.sidebar.number_input("Jump to timestamp (seconds)", min_value=0.0, max_value=max(reader.frame_count / max(reader.fps, 1.0), 0.0), value=float(frame_idx / max(reader.fps, 1.0)))
    if st.sidebar.button("Jump"):
        frame_idx = int(sec * reader.fps)
    selected_view = st.sidebar.selectbox(
        "Select view",
        [
            "broadcast",
            "top crop",
            "bottom_left crop",
            "bottom_right crop",
            "warped top",
            "warped bottom_left",
            "warped bottom_right",
            "stitched",
        ],
    )

    field_settings = resolve_field_settings(cfg, calib_data)

    st.sidebar.header("Field canvas editor")
    field_width_units = st.sidebar.number_input("Field width (units)", min_value=1.0, value=float(field_settings["width_units"]), step=0.1)
    field_height_units = st.sidebar.number_input("Field height (units)", min_value=1.0, value=float(field_settings["height_units"]), step=0.1)
    field_px_per_unit = st.sidebar.number_input("Pixels per unit", min_value=1.0, value=float(field_settings["px_per_unit"]), step=1.0)

    st.sidebar.header("Calibration editor")
    editor_view = st.sidebar.selectbox("Edit view", VIEW_NAMES, key="calib_editor_view")
    point_space = st.sidebar.radio("Point space", ["image_points", "field_points"], horizontal=True)
    active_points = editable_correspondences.setdefault(editor_view, {}).setdefault(point_space, [])
    point_count = len(active_points)
    if point_count:
        point_idx = st.sidebar.slider("Point index", min_value=0, max_value=point_count - 1, value=0)
        pt_x, pt_y = active_points[point_idx]
        new_x = st.sidebar.number_input("Point x", value=float(pt_x), step=1.0)
        new_y = st.sidebar.number_input("Point y", value=float(pt_y), step=1.0)
        active_points[point_idx] = [float(new_x), float(new_y)]

        nudge_col1, nudge_col2, nudge_col3, nudge_col4 = st.sidebar.columns(4)
        nudge_step = st.sidebar.number_input("Nudge step", min_value=0.1, value=1.0, step=0.5)
        if nudge_col1.button("←", use_container_width=True):
            active_points[point_idx][0] -= float(nudge_step)
            st.rerun()
        if nudge_col2.button("→", use_container_width=True):
            active_points[point_idx][0] += float(nudge_step)
            st.rerun()
        if nudge_col3.button("↑", use_container_width=True):
            active_points[point_idx][1] -= float(nudge_step)
            st.rerun()
        if nudge_col4.button("↓", use_container_width=True):
            active_points[point_idx][1] += float(nudge_step)
            st.rerun()

        if st.sidebar.button("Delete selected point"):
            active_points.pop(point_idx)
            st.rerun()
    else:
        st.sidebar.caption("No points in this set yet.")

    add_default = [0.0, 0.0]
    if active_points:
        add_default = [float(active_points[-1][0]), float(active_points[-1][1])]
    add_x = st.sidebar.number_input("New point x", value=float(add_default[0]), step=1.0)
    add_y = st.sidebar.number_input("New point y", value=float(add_default[1]), step=1.0)
    if st.sidebar.button("Add point"):
        active_points.append([float(add_x), float(add_y)])
        st.rerun()

    if st.sidebar.button("Reset edits from config/calib"):
        st.session_state[session_corr_key] = clone_correspondences(correspondences)
        st.rerun()

    export_payload = {
        "field": {
            "width_units": float(field_width_units),
            "height_units": float(field_height_units),
            "px_per_unit": float(field_px_per_unit),
            "origin": list((cfg.get("field") or {}).get("origin", [0, 0])),
        },
        "calibration": {
            "correspondences": copy.deepcopy(editable_correspondences),
        },
    }
    st.sidebar.download_button(
        "Download edited correspondences (json)",
        data=json.dumps(export_payload, indent=2),
        file_name="edited_correspondences.json",
        mime="application/json",
    )

    has_calib_homographies = bool(calib_data and calib_data.get("homographies"))
    use_edited_for_warp = st.sidebar.checkbox(
        "preview_edited_correspondences_for_warp",
        value=not has_calib_homographies,
        help="When off, warp preview uses homographies from --calib (if available), matching render-time behavior.",
    )

    show_rois = st.sidebar.checkbox("show_rois", value=True)
    show_crops = st.sidebar.checkbox("show_crops", value=True)
    show_calib_points = st.sidebar.checkbox("show_calib_points", value=True)
    show_calib_quad = st.sidebar.checkbox("show_calib_quad", value=True)
    show_warp = st.sidebar.checkbox("show_warp", value=True)
    show_metrics = st.sidebar.checkbox("show_metrics", value=True)

    homography_correspondences = editable_correspondences if use_edited_for_warp else correspondences
    homographies = compute_homographies(
        cfg,
        calib_data,
        correspondences=homography_correspondences,
        prefer_correspondences=use_edited_for_warp,
    )
    canvas_size = compute_canvas_size(
        cfg,
        calib_data,
        field_override={
            "width_units": float(field_width_units),
            "height_units": float(field_height_units),
            "px_per_unit": float(field_px_per_unit),
        },
    )

    frame = reader.read_frame(frame_idx)
    display_broadcast = frame.copy()
    if show_rois:
        display_broadcast = draw_rois(display_broadcast, rois)
    if show_crops:
        display_broadcast = draw_crops(display_broadcast, layout)

    crops = extract_crops(frame, layout)
    crop_displays = {k: v.copy() for k, v in crops.items()}

    for view in VIEW_NAMES:
        pairs = editable_correspondences.get(view, {})
        pts = pairs.get("image_points", [])
        if show_calib_points and view in crop_displays:
            crop_displays[view] = draw_points(crop_displays[view], pts, label=view)
        if show_calib_quad and view in crop_displays:
            crop_displays[view] = draw_polygon(crop_displays[view], pts)

    warped: dict[str, np.ndarray] = {}
    if show_warp:
        for view in VIEW_NAMES:
            if view not in crops or view not in homographies:
                continue
            warped[view] = warp_to_canvas(crops[view], homographies[view], canvas_size)
            pairs = editable_correspondences.get(view, {})
            if show_calib_points:
                warped_points = pairs.get("field_points", [])
                warped[view] = draw_points(warped[view], warped_points, label=f"{view}_field")
            if show_calib_quad:
                warped[view] = draw_polygon(warped[view], pairs.get("field_points", []))

    stitched = build_composite(warped) if show_warp else None

    selected_image = resolve_selected_image(selected_view, display_broadcast, crop_displays, warped, stitched)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Broadcast")
        st.image(cv2.cvtColor(display_broadcast, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
    with col2:
        st.subheader("Selected View")
        if selected_image is None:
            st.info("Selected view unavailable for this frame/config.")
        else:
            st.image(cv2.cvtColor(selected_image, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
    with col3:
        st.subheader("Warp/Composite")
        if stitched is not None:
            st.image(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        elif show_warp:
            st.info("No stitched preview (missing homography data).")

    if show_metrics:
        st.subheader("Reprojection metrics (top-down pixels)")
        metrics = compute_reprojection_metrics(editable_correspondences, homographies)
        if metrics:
            st.table(metrics)
        else:
            st.info("No metrics available (missing correspondences/homographies).")

    with st.expander("Resolved config"):
        st.json(
            {
                "video": str(args.video),
                "config": str(args.config),
                "calib": str(args.calib) if args.calib else None,
                "frame": frame_idx,
                "fps": reader.fps,
                "frame_count": reader.frame_count,
                "layout": layout,
                "rois": rois,
                "field": {
                    "width_units": float(field_width_units),
                    "height_units": float(field_height_units),
                    "px_per_unit": float(field_px_per_unit),
                },
                "canvas_size": {"width_px": canvas_size[0], "height_px": canvas_size[1]},
                "editable_correspondences": editable_correspondences,
                "warp_preview_uses_edited_correspondences": use_edited_for_warp,
            }
        )


if __name__ == "__main__":
    main()
