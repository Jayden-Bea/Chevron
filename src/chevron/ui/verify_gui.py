from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from chevron.split.layout import get_layout
from chevron.utils.config import load_config
from chevron.utils.io import read_json, write_json

VIEW_ORDER = ["top", "bottom_right", "bottom_left"]


def _draw_points(img: np.ndarray, points: list[list[float]], color: tuple[int, int, int], prefix: str) -> np.ndarray:
    out = img.copy()
    for idx, (x, y) in enumerate(points):
        pt = (int(round(x)), int(round(y)))
        cv2.circle(out, pt, 4, color, -1)
        cv2.putText(out, f"{prefix}{idx}", (pt[0] + 6, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return out


def _build_field_canvas(width_px: int, height_px: int) -> np.ndarray:
    canvas = np.full((height_px, width_px, 3), 20, dtype=np.uint8)
    step = max(20, min(width_px, height_px) // 20)
    for x in range(0, width_px, step):
        cv2.line(canvas, (x, 0), (x, height_px - 1), (40, 40, 40), 1)
    for y in range(0, height_px, step):
        cv2.line(canvas, (0, y), (width_px - 1, y), (40, 40, 40), 1)
    cv2.rectangle(canvas, (0, 0), (width_px - 1, height_px - 1), (80, 80, 80), 2)
    return canvas


def _compute_homography(image_points: list[list[float]], field_points: list[list[float]]) -> np.ndarray | None:
    if len(image_points) < 4 or len(image_points) != len(field_points):
        return None
    src = np.asarray(image_points, dtype=np.float32)
    dst = np.asarray(field_points, dtype=np.float32)
    homography, _ = cv2.findHomography(src, dst, method=0)
    return homography


def _save_correspondences(video: str, config: str, out_json: str | Path, frame_idx: int, correspondences: dict) -> None:
    write_json(
        out_json,
        {
            "video": str(video),
            "config": str(config),
            "frame_idx": int(frame_idx),
            "correspondences": correspondences,
        },
    )


def _render_single_frame_preview(
    frame: np.ndarray,
    layout: dict,
    correspondences: dict,
    canvas_w: int,
    canvas_h: int,
    views_to_render: list[str] | None = None,
) -> np.ndarray:
    preview = _build_field_canvas(canvas_w, canvas_h)
    active_views = views_to_render if views_to_render is not None else VIEW_ORDER
    for view in active_views:
        if view not in layout:
            continue
        pairs = correspondences.get(view) or {}
        image_points = pairs.get("image_points", [])
        field_points = pairs.get("field_points", [])
        homography = _compute_homography(image_points, field_points)
        if homography is None:
            continue
        x, y, cw, ch = layout[view]
        crop = frame[y : y + ch, x : x + cw]
        warped = cv2.warpPerspective(crop, homography, (canvas_w, canvas_h))
        mask = (warped.sum(axis=2) > 0)[:, :, None]
        preview = np.where(mask, warped, preview)
    return preview


def run_local_verify(video: str, config: str, out_json: str | Path, frame_idx: int = 0) -> dict:
    cfg = load_config(config)

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Unable to read frame {frame_idx} from {video}")

    h, w = frame.shape[:2]
    layout = get_layout(cfg, w, h)
    field = cfg.get("field", {})
    canvas_w = int(float(field.get("width_units", 16.46)) * float(field.get("px_per_unit", 120)))
    canvas_h = int(float(field.get("height_units", 8.23)) * float(field.get("px_per_unit", 120)))

    correspondences = ((cfg.get("calibration") or {}).get("correspondences") or {}).copy()
    out_path = Path(out_json)
    if out_path.exists():
        persisted = read_json(out_path)
        if isinstance(persisted, dict):
            correspondences.update((persisted.get("correspondences") or {}))
    for view in VIEW_ORDER:
        pairs = correspondences.get(view) or {}
        correspondences[view] = {
            "image_points": [[float(x), float(y)] for x, y in pairs.get("image_points", [])],
            "field_points": [[float(x), float(y)] for x, y in pairs.get("field_points", [])],
        }

    print("[chevron] verify(local): click IMAGE point in crop window, then matching FIELD point in field window.")
    print("[chevron] verify(local): keys -> n/space(next view), s(skip view), u(undo pair), c(clear view), q(save+next), esc(cancel)")

    views_to_verify = [view for view in VIEW_ORDER if view in layout]
    if not views_to_verify:
        _save_correspondences(video, config, out_path, frame_idx, correspondences)
        print("[chevron] verify(local): no configured views found in layout.")
        return correspondences
    print(f"[chevron] verify(local): sequential verify order -> {', '.join(views_to_verify)}")

    preview_win = "verify render preview"
    cv2.namedWindow(preview_win, cv2.WINDOW_NORMAL)

    for view_index, view in enumerate(views_to_verify, start=1):
        print(f"[chevron] verify(local): calibrating view -> {view}")
        x, y, cw, ch = layout[view]
        crop = frame[y : y + ch, x : x + cw].copy()
        field_canvas = _build_field_canvas(canvas_w, canvas_h)
        image_points = correspondences[view]["image_points"]
        field_points = correspondences[view]["field_points"]
        pending_image_point: list[float] | None = None

        crop_win = f"verify image :: {view}"
        field_win = f"verify field :: {view}"
        cv2.namedWindow(crop_win, cv2.WINDOW_NORMAL)
        cv2.namedWindow(field_win, cv2.WINDOW_NORMAL)

        def on_crop_click(event, mx, my, flags, param):
            nonlocal pending_image_point
            if event == cv2.EVENT_LBUTTONDOWN:
                pending_image_point = [float(mx), float(my)]

        def on_field_click(event, mx, my, flags, param):
            nonlocal pending_image_point
            if event == cv2.EVENT_LBUTTONDOWN and pending_image_point is not None:
                image_points.append(pending_image_point)
                field_points.append([float(mx), float(my)])
                pending_image_point = None

        cv2.setMouseCallback(crop_win, on_crop_click)
        cv2.setMouseCallback(field_win, on_field_click)

        while True:
            crop_display = _draw_points(crop, image_points, (0, 255, 255), "i")
            field_display = _draw_points(field_canvas, field_points, (255, 180, 0), "f")
            if pending_image_point is not None:
                pending_pt = (int(round(pending_image_point[0])), int(round(pending_image_point[1])))
                cv2.circle(crop_display, pending_pt, 6, (0, 0, 255), 2)
                cv2.putText(crop_display, "pending", (pending_pt[0] + 6, pending_pt[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            status = f"view={view} pairs={min(len(image_points), len(field_points))} pending={'yes' if pending_image_point else 'no'}"
            cv2.putText(crop_display, status, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
            cv2.putText(field_display, status, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
            preview_display = _render_single_frame_preview(
                frame,
                layout,
                correspondences,
                canvas_w,
                canvas_h,
                views_to_render=[view],
            )
            preview_status = f"preview frame={frame_idx} step={view_index}/{len(views_to_verify)}"
            cv2.putText(preview_display, preview_status, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(crop_win, crop_display)
            cv2.imshow(field_win, field_display)
            cv2.imshow(preview_win, preview_display)

            key = cv2.waitKey(20) & 0xFF
            if key in (ord("n"), 32, 13):
                break
            if key == ord("s"):
                print(f"[chevron] verify(local): skipping view -> {view}")
                break
            if key == ord("u"):
                if image_points and field_points:
                    image_points.pop()
                    field_points.pop()
                pending_image_point = None
            if key == ord("c"):
                image_points.clear()
                field_points.clear()
                pending_image_point = None
            if key == ord("q"):
                _save_correspondences(video, config, out_path, frame_idx, correspondences)
                break
            if key == 27:
                cv2.destroyAllWindows()
                raise RuntimeError("Verification cancelled by user")

        cv2.destroyWindow(crop_win)
        cv2.destroyWindow(field_win)
        _save_correspondences(video, config, out_path, frame_idx, correspondences)

    _save_correspondences(video, config, out_path, frame_idx, correspondences)
    cv2.destroyAllWindows()
    return correspondences
