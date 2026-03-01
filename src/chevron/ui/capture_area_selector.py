from __future__ import annotations

from pathlib import Path

from ..utils.io import write_json


def select_capture_area(video_path: str | Path, out_json: str | Path) -> dict[str, int]:
    """Open a scrubbable viewer and allow dragging a capture-area box.

    Controls:
    - Scrub timeline using the frame slider.
    - Press `b` to drag-select a box on the current frame.
    - Press `,` and `.` for previous/next frame.
    - Press Enter or `s` to save.
    - Press `q` or Escape to abort.
    """
    import cv2

    video_path = Path(video_path)
    out_json = Path(out_json)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video for capture-area selection: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        total_frames = 1

    window_name = "chevron capture-area selector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    state = {"frame_idx": 0, "roi": None}

    def _on_seek(idx: int) -> None:
        state["frame_idx"] = max(0, min(total_frames - 1, int(idx)))

    cv2.createTrackbar("frame", window_name, 0, total_frames - 1, _on_seek)

    def _read_frame(frame_idx: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Unable to read frame {frame_idx} from {video_path}")
        return frame

    try:
        while True:
            frame = _read_frame(state["frame_idx"])
            display = frame.copy()

            if state["roi"] is not None:
                x, y, w, h = state["roi"]
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 220, 255), 2)

            help_text = "drag box: [b] save: [Enter/s] step: [,/ .] quit: [q]"
            cv2.putText(display, help_text, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
            cv2.putText(
                display,
                f"frame {state['frame_idx'] + 1}/{total_frames}",
                (14, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 0),
                2,
            )

            cv2.imshow(window_name, display)
            cv2.setTrackbarPos("frame", window_name, int(state["frame_idx"]))

            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                raise RuntimeError("Capture-area selection cancelled by user")
            if key in (13, ord("s")):
                if state["roi"] is None:
                    continue
                x, y, w, h = state["roi"]
                if w <= 0 or h <= 0:
                    continue
                payload = {"x": int(x), "y": int(y), "w": int(w), "h": int(h), "frame_idx": int(state["frame_idx"])}
                write_json(out_json, payload)
                return payload
            if key == ord(","):
                state["frame_idx"] = max(0, state["frame_idx"] - 1)
                continue
            if key == ord("."):
                state["frame_idx"] = min(total_frames - 1, state["frame_idx"] + 1)
                continue
            if key == ord("b"):
                roi = cv2.selectROI(window_name, frame, showCrosshair=True, fromCenter=False)
                x, y, w, h = [int(v) for v in roi]
                if w > 0 and h > 0:
                    state["roi"] = (x, y, w, h)
    finally:
        cap.release()
        cv2.destroyWindow(window_name)
