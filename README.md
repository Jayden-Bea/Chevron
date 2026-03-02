# Chevron

`Chevron` is a Python CLI for step-1 generation of match-level stitched top-down videos from FRC broadcast VODs.

## Scope

Implemented focus:
- Ingest YouTube/local video and produce an OpenCV-friendly H.264 proxy MP4 when possible.
- Detect match boundaries using template matching + debounce state machine OR audio cue matching + fixed match length.
- Split default 3-view broadcast layout (top, bottom-left, bottom-right).
- Calibrate per-view homographies from manual correspondences in config.
- Render stitched top-down match videos and per-frame metadata.

Out of scope:
- Fuel heatmaps (step 2).
- 3D reconstruction.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Quickstart

1) Edit `configs/example_config.yml` for your event:
- Choose a segment mode in config:
  - `segment.mode: template` (legacy visual start/stop matching).
  - `segment.mode: audio_cue` (recommended when the clock/overlay is ambiguous).
- For `template` mode:
  - Set ROI rectangles for start/stop/clock overlays.
  - `start.png` and `stop.png` are matched *inside* those ROIs (or inside the full frame if `rois.start`/`rois.stop` are omitted).
  - Matches are similarity-based (not exact-pixel-equality); if a template is larger than the ROI, Chevron auto-downscales it to fit before matching. Tune `thresholds.start/stop` and optional `matching.template_scale_tolerance_pct` (e.g. `1.0`) for slight overlay size drift.
  - Replace template image paths with screenshots from your broadcast.
- For `audio_cue` mode:
  - Provide a short horn cue clip in `audio_cue.template` (e.g. `configs/templates/horn.wav`).
  - Tune `audio_cue.threshold` for robustness in noisy broadcasts.
  - Set `segment.match_length_s` to control how long Chevron records after each detected cue.
- Set crop rectangles (or use `split` preview helper).
- Add calibration correspondences for each camera view.
- Set `processing.output_fps` (recommended `10`) to control proxy/raw/render output frame rate.

2) Run the full pipeline:

```bash
chevron run --url "https://youtube.com/watch?v=..." --config configs/example_config.yml --out out_dir/
# optional auth fallback for gated videos:
# chevron run --url "https://youtube.com/watch?v=..." --config configs/example_config.yml --out out_dir/ --youtube-cookie "<COOKIE_VALUE>"
# optional fast ingest trim for test clips:
# chevron run --url "https://youtube.com/watch?v=..." --config configs/example_config.yml --out out_dir/ --start-at "00:10:00" --end-at "00:12:00"
# reruns automatically reuse existing ingest output in out_dir/workdir by default
```

For local files:

```bash
chevron run --video /path/to/vod.mp4 --config configs/example_config.yml --out out_dir/
```

`chevron run` now executes in this order:
1. Download/transcode ingest proxy.
2. Segment detection over the pipeline video.
3. Export numbered `matches_raw/match_<n>.mp4` clips from the detected segments.
4. Delete the large ingest proxy (`workdir/proxy.mp4`) to save disk space.
5. Run verify → calibrate → render iteratively on each raw match clip.


### YouTube auth fallback (clear input instructions)

**Fastest path (recommended): no manual cookie copying**

```bash
chevron ingest --url "https://youtube.com/watch?v=..." --out workdir/ --youtube-cookies-from-browser chrome
# or with full pipeline:
chevron run --url "https://youtube.com/watch?v=..." --config configs/example_config.yml --out out_dir/ --youtube-cookies-from-browser chrome
```

You can usually swap `chrome` for `edge` or `firefox` if needed. You can also target a specific browser profile (example: `--youtube-cookies-from-browser "chrome:Default"`).

**Most robust option for stubborn auth failures:** export cookies once to Netscape `cookies.txt` and pass `--youtube-cookies-file` (or env var `CHEVRON_YOUTUBE_COOKIES_FILE`). This avoids browser keychain/cookie DB edge cases in CI and remote environments.

```bash
chevron ingest --url "https://youtube.com/watch?v=..." --out workdir/ --youtube-cookies-file /path/to/cookies.txt
# or
export CHEVRON_YOUTUBE_COOKIES_FILE=/path/to/cookies.txt
chevron run --url "https://youtube.com/watch?v=..." --config configs/example_config.yml --out out_dir/
```

If YouTube blocks anonymous download attempts (403 / "sign in to confirm you're not a bot"), Chevron also supports a user-provided Cookie header fallback directly via `--youtube-cookie` (or env var `CHEVRON_YOUTUBE_COOKIE`).

Chevron automatically shuffles YouTube client and user-agent fallback strategies during ingest, so most users should not need to pass low-level yt-dlp flags manually.
Chevron also prefers H.264/AVC video formats during yt-dlp selection so the generated `proxy.mp4` is directly decodable by OpenCV on more systems.

During ingest, Chevron now emits active progress lines so long downloads do not look stuck:
- `yt-dlp heartbeat: ...` confirms the process is still alive.
- `yt-dlp progress: part=<current .part size> / estimate=<yt-dlp estimated final size>` tracks growth against the expected final size.

If you need to pass a cookie manually, use either of these:

```bash
chevron ingest --url "https://youtube.com/watch?v=..." --out workdir/ --youtube-cookie "<COOKIE_OR_HEADER_BLOCK>"

# or
export CHEVRON_YOUTUBE_COOKIE="<COOKIE_OR_HEADER_BLOCK>"
chevron ingest --url "https://youtube.com/watch?v=..." --out workdir/

# easiest persistent option (no manual cookie extraction):
export CHEVRON_YOUTUBE_BROWSER="chrome"
chevron ingest --url "https://youtube.com/watch?v=..." --out workdir/
```

Notes:
- `--youtube-cookies-from-browser` / `CHEVRON_YOUTUBE_BROWSER` is the simplest path for most users.
- `--youtube-cookies-file` / `CHEVRON_YOUTUBE_COOKIES_FILE` is the most reliable path when browser-cookie extraction is flaky.
- The cookie is only used for YouTube ingest authentication fallback.
- If a provided cookie fails, Chevron automatically retries built-in yt-dlp strategies.
- Avoid sharing or committing cookie values; treat them as sensitive credentials.

## Incremental commands

```bash
chevron ingest --url <vod_url> --out workdir/
chevron segment --video workdir/proxy.mp4 --config configs/example_config.yml --out workdir/segments.json
chevron split --video workdir/proxy.mp4 --segments workdir/segments.json --config configs/example_config.yml --out workdir/splits/
chevron render --video workdir/proxy.mp4 --segments workdir/segments.json --calib workdir/calib/calib.json --config configs/example_config.yml --out out_dir/
```

Optional ingest trim for URL sources:

```bash
chevron ingest --url <vod_url> --out workdir/ --start-at "00:10:00" --end-at "00:12:00"
```

Both flags are required together and must use exact `HH:MM:SS` timestamps.

Capture-area selector controls (`--select-capture-area`):
- Scrub the frame trackbar to find a representative frame.
- Press `b` to drag a rectangle.
- Press `Enter` or `s` to save to `workdir/capture_area.json` (or `--capture-area-out`).
- During `chevron run`, if `workdir/capture_area.json` exists, segmentation uses a cropped proxy (`workdir/proxy_cropped.mp4`) while raw export still uses the full-frame ingest proxy. Verify/calibrate/render then run against per-match clips in `matches_raw/`.


## Verify config (local OpenCV UI)

Use the local verifier to define/adjust calibration correspondences before calibration. It opens native OpenCV windows (no webserver) after segmentation during `chevron run`.

```bash
chevron verify --video workdir/proxy.mp4 --config configs/my_event.yml --out workdir/verify_correspondences.json
```

What the local verifier does:
- Opens one crop window, one field-canvas window, and one live single-frame render preview window while you edit correspondences.
- Click an **image point** in the crop, then click its matching **field point** in the field canvas.
- Keyboard controls: `n`/`Space`/`Enter` (advance to next view), `u` (undo last pair), `c` (clear current view), `q` (save + quit), `Esc` (cancel).
- On-screen status shows `pairs=<count>` and `pending=yes/no`; a pair is only committed when you click an image point then its matching field point. The render preview updates continuously using the current point pairs.
- Verify now calibrates only the main `top` view each run.
- Saves correspondences JSON to the path given by `--out`.

### Quadrilateral vs rectangle

- **Image-space calibration points** can be any arbitrary quadrilateral or scattered points sampled on the carpet plane; they do not need to form an axis-aligned rectangle.
- **Destination field points** are typically laid out on a rectangular top-down canvas because that is the canonical render coordinate system.


## Frame-rate control

Set `processing.output_fps` in your config (e.g. `10`) to control the ingest proxy FPS, raw match clip FPS, and top-down render FPS. This reduces runtime, compute, and storage for high-frame-rate source VODs.

## Outputs

Per match:
- `match_<n>/topdown.mp4`
- `match_<n>/match_meta.json`
  - `src_vod_start_s`, `src_vod_end_s`
  - `fps`, `frame_count`
  - per-frame `frame_idx`, `t_video_s`, optional `t_match_s` (currently null unless OCR integration is enabled later)
  - `config_hash`, `calib_version`

Raw extracted source clips before top-down rendering:
- `matches_raw/match_<n>.mp4`
- `matches_raw/matches_raw.json`

- `chevron run` writes progress checkpoints to `out_dir/workdir/run_status.json` (ingest/segment/raw-export/proxy-delete/verify/calibrate/render stage updates, including per-clip render progress).
- After segmentation and raw-export, `chevron run` launches a local OpenCV verifier per extracted match clip so users only calibrate on in-match frames.
- Optional monitoring knobs in config: `monitoring.segment_progress_interval_s` and `monitoring.render_progress_interval_s` (seconds).
- `chevron run` defaults to `--resume`, so if `workdir/ingest_meta.json` + proxy already exist, ingest is reused instead of re-running.
- On resumed runs, already-rendered match outputs are detected (`match_<n>/topdown.mp4` + `match_meta.json`) and skipped; only missing matches are rendered.
- CLI logs now include both UTC timestamps and elapsed runtime for easier step-by-step time estimation.
- Added `chevron detect --video-dir <mp4_folder> --reference <fuel_element_image> --out <dir> [--config detect.yml] [--combine]` for dense multi-match detection tuned via `detect.*` config keys; outputs are named per input stem (e.g., `match_002_map.png`, `match_002_map.gif`) with optional combined outputs (`match_combined_map.png`, `match_combined_map.gif`) plus `tracked_counts.json`.
Debug artifacts:
- segment score frames in `segment_debug/`
- split layout preview image
- calibration view frame snapshots in `calib_frames/`

## Calibration workflow

1. Run `split` to produce `layout_preview.png` and adjust crop boxes.
2. Run `calibrate` to export per-view calibration frames.
3. Collect 4+ image-to-field correspondences per view and place them in config.
4. Re-run `calibrate` to produce `calib.json` homographies.
5. Run `render`.

## Troubleshooting

- **No segments found**:
  - in `template` mode, lower `thresholds.start`/`thresholds.stop`.
  - in `template` mode, verify ROI boxes tightly contain overlay cues.
  - in `audio_cue` mode, lower `audio_cue.threshold` and use a cleaner horn sample.
- **False starts/stops**:
  - in `template` mode, increase `debounce.min_start_s` / `debounce.min_stop_s` and `debounce.min_match_s`.
  - in `audio_cue` mode, increase `audio_cue.threshold` and/or `audio_cue.min_separation_s`.
- **Segments are too short/long in audio mode**:
  - adjust `segment.match_length_s` to your event timing.
- **AV1 decode / frame-0 read failures in verify/calibrate**:
  - Chevron now requests H.264-first formats from yt-dlp. If your source is still AV1-only, transcode once with `ffmpeg -i input.mp4 -c:v libx264 -pix_fmt yuv420p -an fixed.mp4` and ingest `fixed.mp4`.
- **Warp looks stretched**:
  - add better-distributed correspondence points.
  - verify field coordinate units and `px_per_unit`.
- **Edges ghosting in composite**:
  - tune masks/priority strategy (current implementation uses simple feather blend).
