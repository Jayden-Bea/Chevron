# Chevron

`Chevron` is a Python CLI for step-1 generation of match-level stitched top-down videos from FRC broadcast VODs.

## Scope

Implemented focus:
- Ingest YouTube/local video and normalize to proxy MP4.
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

2) Run the full pipeline:

```bash
chevron run --url "https://youtube.com/watch?v=..." --config configs/example_config.yml --out out_dir/
# reruns automatically reuse existing ingest output in out_dir/workdir by default
```

For local files:

```bash
chevron run --video /path/to/vod.mp4 --config configs/example_config.yml --out out_dir/
```

## Incremental commands

```bash
chevron ingest --url <vod_url> --out workdir/
chevron segment --video workdir/proxy.mp4 --config configs/example_config.yml --out workdir/segments.json
chevron split --video workdir/proxy.mp4 --segments workdir/segments.json --config configs/example_config.yml --out workdir/splits/
chevron calibrate --video workdir/proxy.mp4 --config configs/example_config.yml --out workdir/calib/
chevron render --video workdir/proxy.mp4 --segments workdir/segments.json --calib workdir/calib/calib.json --config configs/example_config.yml --out out_dir/
```


## Verify config (Streamlit UI)

Use the verifier to inspect ROIs, crop layout, calibration correspondences, and warp/composite previews on any frame. For responsive scrubbing, run ingest first and use the normalized proxy video (`workdir/proxy.mp4`).

```bash
chevron verify --video workdir/proxy.mp4 --config configs/my_event.yml --calib workdir/calib.json
```

What the verifier shows:
- Broadcast frame with ROI + crop overlays.
- Per-view crops (`top`, `bottom_left`, `bottom_right`) with optional calibration points.
- Live calibration editing in the sidebar (adjust `image_points`/`field_points`, nudge points, add/delete points, define field width/height + pixel scale, and download edited correspondences JSON).
- Warp preview defaults to calibration-file homographies when `--calib` is provided; enable `preview_edited_correspondences_for_warp` in the sidebar to preview edited points directly.
- Calibration visualization as points (all correspondences) and as a quadrilateral polygon (first 4 points connected in order).
- Warped top-down view per crop and stitched composite preview when homographies are available.
- Reprojection metrics (`avg`, `median`, `p95`) in top-down pixels.

### Quadrilateral vs rectangle

- **Image-space calibration points** can be any arbitrary quadrilateral or scattered points sampled on the carpet plane; they do not need to form an axis-aligned rectangle.
- **Destination field points** are typically laid out on a rectangular top-down canvas because that is the canonical render coordinate system.

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

- `chevron run` writes progress checkpoints to `out_dir/workdir/run_status.json` (ingest/segment/calibrate/render stage updates, including per-match render progress).
- Optional monitoring knobs in config: `monitoring.segment_progress_interval_s` and `monitoring.render_progress_interval_s` (seconds).
- `chevron run` defaults to `--resume`, so if `workdir/ingest_meta.json` + proxy already exist, ingest is reused instead of re-running.
- On resumed runs, already-rendered match outputs are detected (`match_<n>/topdown.mp4` + `match_meta.json`) and skipped; only missing matches are rendered.
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
- **Warp looks stretched**:
  - add better-distributed correspondence points.
  - verify field coordinate units and `px_per_unit`.
- **Edges ghosting in composite**:
  - tune masks/priority strategy (current implementation uses simple feather blend).

