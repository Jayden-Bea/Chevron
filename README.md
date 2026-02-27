# Chevron

`Chevron` is a Python CLI for step-1 generation of match-level stitched top-down videos from FRC broadcast VODs.

## Scope

Implemented focus:
- Ingest YouTube/local video and normalize to proxy MP4.
- Detect match boundaries using template matching + debounce state machine.
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
- Set ROI rectangles for start/stop/clock overlays.
- Replace template image paths with screenshots from your broadcast.
- Add your own `configs/templates/start.png` and `configs/templates/stop.png` (not bundled in-repo).
- Set crop rectangles (or use `split` preview helper).
- Add calibration correspondences for each camera view.

2) Run the full pipeline:

```bash
chevron run --url "https://youtube.com/watch?v=..." --config configs/example_config.yml --out out_dir/
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
  - lower `thresholds.start`/`thresholds.stop`.
  - verify ROI boxes tightly contain overlay cues.
  - ensure template images are from the same broadcast style/resolution.
- **False starts/stops**:
  - increase `debounce.min_start_s` / `debounce.min_stop_s`.
  - increase `debounce.min_match_s`.
- **Warp looks stretched**:
  - add better-distributed correspondence points.
  - verify field coordinate units and `px_per_unit`.
- **Edges ghosting in composite**:
  - tune masks/priority strategy (current implementation uses simple feather blend).

