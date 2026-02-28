# Template images

This folder is intentionally text-only to keep PR tooling compatible.

Add your own broadcast-specific templates here, for example:
- `start.png`
- `stop.png`
- `horn.wav` (short audio cue clip used for `segment.mode: audio_cue`)

Then point `templates.start`/`templates.stop` and optional `audio_cue.template` in `configs/example_config.yml` (or your own config) to those files.
