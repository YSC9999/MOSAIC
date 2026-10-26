# 4. Code Walkthrough

One section per file. You don't need to read the code to use the project, but this tells you what each file is for and where to look.

Everything is in the `mosaic/` folder. Arrows (→) show which module uses which.

```
contracts ← config ← manifest
   ↑          ↑          ↑
ingestion ← pipeline ← cli
   ↑
synthetic        seeding, logging_utils, artifacts, environment (stand-alone helpers)
```

---

## `contracts.py` — "what valid data looks like"

- **Validators** (`require_finite`, `require_unit_interval`, `require_int`, …): tiny functions that either return the cleaned value or raise `ContractError` saying exactly what's wrong. Note that `True`/`False` are *rejected* as numbers (in Python `True` counts as `1`, a classic source of silent bugs).
- **`Box`** — `[x1, y1, x2, y2]` in pixels; `(x1, y1)` top-left, `(x2, y2)` bottom-right. Provides `width`, `height`, `area`, `centroid` (centre point) and `check_within(frame_w, frame_h)`.
- **`Detection`** — one detected object in one frame (box, confidence, class name, frame index, timestamp, optional feature vector).
- **`ClarityReport`** — the video-quality score plus its components; an unavailable component is `None`, and `unavailable_components` lists them.
- **`ObjectEvidence`, `MaskEvidence`** — optional extra evidence; neither can set the violence label.
- **`TrackToken`** — one person at one moment on one local track (box, observed-vs-predicted flag, visibility, centroid, velocity, acceleration, optional extras). *"Local track"* means the ID only has meaning inside one video; it is **not** a person's identity.
- **`TrackSummary`**, **`DecodeDiagnostics`**, **`WindowDiagnostics`**, **`WindowOutput`** — summaries and the final answer record.
- Every class has `to_dict()` / `from_dict()` (and `WindowOutput` has JSON helpers) so data can be saved and reloaded exactly.

## `config.py` — settings

- Section classes: `ExperimentConfig`, `PathsConfig`, `DecodeConfig`, `WindowConfig`, `LoggingConfig`, `PipelineConfig`; all combined in `Config`. Each validates itself on creation.
- `load_config(path, overrides)` — YAML file (optional) + `--set` overrides → validated `Config`.
- `config_from_dict`, `config_to_dict`, `dump_config` — convert between settings objects, dictionaries and YAML.
- `config_hash`, `experiment_id` — fingerprints (see Step 4 in the step-by-step doc).
- `KNOWN_STAGES` — the list of pipeline stage names that exist; later phases add to it.

## `seeding.py` — repeatable randomness

`seed_everything(seed)`, `derive_seed(base, *names)`, `make_rng(base, *names)`.

## `logging_utils.py` — messages

`setup_logging(level, log_file)`, `get_logger(name)`, `log_event(logger, event, **fields)`, `shutdown_logging()` (releases the log file — important on Windows, where open files can't be deleted).

## `artifacts.py` — output locations

`ArtifactLayout(root).ensure()` creates the standard folders; `run_dir(run_id)` creates a safe per-run folder; `make_run_id(experiment_id)` adds a UTC timestamp.

## `environment.py` — machine & software record

`collect_environment()`, `write_environment_record(path)`, `validate_environment_record(record)`. Reads library versions from package metadata (without importing them) and the GPU from `nvidia-smi`.

## `ingestion.py` — video → frames

- `FrameRecord` — `(frame_index, timestamp, frame)`. A frame is a NumPy array of shape *(height, width, 3)* with values 0–255 in **BGR** colour order (OpenCV's convention: blue, green, red).
- `VideoReader(path)` — reads a file. `ArrayFrameSource(frames, fps)` — wraps frames already in memory.
- Both implement `iter_frames()` (a generator of `FrameRecord`) and `.diagnostics` (a `DecodeDiagnostics`).
- `probe_video(path)` — decode a whole file and *return* the report (never raises) — used by the manifest.
- `DecodeError` — raised only when a source can't be used at all.
- Status values: **`ok`**, **`partial`** (some frames decoded but something went wrong), **`empty`** (no frames), **`failed`** (couldn't open).

## `synthetic.py` — fake videos with known answers

`generate_synthetic(scenario, seed, width, height, fps, duration_seconds)` → `SyntheticVideo` with `.frames`, `.ground_truth` (exact box + visible fraction per frame), `.events` (e.g. when the burst starts), `.frame_source()`, `.write(path)` (saves an `.avi`), `.write_ground_truth(path)`.

## `manifest.py` — list of clips

`ManifestEntry` (one clip), `Manifest` (metadata + entries; `read`, `write`, `from_dict`, `summary`, `failed_entries`), `build_entry`, `build_manifest`, `make_meta` (records the window policy), `file_sha256`.

## `pipeline.py` — the assembly line

`Pipeline.from_config(cfg).run(source)` → `RunDiagnostics` (status, frame count, frame shape, full decode report). Never raises for bad video.

## `cli.py` and `__main__.py` — commands

- `python -m mosaic smoke (--video PATH | --synthetic SCENARIO) [--config FILE] [--set KEY=VALUE …]`
- `python -m mosaic make-fixtures [--out FOLDER]`
- Exit codes: `0` ok · `1` configuration problem · `2` video problem.

## Supporting files

| File | Purpose |
|---|---|
| `configs/default.yaml` | Editable, commented settings file |
| `pytest.ini` | Test runner settings and the four test-tier markers |
| `scripts/run_tests.py` | One command per tier |
| `tests/conftest.py` | Auto-marks tests by folder, blocks the network, creates shared test videos |
