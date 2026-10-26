# 8. Notes, Decisions, and Open Items

This document records architectural decisions made during Phase 0, notes about the working environment, and prerequisites for starting Phase 1.

---

## 1. Architectural Decisions Made in Phase 0

### Typed Data Contracts in Pure Python
- **Decision:** We used standard Python `@dataclass(frozen=True)` with explicit validation in `__post_init__` instead of bringing in third-party validation libraries like `pydantic`.
- **Reason:** It keeps our core foundation dependency-free, easy to understand for beginners, and fast. If data structures grow significantly in later phases, adopting Pydantic can be revisited.

### TrackToken Confidence for Predicted Observations
- **Decision:** In `TrackToken`, `detector_confidence` is `Optional[float]`. When `observed=True`, it must be a finite float in `[0, 1]`. When `observed=False` (Kalman-predicted), it must be `None`.
- **Reason:** Enforces the plan's principle: *never fabricate success-shaped defaults*. A person who was not seen by the detector has no detector confidence.

### Standalone Manifest Construction
- **Decision:** The manifest structure (`Manifest` and `ManifestEntry`) stores both dataset metadata and clip-level decoding diagnostics. It can be built and verified independently without loading neural networks.
- **Reason:** Inspecting and cleaning dataset clips should occur before training starts. Corrupt clips are flagged early with `decode_status: "failed"` and documented reasons.

### Network Isolation in Fast Test Tiers
- **Decision:** In `tests/conftest.py`, standard socket connections are monkeypatched to raise a runtime error during `unit` and `integration` test runs.
- **Reason:** Guarantees that regression tests never secretly make external web requests or download files behind the scenes.

---

## 2. Environment Notes

### Existing `mosaic-fl-baseline` Editable Package
- **Finding:** Your Python environment had an existing editable package installation registered under the name `mosaic-fl-baseline` pointing to a missing `src` directory from an earlier test.
- **Action:** Our codebase is created under the clean package namespace `mosaic`. There is no conflict or overlap. The stale package can be uninstalled anytime with `python -m pip uninstall -y mosaic-fl-baseline` if desired.

### GPU & Compute Hardware
- **Processor:** 13th Gen Intel Core i9-13900HX (32 logical threads)
- **RAM:** 32 GB
- **GPU:** NVIDIA GeForce RTX 4070 Laptop GPU (8,188 MiB VRAM) with CUDA 12.6 driver support
- **Implication:** The 8 GB VRAM is well suited for running RT-DETR or lightweight YOLO person detectors along with the temporal models planned for Phases 4–7.

---

## 3. Phase 1 Fixes & Architectural Alignment

The following issues from the initial Phase 1 pass were identified and resolved:

1. **Unified Manifest Metadata & Corrupt Clip Reporting (`mosaic/dataset.py`, `mosaic/manifest.py`):**
   - `scan_rwf_directory` now records full manifest metadata (`schema_version`, `dataset_name`, `window_policy`, `clip_aggregation`, `seed`, `val_carve_ratio`) and converts cleanly to the Phase 0 `Manifest` contract via `to_phase0_manifest()`.
   - Corrupt, empty, or undecodable clips are never given fabricated frame counts (`frame_count=1`); they are recorded explicitly in `corrupt_entries` with their `decode_status` and `error` messages.
   - Added a 3-keyframe perceptual hash (`frame_signature`) alongside exact SHA-256 hashing to detect near-duplicate clips across splits (Plan Section 3.3 item 7).
2. **Lazy ML Imports in CLI (`mosaic/cli.py`):**
   - PyTorch and torchvision are imported lazily inside Phase 1+ commands so Phase 0 commands (`smoke`, `make-fixtures`) run without importing PyTorch and remain compatible with minimal `requirements.txt`-only environments.
3. **Reachable Padding, Validity Masks & Time Deltas (`mosaic/sampling.py`, `mosaic/dataset.py`, `mosaic/models/temporal_baseline.py`):**
   - Short clips (`len(frames) < num_frames`) now properly invoke `pad_or_truncate_frames` and produce accurate boolean `validity_mask` tensors (`False` on padded tail frames) and per-step `time_deltas` (`0.0` at step 0 and padded frames).
   - `BaselineTemporalClassifier` feeds `time_deltas` into the recurrent layer and gathers the hidden state at the last valid frame index per sample (`validity_mask`), preventing padded frames from corrupting predictions.
   - `LightCNNBackbone` preserves a `4x4` spatial grid with `GroupNorm`, and frozen backbones keep normalization layers in `.eval()` mode during `model.train()`.
4. **Training Loop & Baseline Evaluation Fixes (`mosaic/training.py`, `mosaic/models/baselines.py`):**
   - Fixed a bug where `train_targets` and `train_energies` were collected in two separate passes over a shuffled `train_loader`, which previously misaligned features and labels for `MotionEnergyClassifier`.
   - `train_baseline_classifier` now logs structured events to `run.log`, reloads `best_checkpoint.pt` before final evaluation, supports `load_checkpoint_and_evaluate` for exact reproduction from disk, uses strict `>` comparisons for `beats_majority` / `beats_motion_energy`, and writes explicit `comparison_notes` when a trivial baseline is not strictly beaten.

---

## 4. Open Items Before Starting Phase 1

To proceed with **Phase 1 (Dataset Ingestion and Baseline Temporal Classifier)**, the following need to be addressed:

1. **RWF-2000 Dataset Availability:**
   - Where is the RWF-2000 dataset stored on your machine? (If not yet downloaded, we will document how to download the official repository or release).
2. **Train / Validation / Test Split Verification:**
   - RWF-2000 conventionally ships with `train/` (1,600 clips) and `val/` (400 clips). As defined in section 3.3 of the plan:
     - The official `val/` folder will be preserved as the held-out test set.
     - A dedicated validation split will be carved from `train/` using a deterministic, seeded split.
3. **ML Requirements Installation:**
   - For Phase 1, PyTorch and torchvision are required. They are already installed in your system Python, but when training begins, we will activate the ML dependency set defined in `requirements-ml.txt`.
