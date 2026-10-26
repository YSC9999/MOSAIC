# Phase 0 — Repository & Experiment Foundation

> **Goal of Phase 0:** Build a reliable, testable workshop *before* writing any Machine Learning code.

---

## 1. Why Not Jump Straight Into AI Models?

Imagine you are building a race car. Before you drop a 1,000-horsepower engine into the frame, you need:
- **Brakes and steering** that work predictably.
- **Gauges** (speedometer, temperature, fuel) that never lie to you.
- **Standardized bolts and connectors** so parts fit together without leaking oil.
- **A test track** where you can safely test the car.

In Machine Learning projects, jumping straight into neural networks without a foundation leads to disaster:
1. **Silent bugs:** A corrupted video file returns black frames, and your model secretly trains on garbage without warning you.
2. **Irreproducible results:** You get 85% accuracy on Tuesday, change one line, get 70% on Wednesday, change it back, and get 78%—because random numbers weren't locked down.
3. **Mismatched data:** Module A outputs bounding boxes as `[x, y, width, height]`, while Module B expects `[x1, y1, x2, y2]`.

**Phase 0 solves all of these problems up front.** Phase 0 intentionally uses **zero Machine Learning** so that the foundation is fast, lightweight, and 100% deterministic.

---

## 2. The Big Picture of MOSAIC-FL (Bounded Pipeline)

Our overall project takes a short (~5 second) surveillance video clip and answers two things:
1. **Scene Decision:** Is this clip `violent` or `non_violent`? (Plus a confidence score from `0.0` to `1.0`).
2. **Spatial Evidence:** Where are the people in the video, and how are they moving over time?

Here is the assembly line we are building across Phases 0–8:

```text
[Video File]
     │
     ▼
 1. Ingestion (Phase 0) ────────► Reads video frames, assigns exact timestamps, checks for corruption
     │
     ▼
 2. Clarity Profiler (Phase 2) ─► Measures video quality (sharpness, resolution, lighting)
     │
     ▼
 3. Motion Gate (Phase 3) ──────► Skips frames where nothing moved to save compute
     │
     ▼
 4. Person Detector (Phase 4) ──► Draws bounding boxes [x1, y1, x2, y2] around every person
     │
     ▼
 5. Masks & Kinematics (Phase 5)► Extracts body outlines (silhouettes), speed, and acceleration
     │
     ▼
 6. Multi-Object Tracker (Phase 6)► Follows each person across time and packages "Track Tokens"
     │
     ▼
 7. Temporal Classifier (Phase 1 & 7)► Watches how the scene/tracks evolve over time -> violent / non_violent
```

### The Golden Rule of Our Codebase
> **"Say so when something is missing; never pretend."**
> - If a video file is corrupted, we report `status = "failed"` with the exact reason—we never invent blank frames.
> - If a person is hidden behind a wall and predicted by math (Kalman filter) rather than seen by the camera, we mark `observed = False` and set `detector_confidence = None`—we never fake a `0.0` confidence.

---

## 3. Core Concepts Taught in Phase 0

### 3.1 What is a "Data Contract"? ([`mosaic/contracts.py`](../mosaic/contracts.py))
When one part of a program hands data to another part, they need a strict agreement—a **contract**—about what that data looks like.
For example:
- A **Bounding Box (`Box`)** is four numbers `[x1, y1, x2, y2]` marking the top-left corner `(x1, y1)` and bottom-right corner `(x2, y2)` of a rectangle in pixel coordinates.
  - What if a bug produces `x2 < x1` (a box with negative width) or `NaN` ("Not a Number")?
  - Our `Box` contract checks these rules the instant a `Box` is created. If anything is invalid, it immediately stops with a `ContractError` telling you which value broke the rule.
- **Immutability (`frozen=True`):** Once a contract object is created, no later code can accidentally overwrite its fields behind your back.

### 3.2 What is "Configuration & Behavioural Hashing"? ([`mosaic/config.py`](../mosaic/config.py))
Instead of scattering "magic numbers" (like `seed = 1337` or `num_frames = 16`) across 20 Python files, we put every tunable setting in one YAML file ([`configs/default.yaml`](../configs/default.yaml)) and validate it with [`mosaic/config.py`](../mosaic/config.py).
- **Typo Protection:** If you type `windw:` instead of `window:` in your YAML file, most programs silently ignore your typo and use default settings without telling you! Our config loader rejects any unknown setting name immediately.
- **Config Hash (`config_hash`):** How do you know if two experiment folders used the exact same scientific settings?
  - We take all settings that affect the output (excluding folder paths and log levels) and pass them through **SHA-256** (a mathematical fingerprinting function).
  - The first 12 characters of that fingerprint become the **Config Hash** (e.g., `7cb85650db4e`). Change even one hyperparameter (like learning rate from `0.0001` to `0.0002`), and the hash changes completely!

### 3.3 What is a "Random Seed"? ([`mosaic/seeding.py`](../mosaic/seeding.py))
Computers cannot flip a physical coin. Instead, they use mathematical formulas that generate sequences of numbers that *look* random, starting from an initial number called a **seed**.
- If you start with `seed = 1337`, the formula produces the **exact same sequence of numbers** every single time.
- **Sub-seeds (`derive_seed`):** We derive separate seeds for each component (e.g., `"synthetic/background"`, `"rwf_stratified_split"`, `"dataset_augment"`) from the master seed so changing one component never disturbs the random numbers of another component.

### 3.4 How Does Video Decoding Work? ([`mosaic/ingestion.py`](../mosaic/ingestion.py))
A video file (`.avi` or `.mp4`) is a container holding compressed images (**frames**) and a frame rate (**FPS**, frames per second).
- A 5-second video at `30 FPS` contains `5 × 30 = 150` still images.
- Each frame is a 3D grid of pixel numbers with shape `(Height, Width, 3)`, where `3` is the three color channels (**Blue, Green, Red** in OpenCV's convention), each from `0` to `255`.
- **Why we compute timestamps ourselves:** Different video players sometimes round timestamps differently. We compute every frame's timestamp deterministically as:
  $$\text{timestamp} = \frac{\text{frame\_index}}{\text{FPS}}$$
  Frame 0 is at `0.000s`, Frame 1 at `0.0333s`, Frame 30 at `1.000s`.

### 3.5 Why Do We Need "Synthetic Videos"? ([`mosaic/synthetic.py`](../mosaic/synthetic.py))
The real-world dataset we use (**RWF-2000**) only tells us whether an entire 5-second video clip is a `Fight` or `NonFight`. Nobody manually drew bounding boxes around every person in all 300,000 frames of RWF-2000!
- So how can we test whether our future **Motion Gate** (Phase 3) or **Person Tracker** (Phase 6) calculates speed, acceleration, and occlusion recovery accurately?
- We generate **Synthetic Videos** where *we* draw a colored rectangle moving on a textured background along a mathematical path we control. Because we drew it, we know the **100% exact ground-truth bounding box and visibility fraction** in every single frame!
- We built four synthetic test scenarios:
  1. `static`: The rectangle never moves (tests that the motion gate skips static frames).
  2. `moving`: Moves smoothly in a straight line (tests velocity calculation).
  3. `burst`: Still for 1 second $\rightarrow$ sudden fast movement $\rightarrow$ still again (tests motion burst detection).
  4. `occlusion`: Moves behind a solid wall (0% visible) and emerges on the other side (tests that the tracker remembers a hidden person without losing their ID).

---

## 4. Tools & Libraries Used in Phase 0 (And Why)

| Tool / Library | What It Does | Why We Chose It Over Alternatives |
| :--- | :--- | :--- |
| **Python 3.10** | Programming language | Universal standard for AI/Computer Vision; clean, readable syntax. |
| **NumPy (`numpy`)** | Fast math on multi-dimensional arrays | OpenCV video frames *are* NumPy arrays; 100× faster than Python loops. |
| **OpenCV (`opencv-python`)** | Video decoding, image resizing, drawing | Bundles its own video codecs (no separate FFmpeg setup needed) and provides all image operations needed in later phases. |
| **PyYAML (`yaml`)** | Reads/writes `.yaml` configuration files | Supports human-readable comments (unlike JSON). |
| **PyTest (`pytest`)** | Automated test runner | Simple `assert` syntax, test markers (`unit`, `integration`), and fixtures. |
| **Python `dataclasses`** | Defines structured data records | Built into Python (zero extra dependencies) and allows custom validation in `__post_init__`. |

---

## 5. File-by-File Map of Phase 0

1. [`mosaic/contracts.py`](../mosaic/contracts.py) — Defines `Box`, `Detection`, `ClarityReport`, `ObjectEvidence`, `MaskEvidence`, `TrackToken`, `TrackSummary`, `DecodeDiagnostics`, `WindowDiagnostics`, and `WindowOutput`.
2. [`mosaic/config.py`](../mosaic/config.py) & [`configs/default.yaml`](../configs/default.yaml) — Defines and validates all configuration sections, CLI `--set` overrides, `config_hash`, and `experiment_id`.
3. [`mosaic/seeding.py`](../mosaic/seeding.py) — Implements `seed_everything`, `derive_seed`, and `make_rng`.
4. [`mosaic/environment.py`](../mosaic/environment.py) — Records OS, CPU, RAM, NVIDIA GPU (`nvidia-smi`), and exact library versions into `environment.json` for every run.
5. [`mosaic/artifacts.py`](../mosaic/artifacts.py) — Manages the `artifacts/` directory structure (`runs/`, `manifests/`, `checkpoints/`, `fixtures/`, etc.).
6. [`mosaic/logging_utils.py`](../mosaic/logging_utils.py) — Writes human-readable and machine-searchable (`event=...`) logs to console and `run.log`.
7. [`mosaic/ingestion.py`](../mosaic/ingestion.py) — `VideoReader` (for video files) and `ArrayFrameSource` (for in-memory arrays), plus `probe_video`.
8. [`mosaic/synthetic.py`](../mosaic/synthetic.py) — Generates the 4 deterministic synthetic benchmark videos (`static`, `moving`, `burst`, `occlusion`) and their ground-truth JSON files.
9. [`mosaic/manifest.py`](../mosaic/manifest.py) — Builds and validates dataset manifests (`Manifest`, `ManifestEntry`) and computes SHA-256 file hashes.
10. [`mosaic/pipeline.py`](../mosaic/pipeline.py) & [`mosaic/cli.py`](../mosaic/cli.py) — The pipeline runner and command-line interface (`python -m mosaic smoke` and `python -m mosaic make-fixtures`).

---

## 6. Try It Yourself (Commands)

Run these in PowerShell from the project root:

```powershell
# 1. Run a smoke test on an in-memory synthetic video (no files needed)
python -m mosaic smoke --synthetic burst

# 2. Generate the 4 synthetic .avi videos and their ground-truth JSONs on disk
python -m mosaic make-fixtures --out artifacts/fixtures

# 3. Run the smoke test on a real encoded .avi video file
python -m mosaic smoke --video artifacts/fixtures/occlusion.avi
```
