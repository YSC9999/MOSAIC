# 2. Tools, Libraries and Why We Chose Them

Every tool below is described the same way: **what it is → what we use it for → why we chose it → what the alternatives were**.

"Better" is always *for this project*. None of these choices is universally best, and I note the trade-offs honestly.

Versions are the exact ones pinned in [`requirements.txt`](../requirements.txt) and verified on your machine.

---

## A. Tools used **right now** (Phase 0)

### Python 3.10
- **What:** a programming language. Your machine has 3.10.11.
- **Use:** all of our code.
- **Why:** almost every machine-learning library (PyTorch, OpenCV, detectors, trackers) is written for Python first. Its code reads close to English, which matters when you are learning.
- **Alternatives:** C++ (faster programs, but far slower to write and debug; ML ecosystem is thinner), MATLAB (paid, less common for deep learning), Julia (smaller ecosystem).
- **Trade-off:** pure Python is slow for heavy number-crunching — but the heavy work is done inside libraries (NumPy, OpenCV, PyTorch) written in fast languages, so this rarely matters.

### NumPy `2.2.6`
- **What:** a library for fast math on big tables of numbers ("arrays").
- **Use:** a video frame *is* an array of numbers (height × width × 3 colour values). We store frames, make the synthetic videos' random background texture, and compare frames with NumPy.
- **Why:** it is the universal foundation — OpenCV frames *are* NumPy arrays and PyTorch tensors convert to/from them for free. Operations run in compiled code, hundreds of times faster than Python loops.
- **Alternatives:** plain Python lists (far too slow), PyTorch tensors alone (heavier install; not needed in Phase 0).

### OpenCV (`opencv-python` `5.0.0.93`)
- **What:** the standard computer-vision library — reads/writes video, resizes, blurs, converts colours, measures sharpness, etc.
- **Use now:** reading video files frame-by-frame and writing our synthetic test videos (`ingestion.py`, `synthetic.py`).
- **Use later:** frame differencing for the motion gate (Phase 3), Laplacian-variance sharpness for the clarity score (Phase 2), blur/noise/compression for the degradation toolkit (Phase 2), drawing boxes on debug videos (Phase 6).
- **Why:** one install gives us video decoding *and* every classic image operation our plan needs. The pip package bundles its own video-decoding support, so no separate FFmpeg installation is required.
- **Alternatives:** *imageio / PyAV / decord* (good at video reading only — we'd still need OpenCV for the image operations), *calling FFmpeg directly* (powerful but clunky from Python), *scikit-image / Pillow* (image-only, no video).
- **Trade-off:** OpenCV reports a video's frame count and frame rate from the file header, which can occasionally be wrong. That's exactly why our code double-checks (see `ingestion.py`).

### PyYAML `6.0.3`
- **What:** reads and writes **YAML** files — a human-friendly settings format.
- **Use:** `configs/default.yaml`; also saving a snapshot of the settings used in every run.
- **Why:** YAML allows **comments** (JSON does not), so the settings file can explain itself, and it is easy to edit by hand.
- **Alternatives:** JSON (no comments), TOML (also fine; YAML is more common in ML projects), Python files as config (flexible but unsafe/unstructured).
- **Safety note:** we only use `safe_load`, which can't execute code hidden inside a file.

### pytest `9.1.1`
- **What:** a framework for writing and running automated tests.
- **Use:** all 291 tests.
- **Why:** tests are plain functions with plain `assert` statements — very little ceremony. It also gives us **parametrization** (run one test against many inputs), **fixtures** (shared setup, e.g. generating test videos once), and **markers** (label tests as "unit", "integration", … so we can run one tier at a time).
- **Alternatives:** Python's built-in `unittest` (more boilerplate, weaker parametrization), `nose` (obsolete).

### `dataclasses` (built into Python)
- **What:** a way to define a small record type — "a Box has x1, y1, x2, y2".
- **Use:** every data contract (`Box`, `TrackToken`, `WindowOutput`, …) and every config section.
- **Why:** no extra installation; clear, readable definitions. We add our own validation in each record's `__post_init__` so a malformed object **cannot be created**.
- **Alternative — pydantic:** a popular library that does validation automatically with less code. We deliberately did not add it: it is another dependency, and writing the checks by hand makes every rule visible to someone learning. If the contracts grow much larger, switching to pydantic is a reasonable future decision.

### JSON (built into Python)
- **What:** a universal text format for structured data.
- **Use:** manifests, diagnostics, environment records, ground-truth files.
- **Why:** both humans and programs can read it; supports nested data (a window contains tracks which contain boxes). We write it with sorted keys so identical data produces identical files.
- **Alternatives:** CSV (can't nest data), Parquet/HDF5 (efficient for huge tables; overkill here), pickle (Python-only and unsafe to load from untrusted sources).

### `argparse` (built into Python)
- **What:** builds command-line interfaces (`python -m mosaic smoke --synthetic burst`).
- **Why:** no installation, automatic `--help`. Alternatives (`click`, `typer`) are nicer but add a dependency we don't need.

### `logging` (built into Python)
- **What:** the standard way to write status messages with timestamps and severity levels.
- **Why:** configurable (console + file), standard, no dependencies. Preferred over `print` because messages can be filtered by level and saved to `run.log`.

### `hashlib` — SHA-256 (built into Python)
- **What:** computes a "fingerprint" of data: change one byte and the fingerprint changes completely.
- **Use:** (a) the **config hash** that identifies a set of settings; (b) a **file hash** per video so we can later detect duplicate clips (leakage between train and test).
- **Why:** collision-resistant and universally understood.

### `subprocess` + `nvidia-smi`
- **What:** `nvidia-smi` is NVIDIA's own tool that lists graphics cards; `subprocess` lets Python run it.
- **Use:** recording which GPU the run used (yours: *NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB*).
- **Why:** asking `nvidia-smi` avoids importing PyTorch (a very large library) just to learn the GPU name — so Phase 0 stays lightweight. If the tool is missing, the record says so explicitly.

### `venv` + `pip` + `requirements.txt` (our "dependency policy")
- **What:** `venv` creates an isolated Python installation; `pip` installs packages; `requirements.txt` lists exact versions.
- **Why:** "works on my machine" is the biggest source of irreproducible results. Pinning exact versions means anyone can rebuild the same environment. We **verified** this: a brand-new empty environment built from `requirements.txt` passes all tests (see [08_notes_and_open_items.md](08_notes_and_open_items.md)).
- **Rule we follow:** no package is added merely because it was *mentioned* in the plan; it must be justified by a component that needs it.

### PowerShell / Windows commands
- You are on Windows; all commands in these docs are written for **PowerShell** and are run from the project folder.

---

## B. Tools planned for **later phases** (not installed/used yet)

These are listed so the plan makes sense. We will re-check each one (license, speed, memory) when its phase starts.

### PyTorch `2.12.0+cu126` and torchvision `0.27.0+cu126` *(already on your machine; first used in Phase 1)*
- **What:** the most widely used library for building and training neural networks. "cu126" means the version that can use your NVIDIA GPU (CUDA 12.6).
- **Why:** dominant in research, huge community and model zoo, good GPU support. **Alternatives:** TensorFlow/Keras (also capable; research code is now mostly PyTorch), JAX (powerful, steeper learning curve).
- **Why not in Phase 0:** Phase 0 has no model. We even *test* that the Phase 0 pipeline runs without importing PyTorch.

### RT-DETR (person detector, Phase 4)
- **What:** a pretrained neural network that finds objects (here: people) in an image and returns boxes.
- **Why preferred (per the plan):** it does not use "non-maximum suppression" (NMS), a clean-up step that tends to *delete* real people when they overlap heavily — exactly the situation in fights. The CrowdDet paper in your folder documents that failure.
- **Fallback:** a small YOLO model. The final choice is made at the start of Phase 4 after checking speed, memory and the licence of the pretrained weights. (Your 8 GB GPU should be able to run RT-DETR.)

### ByteTrack-style tracker with a Kalman filter (Phase 6)
- **What:** follows each person from frame to frame and gives them a consistent local ID. A **Kalman filter** predicts where a person should be next (so a briefly hidden person isn't "lost"); ByteTrack keeps even low-confidence boxes because partly hidden people produce those.
- **Why:** works well with overlapping people and needs no face/appearance matching (important: we explicitly do not do identity recognition).
- **We expect to write our own small version** rather than use a ready-made library, because our video can have *irregular* gaps between frames after motion gating, which off-the-shelf trackers usually don't expect.

### GRU (Phase 1 / 7)
- **What:** a small neural network that reads a sequence (what happened frame after frame) and keeps a "memory" of the past.
- **Why:** simple, light, proven for short sequences. Alternatives (Transformers, Mamba, TCN) are listed in the plan as later options.

### Git (installed, version 2.47.0)
- **What:** version control — records every change so you can go back.
- **Status:** installed on your machine but the project folder is not yet a Git repository; we only added a `.gitignore`. See open items.
