# 3. Phase 0 — Step by Step

For every step: **What** we did → **Why** it matters → **Where** it lives → **How we know it works**.

The order below is the order we built things, and each step relies on the earlier ones.

---

## Step 1 — Inspect your machine

- **What:** checked Python version, installed libraries, GPU, RAM, Git.
- **Why:** the plan says the detector choice depends on hardware; and we must not install things blindly.
- **Found:** Python 3.10.11 · NVIDIA RTX 4070 Laptop GPU (8 GB) · 32 GB RAM · i9-13900HX CPU · NumPy, OpenCV, PyYAML, pytest, PyTorch (GPU build) already installed · Git installed.
- **Also found:** a leftover package called `mosaic-fl-baseline` from an earlier attempt, pointing at a folder (`Project/src`) that no longer exists. It is harmless and does not clash with our package, which is named `mosaic` (see [08_notes_and_open_items.md](08_notes_and_open_items.md)).

## Step 2 — Decide the dependency policy and create the project skeleton

- **What:** created the folders (`mosaic/`, `tests/`, `configs/`, `docs/`, `scripts/`), `requirements.txt` with exact versions, `.gitignore`, and `pytest.ini`.
- **Why:** a fixed structure and exact versions make the project reproducible.
- **Where:** [`requirements.txt`](../requirements.txt), [`requirements-ml.txt`](../requirements-ml.txt), [`pytest.ini`](../pytest.ini).
- **Decision:** machine-learning libraries (PyTorch) are kept in a **separate** file because Phase 0 doesn't need them; this keeps the Phase 0 environment small and lets us *prove* Phase 0 doesn't depend on them.
- **How verified:** a brand-new virtual environment built from `requirements.txt` alone passes every test.

## Step 3 — Typed data contracts (`mosaic/contracts.py`)

- **What:** wrote precise definitions of the data that flows between modules: `Box`, `Detection`, `ClarityReport`, `ObjectEvidence`, `MaskEvidence`, `TrackToken`, `TrackSummary`, `DecodeDiagnostics`, `WindowDiagnostics`, `WindowOutput`.
- **Why:** if the tracker produces a box with `x2 < x1`, or a confidence of 1.7, or `NaN`, we want to **crash immediately at the border between modules**, not discover garbage in a final result.
- **Key rules built in:**
  - boxes: finite, non-negative, positive width and height;
  - every score/confidence: a finite number in `[0, 1]`;
  - unavailable information is `None` — **never** a fake `0.0`;
  - a *predicted* (Kalman) token must have `observed=False` and **no** detector confidence (an unseen person has no detector confidence);
  - the final `label` can only be `violent` or `non_violent` — there is **no** goon/role/instigator field anywhere.
- **Extra decision:** `TrackToken.detector_confidence` is optional (`None` for predicted tokens). The plan lists it as a plain field; making it optional is what lets us forbid "success-shaped" fake values. This is documented in the open-items file.

## Step 4 — Configuration (`mosaic/config.py`, `configs/default.yaml`)

- **What:** one typed settings object with sections: `experiment`, `paths`, `decode`, `window`, `logging`, `pipeline`.
- **Why:** all tunable values in one place; typos and bad values are rejected with a message naming the exact setting.
- **Details:**
  - **Defaults** are in the code *and* mirrored in the YAML (a test fails if they ever disagree).
  - **Unknown keys are errors** — `windw:` instead of `window:` can't silently do nothing.
  - **Type checks** — `seed: "abc"` or `seed: true` is refused.
  - **Value checks** — negative window length, a stride larger than the window, unknown stage names, etc.
  - **Overrides** from the command line: `--set experiment.seed=7`.
  - **Config hash** — a 12-character fingerprint of every setting that can affect results (paths, logging and the run's label are excluded because they can't change results). Later, cached results are matched by this hash.
  - **Experiment ID** — `<name>-<hash>-s<seed>`, e.g. `phase0-6ad49796a489-s1337`. Same settings → same ID.
- **The window policy** is fixed here (plan section 2.2): by default *one window = the whole clip* and clip scores use the *mean*. These are recorded so later results can't be misread.

## Step 5 — Seeding (`mosaic/seeding.py`)

- **What:** `seed_everything` seeds Python's and NumPy's random generators; `derive_seed`/`make_rng` give each component its own independent, reproducible random stream from one master seed.
- **Why:** computers produce "random" numbers from a starting number (the *seed*). Same seed → same numbers → repeatable experiments. Separate streams stop one component (e.g. data sampling) from disturbing another (e.g. noise generation).
- **Detail:** PyTorch seeding is optional and imported only on request, so Phase 0 never loads PyTorch.

## Step 6 — Logging, artifacts layout, environment record

- **Logging (`logging_utils.py`):** one format (`time | LEVEL | name | message`), console + optional file, and `event=name key=value` lines that are easy to search. Logs are for **humans**; structured **diagnostics** (JSON) are for **programs**.
- **Artifacts (`artifacts.py`):** fixed output folders — `manifests/ checkpoints/ metrics/ reports/ runs/ token_cache/ fixtures/ logs/`. Each run gets its own folder named `<experiment id>_<UTC timestamp>`; names containing `..` or slashes are refused so nothing can be written outside `artifacts/`.
- **Environment record (`environment.py`):** every run writes `environment.json` — OS, Python version, CPU, RAM, GPU, versions of key libraries. Results are only meaningful together with the machine that produced them. If something is unavailable (e.g. no GPU tool) the record says *why*.

## Step 7 — Video ingestion (`mosaic/ingestion.py`)

- **What:** reads a video file into frames. Each frame gets an **index** (0, 1, 2…) and a **timestamp** = `index ÷ FPS`.
- **Why we compute timestamps ourselves:** different video codecs report timestamps slightly differently. Computing `index ÷ FPS` ourselves makes them identical on every machine and every replay.
- **Plan rules enforced:**
  - *Deterministic replay* — same file → same frames/indices/times.
  - *Explicit failure* — missing, unreadable, zero-byte or invalid-frame-rate files raise a `DecodeError` and are reported.
  - *No silent substitution* — if a frame can't be decoded we **stop and report**; we never repeat the previous frame or insert a blank one.
  - *Honest accounting* — the report states how many frames were decoded vs. what the file claimed (some files lie by a frame or two, so a small tolerance is configurable).
- **Two kinds of source, one report format:** real files (`VideoReader`) and in-memory arrays (`ArrayFrameSource`, used for synthetic videos) produce the *same* `DecodeDiagnostics` structure, so later code can't tell them apart.

## Step 8 — Synthetic videos (`mosaic/synthetic.py`)

- **What:** a generator that draws a coloured rectangle (standing in for a person) on a fixed textured background and moves it along a scripted path. It reports the **exact** box in every frame.
- **Why:** RWF-2000 has no answer key for "where is the person in frame 37?", so we couldn't test motion gating, tracking or speed calculations against known truth. Here we know the truth because we drew it.
- **Four scenarios:**

| Scenario | What happens | Will be used to test |
|---|---|---|
| `static` | Nothing moves; all frames identical | Motion gate skips static video (Phase 3) |
| `moving` | Steady straight-line motion | Tracking, velocity, acceleration (Phase 6) |
| `burst` | Still → sudden fast move → still | Gate must keep the burst, skip the rest (Phase 3) |
| `occlusion` | Rectangle passes *behind a wall* and reappears | Tracker must survive hiding, "predicted" tokens (Phase 6) |

- **Deterministic:** same seed → byte-identical video; different seed → different background.

## Step 9 — Manifest (`mosaic/manifest.py`)

- **What:** a checked JSON list of clips: path, label, frame count, FPS, resolution, duration, decode status (+ optional split and file fingerprint) with metadata including the window policy.
- **Why:** corrupt or empty clips are discovered **up front** and listed (`decode_status: failed`, with a reason) instead of crashing a training run hours later. Parsing is strict and reports **every** bad entry with its position at once.

## Step 10 — Pipeline skeleton and command line (`pipeline.py`, `cli.py`)

- **What:** `Pipeline.from_config(cfg)` builds the pipeline from settings alone (no model files). In Phase 0 it has one stage — ingestion. `python -m mosaic smoke …` loads settings, reads a video and writes a run folder containing `config.yaml`, `environment.json`, `diagnostics.json` and `run.log`.
- **Why:** the "smoke command" is the Phase 0 exit-gate test: *if this works from a clean install, the foundation is sound.* Decode problems never crash the pipeline; they come back as diagnostics, and the command exits with code `2` (bad config → `1`).

## Step 11 — Tests (`tests/`)

- **What:** 276 unit tests + 15 integration tests, in separate folders that automatically get the matching marker. A single command exists for each tier: `python scripts/run_tests.py <tier>`.
- **Network off:** during unit and integration tests, network connections are **blocked** by the test setup. If any test tried to download something, it would fail. This proves the "no network needed" requirement instead of just claiming it.
- See [05_testing_guide.md](05_testing_guide.md).

## Step 12 — Exit-gate verification

The plan's Phase 0 exit gate:

| Gate requirement | Result |
|---|---|
| A clean environment can run a smoke command that loads config, reads a fixture video and writes diagnostics | ✅ verified in a brand-new virtual environment built only from `requirements.txt` |
| The unit tier runs without network access, model weights or the dataset | ✅ network blocked by the test setup; no weights or dataset exist yet; PyTorch isn't even installed in the clean environment |

See [08_notes_and_open_items.md](08_notes_and_open_items.md) for the few Phase 0 plan items that are only partly done (they need the real dataset).
