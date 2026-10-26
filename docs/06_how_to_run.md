# 6. How to Run Everything

This document provides ready-to-copy commands for Windows PowerShell. Run all commands from the project root directory (`c:\Users\yscha\OneDrive\Desktop\SVNIT\Projects\HCI\Project`).

---

## 1. Environment Setup

### Using your existing Python environment
Your system Python already has all necessary Phase 0 packages installed:
```powershell
python -m pip install -r requirements.txt
```

### (Optional) Testing in an isolated clean virtual environment
To verify that everything works in a fresh, isolated environment without PyTorch:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 2. Running Automated Tests

We use a test runner script (`scripts/run_tests.py`) that manages test tiers:

### Run unit tests only (fast, isolated logic tests)
```powershell
python scripts/run_tests.py unit
```
*Expected: 292 passed.*

### Run integration tests only (module boundaries & CLI checks)
```powershell
python scripts/run_tests.py integration
```
*Expected: 18 passed.*

### Run fast tests (both unit and integration)
Run this after making any code change:
```powershell
python scripts/run_tests.py fast
```
*Expected: 310 passed.*

### Verbose mode or running specific tests
To see the names of every individual test being executed:
```powershell
python scripts/run_tests.py unit -v
```

To run only tests matching a keyword (e.g. `occlusion` or `contracts`):
```powershell
python scripts/run_tests.py unit -k occlusion
```

---

## 3. Running the Smoke Test

The `smoke` command executes the Phase 0 pipeline end-to-end: loads config, ingests a video, validates contracts, runs the pipeline, and writes output diagnostics.

### Smoke test with an in-memory synthetic video
No video files needed on disk:
```powershell
python -m mosaic smoke --synthetic burst
```
You can choose from four scenarios: `static`, `moving`, `burst`, or `occlusion`.

### Smoke test with overrides
You can override any configuration setting from the command line without editing YAML:
```powershell
python -m mosaic smoke --synthetic occlusion --set experiment.seed=42 --set logging.level=DEBUG
```

### Smoke test on a real video file
Once you have an `.avi` or `.mp4` video:
```powershell
python -m mosaic smoke --video path/to/your/video.avi
```

---

## 4. Generating Synthetic Test Video Fixtures

To generate the 4 standard synthetic benchmark videos (`.avi`) and their ground-truth bounding box JSON files:

```powershell
python -m mosaic make-fixtures --out artifacts/fixtures
```

This creates:
- `artifacts/fixtures/static.avi` & `static_ground_truth.json`
- `artifacts/fixtures/moving.avi` & `moving_ground_truth.json`
- `artifacts/fixtures/burst.avi` & `burst_ground_truth.json`
- `artifacts/fixtures/occlusion.avi` & `occlusion_ground_truth.json`

Now you can test video decoding on an actual encoded file:
```powershell
python -m mosaic smoke --video artifacts/fixtures/burst.avi
```

---

## 5. Phase 1: Dataset & Baseline Model Commands

### Generate a self-contained Mock RWF-2000 dataset
Creates realistic Fight and NonFight video clips in standard RWF folder layout without downloading anything:
```powershell
python -m mosaic make-mock-rwf --out data/rwf2000_mock --clips-per-class 4 --num-frames 16
```

### Scan dataset, carve stratified validation split, and verify anti-leakage
Scans the dataset folder, computes SHA-256 hashes, verifies that splits are pairwise disjoint, and saves the verified manifest:
```powershell
python -m mosaic scan-dataset --data-dir data/rwf2000_mock --out artifacts/manifests/mock_manifest.json
```

### Train and evaluate the baseline temporal classifier
Trains the CNN backbone + GRU sequence model, computes Accuracy, F1, and ROC-AUC, compares against trivial baselines (Majority Class and Motion Energy), and saves checkpoints:
```powershell
python -m mosaic train-baseline --manifest artifacts/manifests/mock_manifest.json --set train.epochs=3 --set model.backbone=cnn_light
```

---

## 6. Inspecting Run Artifacts

Every time `python -m mosaic smoke` or `python -m mosaic train-baseline` runs, it creates a unique timestamped folder inside `artifacts/runs/`:

```
artifacts/runs/phase0-7cb85650db4e-s1337_baseline_20261024T.../
├── config.yaml              # Exact snapshot of all settings used for this run
├── environment.json        # Machine OS, CPU, RAM, GPU, and package versions
├── best_checkpoint.pt      # Saved PyTorch model weights for best validation epoch
├── final_checkpoint.pt     # Saved PyTorch model weights after final epoch
├── metrics_summary.json    # Accuracy, Precision, Recall, F1, ROC-AUC vs trivial baselines
├── training_history.json   # Per-epoch loss and validation metrics
├── val_predictions.json    # Per-clip ground truth, predicted scores, and labels
└── run.log                 # Human-readable log entries with timestamps
```

To view the metrics summary of the latest baseline run in PowerShell:
```powershell
Get-Content (Get-ChildItem artifacts/runs/*/metrics_summary.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1)
```
