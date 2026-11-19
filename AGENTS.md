# AGENTS.md - how to work in this repository

MOSAIC v3 detects violence in short video clips (RWF-2000 and other datasets): clip class, *when* (time intervals) and *where* (a group
box around the area of action). Read `README.md` for the pipeline and `MOSAIC_v3_Spec.md` for the design and its reasoning.
Curated results are in `results/`. Everything below was verified on Windows 11, Python 3.13.5, an RTX 3050 laptop GPU (6 GB).

## 1. First-time setup

Two environments are used: the main Python (everything except detection) and a `.venv` (detection and tracking, which need `rfdetr`).

```powershell
# 1. core + PyTorch in the main Python
python -m pip install -r requirements.txt
python -m pip install -r requirements-ml.txt --extra-index-url https://download.pytorch.org/whl/cu128   # pick the CUDA build for the driver

# 2. detection .venv (reuses the main environment's torch/numpy/opencv)
python -m venv --system-site-packages .venv
.venv\Scripts\python.exe -m pip install -r requirements-detect.txt

# 3. check the setup
python -m pytest tests -q                                  # all tests pass (279 at the last count, about 20 s)
python -c "import torch; print(torch.cuda.is_available())"  # True on a GPU machine
.venv\Scripts\python.exe -c "import rfdetr, trackers, pycocotools"
```

Model weights download themselves on first use (Swin3D-T from PyTorch, RF-DETR from Roboflow) and need internet once.

## 2. Data

Videos are never committed (`data/` is git-ignored) and should live outside OneDrive-synced folders if possible.
Each dataset is described by one YAML file in `configs/datasets/` (copy `TEMPLATE.yaml`). The project's main dataset:

- **RWF-2000, 30 fps copy**: Kaggle `vulamnguyen/rwf2000` (download manually while logged in; check the licence). Extract it with
  `python scripts/extract_rwf_zip.py <zip> data/RWF-2000_30fps --delete-zip-when-verified`. Result: `data/RWF-2000_30fps/RWF-2000/{train,val}/{Fight,NonFight}/*.avi`
  (2,000 clips, 150 frames at 30 fps; the script gives garbled or very long file names short hashed names and writes `name_map.json`).
- Config: `configs/datasets/rwf2000_30fps.yaml` (official `val` folder = test set; clips of one source video are grouped).

A new dataset needs only a YAML (folder layout, positive/negative class folder names, official split or `test_fraction`, grouping rule).

## 3. Running the pipeline on a dataset

```powershell
python scripts/run_pipeline.py configs/datasets/<name>.yaml --dry-run          # print the commands first
python scripts/run_pipeline.py configs/datasets/<name>.yaml --cache D:/mosaic_cache/<name>
python scripts/run_pipeline.py configs/datasets/<name>.yaml --skip train       # build the caches only
python scripts/run_pipeline.py configs/datasets/<name>.yaml --from-step detect # resume (every step is resumable)
```

Steps: `manifest -> split -> swin -> motion -> detect -> tracks -> skd -> train`. `detect`, `tracks` and `skd` run with `.venv` (`--venv-python` to change it); the other steps use the interpreter that
launched `run_pipeline.py` (it needs torch and OpenCV). Caches go to `--cache` (default `C:/mosaic_cache/<name>`), never into the repo. Motion gating is mandatory (`--gate gdef`,
default, keeps ~46% of frames; `g02` keeps ~28%). Options: `--variants main r1 a4 a2`, `--seeds 42 1337 2024`, `--save-ckpt`.
Rough times on the 3050 laptop for RWF-2000: Swin features ~35 min, gated detection ~1-1.5 h, tracks 2 min, person features ~10 min,
each training run ~2 min.

Then, with a checkpoint (`--save-ckpt`):

```powershell
python scripts/eval_checkpoint.py <ckpt.pt> configs/datasets/<name>.yaml --cache <cache>   # metrics + per-clip probabilities, no training
python scripts/export_outputs.py <ckpt.pt> --out <dir> --n 50 --render 4 --only-violent  # intervals, group boxes, overlay videos (uses MOSAIC_* env vars, see the script)
python scripts/make_results.py --dataset rwf2000                                         # regenerate results/rwf2000/ (or any evaluated dataset name)
```

## 4. Where things go

| Path | Tracked in git? | What |
|---|---|---|
| `mosaic/`, `scripts/`, `tests/`, `configs/` | yes | code, tests, dataset configs |
| `results/<dataset>/` | yes | small curated results (tables, per-run metrics, figures, a few demos); rebuilt by `make_results.py` |
| `artifacts/` | **no** (ignored) | manifests, run metrics, checkpoints, logs, eval outputs |
| `data/`, caches (`C:/mosaic_cache*`), `.venv/` | no | videos, features, environment |

Training writes one JSON per run (`artifacts/runs/<dataset>/<gate>/<name>_s<seed>.json`): accuracy, precision, recall, F1, ROC-AUC, ECE on the
internal-validation and test splits (with and without flip averaging), per-epoch history and the config.

## 5. Decisions already made (do not silently undo)

- Main model: scene features (frozen Video Swin-T) + person branch (per-person GRU + attention pooling), **no GAT** (it never helped; `--gat` enables it), masks on.
- Motion gating is not optional; detector is RF-DETR-Seg Nano; tracker is ByteTrack with the cleaned settings (activate >= 0.5, NMS 0.7).
- Report mean +- std over 3 seeds; select on the internal validation split, never on the test set. Everything differing by < ~1 point is noise.
- Do not claim state of the art: published results are ~94-95%, this work is ~0.89 on RWF-2000.
- Outputs describe the model's attention (an *area of action*); they must not be presented as who is responsible.

## 6. Working agreements

- **Ask before training or any long GPU/CPU job** unless the user has just asked for it. Never rerun training "to be safe".
- **Do not commit or push unless asked.** Never force-push. Push to the branch the user names (`yash`).
- Never delete user files (their notes, PDFs, zips, notebook). Check what a command targets first; deleting tracked old code is fine because git keeps it.
- Before appending to a file with `>>`, check it exists (a missed one once overwrote the spec).
- Changes to generic code must still reproduce RWF-2000 exactly (manifest, split, features, evaluation); add a test when behavior changes.
- Run `python -m pytest tests -q` after code changes. New scripts get a `--help` and a clear error for a missing config or cache.

## 7. Gotchas on this machine

- Shells: Git Bash and PowerShell. Use `.venv\Scripts\python.exe` for detection. Set `PYTHONIOENCODING=utf-8` when printing text with symbols (cp1252 console).
- **OneDrive eats RAM** (1-2 GB) while syncing and restarts by itself; stop it for long jobs (`Stop-Process -Name OneDrive -Force`) and keep caches out of the synced folder.
- Background jobs get killed when free RAM is very low; check free memory first and do not restart a killed job without asking.
- Clip file names can be very long or garbled (Windows 260-character limit): rely on `name_map.json` and the hashed names.
- The RWF data folder is ~12 GB; check disk space before extracting or building caches (~2 GB per dataset).
- Detection is slow on a laptop GPU (CPU mask handling is a third of it); gating and Nano already cut it by about half.

## 8. Known gaps

- The pipeline has been run end to end only on RWF-2000. Another dataset's detection stage (and padded short clips) is untested.
- No checkpoint of the main model is saved yet; cross-dataset evaluation needs one (`--save-ckpt`).
- Time-interval and box outputs are not scored (RWF-2000 has no such labels; CCTV-Fights has frame-level fight intervals).
- Long untrimmed videos must be cut into clips before use.
