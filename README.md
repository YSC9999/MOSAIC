# MOSAIC v3 - violence detection with person tracks (RWF-2000)

Clip-level violence classification plus *when* (time intervals) and *where* (a group box around the area of action).
The design and the reasoning behind it are in `MOSAIC_v3_Spec.md`. Earlier planning documents are kept at the top level.

## Pipeline

```
clip (150 frames @ 30 fps) -> 64 sampled frames -> 7 windows of 16 frames
  |- scene branch   frozen Video Swin-T (Kinetics-400) features per window (original + flipped), cached
  |- motion score   mean absolute frame difference per window
  |- MOTION GATE    64x64 thumbnails; only the kept frames (~46% with the default gate) go on  <- always on
  |     -> RF-DETR-Seg Nano (persons + masks) -> NMS 0.7 -> ByteTrack (activate >= 0.5) -> track IDs
  |     -> 29 person features per person per frame (+ 9 pair features), real time steps between detections
  '- person branch  Linear -> per-person GRU (state carried over missing frames) -> attention pooling -> 128-d
fusion per window: [scene 256 | person 128 | motion | has-people | count/16] -> MLP -> window logit
clip logit = mean of the top-3 window logits; outputs: class, time intervals, group boxes
```

The interaction graph (GAT) is off by default: it never improved accuracy (`--gat` enables it). Masks are kept for
the silhouette features and visualisation; box-only (`--box-only`) matched the full model in every test.

## Layout

| Path | What |
|---|---|
| `mosaic/` | `windows, splits, motion, motion_gate (gate settings), scene_cache, det_cache, track_join, skd, outputs, train_v3`, `models/{person_branch,fusion_v3}`, plus `config, contracts, ingestion, seeding, synthetic` (test fixtures) |
| `scripts/run_pipeline.py` | runs every step in order, resumable (`--dry-run` shows the commands) |
| `scripts/build_*.py` | the individual steps: `manifest, v3_split, scene_cache, motion, det, tracks, skd` (all dataset-independent) |
| `scripts/train_v3.py`, `export_outputs.py` | training (`--save-ckpt` writes the model) and spatio-temporal outputs + overlay videos |
| `scripts/eval_checkpoint.py` | evaluate a saved model on any dataset (cross-dataset tests), no training |
| `scripts/extract_rwf_zip.py`, `scene_probe.py`, `gate_sweep.py` | RWF zip extraction, linear-probe check, gate analysis |
| `configs/datasets/` | one small YAML per dataset (`TEMPLATE.yaml` to start) |
| `tests/` | `python -m pytest tests -q` |
| `results/` | curated, small results per dataset (tables, metrics, figures, demos); regenerate with `scripts/make_results.py` |
| `artifacts/` | generated and heavy files; not in git |

## Run

```
python scripts/run_pipeline.py configs/datasets/rwf2000_30fps.yaml --cache C:/mosaic_cache_30fps
python scripts/run_pipeline.py configs/datasets/rwf2000_30fps.yaml --gate g02   # stronger gate (~28% of frames)
python scripts/export_outputs.py CKPT --out DIR     # intervals + group boxes + overlay videos for test clips
```

Detection needs the isolated `.venv` (rfdetr, trackers, pycocotools, supervision); everything else runs with any
Python that has torch, torchvision, opencv, scikit-learn.

## Other datasets

The pipeline is dataset-independent: a dataset is one YAML file in `configs/datasets/` (copy `TEMPLATE.yaml`): where the clips are,
which folders are the positive and negative class, whether there is an official split, and how clips map to source videos.

```
python scripts/run_pipeline.py configs/datasets/my_dataset.yaml --skip train        # manifest, split, caches
python scripts/eval_checkpoint.py CKPT configs/datasets/my_dataset.yaml             # model trained elsewhere, no training here
python scripts/make_results.py --dataset my_dataset                                 # results/my_dataset/
```

Clips of any length and frame rate work: each clip is sampled to 64 positions (short clips are padded by repeating the last frame; padded
frames are never detected) and always gives 7 windows. Very long untrimmed videos need cutting into clips first.

## Results (RWF-2000 official test set, 400 clips, 3 seeds)

Main model: no GAT, masks on. Full tables, ablations, per-run metrics and demos are in [`results/rwf2000/`](results/rwf2000/README.md).

| Setting | Frames detected | Test accuracy | AUC |
|---|---|---|---|
| Nano + default gate (current pipeline) | 46% | 0.888 ± 0.004 | 0.958 |
| Nano + strong gate | 28% | 0.879 ± 0.008 | 0.950 |
| Large, no gate | 100% | 0.885 ± 0.009 | 0.959 |

Published results are about 94-95% (CUE-Net 94.00%; a Nov 2025 preprint reports 95.25%), so this is not yet state of the art.
Next levers: spatial cropping, a stronger backbone, a pose branch.
