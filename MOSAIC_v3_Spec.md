# MOSAIC v3 — Accuracy-First Violence Detection Spec

Status: draft, 2026-11-17. Supersedes the Phase 7/8 work in `scratch/` and
`artifacts/reports/temporal_fusion_report.json` / `phase8_*`, which used synthetic
track tensors, hardcoded baselines and formula-generated robustness curves and
must not be cited.

Scope change: the HCI/edge parts (clarity block, motion gating, edge latency,
federated learning) are deferred. This spec targets accuracy and explainable
outputs on RWF-2000.

## 0. Constraints this spec is written against

| Fact (measured) | Consequence |
|---|---|
| Local RWF-2000 copy (HF `A1mal/RWF-2000-Dataset`) is **50 frames @ 10 fps** for all 1,989 clips | No resampling to 64; windows built on native frames (§1). Results are not strictly comparable to papers using the 30 fps original |
| GPU: RTX 3050 Laptop, **6 GB** | Swin is frozen and its features are cached (Stage A). Fine-tuning is an optional Stage B (§10) |
| Python 3.13, Windows | Video I/O via OpenCV (already in `mosaic/ingestion.py`); Decord dropped |
| 1,273 train-fit clips come from **610 source videos** (up to 22 clips each) | Internal validation split is grouped by source video (§10) |
| 11 exact duplicates already removed by the manifest | Keep using the deduplicated manifest (1,589 train / 400 val) |

## Overview

```
RWF-2000 clip (50 frames @ 10 fps)
  │
  ▼
[1] Loader ── native frames → W windows of 16 frames (W = 6 for 50 frames)
  │
  ├──► [2] Motion score m_w (per window)
  ├──► [3] RF-DETR-Seg ── person boxes + full-size masks (cached)
  │          ▼
  │     [4] ByteTrack (boxes only) ── track IDs (joined by detection index)
  │          ▼
  │     [5] Silhouette-kinematic descriptors (29 per person per frame)
  │          ▼
  │     [6] Person branch: per-frame GATv2 → per-person GRU → attention pooling
  │
  └──► [7] Scene branch: Video Swin-T (K400), frozen, cached features
             ▼
        [8] Fusion → window logit → clip logit (top-k mean)
             ▼
        [9] Outputs: (1) clip class  (2) [start, end]  (3) person tracks
```

## 1. Loader

- OpenCV decode. Record N, fps, (H0, W0). Timestamps t_i = i / fps.
- Frame set: if N ≤ 64 use all N native frames (no duplicates). If N > 64
  (e.g. the 30 fps original is used later), take 64 distinct frames at
  round(i·(N−1)/63). Let T_s be the number of sampled frames.
- If T_s < 16: pad by repeating the last frame, mark padded frames
  `is_pad = 1`, exclude them from all rate computations. Log the clip.
- Windows: W = ceil((T_s − 16) / 8) + 1, starts s_k = round(k·(T_s − 16)/(W − 1)).
  For T_s = 50: W = 6, starts 0, 7, 14, 20, 27, 34 (covers every frame).
  For T_s = 64: W = 7, starts 0, 8, …, 48.
- Window time span: [t_{s_k}, t_{s_k+15} + 1/fps] from the real timestamps.
- Frame copies: original RGB (detector), 112×112 grayscale (motion), 224×224 RGB
  (direct bilinear resize, ImageNet mean/std, layout B×C×T×H×W) for Swin.
  The aspect distortion (4:3 and 16:9 sources) is accepted and documented.

## 2. Motion score

d_t = mean(|g_{t+1} − g_t|) on 112×112 grayscale in [0, 1];
m_w(k) = mean(d_{s_k} … d_{s_k+14}). A plain feature, nothing is skipped.
Standardized with train-fit statistics like every other feature.

## 3. Detection and masks

- `rfdetr` segmentation model, frozen, threshold 0.10, persons only (class looked
  up by name). Start with the Large variant; if throughput on 6 GB is too low,
  drop to Medium and record the choice. Verify the exact class name of the
  installed `rfdetr` version before writing code.
- Clamp boxes to the frame; drop boxes < 4 px wide or < 8 px tall.
- mask_valid = ≥ 16 mask pixels and IoU(mask bbox, det box) ≥ 0.5.
- Store per detection: `det_idx` (index within frame), box, score, mask as COCO
  RLE (pycocotools), area, centroid, mask bbox; per frame: pairwise mask overlaps.
- Cache: `cache/det/{split}/{clip}.pkl`, keyed with model name, version, threshold.

## 4. Tracking

- ByteTrack from `trackers`, boxes + scores only. High 0.25, low 0.10,
  lost-track patience 16 frames, one fresh tracker per clip.
- **Join by detection index, not coordinates.** First check whether the tracker
  returns the input boxes unchanged. If yes, join on `det_idx`. If it returns
  filtered boxes, match to cache rows by IoU ≥ 0.99. Log the per-clip match rate
  and fail loudly if < 95% of tracker outputs match a cache row.
- Unmatched rows and tracks seen in < 3 frames get ID −1.
- A track belongs to a window if observed in ≥ 4 of its 16 frames. Cap 16 people
  per window, ranked by observed frames, then mean score, then lowest ID. The
  uncapped count is kept as a feature.

## 5. Silhouette-kinematic descriptor (SKD)

Unchanged from the original 29 node / 9 edge feature design (geometry 6,
shape 7, kinematics 9, interaction 3, quality/flags 4). Cache
`cache/skd/{clip}.npz` (float16). Additional rules:

- All rates use real timestamp differences Δt; assert Δt > 0 (padded frames
  excluded). At 10 fps, Δt between consecutive observations is ≥ 0.1 s.
- No previous observation → kinematics 0, has_prev 0. Acceleration needs 2
  previous observations, jerk 3.
- Invalid mask → shape features 0, mask_valid 0.
- Velocities clipped to ±10 heights/s, other rates ±20.
- Standardize with train-fit mean/std (§10), clip to ±5.
- Horizontal flip: cx → 1 − cx; negate vx, edge dx, sin 2θ. cos 2θ unchanged.

## 6. Person branch

As designed: Linear 29→64 + LayerNorm + GELU; 2 dense masked GATv2 layers
(4 heads × 16), edge features 9→64 inside attention, self-loops, residual +
LayerNorm; per-person GRU (128) carrying state over missing frames; attention
pooling a_i = softmax(wᵀ tanh(W h_i)) over valid people → p (128). Weights a_i
kept for Output 3. No qualifying people → learned 128-d empty vector,
has_people = 0.

## 7. Scene branch

- `torchvision.models.video.swin3d_t`, `KINETICS400_V1`, head → identity (768-d).
- **Stage A (default): frozen.** Precompute 768-d features for every window,
  original and horizontally flipped, into `cache/swin/{split}/{clip}.npy`.
  Linear 768→256 is trained.
- Random resized crop / colour jitter are not available in Stage A (features are
  cached). That is accepted for Stage A.
- Stage B (optional, §10) fine-tunes stage 4 only.

## 8. Fusion and window score

[scene 256 ‖ person 128 ‖ m_w ‖ has_people ‖ count/16] = 387 → Linear 128 → GELU
→ Dropout 0.3 → Linear 1 = window logit z_k.

Modality dropout (training only): per sample, zero the scene vector with p 0.2,
otherwise zero the person vector with p 0.2. Never both.

## 9. Outputs

**Output 1 — clip class.** Clip logit = mean of the top-k window logits,
k = ceil(0.4·W) (k = 3 for W = 6 or 7). Violent if sigmoid ≥ 0.5.

**Output 2 — interval** (clips classified violent only): p_k = sigmoid(z_k),
size-3 median filter (edges replicated); windows with p ≥ 0.5 are violent; merge
consecutive ones into [start of first, end of last]; if none, use the argmax
window. Report all intervals; primary = the one containing the argmax window.

**Output 3 — people.** In violent windows draw all tracked boxes, coloured by
track ID; highlight the 2 tracks with the highest mean attention weight, labelled
"most attended (model attention, not verified involvement)". Render on the
sampled frames at native fps.

## 10. Training

**Splits.** Start from the deduplicated official train (1,589). Source video ID =
filename with the trailing `_<n>` removed. Hold out ~10% of *source videos*
(stratified by label, seed 42; ≈ 160 clips) as internal validation. No source
video appears on both sides. Feature statistics come from the remaining clips.
The official val (400) is the test set, touched only for final reporting.

**Loss.**
- Clip BCE on the clip logit.
- Window negatives: for non-violent clips, every window is non-violent, so add
  BCE(z_k, 0) on all windows with weight λ = 0.5. Violent clips get no window
  loss (their window labels are unknown). This gives the non-top-k windows a
  training signal and grounds Output 2.

**Stage A optimizer.** AdamW, wd 0.05, lr 3e-4, 2-epoch linear warmup then cosine,
30 epochs, batch 8 clips (no accumulation needed with cached features), fp16.
Early stopping: patience 6 on internal-val clip accuracy, ties by lower loss.

**Augmentation.** Horizontal flip p 0.5, applied jointly to the cached flipped
Swin features and the flipped SKD.

**Stage B (only if Stage A is promising and time allows).** Unfreeze Swin stage 4
+ final norm, lr 3e-5 there and 3e-4 elsewhere, 1 clip per step with gradient
accumulation 8, gradient checkpointing, fp16; frames decoded on the fly with
crop/jitter (same parameters across a window's 16 frames). Abort if it does not
fit in 6 GB.

**Reproducibility.** Seeds for Python, NumPy, PyTorch; `cudnn.deterministic=True`,
`torch.use_deterministic_algorithms(True, warn_only=True)`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. **Three independent training runs** (seeds 42,
1337, 2024), each trained from scratch; report mean ± std.

## 11. Ablations (each trained separately, 3 seeds)

| ID | Variant | Question |
|---|---|---|
| B0 | Majority class | Floor |
| B1 | Logistic regression on m_w (top-k pooled) | Is motion alone enough? |
| A1 | Full model | — |
| A2 | Scene only | How much is the Swin backbone alone? |
| A3 | Person only | Does the person branch carry signal by itself? |
| A4 | Person box-only (drop the 7 shape features + mask overlaps) | Are masks worth it? |
| A5 | No GAT (node embedding straight into the GRU) | Does interaction modelling help? |
| A6 | Clip aggregation mean instead of top-k | Is MIL pooling needed? |

The person branch is only claimed to help if A1 beats A2 by more than the
seed std.

## 12. Evaluation

- **Output 1:** accuracy, precision, recall, F1, ROC-AUC, ECE on the 400 test
  clips, mean ± std over 3 seeds. Context: the RWF-2000 paper (in repo,
  `2003.09163v2.pdf`) reports 87.25% on the 30 fps original.
- **Output 2:** hand-annotate [start, end] on 50 test fight clips. Report mean
  tIoU and share with tIoU ≥ 0.5, alongside two baselines: *whole clip* and
  *argmax window only*.
- **Output 3:** qualitative examples; people per window; share of windows with
  ≥ 2 people; share of clips where a "most attended" track exists.
- **Runtime:** offline per-clip time per stage (detection, tracking, SKD, Swin
  features, model).

## 13. Build order and gates

| Step | Deliverable | Gate before moving on |
|---|---|---|
| 1 | Grouped split + manifest update | Zero source-video overlap between fit and internal val |
| 2 | Detection cache (all clips) | Rendered spot-check of 20 clips; persons/frame stats logged |
| 3 | Tracking + index join | Match rate ≥ 95% on every clip; ID −1 share reported |
| 4 | SKD cache + feature stats | No NaN/inf; Δt > 0 asserted; flip test passes |
| 5 | Swin feature cache (orig + flip) | Shapes W×768; deterministic re-run matches |
| 6 | Model + training loop (Stage A) | Overfits 32 clips to ~100%; unit tests for GAT masking, empty-window path, top-k |
| 7 | Full runs: A1, A2, B0, B1 (3 seeds) | A2 alone should land well above the Phase 1 0.70 |
| 8 | Remaining ablations A3–A6 | — |
| 9 | Output 2 annotation + tIoU; Output 3 renders | — |
| 10 | DVD subset: Output 2 out of domain (§15.1) | Only after A1 vs A2 is settled |
| 11 | RLVS: cross-dataset Output 1 (§15.2) | Overlap check with RWF done first |

## 14. Reuse from the existing repo

- Keep: `config.py`, `manifest.py` (add source-video grouping), `ingestion.py`
  (OpenCV), `seeding.py`, `artifacts.py`, `environment.py`, `metrics.py`,
  `logging_utils.py`, test infrastructure.
- Retire for now: `clarity.py`, `motion_gate.py`, `degradations.py` (kept on
  disk for later HCI/edge work), `detector.py` (Faster R-CNN), `tracker.py`
  (homegrown ByteTrack; kept as a fallback if `trackers` misbehaves),
  `temporal.py` fusion model, `scratch/train_and_evaluate_phase7.py`,
  `scratch/run_phase8_validation.py`.
- New dependencies to pin after a feasibility check on Python 3.13 / Windows:
  `rfdetr`, `trackers`, `pycocotools`.

## 15. External evaluation (after the RWF gates)

No retraining or fine-tuning on either dataset. Same preprocessing as RWF:
resample to 10 fps, same windows, same detector/tracker settings, same train-fit
feature statistics, threshold fixed at 0.5.

### 15.1 DVD subset — Output 2 out of domain

- Confirm the dataset version, licence and annotation format before downloading.
- Videos are long: slide the 16-frame / stride-8 windows over the whole video.
  Top-k clip aggregation is not used here; the interval logic of §9 is unchanged.
- Metrics: frame-level ROC-AUC, mean tIoU, share with tIoU ≥ 0.5,
  window-level classification AUC (to size the domain shift separately from
  localization quality).
- Baselines: whole video, argmax window only, thresholded m_w, and A2
  (scene only) with the same interval logic.

### 15.2 RLVS — cross-dataset Output 1

- Remove clips whose frame signature matches any RWF clip before evaluating.
- Evaluate A1, A2 and A3 (3 seeds each). Report accuracy/F1 at 0.5 and ROC-AUC,
  and the drop from RWF test for each variant.
- Main question: does the person branch transfer better than the scene branch
  (A1 or A3 drop < A2 drop)?
- Caveat to state: RLVS non-violent clips are everyday/sports scenes and may be
  easier than RWF, so a small drop is not by itself evidence of robustness.

## 16. Repo layout

New code is added beside the existing package; nothing is deleted.

```
mosaic/
  windows.py            # §1 frame selection + window spans
  splits.py             # §10 source-video grouped split
  motion.py             # §2 m_w
  det_cache.py          # §3 RF-DETR-Seg wrapper + RLE cache
  track_join.py         # §4 ByteTrack + index join + window membership
  skd.py                # §5 node/edge descriptors + stats + flip
  scene_cache.py        # §7 Swin feature cache
  models/person_branch.py, models/fusion_v3.py
  train_v3.py           # §10 Stage A/B
  outputs.py            # §9 interval logic + rendering
scripts/
  build_det_cache.py, build_tracks.py, build_skd.py, build_scene_cache.py,
  train_v3.py, evaluate_v3.py, eval_external.py
configs/v3.yaml
tests/unit/test_windows.py, test_splits.py, test_skd.py, test_track_join.py,
  test_person_branch.py, test_outputs.py
```

`config.py` rejects unknown keys, so the v3 sections (windows, det, track, skd,
scene, train_v3) are added to its schema; existing sections stay valid. The
invalid Phase 7/8 reports are moved to `artifacts/reports/_invalid/` with a
README explaining why.

## 17. Changes toward SOTA (agreed 2026-11-17)

1. **Official 30 fps RWF-2000.** Request access via the authors' GitHub; build and
   debug on the 10 fps copy meanwhile. Only numbers on the official data are
   compared with published results.
2. **Stronger frozen backbone, extracted once on a cloud GPU.** Candidates:
   Swin3D-T (baseline), VideoMAE V2 (ViT-B/L, K710), InternVideo2 (check licence).
   Cache pooled + per-time-step tokens for original, flip and ~4 crop/jitter
   views. Choose by linear probe on internal_val only. Optional later: partial
   fine-tune / LoRA on the cloud GPU.
3. **Pose branch.** Top-down pose (RTMPose via `rtmlib`, or ViTPose) on tracked
   boxes. ~40 extra node features: hip-centred keypoints / box height, wrist,
   ankle and head speeds, joint angles and their rates, min wrist/ankle-to-other
   head/torso distance and its closing speed, mean keypoint confidence. Edge
   features: min wrist-to-head distance and closing speed. Flip must mirror x
   and swap left/right joints (unit-tested). Low-confidence joints zeroed and
   flagged. New ablations A7 (+pose) and A8 (pose replaces mask shape features);
   if masks add nothing, switch to box-only RF-DETR.
4. **Test-time averaging.** Flip TTA and seed ensembling, always reported
   separately from single-model results.

Day-1 tooling: `mosaic/windows.py`, `mosaic/splits.py`, `mosaic/scene_cache.py`,
`scripts/build_v3_split.py`, `scripts/build_scene_cache.py`, `scripts/scene_probe.py`.
