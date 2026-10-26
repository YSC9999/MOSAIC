# MOSAIC-FL — Phase-by-Phase Learning Guides

Welcome! This folder contains **only Markdown (`.md`) files — one for each phase of the project**, written specifically to teach you how everything works from the ground up, assuming **no prior background in Machine Learning or Computer Vision**.

Every guide explains:
1. **What** we are building in that phase (in plain English, with analogies).
2. **Why** each piece is needed before moving to the next step.
3. **Every tool and concept** introduced, why we picked it over alternatives, and how the math/logic works intuitively.
4. **File-by-file walkthrough** of the exact code written in `mosaic/` and `tests/`.
5. **What could go wrong (and the mishaps we fixed)** so you learn real-world engineering debugging.
6. **Commands to run** so you can see it working on your machine.

---

## Reading Order & Progress Tracker

| Phase | Guide File | Status | What You Will Learn |
| :--- | :--- | :--- | :--- |
| **Phase 0** | [Phase_0_Foundation.md](Phase_0_Foundation.md) | ✅ Complete | How to build a bulletproof software foundation before adding AI: data contracts, config hashing, random seeds, video decoding, and synthetic ground-truth videos. |
| **Phase 1** | [Phase_1_Dataset_and_Baseline.md](Phase_1_Dataset_and_Baseline.md) | ✅ Complete | Your first Deep Learning model: how neural networks learn, preventing data leakage (train/val/test splits), frame sampling, CNN + GRU video classification, and trivial sanity baselines. |
| **Phase 2** | [Phase_2_Clarity_and_Degradations.md](Phase_2_Clarity_and_Degradations.md) | ✅ Complete | How to measure video quality (blur, darkness, resolution, compression) and intentionally degrade videos to test robustness. |
| **Phase 3** | [Phase_3_Motion_Gating.md](Phase_3_Motion_Gating.md) | ✅ Complete | How to skip redundant static video frames to save 75%+ compute without missing sudden violent bursts. |
| **Phase 4** | [Phase_4_Person_Detection.md](Phase_4_Person_Detection.md) | ✅ Complete | How fast edge object detectors find people in each frame, extract bounding boxes, and compute person scale ratios. |
| **Phase 5** | [Phase_5_Masks_and_Kinematics.md](Phase_5_Masks_and_Kinematics.md) | ✅ Complete | Why skeleton joints fail in crowds, and how silhouette outlines + box kinematics (speed, acceleration) work instead. |
| **Phase 6** | [Phase_6_Tracking_and_Tokens.md](Phase_6_Tracking_and_Tokens.md) | ✅ Complete | How Kalman filters and Hungarian matching track individuals across frames (even behind obstacles) and build `TrackToken` records. |
| **Phase 7** | [Phase_7_Temporal_Context_Encoding.md](Phase_7_Temporal_Context_Encoding.md) | ✅ Complete | Combining scene features and person track histories to make the final `violent` / `non_violent` decision. |
| **Phase 8** | [Phase_8_Ablations_and_Robustness.md](Phase_8_Ablations_and_Robustness.md) | ✅ Complete | Scientific validation: proving which parts of the system matter most across multiple random seeds, hardware latency profiling, and camera degradations. |
