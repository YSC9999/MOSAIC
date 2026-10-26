# Phase 1 — Dataset Ingestion & Baseline Temporal Classifier

> **Goal of Phase 1:** Teach our program to read the **RWF-2000** violence dataset without "cheating" (data leakage), and train our first Deep Learning video classifier to establish a scientific benchmark.

---

## 1. Machine Learning From Scratch (Plain-English Primer)

If you have never built a Machine Learning model before, here is how it works conceptually.

### 1.1 Traditional Programming vs. Machine Learning
- **Traditional Programming:** You write explicit rules (`if pixel_speed > 50: return "violent"`). This fails on real video because someone jogging fast or a camera panning quickly also has high pixel speed!
- **Supervised Machine Learning:** Instead of writing rules by hand, you build a mathematical network with millions of adjustable knobs (called **weights** or **parameters**) and show it thousands of labeled examples (`Fight` vs. `NonFight`).
  1. **Forward Pass (Guessing):** The model watches a video clip and outputs a number between `0.0` (definitely `NonFight`) and `1.0` (definitely `Fight`), called the **violence score**.
  2. **Loss Function (Grading the Guess):** A mathematical formula (`BCEWithLogitsLoss`, Binary Cross-Entropy) measures how far the model's guess was from the true answer (`1` for Fight, `0` for NonFight). A terrible guess gets a high penalty (loss); a spot-on guess gets a loss near `0`.
  3. **Backward Pass / Backpropagation (Finding Fault):** Using calculus (the chain rule), PyTorch calculates which internal knobs caused the error.
  4. **Optimizer Step (Turning the Knobs):** An algorithm (`AdamW`) nudges each knob slightly in the direction that reduces the error next time.

Repeating this over the training dataset is called **training**. One complete pass through all training clips is called an **epoch**.

---

## 2. The Dataset: RWF-2000 & Preventing "Data Leakage"

### 2.1 What is RWF-2000?
- **Real World Fight 2000 (Cheng et al., ICPR 2020):** 2,000 real surveillance video clips collected from CCTV footage.
- **Length & Frame Rate:** Each clip is **~5 seconds long** at **30 FPS** (~150 frames per clip).
- **Balanced Classes:**
  - `1,000` clips of `Fight` (target = `1`, `violent`)
  - `1,000` clips of `NonFight` (target = `0`, `non_violent`)

### 2.2 What is "Data Leakage" and Why Do We Need Three Splits?
Imagine a professor gives students last year's exam to study from, and then accidentally uses the **exact same questions** on the final exam. A student who just memorized the answer key gets 100%, even if they learned nothing! In Machine Learning, this is called **Data Leakage**.

To prevent data leakage, we split the dataset into **three strictly separate sets**:

```text
Official RWF-2000 Folders
├── train/ (1,600 clips) ──┬──► 80% -> Train Set (1,280 clips)
│                          │         Used to update the neural network's weights.
│                          │
│                          └──► 20% -> Carved Validation Set ("val_carved", 320 clips)
│                                    Used as a "practice exam" after each epoch to pick
│                                    the best model checkpoint and tune settings.
│
└── val/   (400 clips)   ─────► 100% -> Held-Out Test Set ("test", 400 clips)
                                     Locked in a vault. Never used to tune settings!
```

### 2.3 Our Three Anti-Leakage Protections ([`mosaic/dataset.py`](../mosaic/dataset.py))
1. **Stratified Split Carving:** When we carve 20% of `train/` into `val_carved`, we shuffle `Fight` and `NonFight` clips separately with our fixed random seed so every split is guaranteed to remain balanced (50% Fight, 50% NonFight).
2. **SHA-256 Cryptographic File Hash Check:** What if someone copied a video from `train/` into `val/` and renamed it? We compute the SHA-256 hash of every file's bytes. If any two clips across different splits share a filename or SHA-256 hash, `verify_disjoint()` immediately halts with a `DataLeakageError`.
3. **Perceptual Frame-Signature Check (`compute_frame_signature`):** What if two clips are near-duplicates (e.g., re-saved with slightly different headers)? We sample 3 keyframes (start, middle, end), shrink them to `8×8` grayscale grids, and compute a 48-character perceptual signature to flag near-duplicates across splits.

---

## 3. Frame Sampling, Padding & Time Deltas ([`mosaic/sampling.py`](../mosaic/sampling.py))

A 5-second clip has ~150 frames. Adjacent frames (`0.033` seconds apart) look almost identical. Feeding all 150 frames into a 2D CNN at once wastes memory and compute.
Instead, we sample **`num_frames = 16` evenly spaced frames** across the clip:
- For a 150-frame clip, `sample_frame_indices(150, 16, strategy="uniform")` picks frames `[0, 10, 20, 30, ..., 149]`.
- **What if a clip is shorter than 16 frames (say, only 6 frames)?**
  - `pad_or_truncate_frames` repeats the last frame 10 times (`edge` padding) to reach length 16, and outputs a boolean **`validity_mask`**:
    `[True, True, True, True, True, True, False, False, ..., False]`
  - That mask tells our neural network: *"Only the first 6 frames are real; ignore frames 7–16!"*
- **Per-Step Time Deltas (`time_deltas`):**
  - We also pass the time gap $\Delta t = t_i - t_{i-1}$ between consecutive sampled frames into the model. Why? Because in Phase 3, our **Motion Gate** will skip static frames unevenly! Giving the model $\Delta t$ lets it know whether two frames were `0.03s` apart or `1.5s` apart.

---

## 4. How the Baseline Neural Network Works ([`mosaic/models/temporal_baseline.py`](../mosaic/models/temporal_baseline.py))

Our `BaselineTemporalClassifier` has three stages connected in sequence:

```text
Input Tensor: [Batch, T=16 frames, C=3 RGB channels, H=224, W=224]
      │
      ▼
┌──────────────────────────────────────────────────────────┐
│ 1. Spatial Backbone (ResNet-18 w/ ImageNet Norm or Light)│
│    Normalizes RGB pixels by ImageNet mean/std and        │
│    extracts a 512-d spatial feature vector f_t per frame.│
└──────────────────────────────────────────────────────────┘
      │  Computes frame-to-frame feature motion: Δf_t = f_t - f_{t-1}
      │  Applies LayerNorm + Clip-Consistent Channel Dropout
      │  + Concatenates per-step time_deltas [Batch, T=16, 1]
      ▼
┌──────────────────────────────────────────────────────────┐
│ 2. Temporal Sequence Model (GRU - Gated Recurrent Unit)  │
│    Reads [f_t, Δf_t, Δt_t] in chronological order and    │
│    pools across ALL valid frames (Last + Mean + Max).    │
└──────────────────────────────────────────────────────────┘
      │  Outputs: [Batch, 3 × Hidden_Dim = 384]
      ▼
┌──────────────────────────────────────────────────────────┐
│ 3. Binary Classification Head (LayerNorm + Linear)       │
│    Maps the pooled temporal representation to a single   │
│    violence probability in [0.0, 1.0].                   │
└──────────────────────────────────────────────────────────┘
```

### Why Two Backbones (`resnet18` vs `cnn_light`)?
- **`resnet18`:** A famous 18-layer Convolutional Neural Network pre-trained on millions of photographs (ImageNet). Used for full RWF-2000 training on your RTX 4070 GPU.
- **`cnn_light`:** A compact 2-stage CNN with `GroupNorm` and a `4×4` spatial grid (`AdaptiveAvgPool2d((4, 4))`). It needs zero internet downloads, preserves where objects are on a `4×4` grid, and runs in milliseconds for unit tests and smoke runs.

### Why a GRU (Gated Recurrent Unit)?
A single still photo cannot tell you whether an outstretched arm is a punch or a friendly wave—violence is defined by **motion over time**. A **GRU** is a recurrent neural network with mathematical "gates" (*reset* and *update* gates) that learn what to remember and what to forget across the 16 frames.

---

## 5. Why Do We Need "Trivial Baselines"? ([`mosaic/models/baselines.py`](../mosaic/models/baselines.py))

How do you know if your deep neural network actually learned anything useful? You compare it against two "mindless" baselines:
1. **Majority Class Classifier (`MajorityClassClassifier`):** Counts which label was most common in the training set and guesses that label every single time. On a 50/50 balanced dataset, it gets **50.0% accuracy**. Any deep model that cannot beat 50% has learned nothing.
2. **Motion Energy Classifier (`MotionEnergyClassifier`):** Measures the average pixel difference $|I_t - I_{t-1}|$ between consecutive frames and predicts `violent` if the average movement exceeds a learned threshold.
   - Fights usually have high movement, so this simple formula gets ~60–70% accuracy without knowing what a person or a punch is! Comparing our neural network against `MotionEnergyClassifier` tells us whether our model is genuinely recognizing violent behavior or merely acting as an expensive motion sensor.

---

## 6. Mishaps We Caught and Fixed in Phase 1

Studying the initial Phase 1 code revealed several critical engineering bugs that we fixed:
1. **Shuffled DataLoader Bug:** The training script originally looped over `train_loader` (which had `shuffle=True`) once to get labels, and a *second* time to get motion energies—pairing each video's motion energy with a random different video's label! We fixed it to extract both in a single pass.
2. **Dead Padding Code:** `sample_frame_indices` previously duplicated frame indices on short clips, so `pad_or_truncate_frames` and `validity_mask` were never triggered. Now short clips are properly edge-padded with `validity_mask = False` on padded frames, and the GRU reads the hidden state at the last valid frame instead of the padded tail.
3. **1×1 Global Pooling Erasing Object Positions:** `LightCNNBackbone` originally averaged the entire frame into a `1×1` pixel and froze random weights, causing the model to predict `non_violent` (`0.47`) for every video (`F1 = 0.0`). Switching to a `4×4` spatial grid with `GroupNorm` raised mock validation accuracy from **50% (`F1 = 0.0`) to 100% (`F1 = 1.000`)**.
4. **Corrupt Clip Fabrication:** `scan_rwf_directory` previously set `frame_count = meta.decoded_frame_count or 1`, turning unreadable corrupt videos into fake 1-frame entries. Corrupt clips are now isolated into `corrupt_entries` with explicit error messages.

---

## 7. Real-World Discoveries on the Downloaded RWF-2000 Dataset

When we ran our verified scanner (`python -m mosaic scan-dataset`) on the official **2,000-clip RWF-2000 dataset** (`11.47 GB`), our anti-leakage checks uncovered three real-world dataset issues that most published papers overlook:

1. **11 Exact Byte-for-Byte Duplicate Videos Across Splits (Different Filenames, Identical SHA-256!):**
   - Our SHA-256 check discovered **11 byte-for-byte identical `.avi` files** saved under different YouTube IDs inside the official RWF-2000 dataset—including **6 clips in `train/NonFight/` that are exact copies of clips in `val/NonFight/` (`test`)** (e.g., `train/NonFight/aGfwZ3E4_0.avi` == `val/NonFight/aZxUVVqk_0.avi`) and **5 within-train duplicates**.
   - **Our Fix:** [`scan_rwf_directory`](../mosaic/dataset.py) deduplicates those 11 files *before* carving `val_carved` while keeping the official 400-clip `test` set intact, leaving **1,989 unique, leak-free clips**.
2. **Multi-Clip Source Videos & 209 Shared Filenames (`Fight/` vs `NonFight/`):**
   - The `1,589` unique clips in RWF-2000's `train/` folder were cut from **781 unique YouTube/CCTV source videos** (`<video_id>_0.avi`, `<video_id>_1.avi`, `<video_id>_2.avi`, ... up to 22 clips from a single camera!), and `209` clips even share identical filenames across `train/Fight/` and `train/NonFight/`.
   - If you split individual clips randomly, `<video_id>_0.avi` goes into `train` while `<video_id>_1.avi` from the **exact same CCTV camera** goes into `val_carved` (camera-background leakage!).
   - **Our Fix:** [`scan_rwf_directory`](../mosaic/dataset.py) strips the `_<clip_number>` suffix (`_source_video_id`) and groups all clips from the same source camera into a single unit during stratified carving. Result: **`0` shared source cameras between `train` (`1,273 clips`: `639 Fight`, `634 NonFight`) and `val_carved` (`316 clips`: `158 Fight`, `158 NonFight`)**, and cross-split near-duplicates dropped from `9` down to **`1`**.
3. **Why the "Majority Class" Baseline Has `F1 = 0.667` (The 100% False-Positive Trap):**
   - Because removing the 11 `NonFight` duplicates left `639 Fight` vs `634 NonFight` in `train`, the most common class in `train` is `Fight` (`50.2%`).
   - A `MajorityClassClassifier` therefore predicts `Fight` (`1`) for **every single video**. That gives `100% Recall` (it never misses a fight because it calls everything a fight!) and `50% Precision`, which plugs into $F_1 = \frac{2 \times 0.5 \times 1.0}{0.5 + 1.0} = 0.667$—even though its **Accuracy is 50.0%**, its **ROC-AUC is 0.500** (pure coin flip), and its **False Positive Rate is 100.0%**! This is why we always check **Accuracy + ROC-AUC + False Positive Rate** together rather than $F_1$ alone.

---

## 8. Real RWF-2000 Phase 1 Baseline Results (RTX 4070 GPU)

After adding ImageNet mean/std normalization, frame-to-frame feature motion ($\Delta f_t = f_t - f_{t-1}$), clip-consistent channel dropout, and masked temporal pooling (`last + mean + max` across all valid frames), we trained the **Phase 1 Baseline Temporal Classifier** (`ResNet-18` frozen backbone + `GRU`) on the `1,273` training clips and evaluated it on both the **camera-isolated carved validation set (`val_carved`, `N = 316`)** and the **official held-out RWF-2000 test set (`test`, `N = 400`)**:

| Model / Split | Accuracy | Precision | Recall | F1 Score | ROC-AUC | TP / TN / FP / FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class (`val_carved`, N=316)** | 50.00% | 50.00% | 100.0% | 0.667 | 0.500 | 158 / 0 / 158 / 0 |
| **Motion Energy (`val_carved`, N=316)** | 61.08% | 58.97% | 72.78% | 0.652 | 0.709 | 115 / 78 / 80 / 43 |
| **Learned Baseline (`val_carved`, N=316)** | **70.25%** | **75.81%** | **59.49%** | **0.667** | **0.772** | **94 / 128 / 30 / 64** |
| **Learned Baseline (`test`, Held-Out N=400)** | **75.25%** | **75.63%** | **74.50%** | **0.751** | **0.835** | **149 / 152 / 48 / 51** |

- **Why `Test` (`75.25%` Acc, `0.835` ROC-AUC) Is Higher Than `val_carved` (`70.25%` Acc, `0.772` ROC-AUC):**
  - Because `val_carved` is strictly **camera-grouped** (`0` shared YouTube source IDs with `train`), every single validation clip comes from an unseen camera environment, making `val_carved` a tough, zero-shortcut generalization test. On the official 400-clip RWF-2000 benchmark split (`test`), the same checkpoint achieves **75.25% accuracy** and **0.835 ROC-AUC** using only a frozen 2D ResNet-18 backbone on 16 frames!
- **Why Whole-Frame Features Top Out Around ~75% (Motivating Phase 4):**
  - Whole-frame ResNet-18 averages features across the entire `224×224` image (~85% background walls, floors, trees, and cars). In **Phase 4**, we will add **YOLOv26n Person Detection + ROI Align** so the temporal model focuses directly on the *people* and their interactions rather than the room background.

---

## 9. Try It Yourself (Commands)

```powershell
# 1. Download and extract the full RWF-2000 dataset (resumable, sanitizes filenames)
python -m mosaic download-rwf

# 2. Scan the full RWF-2000 dataset and build the verified camera-grouped manifest
python -m mosaic scan-dataset

# 3. Train and evaluate the baseline temporal classifier on RWF-2000 (uses disk feature cache after run 1)
python -m mosaic train-baseline --set model.pretrained=true --set model.dropout=0.35 --set data.batch_size=16 --set train.epochs=10 --set train.early_stopping_patience=6 --set train.learning_rate=0.0002 --set train.weight_decay=0.02
```
