# MOSAIC-FL Phase 1 — Dataset Ingestion & Baseline Temporal Classifier

This guide explains **Phase 1** of MOSAIC-FL for readers who are completely new to Machine Learning (ML) and Deep Learning (DL).

Every tool, concept, and line of architectural reasoning is explained from first principles: what it is, why it exists, what alternatives were considered, and why our choice was selected.

---

## 1. The Big Picture: What is Phase 1 Doing?

In **Phase 0**, we built the foundation of our software:
- Typed data contracts (rules for how information must look).
- Validated configuration (preventing typos and silent bugs).
- Deterministic random seeding (making experiments 100% reproducible).
- Video decoding and synthetic test videos.

Now, in **Phase 1**, we take our first step into **Machine Learning**:
1. We teach our system how to read real-world video clips from the **RWF-2000 dataset**.
2. We organize the videos into three strict groups (**Train**, **Validation**, and **Test**) so our model can learn without "cheating".
3. We extract a sequence of video frames from each clip in a completely repeatable way.
4. We build our first **Temporal Classifier** — a deep neural network that watches the sequence of frames and predicts whether violence is happening.
5. We establish **trivial baseline floors** (simple rules like "always guess non-fight" or "measure raw pixel movement") to prove that our neural network is genuinely learning rather than guessing.

```
+-------------------------------------------------------------------------+
|                              Phase 1 Overview                           |
|                                                                         |
|  +--------------------+         +--------------------+                  |
|  |   RWF-2000 Clips   |  --->   |  Split Carver &    |                  |
|  |  (Fight / NonFight)|         |  Disjoint Checker  |                  |
|  +--------------------+         +--------------------+                  |
|                                           |                             |
|                                           v                             |
|                                 +--------------------+                  |
|                                 | Deterministic      |                  |
|                                 | Frame Sampler      |                  |
|                                 +--------------------+                  |
|                                           |                             |
|                                           v                             |
|                                 +--------------------+                  |
|                                 | PyTorch Dataset &  |                  |
|                                 | Seeded DataLoader  |                  |
|                                 +--------------------+                  |
|                                           |                             |
|                       +-------------------+-------------------+         |
|                       |                                       |         |
|                       v                                       v         |
|             +-------------------+                   +-------------------+
|             | Trivial Baselines |                   | Baseline Temporal |
|             | - Majority Class  |                   | Model (Backbone + |
|             | - Motion Energy   |                   | GRU Sequence Head)|
|             +-------------------+                   +-------------------+
|                       |                                       |         |
|                       +-------------------+-------------------+         |
|                                           |                             |
|                                           v                             |
|                                 +--------------------+                  |
|                                 | Evaluation Metrics |                  |
|                                 | (Acc, F1, ROC-AUC) |                  |
|                                 +--------------------+                  |
+-------------------------------------------------------------------------+
```

---

## 2. Core Concepts: Machine Learning From Scratch

If you have never worked with Machine Learning, here are the essential concepts you need to know.

### 2.1 What is "Supervised Learning"?
In traditional programming, a human programmer writes explicit rules:
```python
if person_is_running and arms_flailing:
    return "violence"
```
In **Supervised Machine Learning**, we do not write the rules by hand. Instead:
1. We show a model thousands of examples of video clips.
2. Each video has a **ground truth label** (e.g. `Fight` or `NonFight`) assigned by human experts.
3. The model makes a guess.
4. If the guess is wrong, a mathematical formula (called the **Loss Function**) measures how wrong the guess was.
5. An algorithm (called an **Optimizer**) slightly adjusts millions of tiny mathematical dials inside the model (called **Weights** or **Parameters**) so that next time, its guess is closer to the truth.

### 2.2 What is the RWF-2000 Dataset?
- **Name:** Real World Fight 2000 (RWF-2000).
- **Source:** Created by researchers (Cheng et al., 2020) from surveillance cameras capturing real-life physical altercations and normal human activities.
- **Size:** 2,000 video clips in total.
  - 1,000 clips showing violent fights (`Fight`).
  - 1,000 clips showing normal human activities like walking, jogging, hugging, or talking (`NonFight`).
- **Characteristics:**
  - Each clip is approximately **5 seconds long**.
  - Recorded at approximately **30 frames per second** (about 150 frames per clip).
  - Realistic surveillance conditions: varying lighting, camera angles, resolutions, and crowd sizes.

---

## 3. Data Integrity: Splits and Avoiding "Data Leakage"

One of the most dangerous traps in Machine Learning is **data leakage**. Data leakage happens when a model accidentally sees answers during training that it should only see during testing.

### 3.1 Why Three Splits?
Imagine a student preparing for an exam:
1. **Training Set (`train`)**: The textbook exercises the student studies every day. The model updates its parameters on these examples.
2. **Carved Validation Set (`val_carved`)**: A practice mock exam the teacher gives to check if the student is understanding the material, and to decide which study habits work best (hyperparameter tuning). The model never directly learns from this, but the engineer uses it to pick the best model settings.
3. **Held-Out Test Set (`test`)**: The final board exam. This is locked in a vault and only touched at the very end to measure the model's true real-world performance.

In RWF-2000:
- The authors provided two folders: `train/` (1,600 clips) and `val/` (400 clips).
- To preserve absolute scientific rigor, **we treat the official `val/` folder as our locked held-out test set (`test`)**.
- We carve out 20% of the official `train/` folder using a deterministic random seed to create our `val_carved` set. The remaining 80% becomes our `train` set.

### 3.2 What is "Stratification"?
If you randomly split a dataset, you might accidentally put 90% of the fights into the training set and almost none into the validation set. That would make the validation test unrepresentative.
**Stratification** guarantees that every single split has the exact same proportion of classes — in our case, **50% Fight and 50% NonFight**.

### 3.3 The SHA-256 Disjointness Check
How do we mathematically prove that no video appears in more than one split?
We compute the **SHA-256 cryptographic hash** of every single video file. If two files have the exact same hash, they contain identical bytes.
Our pipeline runs an automated check verifying:
$$\text{train} \cap \text{val\_carved} = \emptyset \quad \text{and} \quad \text{train} \cap \text{test} = \emptyset \quad \text{and} \quad \text{val\_carved} \cap \text{test} = \emptyset$$
If a duplicate is ever found, the code immediately halts with a `DataLeakageError`.

---

## 4. Video Sampling & Preprocessing

A video is just a rapid sequence of still pictures (frames). A 5-second video at 30 FPS has 150 frames. Feeding all 150 high-resolution frames into a deep neural network would require massive amounts of memory and compute, most of which is redundant (consecutive frames 0.03 seconds apart look almost identical).

### 4.1 Frame Sampling: How Many Frames Do We Need?
For video classification, state-of-the-art models typically sample **16 or 32 evenly spaced frames** across the clip.
- If a clip has 150 frames and we need 16 frames, we pick frames at step intervals: $0, 9, 19, 28, \dots, 149$.
- **Bounding guarantee:** Our frame sampler mathematically guarantees that it will never request a frame index outside $[0, \text{total\_frames} - 1]$.

### 4.2 Handling Irregular Video Lengths: Padding vs Truncation
What happens if a corrupt or short video only has 10 frames, but our model expects 16?
- **Padding Policy:** We duplicate the final frame until we reach 16 frames, and we create a boolean validity mask (`[True, True, ..., False]`) so the model knows which frames were real and which were padded.
- **Truncation Policy:** If a video is too long, we uniformly sample down to the exact target count.
This behavior is 100% deterministic: the same video input always yields the exact same tensor.

### 4.3 Data Augmentation: Why Train-Only?
When training, we can slightly alter the video (e.g. flipping it horizontally like a mirror) so the model learns that a fight happening left-to-right is the same as a fight happening right-to-left.
**Critical Rule:** Augmentations must **NEVER** be applied to validation or test videos. Validation and test videos must remain pristine and unaugmented so that our evaluation reflects reality.

---

## 5. Tools & Frameworks Used in Phase 1

### 5.1 PyTorch (`torch`)
- **What it is:** The world's leading open-source deep learning framework, developed by Meta AI and the open-source community.
- **Why we chose it:**
  1. *Dynamic Computation Graph (Eager Execution):* PyTorch lets you debug neural networks like standard Python code with standard breakpoints, print statements, and stack traces.
  2. *Hardware Acceleration:* Seamlessly executes tensor math on NVIDIA GPUs using CUDA without changing model code.
  3. *Industry Standard:* The majority of modern computer vision research is written in PyTorch.
- **Alternatives considered:**
  - *TensorFlow / Keras:* Widely used in industry, but more rigid computation graphs and less prevalent in recent cutting-edge video recognition research.
  - *JAX:* Extremely fast on TPUs, but smaller ecosystem for video ingestion and pre-trained computer vision backbones.

### 5.2 PyTorch `Dataset` and `DataLoader`
- **`torch.utils.data.Dataset`:** A clean Python interface representing a collection of data. It only requires implementing two methods: `__len__()` (how many clips exist) and `__getitem__(index)` (how to load clip number $i$).
- **`torch.utils.data.DataLoader`:** A manager that batches clips together (e.g. 8 clips at a time), shuffles training clips deterministically using our random seed, and sends batches to the GPU.

---

## 6. Model Architecture: Baseline Temporal Classifier

The goal of Phase 1 is to build a reliable **baseline** temporal classifier. A baseline is a clean, simple, solid model that serves as the benchmark against which all future complex modules (object detection, multi-person tracking, kinematics) will be measured.

### 6.1 Architecture Breakdown

```
Input: Video Clip of T frames [B, T, C, H, W]
                      |
                      v
      +-------------------------------+
      |   Spatial Feature Extractor   |  Extracts a compact 1D vector (e.g., 512 numbers)
      |   (ResNet-18 / ConvNet)       |  summarizing the visual content of each frame.
      +-------------------------------+
                      |
                      v
          Sequence of T Feature Vectors: [B, T, D]
                      |
                      v
      +-------------------------------+
      |  Temporal Sequence Model      |  Reads the sequence over time, maintaining a
      |  (Gated Recurrent Unit / GRU) |  hidden memory of past movements and actions.
      +-------------------------------+
                      |
                      v
           Final Temporal Summary Vector: [B, H]
                      |
                      v
      +-------------------------------+
      |  Classification Head          |  Dense layer + Sigmoid activation, outputting
      |  (Linear -> Sigmoid)          |  a single probability: Violence Score in [0, 1].
      +-------------------------------+
```

### 6.2 What is a Feature Backbone (ResNet-18)?
A Convolutional Neural Network (CNN) excels at looking at 2D image pixels and finding edges, textures, shapes, and objects. **ResNet-18** is a classic, lightweight, 18-layer deep CNN.
Instead of training a CNN from scratch (which requires millions of images), we can use a frozen backbone that converts each $224 \times 224$ frame into a rich 512-dimensional summary vector.

### 6.3 What is a GRU (Gated Recurrent Unit)?
A single frame cannot tell you if someone is punching or just stretching their arm. Violence is a **temporal process** — it happens over time.
- A **Recurrent Neural Network (RNN)** is a neural network with memory. As it reads frame 1, frame 2, frame 3, it carries forward a **hidden state** that remembers what happened previously.
- A **GRU (Gated Recurrent Unit)** is an advanced form of RNN that uses mathematical "gates" (an *update gate* and a *reset gate*) to decide what to remember and what to forget.
- **Why GRU instead of standard RNN?** Standard RNNs suffer from the "vanishing gradient problem" (they forget things that happened more than a few frames ago). GRUs solve this.
- **Why GRU instead of LSTM (Long Short-Term Memory)?** GRUs have fewer parameters than LSTMs, train faster, use less memory, and achieve virtually identical accuracy on short 5-second video clips.

---

## 7. Trivial Reference Baselines: Why Are They Essential?

Before you declare that your neural network has an 80% accuracy, you must answer: **what score would a completely mindless program get?**

If 80% of your dataset is `NonFight`, a program that does nothing and always guesses `NonFight` will achieve 80% accuracy without looking at a single pixel!

To ensure scientific honesty, Phase 1 implements two **trivial reference baselines**:

### 7.1 The Majority Class Baseline
- **How it works:** Finds whichever class was most frequent in the training set and always predicts that class.
- **Purpose:** Sets the absolute bottom floor of performance. If a deep neural network cannot beat the majority class baseline, it has learned nothing.

### 7.2 The Motion Energy Baseline
- **How it works:** Computes the average pixel difference between consecutive frames:
  $$\Delta I = \frac{1}{HW} \sum_{x, y} |I_t(x, y) - I_{t-1}(x, y)|$$
  High movement yields a high motion energy score. A simple threshold predicts violence if movement is above average.
- **Purpose:** Fights naturally involve vigorous movement. Does our deep learning model actually understand violent interactions, or is it just acting as an expensive motion detector? The motion energy baseline answers this question.

---

## 8. Evaluation Metrics Explained Simply

When evaluating a binary classifier ($1 = \text{Fight}$, $0 = \text{NonFight}$), we compute four basic counts:
- **True Positive (TP):** A real fight correctly identified as a fight.
- **True Negative (TN):** A normal video correctly identified as normal.
- **False Positive (FP):** A normal video falsely accused of being a fight (false alarm).
- **False Negative (FN):** A real fight that the model missed (missed detection).

From these counts, we compute:
1. **Accuracy:** $\frac{\text{TP} + \text{TN}}{\text{Total}}$
   - Percentage of total correct predictions.
2. **Precision:** $\frac{\text{TP}}{\text{TP} + \text{FP}}$
   - When the alarm sounds, how often is there really a fight? (High precision means few false alarms).
3. **Recall (Sensitivity):** $\frac{\text{TP}}{\text{TP} + \text{FN}}$
   - Out of all real fights, what fraction did the model catch? (High recall means few missed fights).
4. **F1-Score:** $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
   - The harmonic mean of precision and recall. Balances false alarms against missed detections.
5. **ROC-AUC (Receiver Operating Characteristic - Area Under Curve):**
   - Measures how well the model separates fights from non-fights across *all possible confidence thresholds* (from 0.0 to 1.0).
   - $1.0$ is perfect separation; $0.5$ is pure random coin-flipping.

---

## 9. Summary of Phase 1 Deliverables

| Deliverable | Purpose | File Location |
| :--- | :--- | :--- |
| **Data Configuration** | Validated settings for RWF dataset, sampling, and batching | [mosaic/config.py](file:///c:/Users/yscha/OneDrive/Desktop/SVNIT/Projects/HCI/Project/mosaic/config.py) |
| **RWF Manifest & Split** | Stratified split, SHA-256 checks, leakage prevention | [mosaic/dataset.py](file:///c:/Users/yscha/OneDrive/Desktop/SVNIT/Projects/HCI/Project/mosaic/dataset.py) |
| **Deterministic Sampler** | Bounded frame selection with padding/truncation policy | [mosaic/sampling.py](file:///c:/Users/yscha/OneDrive/Desktop/SVNIT/Projects/HCI/Project/mosaic/sampling.py) |
| **Window Aggregator** | Mean and max window-to-clip scoring | [mosaic/aggregation.py](file:///c:/Users/yscha/OneDrive/Desktop/SVNIT/Projects/HCI/Project/mosaic/aggregation.py) |
| **Trivial Baselines** | Majority class and motion energy sanity floors | [mosaic/models/baselines.py](file:///c:/Users/yscha/OneDrive/Desktop/SVNIT/Projects/HCI/Project/mosaic/models/baselines.py) |
| **Baseline Temporal Model** | Visual backbone + GRU sequence classifier | [mosaic/models/temporal_baseline.py](file:///c:/Users/yscha/OneDrive/Desktop/SVNIT/Projects/HCI/Project/mosaic/models/temporal_baseline.py) |
| **Evaluation Metrics** | Accuracy, Precision, Recall, F1, ROC-AUC | [mosaic/metrics.py](file:///c:/Users/yscha/OneDrive/Desktop/SVNIT/Projects/HCI/Project/mosaic/metrics.py) |
| **Training Engine** | Reproducible training loop, checkpointing, diagnostics | [mosaic/training.py](file:///c:/Users/yscha/OneDrive/Desktop/SVNIT/Projects/HCI/Project/mosaic/training.py) |
| **Mock RWF Fixture** | Offline synthetic dataset for automated testing | [mosaic/synthetic_rwf.py](file:///c:/Users/yscha/OneDrive/Desktop/SVNIT/Projects/HCI/Project/mosaic/synthetic_rwf.py) |
| **Unit & Integration Tests**| 100% test coverage for Phase 1 components | `tests/unit/test_phase1_*.py`, `tests/integration/` |
