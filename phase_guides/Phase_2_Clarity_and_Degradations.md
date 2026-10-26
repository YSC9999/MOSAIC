# Phase 2 — Clarity Profiler & Seeded Degradation Toolkit

> **Goal of Phase 2:** Build an optical quality assessor (Block 1) that scores how clear and reliable an incoming video is *without* guessing or cheating, and create a reproducible degradation toolkit to test our system against degraded real-world edge cameras (blur, low light, compression, packet loss, sensor noise).

---

## 1. Why Do We Need Quality Scoring? (Plain-English Primer)

Imagine a security camera in a parking garage:
- At 2:00 PM on a clear day, the image is sharp, bright, and high-resolution.
- At 2:00 AM in a rainstorm, the lens is covered in water droplets, headlights cause blinding glare, or the scene is pitch black.

### The "Silent Failure" Problem in AI
Most computer vision systems blindly process whatever image they receive. If a camera lens is smeared with mud or pointing into total darkness:
* A dumb system will output: *"Violence score = 0.05 (Safe)"*.
* A human operator thinks: *"Great, the garage is safe!"*
* In reality, **the camera was completely blind**.

To prevent silent failures, our pipeline measures **input reliability** (called the **Clarity Score**) and attaches it to every analysis window. If the clarity drops to `0.20`, the security center receives an alert: *"Camera 4 is too degraded to monitor reliably—send maintenance."*

---

## 2. The 5 Clarity Components ([`mosaic/clarity.py`](../mosaic/clarity.py))

Our clarity profiler evaluates five independent optical and physical dimensions of the video, mapping each component into a normalized score between `0.0` (unusable) and `1.0` (crystal clear):

```text
Input Video Frame [H, W, 3]
       │
       ├──► 1. Resolution Component  (Pixel density vs. 720p baseline)
       ├──► 2. Sharpness Component   (Edge strength via Laplacian variance)
       ├──► 3. Exposure Component    (Luminance balance & clipping penalty)
       ├──► 4. Compression Component (8×8 DCT grid blockiness ratio)
       └──► 5. Person Scale Component (EXPLICITLY NONE until Phase 4!)
       │
       ▼
  ClarityReport: score in [0.0, 1.0] + component breakdown + source_profile
```

### 2.1 Resolution Component (`resolution_component`)
- **What it measures:** The physical pixel density of the camera sensor.
- **How it works:** We compare the frame's diagonal pixel count against a standard **720p (`1280×720`)** benchmark:
  $$\text{Resolution Score} = \min\left(1.0, \frac{\sqrt{W^2 + H^2}}{\sqrt{1280^2 + 720^2}}\right)$$
  - `1080p (1920×1080)`: `1.00` (optimal)
  - `720p (1280×720)`: `1.00` (baseline standard)
  - `480p (640×480)`: `~0.54`
  - `240p (320×240)`: `~0.27`

### 2.2 Sharpness Component (`sharpness_component`)
- **What it measures:** Optical focus and motion blur.
- **How it works (Laplacian Variance):**
  A blurry image has smooth, gradual transitions between colors. A sharp image has crisp, abrupt transitions (high spatial derivatives).
  1. We convert the frame to grayscale and apply the **Laplacian operator** ($\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$), which calculates the second derivative across the image.
  2. We measure the **variance** of the Laplacian.
  3. A completely uniform/blank frame has variance $= 0.0$ (`score = 0.0`). A sharp CCTV image has variance $> 500$ (`score = 1.0`).

### 2.3 Exposure Component (`exposure_component`)
- **What it measures:** Lighting balance, dark shadows, and overexposed washed-out highlights.
- **How it works:**
  1. **Luminance Balance:** We calculate the mean brightness $\mu \in [0, 255]$. Optimal CCTV lighting sits near the center ($\mu \approx 128$). We measure distance from center: $1.0 - 2 \times |\frac{\mu}{255} - 0.5|$.
  2. **Clipping Penalty:** We penalize "crushed blacks" (pixels $\le 5$) and "blown-out whites" (pixels $\ge 250$). If 40% of the screen is pure pitch-black shadow, the clipping penalty knocks the score down heavily.
  - Result: Pitch-black night footage gets `0.00`; sun glare gets `0.00`; balanced daylight gets `~0.95`.

### 2.4 Compression Component (`compression_component`)
- **What it measures:** Artifacts from low-bitrate streaming or aggressive video encoding.
- **How it works (8×8 DCT Grid Analysis):**
  Video compression algorithms (like H.264 and JPEG) divide images into **8×8 pixel blocks**. Under low bandwidth, block edges become visible as harsh grid lines ("blockiness").
  - We calculate the ratio between pixel jumps *across* the 8-pixel block boundaries versus pixel jumps *inside* the blocks.
  - Clean video: ratio $\approx 1.0$ (`score = 1.0`).
  - Heavily blocky video: boundary ratio $> 1.8$ (`score drops toward 0.0`).

### 2.5 Person Scale Component (`person_scale_component`)
- **Why is it `None`?**
  In Phase 2, we have not added an object/person detector yet (that happens in Phase 4).
  - Rather than faking a default number like `0.5`, our typed contract ([`ClarityReport`](../mosaic/contracts.py#L327-L374)) explicitly sets `person_scale_component = None`.
  - When a component is `None`, [`combine_clarity_components`](../mosaic/clarity.py) automatically renormalizes the weights over the remaining 4 components, keeping the final score in `[0.0, 1.0]`.

---

## 3. The Seeded Camera Degradation Toolkit ([`mosaic/degradations.py`](../mosaic/degradations.py))

Real edge deployments do not receive pristine 1080p footage. To test how resilient our pipeline is before deploying to actual hardware, we built a **deterministic degradation toolkit** covering the 7 simulated camera conditions specified in Section 3.4 of the plan:

| Degradation | What It Simulates | Mathematical Implementation |
| :--- | :--- | :--- |
| **`resolution`** | Cheap low-tier sensors | Downscales frame by up to $80\%$ via area interpolation, then upsamples back. |
| **`blur`** | Out-of-focus lenses or camera shake | Gaussian filter with kernel size $k$ scaling up to $25\times 25$ and $\sigma$ up to $5.0$. |
| **`compression`** | Low-bandwidth RTSP stream packet compression | Encodes and decodes via OpenCV JPEG DCT quantization (quality factor dropping from 95 down to 5). |
| **`low_light`** | Nighttime or unlit corridors | Crushes shadows with power-law gamma curve $I^\gamma$ ($\gamma$ scaling from $1.0$ to $3.0$) and scales brightness down to $15\%$. |
| **`noise`** | High-ISO sensor thermal noise in darkness | Injects zero-mean Gaussian perturbation $\mathcal{N}(0, \sigma^2)$ ($\sigma$ scaling up to $45.0$). |
| **`frame_drop`** | Network packet loss / RTSP frame skips | Drops intermediate frames with probability up to $70\%$, **while preserving original timestamps** for temporal stability. |
| **`occlusion`** | Physical obstruction (tree branch, pillar, insects) | Places a dark gray rectangular mask over up to $40\%$ of the screen at a deterministic seed-derived position. |

### Why "Seeded & Deterministic"?
Every degradation accepts a random `seed` and a `severity` parameter in `[0.0, 1.0]`. Running the exact same clip with `seed=1337, severity=0.6` on Windows, Linux, or a server produces the **exact same pixel values every single time**. This guarantees our scientific ablation experiments in Phase 8 are 100% reproducible.

---

## 4. Visualizing the 7 Simulated Camera Conditions

Below is the visual artifact generated by running all 7 degradations on an actual RWF-2000 surveillance camera:

![Phase 2 Degradations and Clarity](file:///C:/Users/yscha/.gemini/antigravity-cli/brain/68bdc3f2-7c90-4346-b91d-8cd22e91ae87/phase2_degradations_and_clarity.png)

---

## 5. Exit Gate Findings: The "Sensor Noise" Bug & The Donoho MAD Fix

When we first tested our clarity profiler across 30 real RWF-2000 CCTV test videos (`python -m mosaic profile-clarity --split test --max-clips 30`), we caught an unexpected mathematical anomaly:
- **Optical Blur, Low Light, Resolution, and Compression** all reduced the clarity score as expected.
- But **Thermal Sensor Noise** originally *increased* the score from `0.770` to `0.891` (`+0.120`)!

### Why Did Naive Sharpness Think Noise Was "Sharp"?
* The Laplacian operator measures sharp edges by taking the second spatial derivative ($\nabla^2 I$).
* Optical edges are **spatially coherent lines and curves** (adjacent pixels along an edge change together).
* Gaussian sensor noise adds random, high-frequency spikes to **every single isolated pixel**.
* A naive second derivative sees those millions of random pixel spikes and calculates massive mathematical variance, foolishly rating a grainy, noisy camera as "ultra-sharp"!

### How We Fixed It: The Donoho & Johnstone MAD Noise Estimator
To fix this, we upgraded [`mosaic/clarity.py`](../mosaic/clarity.py) with two signal-processing techniques:
1. **Gaussian Pre-Smoothing ($3\times 3, \sigma=0.75$):** Smooths out isolated single-pixel white noise while preserving coherent multi-pixel edges.
2. **Donoho & Johnstone Median Absolute Deviation (MAD) Noise Estimator:**
   We pass the image through a specialized high-pass filter:
   $$\text{Kernel} = \frac{1}{6} \begin{bmatrix} 1 & -2 & 1 \\ -2 & 4 & -2 \\ 1 & -2 & 1 \end{bmatrix}, \quad \hat{\sigma}_{\text{noise}} = \frac{\text{median}(|\text{Residual}|)}{0.6745}$$
   * On clean CCTV, $\hat{\sigma}_{\text{noise}} \approx 0.0$ (zero penalty).
   * On sensor noise, $\hat{\sigma}_{\text{noise}} > 18.0$, triggering a noise-cleanliness attenuation penalty.

### Verified Results on 30 Real RWF-2000 Test Videos
With the fix in place, re-running `python -m mosaic profile-clarity --split test --max-clips 30` saved the verified results to [`artifacts/reports/clarity_profile_report.json`](../artifacts/reports/clarity_profile_report.json):

| Condition (Severity = 0.6) | Mean Clarity | Change vs. Clean | Physical Effect |
| :--- | :---: | :---: | :--- |
| **Clean CCTV (Unaltered)** | **0.814** | Baseline | Typical 720p surveillance footage |
| **Low-Light Gamma Crushing** | **0.572** | **-0.243** | Shadow crushing triggered exposure penalty |
| **Optical / Defocus Blur** | **0.641** | **-0.173** | Coherent high-frequency edges blurred |
| **Sensor Thermal Noise** | **0.779** | **-0.036** | **Noise estimator penalizes high-ISO sensor snow** |
| **JPEG / DCT Compression** | **0.785** | **-0.029** | 8×8 block boundary artifacts penalized |
| **Resolution Downscale** | **0.789** | **-0.025** | Spatial resolution downsampled |
| **Synthetic Occlusion** | **0.808** | **-0.007** | Obstacle shadow clipping penalty triggered |
| **Frame Dropping** | **0.813** | **-0.002** | Temporal frame skipping (per-frame quality intact) |

**Every single degradation now reliably decreases clarity**, successfully clearing the **Phase 2 Exit Gate**.

---

## 6. Try It Yourself (Commands)

```powershell
# 1. Run all unit and integration tests for Phase 2
pytest tests/unit/test_clarity.py tests/unit/test_degradations.py tests/integration/test_phase2_integration.py

# 2. Profile clarity across real RWF-2000 test clips and all 7 degradations
python -m mosaic profile-clarity --split test --max-clips 30

# 3. Test a specific degradation through the pipeline
python -m mosaic smoke --synthetic moving --set pipeline.stages=[ingestion,clarity] --set degradation.enabled=true --set degradation.kind=blur --set degradation.severity=0.8
```
