# Phase 4 — Person & Object Detection (Block 3)

> **Goal of Phase 4:** Add a fast, edge-optimized visual object detector (Block 3) that localizes people in retained video frames, extracts standardized bounding boxes (`Box`) and optional visual feature vectors, and activates the final component of the Phase 2 Clarity Profiler: the **Person Scale Component**.

---

## 1. Why Do We Need Person Detection? (Plain-English Primer)

In Phase 1, our baseline model looked at the **entire video frame as a single giant image** (whole-frame 2D CNN). While this reached 75.25% test accuracy, it suffered from a fundamental weakness:
* **Background Confusion:** If a violent altercation happens in a small corner of an airport lobby while 20 people calmly sit in the foreground, whole-frame pooling blurs the violent struggle into background noise.
* **Camera Shake & Environmental Motion:** A tree swaying in the wind or a bus driving past the camera generates huge whole-frame pixel shifts that look like "violence" to a naive whole-frame model.

### Focusing Attention on Human Actors
Violence is inherently an interaction between **people**. 
By detecting exactly where people are in each frame:
1. The AI ignores swaying trees, passing cars, and empty background walls.
2. The AI can track individual movements (Phase 6), measure person-to-person distances, and detect physical altercations (punches, grapples, falls) directly where the action happens.

```text
Motion-Gated Frame Stream (Retained Frames Only)
        │
        ▼
┌────────────────────────────────────────────────────────┐
│         Block 3: Person & Object Detection             │
│    (FasterRCNN-MobileNetV3-320-FPN / SSDLite320)       │
│                 (~9.2 ms/frame on CPU)                 │
└──────────────────────────┬─────────────────────────────┘
                           │
       ┌───────────────────┴───────────────────┐
       ▼                                       ▼
  [Person Bounding Boxes]               [Person Scale Signal]
  • x1, y1, x2, y2 coordinates          • Ratio: median person height vs frame
  • Detection confidence (e.g. 98%)     • Completes the Phase 2 Clarity Profiler!
  • Optional 128-dim visual feature     • Low score if people are too distant
       │                                       │
       ▼                                       ▼
  Forwarded to Tracking (Phase 6)       Attached to ClarityReport
```

---

## 2. Core Architecture & Engineering Design ([`mosaic/detector.py`](../mosaic/detector.py))

### 2.1 The Standardized Detection Contract ([`mosaic/contracts.py`](../mosaic/contracts.py))
Every detected person produces an immutable, self-validating [`Detection`](../mosaic/contracts.py) contract:

```python
@dataclass(frozen=True)
class Detection:
    box: Box                                # [x1, y1, x2, y2] strictly non-negative with x2 > x1, y2 > y1
    confidence: float                       # Score in [0.0, 1.0]
    class_name: str                         # Filtered to "person"
    frame_index: int                        # Exact original frame index
    timestamp: float                        # True physical timestamp
    pooled_feature: Optional[List[float]]   # Optional 128-dim appearance embedding
```

### 2.2 Selecting the Pretrained Edge Backbone
We evaluated two edge-friendly backbones on real CCTV footage:

1. **FasterRCNN-MobileNetV3-320-FPN (Our Default Choice):**
   - Incorporates a **Feature Pyramid Network (FPN)**, allowing it to detect people across multiple spatial scales (both foreground pedestrians and distant background actors).
   - Achieves high confidence ($>95\%$) on challenging, low-resolution CCTV scenes.
   - Executes in **$\approx 9.2\text{ ms per frame}$ ($109\text{ FPS}$) on CPU**.
2. **SSDLite320-MobileNetV3 (Alternative Tier-C Model):**
   - Single-shot MobileNet detector with only $3.4\text{ million parameters}$.
   - Slightly faster ($8.5\text{ ms}$), but less sensitive to small/partially occluded people.
3. **MockDetector (Zero-Weight Testing Engine):**
   - Instantaneous ($0\text{ ms}$), deterministic detector for unit tests that requires zero neural network weights or network downloads.

### 2.3 Coordinate Preservation & Boundary Clamping
Camera frames can be resized or letterboxed by internal neural networks. Our detector strictly maps every bounding box back to the **original unscaled pixel coordinates** $[0, W] \times [0, H]$.
* Coordinates are strictly clamped so $x_1 \ge 0, y_1 \ge 0, x_2 \le W, y_2 \le H$.
* Degenerate boxes with area $< 100\text{ px}^2$ or height $< 16\text{ px}$ are safely pruned.

---

## 3. Completing the Clarity Profiler: The Person Scale Component

In Phase 2, `person_scale_component` was left explicitly as `None` because person bounding boxes did not yet exist. In Phase 4, we activate it!

### Why Person Scale Matters for Video Quality
* If a camera is mounted 50 meters away and people appear as tiny 8-pixel tall specks, optical sharpness and 1080p resolution are irrelevant—the system simply cannot see human limbs or punches.
* If people are at least **$25\%$ of the frame height**, individual body parts and micro-movements are fully resolvable.

### Mathematical Formulation ([`mosaic/clarity.py`](../mosaic/clarity.py))
When detections are available:
1. Extract the heights $h_i = y_2 - y_1$ of all detected persons in the window.
2. Compute the **median person height**: $h_{\text{median}} = \text{median}(\{h_i\})$.
3. Compute the scale ratio against frame height $H$:
   $$\text{Scale Ratio} = \frac{h_{\text{median}}}{H}$$
4. Map into a quality score in $[0.0, 1.0]$:
   $$\text{person\_scale\_component} = \text{clip}\left(\frac{h_{\text{median}} / H}{0.25}, 0.0, 1.0\right)$$

* **Graceful Degradation:** If no people are detected in a calm empty room, `person_scale_component` remains `None`, and the composite Clarity score automatically renormalizes over the remaining optical components (resolution, sharpness, exposure, compression).

---

## 4. Empirical Evaluation on Real RWF-2000 CCTV Clips

We ran the detector across 20 held-out clips from the RWF-2000 validation split (`val_carved`) using:
```bash
python -m mosaic detect-persons --split val_carved --max-clips 20
```

The resulting evaluation report ([`artifacts/reports/detector_report.json`](../artifacts/reports/detector_report.json)) shows:

| Metric | Measured Value | What It Means for Deployment |
| :--- | :---: | :--- |
| **Inference Latency** | **$9.2\text{ ms / frame}$** | **$109\text{ FPS}$ on standard CPU** (comfortably beats real-time $30\text{ FPS}$ requirement) |
| **Persons Detected Per Frame** | **$1.98\text{ people}$** | Perfectly captures the pairs of interacting individuals in RWF-2000 |
| **Mean Person Scale Component**| **$0.982$** | People in RWF-2000 clips are well-scaled and fully resolvable for tracking |
| **Composite Window Clarity** | **$0.695$** | Reflects typical real-world commercial CCTV quality |

---

## 5. Visual Artifact: Rendered Detections & Person Scale

We generated a visual artifact located at:
`artifacts/phase4_person_detections.png`

The visualization displays:
1. **Four Real CCTV Clips:** 2 violent fights (red headers) and 2 non-violent activities (blue headers).
2. **Emerald Bounding Boxes:** Tight localization of each person with detector confidence (e.g. `person 98%`) and pixel height ($h = 245\text{px}$).
3. **Median Scale Annotations:** Real-time display of person height relative to frame height ($h/H$) and the resulting person scale quality score.

---

## 6. Exit Criteria Checklist for Phase 4

- [x] **Block 3 Detector Module:** Implemented in [`mosaic/detector.py`](../mosaic/detector.py) supporting `fasterrcnn_mobilenet_320`, `ssdlite320`, and `MockDetector`.
- [x] **Configuration Integration:** `DetectorConfig` registered in [`mosaic/config.py`](../mosaic/config.py) and added to [`configs/default.yaml`](../configs/default.yaml).
- [x] **Clarity Person Scale Activation:** `calculate_person_scale_component` implemented in [`mosaic/clarity.py`](../mosaic/clarity.py) and verified.
- [x] **Pipeline Integration:** Wired into [`mosaic/pipeline.py`](../mosaic/pipeline.py) with lazy importing to preserve Phase 0 zero-weight ingestion guarantees.
- [x] **CLI Subcommand:** `detect-persons` subcommand implemented in [`mosaic/cli.py`](../mosaic/cli.py).
- [x] **Unit & Integration Tests:** 12 dedicated tests in [`tests/unit/test_detector.py`](../tests/unit/test_detector.py) and [`tests/integration/test_phase4_integration.py`](../tests/integration/test_phase4_integration.py).
- [x] **Full Regression Pass:** All **356 tests** across the codebase pass cleanly in $< 25\text{ seconds}$.
- [x] **Empirical Metrics:** Evaluated on RWF-2000 clips with verified **$9.2\text{ ms}$ latency** and **$1.98$ persons/frame** recorded in [`artifacts/reports/detector_report.json`](../artifacts/reports/detector_report.json).
- [x] **Visual Artifact:** Generated [`phase4_person_detections.png`](file:///C:/Users/yscha/.gemini/antigravity-cli/brain/68bdc3f2-7c90-4346-b91d-8cd22e91ae87/phase4_person_detections.png).
