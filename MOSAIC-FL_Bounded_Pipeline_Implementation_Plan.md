# MOSAIC-FL Bounded Pipeline — Implementation Plan

## 1. Purpose and scope

This document defines the first implementation phase of MOSAIC-FL. It is intentionally limited to the red-bounded on-device pipeline in the modular architecture diagram:

1. Camera input and video ingestion
2. Block 1 — Clarity estimation
3. Block 2 — Motion gating
4. Block 3 — Visual backbone and person detection
5. Block 4 — Threat/object evidence and masks
6. Support path — silhouette and box kinematics
7. Block 5 — Multi-object tracking and Track Tokens
8. Block 6 — Temporal context encoding

The initial dataset is **RWF-2000**.

The objective is to build a working, testable pipeline that detects violent versus non-violent activity while preserving spatial evidence about people, tracks, masks/kinematics and objects.

### Explicitly out of scope

The following are not part of this implementation:

- Goon/instigator detection or role attribution.
- Victim, bystander or participant classification.
- Interaction graph and role reasoning.
- Ground-plane projection or multi-camera fusion.
- Cross-camera association or appearance-based Re-ID.
- Evidence-packet federation.
- Federated training, aggregation and client clustering.
- Site gateway logic.
- Operator console and human feedback loop.
- Audio processing.
- Identity recognition or face recognition.
- Any claim that the system infers intent, political affiliation or suspiciousness.

The system may produce per-person tracks and temporal features, but those outputs must not be interpreted as responsibility or causality.

---

## 2. Problem definition for this phase

Given a single video clip from RWF-2000, the system must:

1. Ingest and timestamp the video.
2. Estimate input clarity and preserve quality metadata.
3. Select informative frames using motion gating without losing short violent events.
4. Detect people and optionally relevant objects.
5. Generate silhouette or box-kinematic evidence.
6. Maintain person tracks through time.
7. Convert spatial observations into a stable Track Token representation.
8. Encode scene and track history over a temporal window.
9. Output a scene-level `violent` or `non_violent` prediction with confidence.

The system must also expose intermediate outputs for debugging and evaluation:

- Selected and skipped frame indices.
- Motion scores and motion regions.
- Clarity score and quality components.
- Person detections.
- Object detections.
- Masks or mask availability status.
- Track IDs and track lifecycle events.
- Track observations versus Kalman-predicted observations.
- Centroids, velocity and acceleration.
- Temporal window metadata.
- Scene violence score.

### 2.1 Formal output contract

For each processed video window:

```text
WindowOutput:
  window_id
  source_id
  start_timestamp
  end_timestamp
  label: violent | non_violent
  violence_score: float in [0, 1]
  temporal_confidence: float in [0, 1]
  clarity_score: float in [0, 1]
  processed_frame_count
  skipped_frame_count
  tracks: list[TrackSummary]
  objects: list[ObjectEvidence]
  diagnostics: WindowDiagnostics
```

For each person observation:

```text
TrackToken:
  track_id
  timestamp
  frame_index
  box: [x1, y1, x2, y2]
  detector_confidence
  observed: bool
  visibility_score
  mask_available: bool
  silhouette_features: optional vector
  centroid: [cx, cy]
  velocity: [vx, vy]
  acceleration: [ax, ay]
  clarity_score
  object_evidence: optional list
```

`track_id` identifies a local temporal track only. It is not an identity and must not be reused as a person identity across cameras or deployments.

### 2.2 Window and clip semantics

RWF-2000 clips are short (roughly 5 seconds), so the window policy must be fixed early because it affects the loader, the temporal model and evaluation.

- A window is defined by `window_seconds` and `stride_seconds` in configuration. The initial default is one window per clip (window covers the whole clip), with shorter overlapping windows treated as an experiment, not a default.
- Windows are defined on source timestamps, not on processed-frame counts, so gating does not change window boundaries.
- The clip-level prediction used for RWF-2000 evaluation is an explicit, configured aggregation of window scores (initially mean, with max as an ablation). It must be defined before any metric is reported.
- Temporal training uses a fixed number of time steps per window with a documented padding/truncation policy, plus a per-step time delta so irregular (gated) sampling is visible to the model.

---

## 3. Dataset strategy: RWF-2000

### 3.1 Primary role of RWF-2000

RWF-2000 is the primary dataset for:

- Binary fight/non-fight classification.
- Temporal violence scoring.
- End-to-end scene-level evaluation.
- Controlled robustness experiments.
- Motion-gating recall analysis.

The dataset does not provide complete annotations for every module in the bounded architecture.

### 3.2 What RWF-2000 does not provide

RWF-2000 should not be treated as a fully annotated detection or tracking dataset. It does not reliably provide:

- Person bounding-box ground truth.
- Person identity or track ground truth.
- Silhouette masks.
- Weapon/object boxes.
- Weapon ownership.
- Instigator labels.
- Calibrated multi-view footage.

Therefore:

- Scene classification is supervised directly with RWF-2000 labels.
- Detection, segmentation and tracking use pretrained components initially.
- Module-level spatial evaluation is performed on appropriate auxiliary datasets when added.
- RWF-2000 spatial outputs are treated as intermediate evidence unless independently validated.

### 3.3 Dataset preparation requirements

Before model development:

1. Verify the official split and avoid leakage. RWF-2000 is expected to ship with train and validation folders only (to be confirmed on the downloaded copy). Treat the official validation folder as the held-out test set, and carve a separate validation set out of the official train folder using a seeded, stratified split that is recorded in the manifest. Tune thresholds and hyperparameters on this carved validation set only.
2. Build a manifest containing:
   - Clip path.
   - Label.
   - Frame count.
   - Source FPS.
   - Resolution.
   - Duration.
   - Decode status.
3. Detect and report corrupt, empty or undecodable clips.
4. Define a deterministic frame-sampling policy.
5. Keep all augmentation and degradation settings reproducible through configuration and seeds.
6. Ensure train-only transforms are not applied to validation/test clips.
7. Check for duplicate or near-duplicate clips across splits (for example by file hash and a cheap frame-signature comparison) and report any found.
8. Record the chosen window policy (section 2.2) in the manifest metadata.

### 3.4 Simulated camera conditions

RWF-2000 will be transformed to approximate heterogeneous edge inputs:

- Original/high-quality input.
- 720p-equivalent.
- 480p-equivalent.
- 240p-equivalent.
- Blur.
- H.264-like compression.
- Low-light gamma adjustment.
- Sensor noise.
- Frame dropping.
- Synthetic partial occlusion.

These are controlled robustness conditions, not real multi-camera data.

The degradations are produced by a dedicated, seeded degradation toolkit (built in Phase 2) so that every degraded variant can be regenerated exactly from configuration. Each degradation has a severity parameter and is applied identically to every frame of a clip unless it is explicitly temporal (frame dropping).

---

## 4. Proposed technology direction

The implementation should remain modular so components can be replaced without changing downstream contracts.

### 4.1 Core software direction

- Python.
- PyTorch for model code and training.
- OpenCV or an equivalent reliable video-decoding layer.
- A configurable detector interface.
- ByteTrack-style tracking with Kalman prediction.
- GRU or ConvGRU for the initial temporal model.
- PyTest for unit and integration tests.

The exact package selection should follow the project environment and dependency policy when implementation begins. No package should be added merely because it is mentioned here; it should be justified by the selected component and validated after installation.

### 4.2 Initial model defaults

These are defaults, not irreversible decisions:

| Area | Initial direction | Later alternatives |
|---|---|---|
| Clarity | Deterministic quality score | Learned no-reference IQA |
| Motion gate | Low-resolution frame differencing | H.264 motion vectors, MOG2, small optical flow |
| Person detector | Pretrained RT-DETR or YOLO interface | Crowd-aware detector, lighter Tier-C model |
| Object evidence | Optional pretrained/open-vocabulary interface | Closed-set weapon detector |
| Masks | Segmentation head or pretrained lightweight segmentation | EdgeSAM/MobileSAM |
| Tracking | ByteTrack-style tracker | OC-SORT, BoT-SORT |
| Temporal model | GRU | TCN, Transformer, Mamba-style model |

The first implementation should use one concrete default per interface, not multiple competing implementations at once.

The concrete person-detector default is chosen at the start of Phase 4 after a short feasibility check (available hardware, latency, memory, weight license) and is recorded in configuration. RT-DETR is preferred if the hardware allows; otherwise a small YOLO model is used behind the same interface.

In this phase, pretrained components (detector, segmentation, tracker logic) are used frozen. The only trained parts are the baseline temporal model and the Block 6 temporal/fusion head.

---

## 5. Architecture and module contracts

### 5.1 Ingestion

**Input**

- Video path or stream adapter.
- Decode configuration.

**Output**

- Frames.
- Frame indices.
- Timestamps.
- Source FPS and resolution.
- Decode diagnostics.

**Requirements**

- Deterministic replay from a file.
- Explicit handling of decode failure.
- No silent frame substitution.
- Stable timestamp behavior after frame dropping.

### 5.2 Block 1 — Clarity estimation

**Purpose:** quantify the reliability of the current input without pretending that clarity is a model prediction.

Candidate signals:

- Normalized resolution.
- Sharpness, such as Laplacian variance.
- Exposure or luminance statistics.
- Noise/compression indicators.
- Median detected-person height when detections exist.

**Output**

```text
ClarityReport:
  score: float in [0, 1]
  resolution_component
  sharpness_component
  exposure_component
  compression_component
  person_scale_component
  source_profile
```

The score must be deterministic for a fixed frame/window and must expose its components for debugging.

### 5.3 Block 2 — Motion gating

**Purpose:** avoid processing redundant frames while retaining violent-frame recall.

Initial behavior:

- Compute low-cost frame difference on downsampled grayscale frames.
- Compare against the last processed frame.
- Process a frame when motion exceeds threshold.
- Enforce a minimum keep-alive rate.
- Enter burst mode when motion rises sharply.
- Record all skipped and retained frame indices.

The gate must not change the semantic timestamp of an observation. When a frame is skipped, later modules may use tracker prediction, but that prediction must be marked `observed = false`.

### 5.4 Block 3 — visual backbone and person detection

**Purpose:** provide person detections and reusable visual features.

Required detector output:

```text
Detection:
  box
  confidence
  class_name
  feature_reference or pooled_feature
  frame_index
  timestamp
```

The detector interface must support:

- Batch inference where available.
- Empty detections.
- Low-confidence detections.
- Different input resolutions.
- CPU and accelerator execution where supported.

The detector must be evaluated independently from the RWF-2000 classifier.

### 5.5 Block 4 — threat/object evidence and masks

This block has two independently switchable paths:

1. Object/threat evidence.
2. Silhouette/mask evidence.

Object evidence is auxiliary. It must not directly produce the scene violence decision.

Object output:

```text
ObjectEvidence:
  object_id
  category
  box
  confidence
  frame_index
  associated_track_id: optional
```

Mask output:

```text
MaskEvidence:
  mask_or_reference
  confidence
  visible_fraction
  contour_features
  frame_index
```

Fallback hierarchy:

1. Valid silhouette/mask.
2. Box geometry and kinematics.
3. No spatial feature, with explicit missing-modality metadata.

The system must not silently fabricate masks when a segmentation result is unavailable.

### 5.6 Support path — silhouette and box kinematics

Per-track features may include:

- Box width, height and area.
- Aspect ratio.
- Centroid.
- Displacement.
- Velocity.
- Acceleration.
- Optional jerk.
- Mask area.
- Visible fraction.
- Contour descriptors.
- Neighbor-independent local motion.

This representation describes observed movement. It must not be named or exposed as an instigator score.

### 5.7 Block 5 — multi-object tracking

**Purpose:** maintain local temporal continuity.

Initial tracker requirements:

- Kalman prediction.
- Detection-to-track association.
- Track creation and termination.
- Occlusion tolerance.
- Low-confidence detection handling.
- Predicted-only observations.
- Track history retention.
- No appearance Re-ID.

Track lifecycle events:

```text
TrackStarted
TrackUpdated
TrackPredicted
TrackRecovered
TrackEnded
```

### 5.8 Track Token construction

Track Token construction must be a dedicated adapter between tracking and temporal modelling. This prevents downstream code from depending directly on the detector or tracker implementation.

The adapter must:

- Normalize feature dimensions.
- Preserve source metadata.
- Mark unavailable modalities.
- Keep observed/predicted state.
- Validate coordinate ranges.
- Reject malformed boxes rather than silently passing them downstream.

### 5.9 Block 6 — temporal context encoding

The first version should support:

- Scene-level temporal features from pooled frame/backbone features.
- Track-level temporal features from Track Token sequences.
- Masking of missing observations.
- Masking of predicted-only updates.
- Variable numbers of tracks.
- Fixed-length training windows or a documented padding policy.

The required prediction is:

```text
TemporalOutput:
  violence_score
  predicted_label
  confidence
  window_metadata
```

The temporal model must not output instigator or role labels.

---

## 6. Phase-wise implementation plan

Each phase has a deliverable, dependencies, unit tests, integration tests and an exit gate. A phase is complete only when its gate passes.

### Phase 0 — Repository and experiment foundation

**Goal:** establish reproducible project structure before implementing model logic.

Tasks:

1. Define package boundaries around ingestion, clarity, gating, detection, masks, tracking, tokens, temporal modelling and evaluation.
2. Define configuration loading and validation.
3. Define typed data contracts.
4. Define logging and diagnostics conventions.
5. Define deterministic seeds and experiment identifiers.
6. Define artifact directories for manifests, checkpoints, metrics and reports.
7. Record the verified RWF-2000 split and dataset license/provenance information.
8. Record the execution environment (OS, Python, library versions, CPU/GPU, memory) in every run, and decide the dependency policy (pinned lockfile) before adding packages.
9. Build a deterministic synthetic video generator (for example moving rectangles with known positions, static scenes, sudden-motion scenes, and scenes with scripted occlusion) so that gating, tracking and kinematics can be tested against known ground truth without any dataset or model download.
10. Define test tiers and markers (unit, integration, model-smoke, slow) and a single documented command for each tier.

Unit tests:

- Configuration defaults load correctly.
- Invalid configuration values fail explicitly.
- Data-contract validation rejects malformed boxes and scores.
- Seeds produce reproducible sampling.
- Manifest parsing handles valid and invalid entries.
- The synthetic video generator is deterministic for a given seed and reports its ground-truth boxes and timestamps.
- Environment record is written and contains all required fields.
- Frame timestamps are monotonic and match the declared FPS for the fixture videos.

Integration tests:

- A tiny fixture video can be opened, decoded and represented in the manifest.
- Configuration can instantiate a minimal pipeline without model weights.
- A synthetic video and a real tiny fixture both pass through ingestion and produce the same diagnostics schema.

Exit gate:

- A clean environment can run a smoke command that loads configuration, reads a fixture video and writes diagnostics.
- The unit tier runs without network access, model weights or the dataset.

### Phase 1 — Dataset ingestion and baseline temporal classifier

**Goal:** produce a reliable RWF-2000 scene-level baseline before adding spatial complexity.

Tasks:

1. Implement dataset manifest loading.
2. Implement train/validation/test loaders.
3. Implement deterministic frame sampling.
4. Implement basic preprocessing.
5. Implement a minimal temporal model.
6. Train and evaluate a baseline on RWF-2000.
7. Save metrics and predictions per clip.
8. Implement trivial reference baselines (majority class and a simple motion-energy classifier) as a sanity floor.
9. Implement the configured window-to-clip aggregation from section 2.2.

Unit tests:

- Dataset labels map only to the expected two classes.
- Sampling never requests frames outside the clip.
- Padding and truncation are deterministic.
- Train transforms do not leak into evaluation.
- Model output shape is correct for batch size one and greater.
- Binary loss and metric calculations handle both classes.
- Train, carved-validation and test sets are pairwise disjoint by clip and by file hash.
- Window-to-clip aggregation is deterministic and handles a single window.
- The majority baseline gives the expected metrics on a balanced fixture.

Integration tests:

- A loader batch passes through preprocessing and the temporal model.
- A tiny overfit test confirms that the model can memorize a very small fixture subset.
- Evaluation output can be reloaded and reproduced.
- Training with the same seed twice on the smoke subset gives matching metrics within tolerance.

Exit gate:

- End-to-end baseline training and evaluation complete on a small smoke subset.
- Full baseline metrics are recorded.
- No dataset leakage is observed.
- The learned baseline is compared against the trivial baselines, and any case where it does not beat them is documented.

### Phase 2 — Clarity profiler and degradation toolkit

**Goal:** add quality metadata without coupling quality estimation to downstream predictions, and build the seeded degradation toolkit needed to verify it.

Tasks:

1. Implement individual clarity signal calculators.
2. Normalize signals with documented bounds.
3. Combine them into a score in `[0, 1]`.
4. Support frame-level and window-level aggregation.
5. Store all components in diagnostics.
6. Implement the seeded degradation toolkit from section 3.4 (resolution, blur, compression, low-light, noise, frame dropping, synthetic occlusion), each with a severity parameter.
7. Report `person_scale_component` as explicitly unavailable until detections exist (it is filled in after Phase 4), rather than substituting a default.

Unit tests:

- Blank frames receive a low or explicitly invalid quality result.
- Sharpness changes affect the sharpness component predictably.
- Resolution changes affect only the intended component.
- Scores remain within `[0, 1]`.
- NaN, infinity and missing metadata are handled explicitly.
- Identical input produces identical output.
- Each degradation is deterministic for a fixed seed and severity.
- Increasing severity changes the targeted property monotonically (for example blur lowers sharpness, downscaling lowers resolution).
- Output shape, dtype and timestamps are preserved except where the degradation intentionally changes them (resolution, frame dropping).
- Missing person-scale input yields an explicit unavailable marker and a documented renormalization of the score.

Integration tests:

- Ingestion metadata flows into the clarity profiler.
- Clarity reports are attached to window outputs.
- Quality metadata survives serialization and reloading.
- A degraded clip regenerated from the saved configuration is identical to the original degraded clip.

Exit gate:

- The pipeline reports clarity scores for every test window.
- Score distributions are inspected across original and degraded RWF-2000 variants.
- The clarity score decreases with degradation severity on average, and cases where it does not are documented.

### Phase 3 — Motion gate

**Goal:** reduce redundant computation while preserving violence recall.

Tasks:

1. Implement low-resolution frame differencing.
2. Add threshold and minimum keep-alive configuration.
3. Add motion burst behavior.
4. Emit selected/skipped frame records.
5. Add a no-gate mode for controlled comparison.
6. Connect skipped-frame metadata to tracker prediction behavior later.
7. Define the frame-level recall protocol before measuring: because RWF-2000 has no frame-level labels, use stated proxies such as agreement between gated and ungated clip predictions, and coverage of high-motion frames (top motion-energy frames per clip) by retained frames. Tune gate thresholds on the carved validation set only.

Unit tests:

- Identical frames are classified as low motion.
- Large frame changes are classified as motion.
- Keep-alive processing occurs under static input.
- Burst mode increases processing frequency after a motion spike.
- First-frame behavior is deterministic.
- Output indices are ordered and non-duplicated.
- Threshold edge cases behave correctly.
- On synthetic videos, a scripted motion burst is retained and a long static segment is skipped except for keep-alive frames.
- Retained frames keep their original timestamps and indices.

Integration tests:

- Ingestion → clarity → motion gate produces a valid selected-frame stream.
- Gated and ungated paths produce aligned timestamps.
- The gate can process a short fixture without deadlock or frame loss.
- Gate decisions are identical across repeated runs of the same clip.

Evaluation:

- Percentage of skipped frames.
- Violent-frame recall proxy.
- Scene classification change from gating.
- Processing FPS and latency.

Exit gate:

- Gating reduces processing on static content.
- Violent clips do not show unacceptable recall loss relative to ungated inference.
- Any recall trade-off is documented before proceeding.
- The recall protocol and its limits (proxy, not ground truth) are written down before numerical targets are set.

### Phase 4 — Visual backbone and person detection

**Goal:** add spatial person evidence and a replaceable detector contract.

Tasks:

1. Implement detector interface.
2. Integrate one pretrained detector.
3. Normalize detector outputs into the common Detection contract.
4. Add feature pooling for scene and person regions.
5. Support empty and low-confidence detections.
6. Record detector latency and memory diagnostics.
7. Fix the detector default after the feasibility check described in section 4.2.
8. Map boxes back to original-frame coordinates after any resizing or letterboxing, and filter to the person class with a configurable minimum box size.

Unit tests:

- Detector outputs are normalized to valid coordinates.
- Invalid boxes are rejected.
- Confidence values are range-checked.
- Empty detections produce valid empty outputs.
- Class filtering behaves deterministically.
- Feature pooling has the expected shape.
- Resize/letterbox mapping returns boxes to the correct original coordinates (verified on synthetic frames with known boxes).
- Boxes below the minimum size are filtered consistently.
- Mock detector outputs and real detector outputs satisfy the same contract tests.

Integration tests:

- Motion-gated frames reach the detector with correct timestamps.
- Detector outputs can be rendered or serialized.
- Detector outputs can be consumed by the mask and tracker adapters.
- CPU smoke inference completes on a fixture clip.
- Detections on a degraded clip stay within the original frame bounds after coordinate mapping.

Evaluation:

- Detector latency.
- Number of detections per frame.
- Detection stability under degraded input.
- Independent detector metrics when an annotated detection dataset is introduced.

Exit gate:

- A detector can be swapped through configuration without changing downstream contracts.
- Spatial detections are available for every selected frame.

### Phase 5 — Threat/object evidence and silhouette support

**Goal:** add optional object and mask evidence with explicit fallbacks.

Tasks:

1. Implement object-evidence interface.
2. Implement one object detector or a stub adapter with clearly marked unavailable status.
3. Implement mask/silhouette interface.
4. Implement box-kinematic fallback.
5. Calculate visibility and contour features where masks exist.
6. Record missing-modality reasons.

Unit tests:

- Object boxes are validated and normalized.
- Unknown object classes are handled explicitly.
- Mask dimensions match the associated frame/box contract.
- Invalid masks trigger fallback rather than silent success.
- Visibility fraction remains within `[0, 1]`.
- Box-only kinematics work when masks are absent.
- Object evidence never directly sets the violence label.

Integration tests:

- Person detections flow into mask extraction.
- Masks or fallback geometry flow into Track Token construction.
- Object detections can be associated with nearby local tracks without requiring Re-ID.
- Removing the mask stream still produces a valid temporal input.
- Removing the object stream still produces a valid temporal input.

Evaluation:

- Mask availability rate.
- Fallback rate.
- Object evidence precision/recall on an appropriate object dataset.
- False positives on harmless object categories where annotations exist.

Exit gate:

- Both mask-present and mask-absent paths work.
- The pipeline remains functional when object evidence is unavailable.

### Phase 6 — Multi-object tracking and Track Tokens

**Goal:** provide stable local temporal person representations.

Tasks:

1. Implement tracker adapter.
2. Add Kalman prediction.
3. Add association and lifecycle handling.
4. Mark observed versus predicted-only observations.
5. Add track history storage.
6. Implement Track Token construction and validation.
7. Add visualization/debug export for tracks.
8. Express box and kinematic features in frame-normalized coordinates (in addition to pixels) so tokens are comparable across resolutions.
9. Add an offline Track Token cache per window, keyed by configuration hash and source clip, so the temporal model can be trained and ablated without re-running detection and tracking.

Unit tests:

- A stationary detection retains a stable track ID.
- A moving detection produces sensible centroid motion.
- Short missing detections create predicted-only observations.
- Stale tracks terminate after the configured timeout.
- Track IDs are unique within a clip.
- Track tokens preserve observation state.
- Velocity and acceleration are computed with correct timestamp spacing.
- Empty frames do not corrupt active tracks.
- Malformed detections cannot create malformed tokens.
- Lifecycle events occur in a valid order (a track cannot end before it starts or be recovered without having been predicted).
- Identical input produces identical track IDs and tokens.
- Normalized coordinates stay in range and are unchanged when the same synthetic scene is rendered at two resolutions.
- Token cache round-trips exactly and is invalidated when the configuration hash changes.

Integration tests:

- Detector → tracker → Track Token flow works on a fixture sequence.
- Motion-gated skipped frames result in marked predictions, not fabricated observations.
- Mask features and object evidence attach to the correct local track when available.
- Track histories can be passed to the temporal encoder.
- A complete clip produces stable track diagnostics.
- On a synthetic scripted-occlusion video, the track survives the occlusion within the configured timeout and no new ID is created for the same moving object.
- Tokens read from the cache are identical to tokens produced live.

Evaluation:

- Track duration.
- Track fragmentation.
- ID switches.
- Observed/predicted ratio.
- HOTA/IDF1/MOTA when evaluated on an annotated tracking dataset.

Exit gate:

- Tracks remain usable under moderate frame dropping and synthetic occlusion.
- Track Token serialization and replay are deterministic.
- Tokens for the full train and test sets can be cached, with failed clips listed explicitly.

### Phase 7 — Temporal context encoding with spatial evidence

**Goal:** make the bounded pipeline perform end-to-end scene-level violence classification.

Tasks:

1. Define scene feature pooling.
2. Define per-track sequence batching and masking.
3. Implement GRU or ConvGRU.
4. Mask predicted-only observations appropriately.
5. Fuse scene, track, kinematic, mask and optional object features.
6. Produce violence score and binary label.
7. Support modality ablation.
8. Add the per-step time delta as a model input so irregular (gated) sampling is visible to the temporal model.
9. Train from the cached tokens with detector and tracker frozen; choose the decision threshold on the carved validation set only, and report calibration (for example expected calibration error) alongside accuracy.
10. Define `temporal_confidence` explicitly (for example distance of the score from the threshold, or a calibrated probability) and document it.

Unit tests:

- Variable-length sequences are padded and masked correctly.
- Predicted-only observations do not update the track state when configured not to.
- Empty-track windows still produce valid scene predictions.
- Missing mask/object features are handled through modality masks.
- Temporal output is in the defined schema.
- Score and label thresholds are deterministic.
- Batch and single-window inference agree within numerical tolerance.
- Changing only the time deltas changes the output when expected, and uniform deltas match the ungated case.
- Calibration and confidence calculations are in range and deterministic.
- Threshold selection never reads the test set.

Integration tests:

- Ingestion → clarity → gating → detector → masks/kinematics → tracking → tokens → temporal model.
- Ungated and gated pipelines produce comparable output schemas.
- Mask-enabled and box-only paths both complete end to end.
- Object-enabled and object-disabled paths both complete end to end.
- Checkpoint save/load preserves predictions.
- Training from the token cache and running the live pipeline give matching predictions within tolerance.
- A full fixture clip produces spatial and temporal outputs together.

Evaluation:

- Accuracy, precision, recall, F1 and ROC-AUC on RWF-2000.
- Per-class recall.
- Performance under simulated degradation.
- Comparison with no-gating and scene-only baselines.
- Contribution of tracks, kinematics, masks and object evidence.
- Calibration quality of the violence score.

Exit gate:

- The entire bounded pipeline runs on RWF-2000.
- It produces a violent/non-violent result plus spatial diagnostics.
- All optional paths degrade explicitly rather than failing silently.

### Phase 8 — Robustness, ablation and performance validation

**Goal:** validate the engineering claims of the bounded pipeline without making unsupported goon claims.

Experiments:

1. No gate versus motion gate.
2. Scene-only versus scene plus tracks.
3. Box-only versus box plus kinematics.
4. Box/kinematics versus silhouette features.
5. Object stream absent/present.
6. Original versus degraded resolution.
7. Original versus blur/compression.
8. Original versus frame dropping.
9. Original versus synthetic occlusion.
10. CPU versus available accelerator.
11. Gate threshold sweep (frames skipped versus accuracy and recall proxy).
12. Clip-level aggregation choice (mean versus max).

Stochastic experiments are repeated over several seeds (at least three where compute allows), and results are reported as mean and spread rather than a single run.

Unit tests:

- Every experiment configuration validates before execution.
- Metric aggregation handles missing or failed runs explicitly.
- Results include configuration hash and checkpoint identifier.
- Multi-seed aggregation computes mean and spread correctly and excludes failed seeds explicitly.
- Latency measurements record the environment, warm-up policy and sample count.

Integration tests:

- Experiment runner can execute all ablations on a small smoke subset.
- Metrics, predictions and diagnostics are linked to the same run identifier.
- A failed run is reported as failed, not converted into a zero score.
- Re-running one ablation from its saved configuration reproduces its metrics within tolerance.

Exit gate:

- Results are reproducible from saved configuration and checkpoint.
- The report distinguishes measured results from targets and hypotheses.
- Differences between ablations are interpreted against seed-to-seed variation, not single runs.

---

## 7. Testing strategy

### 7.0 Test tiers and fixtures

| Tier | Needs dataset | Needs model weights | Purpose |
|---|---|---|---|
| Unit | No | No | Contracts, numerics, edge cases |
| Integration | No (fixtures only) | No (mocks) | Module boundaries and data flow |
| Model-smoke | No (fixtures only) | Yes | One real pretrained model path works |
| Slow/full | Yes | Yes | RWF-2000 training, evaluation, ablations |

- The unit and integration tiers must pass on every change. Model-smoke runs wherever weights are available. The slow tier runs only for phase exit gates and ablations.
- Fixtures are of two kinds: synthetic videos with known ground truth (preferred for gate, tracker and kinematics tests) and one or two tiny real clips (for decoding and end-to-end checks).
- Every acceptance criterion in section 8 should map to at least one named test. Coverage is reported but is not used as the sole gate.

### 7.1 Unit testing principles

Unit tests should be fast, deterministic and independent of large datasets or model downloads.

They should cover:

- Data contracts.
- Numerical ranges.
- Empty inputs.
- Missing optional modalities.
- Invalid inputs.
- Timestamp/frame-index behavior.
- Boundary thresholds.
- Batch-size variation.
- CPU execution.

Tests must not depend on live camera streams or network access.

### 7.2 Integration testing principles

Integration tests should use:

- Tiny fixture videos.
- Synthetic frame sequences.
- Small deterministic tensors.
- Mock detector outputs where model inference is not the subject of the test.
- One real pretrained model smoke test where the environment supports it.

The tests should verify module boundaries, not just individual functions.

### 7.3 End-to-end testing

At least one end-to-end test must:

1. Load a fixture clip.
2. Decode frames.
3. Calculate clarity.
4. Run motion gating.
5. Run spatial inference.
6. Track detections.
7. Construct Track Tokens.
8. Run temporal inference.
9. Produce a valid WindowOutput.

At least one additional end-to-end test must run with:

- No masks.
- No object evidence.
- Skipped frames.
- No detected persons.

The system must produce explicit degraded outputs in all four cases.

### 7.4 Regression testing

After each phase:

- Preserve a small golden fixture set.
- Store expected schema and key numerical tolerances.
- Compare output shapes, labels and diagnostics.
- Do not require bit-exact neural-network outputs across hardware unless the environment guarantees it.

Regression tests must detect:

- Changed frame sampling.
- Changed track lifecycle.
- Changed output schema.
- Unexpected recall degradation.
- Silent disappearance of optional evidence.

---

## 8. Acceptance criteria

The bounded implementation is acceptable when all of the following are true:

### Functional

- RWF-2000 clips can be loaded using a verified split.
- A complete clip can pass through the bounded pipeline.
- The pipeline outputs violent/non-violent classification.
- Person detections, tracks and temporal diagnostics are available.
- Mask and object paths are optional and explicit.
- No module emits a goon or instigator label.

### Reliability

- Decode failures and invalid inputs are reported explicitly.
- Missing modalities are represented as missing, not replaced by success-shaped defaults.
- Predicted-only tracker observations are distinguishable from observed detections.
- Outputs have validated coordinate and confidence ranges.

### Testing

- Unit tests cover each module and its boundary conditions.
- Integration tests cover the complete data path.
- End-to-end smoke tests complete on a small fixture dataset.
- RWF-2000 evaluation completes on the selected split.
- Ablation results are reproducible from saved configuration.

### Performance

The following are targets to measure, not assumed achievements:

- Meaningful frame reduction on mostly static clips.
- Violent-frame recall close to the ungated baseline.
- Stable tracking under moderate frame skipping.
- Predictable latency and memory usage.
- Acceptable scene-level classification performance relative to the baseline.

The exact numerical thresholds should be finalized after the ungated baseline is measured. It is not valid to claim a target such as 98% retained violent-frame recall before defining the frame-level evaluation protocol.

---

## 9. Evaluation and reporting rules

Every report must distinguish:

1. Directly measured results.
2. Results from pretrained components.
3. Results from auxiliary datasets.
4. Simulated degradation results.
5. Proposed future targets.

RWF-2000 results should be reported as scene-level violence results. Person detection, segmentation, tracking and object results should be reported separately unless a valid annotation protocol exists.

The report must not claim:

- Instigator detection.
- Causal responsibility.
- Intent inference.
- Privacy guarantees beyond the implemented behavior.
- General multi-camera performance.
- Federated-learning performance.

---

## 10. Deferred expansion path

Once the bounded pipeline is stable, future work can be added in this order:

1. Evidence-packet schema.
2. Ground-plane multi-view fusion.
3. Operator review interface.
4. Interaction graph and role reasoning.
5. Instigator-specific annotations.
6. Federated client simulation.
7. Clarity-aware clustered aggregation.
8. Human feedback and semi-supervised training.

Each future module should consume the existing Track Token or WindowOutput contracts rather than bypassing them.

---

## 11. Final implementation boundary

The first implementation is a **spatially grounded temporal violence detector**:

- Spatially grounded because it detects people, objects and local tracks.
- Temporally grounded because it models frame and track history.
- Not an instigator detector because it does not infer roles, responsibility or causality.

The most important engineering objective is to make this bounded pipeline reliable, observable and testable before adding the more sensitive and difficult goon-detection stage.

---

## 12. Revision notes

This revision adds the following without changing the phase order or scope:

- Window and clip semantics (2.2), and train/validation/test handling for RWF-2000 (3.3).
- Ownership of the degradation toolkit (Phase 2) and a detector-selection step (Phase 4).
- Synthetic ground-truth fixtures and test tiers (Phase 0, 7.0).
- Trivial baselines (Phase 1) and a recall-proxy protocol (Phase 3).
- Coordinate mapping tests (Phase 4), normalized coordinates and a Track Token cache (Phase 6).
- Time-delta input, calibration and threshold discipline (Phase 7).
- Multi-seed reporting and two more ablations (Phase 8).
