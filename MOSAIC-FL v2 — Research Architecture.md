# MOSAIC-FL v2 — Research Architecture

Oct 14, 2026 · @Arshad Khatib

## Problem and gap analysis

The v1 draft covers your sketch well but stops short of solving the problem in three places: it never defines "goon" operationally, it makes multi-device inference optional, and its HCI evaluation is a single module. v2 closes those gaps and shrinks v1's thirteen modules to one core pipeline plus optional plug-ins.

**Problem, restated.** K edge cameras differ in resolution, frame rate, compute and image clarity. Without sharing raw video, they must jointly learn to output (1) violent or non-violent per time window, (2) which individuals are instigating, and (3) which objects, such as weapons, are involved and who holds them. Faces and bodies are partly hidden throughout.

| # | Gap in v1 | Why it matters | v2 fix |
| --- | --- | --- | --- |
| 1 | "Goon" is never defined. MIL on video-level labels only says which clips are violent | The model learns to flag anyone in a violent clip, so it cannot separate instigator from victim or bystander | Define the instigator as the track that first initiates contact or aggression (temporal precedence plus outgoing interaction). Add an initiation-order term to the graph and a small annotated validation set |
| 2 | Multi-view fusion is optional, and cross-camera person matching is ruled out (no Re-ID) | The problem is about multiple devices. Without cross-camera association there is nothing to fuse | Fuse on a shared ground-plane grid through homography, so no appearance Re-ID is needed. Devices send per-cell evidence, not identities |
| 3 | FL clients need labels, but v1 only gestures at operator feedback and teacher labels | Real clients have almost no labels, so local training stalls | Semi-supervised FL: confidence-gated pseudo-labels from the on-device model plus operator confirmations. Heavy teacher models run only on public data |
| 4 | Heterogeneity is treated mostly as label skew | Resolution and compression are feature shift. They break normalisation statistics and detector scale | Clarity-conditioned adapters, private norm layers, and a resolution-consistency loss across simulated downscales |
| 5 | Occlusion robustness is asserted ("silhouettes degrade gracefully"), not designed | Masks merge or fragment in dense crowds, so the skeleton-free claim may not hold | An occlusion-aware token (visible fraction, mask confidence) plus a box-kinematics fallback. The ablation decides whether silhouettes beat skeletons |
| 6 | The HCI contribution is one console module | This is an HCI project, and reviewers will look for a user-facing study | An operator study on alert design, explanation, trust calibration and privacy (section 7) |
| 7 | Seven novelty claims | Too diffuse. Each needs its own evidence | Three headline contributions. The rest become design choices |
| 8 | Dual use appears only as one risk row | Flagging protesters has real harms if wrong | Human-confirmed alerts only, no identity, audit logs. Measure the false-alarm cost on peaceful crowds |

## What your papers and TraceNet contribute

TraceNet already contains the hardest-to-build piece of v2, a calibrated ground-plane multi-camera fusion pipeline, and one of its own measurements supports dropping appearance Re-ID. The two PDFs you attached supply the weak-supervision recipe and the crowd-detection argument.

### From the attached papers

| Source | What it gives MOSAIC-FL | Limit for this problem |
| --- | --- | --- |
| Choqueluque-Roman and Camara-Chavez, Sensors 2022 | Weak supervision from video-level labels: person tubes from a detector, motion and a tracker, scored by multiple-instance learning. Reports 88.7% on RWF-2000, 92.88% on RLVSD and 97.3% on Hockey Fight | Tubes are linked from detector boxes, so (my inference) they inherit the detector's failures in dense crowds. No weapons, no federation, no device heterogeneity |
| Chu et al., CrowdDet, CVPR 2020 | Explains why NMS-based person detectors drop overlapped people, and fixes it with EMD loss and Set NMS (+4.9 AP on CrowdHuman with an FPN-Res50 baseline) | A detector paper only. It motivates the detector choice, not the violence logic |
| Your reference list, items 1 to 5 (federated action recognition) | The FL toolbox. FedFSLAR runs few-shot 3D-CNN meta-learning under FedAvg and FedProx. The personalised-FL paper splits the model into private and public parameters and reports 97.79% on UCF-101 under non-IID data. The DFKI paper trains LSTM and Transformer models on FastPose skeletons for factory gestures with five clients. CF-STGAT clusters clients by attention-layer parameters (PCA then K-means) and beats FedAvg by 0.84 to 7.98 top-1 points on NTU RGB+D 60 and 120 | All are generic action recognition, not violence, crowds or weapons, and the skeleton ones inherit pose failure under occlusion. I could not open FedDet (item 6, poisoning), so it rests on your v1 notes. The Pose to Protect source I found is a 2025 MSc thesis (SIU Carbondale), while v1 cites it as an ICCV workshop paper. Check which one your CVF link points to before citing |

### From TraceNet

| TraceNet asset | Reuse in MOSAIC-FL | Caveat |
| --- | --- | --- |
| LUMPI Fusion Replay (`analytics/lumpi/replay.py`): back-projects each box's foot point through camera calibration onto a ground plane and merges per-camera tracks by co-temporal ground distance. Its notes report 8,385 detections becoming 123 local tracks and 70 identities on one experiment | The base of the cross-view module. It already does geometry-based association without appearance | LUMPI clips are 3.2 s, contain no violence, and use synthetic identity embeddings. They test association geometry only |
| Measured finding (task 5.12): CLIP similarity between different people is 0.86 to 0.96 | Direct evidence for skipping appearance Re-ID, which also suits occluded faces | Measured on TraceNet's own CCTV data, not a controlled study |
| YOLO plus ByteTrack tracklets with best crop, CLIP embedding and BLIP caption | Starting point for the detector, tracker and Track Token | TraceNet standardises every video to 720p at 10 fps. MOSAIC must not, because heterogeneity is the subject. Replace that step with a per-device clarity profile |
| VideoMAE-L (UCF-Crime) assault scanner with 2-second windows | A clip-level baseline and a teacher for pseudo-labels | Its own check: staged fight flagged at 0.97, a street fight missed at 0.50, a normal clip was correctly left unflagged (0.51). Three clips, not a benchmark. No per-person output |
| Chain-snatching rules: proximity spike, victim-fall aspect-ratio flip, chase-vector cosine (kinematics are off by default there) | A hand-written baseline for the interaction scorer | Written for vehicle and person theft, so thresholds need re-tuning |
| Phone-as-camera over WebRTC (MediaMTX), chunked ingest, per-device stream config, telemetry | A real heterogeneous edge testbed: phones at different resolutions and frame rates as FL data sources | The live path currently runs only the vehicle detector |
| Operator console: "Why this matched" panel, acknowledgement stamps, audit log, confirm-before-action for Copilot writes | Starting point for the HCI study (section 7) | No user testing yet. An assault alert was once labelled "Abandoned Object" (task 5.29), a ready-made comprehension failure to design against |
| Ethical guardrails: human in the loop, explainability, audit, no "suspicious intent" classifiers | Adopt as they stand. MOSAIC scores observed physical actions, never intent | Scoring individuals as instigators sits closer to that line than TraceNet's current scope, so it needs an explicit justification |
| RS\_01 camera: 47 accident alerts that were false positives from parked auto-rickshaws (task 5.27) | A motivating example for on-site adaptation to Indian street scenes | One camera, anecdotal |
| Face redaction: fails closed with no detector, and is untested on real footage (task 4.20) | None yet | Do not claim on-device face blurring in the paper until it is tested |

## Architecture at a glance

v2 runs perception and reasoning on each camera, fuses evidence at a site gateway, and trains across devices without any video leaving them.

&#91;embedded content: MOSAIC-FL v2 architecture · 3 tiers, 10 blocks\]

Read the drawing top to bottom. Each camera runs the eight edge steps and emits one evidence packet. The gateway merges packets on a shared floor plan and makes the decision. Operator confirmations return as training labels, and only adapter updates travel to the federated server.

### One window, end to end

1. The clarity score updates, and the motion gate keeps the frames worth processing. Skipped frames are filled by Kalman prediction.
2. On kept frames the detector, weapon prompts and masks run. The tracker follows people across all frames.
3. Track Tokens go through the GRU and the graph, giving each person a role score and an aggression onset time, plus a scene score.
4. The device sends an evidence packet. The gateway places tracks on the floor plan, merges cameras, and the gated decision returns the violence probability, roles and weapon holder.
5. The operator confirms or rejects. The label stays on the device and trains the next round.

Early exit: if the gate sees no motion and the scene score stays low, the device skips masks and the graph for that window, which is where tier C saves most.

## Module specs

The sketch's blocks survive, grouped into ten blocks across three tiers. One interface, the Track Token, sits in the middle, so a block that underperforms can be swapped without touching its neighbours.

| # | Block | Job and default | Swap options | Judge it by |
| --- | --- | --- | --- | --- |
| 1 | Clarity score | Rates each device and window from resolution, sharpness, exposure, bitrate and person height in pixels, giving q in \[0, 1\]. Default: a weighted formula | A learned no-reference image-quality model | Correlation of q with per-device accuracy |
| 2 | Motion gate | Skips frames about 95% identical to the last processed one, plus static regions. Keeps a minimum keep-alive rate and bursts when motion spikes. Default: frame differencing on small grey frames, or H.264 motion vectors | Background subtraction (MOG2). Tiny optical flow on tier A only | Violent-frame recall at a given share of frames skipped |
| 3 | Detector | Person boxes. Default: RT-DETR, which is NMS-free, so overlapped people are not suppressed (the failure CrowdDet documents) | YOLO small with Set NMS on tier C. Open-vocabulary YOLO-World | AP and MR-2 on CrowdHuman |
| 4 | Threat and masks | Weapon yes or no with its holder, from a prompt bank (weapon, knife, stick, stone) with negative prompts for protest props (flag pole, placard, umbrella). Per person: silhouette, edge contour, centroid. Default: YOLO-World-style prompts plus a segmentation head | Closed-set weapon head. EdgeSAM box-prompted masks on tier A. Box-only kinematics on tier C | Weapon mAP and false alarms on props. Mask IoU on occluded people |
| 5 | Tracker | Stable IDs through occlusion. Default: ByteTrack, which keeps low-score boxes, exactly what occluded people produce. No appearance Re-ID | OC-SORT, BoT-SORT | IDF1 and HOTA on MOT20 |
| 6 | Temporal GRU | One GRU per track and one for the scene. Updates are masked, so a frame where a person is only Kalman-predicted does not change the state | TCN, small Transformer, Mamba | Clip AUC and per-track onset error |
| 7 | Graph and roles | Directed interaction graph over tracks. Outputs a role per person (instigator, participant, bystander) and a crowd embedding. Default: Transformer attention with a distance bias (the Actor-Transformers idea) fed silhouette tokens instead of pose | TraceNet-style rules (proximity spike, fall flip, chase vector) on tier C | Instigator AP on the annotated subset |
| 8 | Evidence packet | The only thing that leaves a device: ground-plane tracks, silhouette embeddings, role scores, weapon probability, q, visible fraction | Quantised embeddings | Bytes per window. A leak test: can a model recover identity from packets? |
| 9 | Ground-plane fusion | The gateway maps each camera's foot points to a shared floor plan by homography, matches tracks across cameras by distance at the same instant, and fuses scores weighted by q and visibility. Default: adapted from TraceNet's LUMPI replay | Learned cross-view ground-plane features (the MVDet idea) | Cross-camera association F1. Instigator AP, one view against fused |
| 10 | Gated decision | Cross-attention over scene, crowd, top tracks, weapon and text-anchor tokens. Each token's attention is scaled by a reliability weight from q, detector confidence and visible fraction. Trained with modality dropout, so a missing or swapped stream degrades gracefully | Weighted late fusion | Calibration error. Accuracy with one stream removed |

### The Track Token

Every person in every processed frame becomes one token. Blocks 6 to 8 read only this, which is what makes blocks 3 to 5 swappable.

```yaml
TrackToken:
  track_id: int
  t: timestamp
  box: [x1, y1, x2, y2]
  observed: bool          # false when only Kalman-predicted
  visible_frac: float     # visible mask area / box area
  skd: float[48]          # silhouette-kinematic descriptor
  centroid, vel, acc, jerk   # image plane
  ground_xy: [x, y]       # metres, via the camera homography
  weapon_prob: float
  det_conf: float
  q: float                # clarity of the source device
```

### Instigator score

v1 left "goon" undefined. v2 defines an instigator as the person whose aggression starts first in the window and whose actions are followed by reactions in others, such as flight, a fall or counter-aggression. With a(t) the GRU's per-person aggression score, the onset and its precedence weight are:

```latex
t_i = \min\{t : a_i(t) > \theta\}, \qquad \pi_i = \frac{\exp\!\big(-(t_i - \min_k t_k)/T\big)}{\sum_k \exp\!\big(-(t_k - \min_k t_k)/T\big)}
```

The directed edge e\_ij combines i's closing speed toward j, mask contact, and j's reaction in the next few seconds. The instigator score then uses outgoing influence, incoming influence and precedence:

```latex
g_i = \sigma\Big(\mathrm{MLP}\big([\,h_i,\ \textstyle\sum_j e_{ij},\ \sum_j e_{ji},\ \pi_i\,]\big)\Big)
```

Training is track-level multiple-instance learning: the clip label supervises the maximum of g\_i, as in the Sensors 2022 paper but with tracks instead of tubes. The precedence term is a prior, not a label, so the 200 to 300 annotated clips (section 6) validate it. Mutual fights will not have a single clear first mover, so the console shows roles with confidence, never a bare "goon" label. High incoming influence with a reaction marks a likely victim.

## Federated learning design

v2 federates only the small trainable parts, groups clients by clarity and update similarity, and assumes each client has almost no labels. Resolution and compression shift the input features, so the design treats heterogeneity as feature skew first and label skew second.

| Parameter group | Treatment | Why |
| --- | --- | --- |
| Backbone, detector, text encoder | Frozen, pretrained | Too heavy to train on edge hardware, and stable across sites |
| Silhouette encoder, GRU, graph, decision head, clarity conditioning | Shared within a cluster. Low-rank adapters where a layer is large | Small payload, and this is where the violence knowledge lives |
| Normalisation statistics, resolution adapter, alert thresholds | Private to each device | Feature statistics differ by resolution and compression. FedBN keeps batch norm local for exactly this reason ([arXiv 2102.07623](https://arxiv.org/abs/2102.07623)) |
| Tier C devices | Train only the decision head and threshold, or a width-sliced sub-model | HeteroFL gives each client a subnetwork matched to its compute ([arXiv 2010.01264](https://arxiv.org/abs/2010.01264)) |

### One training round

1. The server sends each client its cluster model (the shared parameters only).
2. The client trains locally with a FedProx proximal term on two label sources: windows the operator confirmed or rejected, and pseudo-labels from its own model, kept only above a confidence threshold. FedMatch studies this labels-scarce setting and adds an inter-client consistency loss ([arXiv 2006.12097](https://arxiv.org/abs/2006.12097)).
3. A resolution-consistency loss makes predictions agree when the same window is downscaled, blurred or re-compressed. This is the part aimed at device heterogeneity, and it is my proposal, not taken from a paper.
4. The client uploads its update and its clarity profile. Slow devices upload asynchronously with a staleness-weighted merge. An asynchronous scheme on embedded action-recognition clients cut training time by about 40% against synchronous FedAvg ([Jain et al.](https://ar5iv.labs.arxiv.org/html/2107.12147)).
5. The server screens updates by cosine distance to the cluster median, then re-clusters. CF-STGAT clusters by attention-layer parameters (PCA, then K-means) and gains 0.84 to 7.98 top-1 points over FedAvg on NTU RGB+D. v2 adds clarity and tier as clustering features.
6. Within each cluster the server averages with clarity-aware weights, then applies a coordinate-wise trimmed mean:

```latex
w_k = \frac{n_k\, q_k^{\gamma}}{\sum_j n_j\, q_j^{\gamma}}
```

The exponent gamma is an open design choice. Weighting by clarity helps the model learn from good data but can starve low-tier devices, so sweep gamma over 0, 0.5 and 1 and report the accuracy gap between the best and worst tier, not only the mean.

### Privacy and robustness

- No raw frames leave a device. Evidence packets carry tracks and embeddings, so run the leak test from block 8 before claiming privacy.
- Secure aggregation and differential privacy are optional layers, reported separately so their cost is visible.
- Poisoning: cosine screening plus the trimmed mean is the baseline. Test it with 10 to 30% label-flipping or backdoored clients. The FedDet paper on skeleton poisoning is in your list, but I could not open it, so read it before borrowing its method.

## Datasets, experiments and ablations

No public dataset has violence, multiple calibrated cameras and per-person instigator labels together, so v2 combines public clip benchmarks with a small staged multi-camera set that you record yourselves.

| Dataset | Role | Note |
| --- | --- | --- |
| RWF-2000, RLVS, Hockey Fight | Clip-level violence accuracy | Sensors 2022 reports 88.7%, 92.88% and 97.3% respectively. Hockey Fight is saturated, so report it but do not rely on it |
| UCF-Crime, XD-Violence | Weakly supervised training and AUC or AP evaluation | Video-level labels only. XD-Violence also has audio, which is out of scope here |
| UCFCrime2Local | Spatial localisation | Sensors 2022 reports 31.92% localisation error, the number to beat |
| CrowdHuman, MOT20 | Detector and tracker under crowding | Tests blocks 3 and 5 on their own |
| Weapon box datasets, plus protest props as negatives | Weapon mAP and false alarms | Collect flags, placards and umbrellas as hard negatives |
| UCLA Protest Image Dataset | Protest-domain context | Images, not video |
| WILDTRACK, MultiviewX | Cross-view ground-plane association | The MVDet benchmarks. Calibrated, occluded crowds, no violence |
| LUMPI (via TraceNet) | Association geometry on real calibrated cameras | 3.2 s clips, no violence, synthetic embeddings |
| MOSAIC-Staged (new) | Multi-view instigator evaluation | See below |

### MOSAIC-Staged: the small dataset worth recording

Record scripted scenes with consenting volunteers and three or four phones at different resolutions and frame rates, using TraceNet's phone camera app and its per-device stream settings. Scenes: a peaceful march, shoving, a staged scuffle with padded props, an object thrown (soft prop), a stick prop, and a crowd dispersing. Because the script says who starts each scene, every clip has instigator ground truth for free, in all views at once. Get the institute's permission first, keep every prop padded, and keep the clips for research use only.

### Simulating heterogeneity

Simulate K = 10 to 20 virtual clients so the results are reproducible.

| Axis | Setting |
| --- | --- |
| Resolution | 1080p, 720p, 480p, 240p per client |
| Clarity | Blur, H.264 at several CRF values, low-light gamma, sensor noise |
| Frame rate | 30, 15, 5 fps |
| Occlusion | Random erasing, crowd copy-paste, mask and scarf overlays on faces |
| Data skew | Dirichlet partition (alpha 0.1, 0.5, 1.0) and one-dataset-per-client splits |
| Compute tier | A, B, C model widths (section 4) |
| Faults | 20 to 50% client dropout per round, 10 to 30% poisoned clients |

### Hypotheses and pass criteria

These are targets to test, not results.

| # | Hypothesis | Pass criterion |
| --- | --- | --- |
| H1 | Ground-plane fusion beats the best single view on instigator AP when people are occluded | Gain on MOSAIC-Staged and on occluded subsets of WILDTRACK-style scenes |
| H2 | Silhouette descriptors lose less than skeletons at 240p with 30% occlusion | Accuracy drop at most half of the OpenPose plus LSTM baseline's drop |
| H3 | Clarity-aware clustering narrows the tier gap | Smaller best-minus-worst tier accuracy than FedAvg, and Tier C up by 3 or more points |
| H4 | Dual gating saves compute without missing violence | At least 40% of frames skipped on mostly static footage with at least 98% violent-frame recall |
| H5 | Federated training stays near centralised | Gap of 3 points or less at alpha 0.5 |
| H6 | Better localisation than tube-based MIL | UCFCrime2Local error below 31.92% |
| H7 | Removing any one stream costs little | At most 5 points of accuracy per removed stream |

### Baselines and ablations

- Baselines: TraceNet's VideoMAE scanner (clip level), the Sensors 2022 tube-MIL method, OpenPose plus LSTM with FedAvg (the skeleton FL family in your list), and a single-camera MOSAIC.
- Representation: RGB only, plus skeleton, plus silhouette descriptors, plus both, under clean, low-resolution and occluded conditions.
- Gating: none, gate 1, gate 2, both, reporting accuracy, fps and share skipped.
- Detector and tracker swaps: RT-DETR against YOLO with and without Set NMS; IoU linking against ByteTrack against OC-SORT.
- Instigator reasoning: no graph, rule-based (TraceNet-style), attention graph; with and without the precedence term.
- Fusion: concatenation, weighted late fusion, reliability-gated with modality dropout.
- Federation: FedAvg, FedProx, FedBN, clustered, clarity-aware clustered, each with and without trimmed mean under attack, plus the gamma sweep.

## HCI evaluation plan

The HCI question is whether the console helps a person reach the right decision with calibrated trust, without alert fatigue and without wrongly accusing someone. A model with a good AUC can still fail all three, so the study measures the operator, not only the detector.

### What the console must do

- **Rank and escalate.** Three levels: suspicious activity, weapon seen, violence. Per-zone thresholds and grouping of repeated alerts control fatigue.
- **Show roles with confidence.** Instigator, participant and bystander appear as silhouette overlays with a confidence, never a bare label and never a face.
- **Explain.** A "why" panel lists the evidence that drove the alert: the tokens, the weapon crop, which camera saw it. TraceNet's "Why this matched" panel is the starting point.
- **Say when evidence is weak.** If a camera's clarity score is low, the console says its evidence was discounted.
- **Close the loop.** Confirm, reject and relabel are one click, and each becomes a training label for the next federated round.
- **Make setup easy.** Pinning four landmarks per camera to the floor plan is an operator task, so measure how long it takes and how accurate it is.

### Study design

A within-subjects study with 12 to 20 participants, each reviewing 20 to 30 staged or public clip windows that include peaceful crowds and confusing look-alikes (a dance, a tug of war, a crowd surge). Real police operators are unlikely to be reachable, so recruit students as proxies and add two or three domain experts for a qualitative check.

| Condition | What the operator sees | What it isolates |
| --- | --- | --- |
| A: wall only | Raw multi-camera video | The unaided baseline |
| B: scores only | Alerts with a violence score | The value of detection alone |
| C: full console | Alerts with roles, evidence and clarity warnings | The value of explanation and uncertainty |

| Measure | Instrument | Answers |
| --- | --- | --- |
| Decision accuracy and time | Task log | Does the system help? |
| False-accusation rate | Share of bystanders or victims marked as instigators | Does it protect the innocent? |
| Trust calibration | Agreement with the system when it is right against when it is wrong (reliance and over-reliance) | Does explanation prevent automation bias? |
| Workload | NASA-TLX | Is it a burden? |
| Usability | System Usability Scale | Can people use it? |
| Alert fatigue | Missed alerts and response time over a long session | Does fatigue control work? |
| Perceived privacy | A short questionnaire on silhouettes against video | Does the privacy design land with users? |

The standard sources for these instruments are Hart and Staveland (NASA-TLX), Brooke (SUS), Lee and See (trust in automation) and Parasuraman and Manzey (complacency and automation bias). I have not re-checked those citations here, so confirm them from the originals.

### Ethics

Show participants only staged and public clips, never footage of real protesters. Get consent and institutional review before recruiting. In the paper, state that the system flags observed physical actions for human review, never intent, identity or political activity, and that no alert triggers action without a human decision.

## Roadmap, risks and open questions

Build in six phases, and stop at each gate to check one claim before adding the next layer. The phases are in order, not to scale, because I do not know your deadline.

&#91;embedded content: roadmap · 6 phases, 6 gates\]

Phase 3 carries the most risk and the most novelty, so record MOSAIC-Staged early.

### Risks and fallbacks

| Risk | Fallback |
| --- | --- |
| Silhouettes fail in dense or low-resolution crowds (H2 fails) | Box kinematics plus patch-level motion features. Keep skeletons as an optional token on high-clarity devices |
| Instigator precedence is ambiguous in mutual fights | Report roles with confidence, use precedence only as a prior, and show "mutual" explicitly |
| Cross-camera association drifts with calibration or clock error | Re-pin landmarks in the console, allow a timestamp tolerance, and fall back to cell-level fusion without identities |
| Protest props cause weapon false alarms | Negative prompts. Weapon evidence alone never triggers violence, and a human confirms |
| Clients have few labels | Confidence-gated pseudo-labels, operator labels, and a teacher trained on public data only |
| Tier C cannot run the pipeline | Box-only kinematics, head-only training, and offloading uncertain windows to the gateway |
| Misuse against peaceful protesters | Human-confirmed alerts only, no identity, an audit log, stated limits in the paper, and the false-accusation measure in the study |

### Open questions for you

1. Which edge devices can you actually use (phones, Jetson-class boards, Raspberry Pi)? This decides the tiers.
2. Is multi-view fusion in scope for this paper? v2 treats it as core. If it is out, H1 and the staged dataset shrink.
3. Can the team record MOSAIC-Staged with consenting volunteers, and annotate 200 to 300 clips for instigators?
4. Is audio out of scope? XD-Violence shows audio helps, but v2 stays visual only.
5. Which venue and deadline? An HCI venue will expect the user study, and that sets the schedule.
6. v1's sketch questions are still open: what P1, P2, "Ty" and "Fles" mean next to the grounding box.

## References

The first group was opened while writing this document. The second comes from your list and v1 notes, and was not opened.

### Opened

- Choqueluque-Roman, Camara-Chavez. Weakly Supervised Violence Detection in Surveillance Video. Sensors 22(12):4502, 2022. [doi:10.3390/s22124502](https://doi.org/10.3390/s22124502) (your PDF)
- Chu, Zheng, Zhang, Sun. Detection in Crowded Scenes: One Proposal, Multiple Predictions. [arXiv 2003.09163](https://arxiv.org/abs/2003.09163) (your PDF)
- Tu et al. [FedFSLAR: A Federated Learning Framework for Few-Shot Action Recognition](https://openaccess.thecvf.com/content/WACV2024W/RWS/html/Tu_FedFSLAR_A_Federated_Learning_Framework_for_Few-Shot_Action_Recognition_WACVW_2024_paper.html). WACV Workshops 2024
- Wu et al. [Video Action Recognition Method Based on Personalized Federated Learning and Spatiotemporal Features](https://www.sciopen.com/article/10.32604/cmc.2025.061396). CMC 83(3), 2025
- [Federated Action Recognition for Smart Worker Assistance Using FastPose](https://www.dfki.de/fileadmin/user_upload/import/16215_FLTA2025_paper_%281%29.pdf). DFKI, 2025
- [Clustered Federated Spatio-Temporal Graph Attention Networks for Skeleton-Based Action Recognition](https://www.mdpi.com/1424-8220/25/23/7277). Sensors 25(23), 2025
- Famouri. [Pose to Protect](https://opensiuc.lib.siu.edu/theses/3395). MSc thesis, Southern Illinois University Carbondale, 2025
- Jain et al. [Federated Action Recognition on Heterogeneous Embedded Devices](https://ar5iv.labs.arxiv.org/html/2107.12147). arXiv 2107.12147
- Hou, Zheng, Gould. [Multiview Detection with Feature Perspective Transformation](https://arxiv.org/abs/2007.07247) (MVDet)
- Li et al. [FedBN: Federated Learning on Non-IID Features via Local Batch Normalization](https://arxiv.org/abs/2102.07623)
- Jeong et al. [Federated Semi-Supervised Learning with Inter-Client Consistency and Disjoint Learning](https://arxiv.org/abs/2006.12097) (FedMatch)
- Diao, Ding, Tarokh. [HeteroFL](https://arxiv.org/abs/2010.01264)
- Zhao et al. [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069) (RT-DETR)
- Zhang et al. [ByteTrack](https://arxiv.org/abs/2110.06864)
- Cheng et al. [YOLO-World](https://arxiv.org/abs/2401.17270)
- Zhou et al. [EdgeSAM](https://arxiv.org/abs/2312.06660)
- TraceNet repository, `manavdhamecha77/tracenet`: README.md, AGENTS.md, `docs/lumpi-evaluation.md`, `analytics/lumpi/replay.py`, `alerts/chain_snatching.py`, `detection/assault_detector.py`

### From your list, not opened

- Cheng, Cai, Li. RWF-2000. ICPR 2020
- Sultani, Chen, Shah. Real-World Anomaly Detection in Surveillance Videos. CVPR 2018
- Wu et al. Not only Look, but also Listen (XD-Violence). ECCV 2020
- Gavrilyuk et al. Actor-Transformers for Group Activity Recognition. CVPR 2020
- Duan et al. SkeleTR. ICCV 2023
- FedDet, federated skeleton poisoning detection (item 6 in your list)

Venues for the arXiv papers are not shown on their pages, so confirm each venue and year before citing.
