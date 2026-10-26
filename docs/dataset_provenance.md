# RWF-2000 Dataset: Provenance and Licensing Information

This file records the official citation, repository source, licensing guidelines, and split structure for the **RWF-2000** dataset used as the primary benchmark in this project.

---

## 1. Citation and Academic Attribution

- **Paper Title:** RWF-2000: An Open Large Scale Video Database for Violence Detection
- **Authors:** Ming Cheng, Kaimin Cai, Ming Li
- **Published In:** 2020 25th International Conference on Pattern Recognition (ICPR), January 2021
- **DOI / Link:** [10.1109/ICPR48876.2021.9412507](https://doi.org/10.1109/ICPR48876.2021.9412507)
- **Official GitHub:** [https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection)

```bibtex
@inproceedings{cheng2021rwf2000,
  title={RWF-2000: An Open Large Scale Video Database for Violence Detection},
  author={Cheng, Ming and Cai, Kaimin and Li, Ming},
  booktitle={2020 25th International Conference on Pattern Recognition (ICPR)},
  pages={4183--4190},
  year={2021},
  organization={IEEE}
}
```

---

## 2. Dataset Structure and Summary

RWF-2000 was collected from YouTube surveillance-style videos to study real-world fighting vs. non-fighting behavior:

- **Total Clips:** 2,000 video clips (each roughly 5 seconds long, ~30 FPS, ~150 frames per clip).
- **Classes:** Balanced binary classification (1,000 `Fight` clips, 1,000 `NonFight` clips).
- **Official Folder Organization:**
  ```text
  RWF-2000/
  ├── train/
  │   ├── Fight/      (800 clips)
  │   └── NonFight/   (800 clips)
  └── val/
      ├── Fight/      (200 clips)
      └── NonFight/   (200 clips)
  ```

---

## 3. Split Strategy for MOSAIC-FL

To prevent data leakage and guarantee clean evaluation (as specified in Section 3.3 of the Implementation Plan):

1. **Test Set (Held-Out):** The official `val/` directory (400 clips: 200 fight, 200 non-fight) is treated strictly as the final held-out test benchmark. It is never used for hyperparameter tuning, model checkpoint selection, or threshold tuning.
2. **Carved Validation Set:** From the official `train/` directory (1,600 clips), a stratified, seeded split (e.g. 20% = 320 clips) is carved out. All threshold adjustments, ablation tuning, and early stopping decisions use this set.
3. **Training Set:** The remaining clips from `train/` (e.g. 80% = 1,280 clips) are used for training baseline and temporal models.
4. **Duplicate Verification:** During manifest creation, SHA-256 hashes of every clip are checked to ensure no video appears across both train and test splits.

---

## 4. Usage Rights and Ethical Considerations

- **Research Purpose:** RWF-2000 is released specifically for academic and non-commercial computer vision research.
- **Privacy Notice:** Videos depict public surveillance footage. MOSAIC-FL operates strictly on bounding boxes, motion kinematics, and silhouette contours. No facial recognition, personal identity matching, or demographic profiling is performed.
