"""Trivial reference baselines for sanity checks.

As specified in Phase 1, Task 8 of the implementation plan:
Every learned baseline must be compared against two trivial baselines:
1. Majority Class Classifier: Always predicts the most common class in the training
   set (or the empirical prior probability). On a balanced dataset (50% Fight,
   50% NonFight), this yields 50% accuracy.
2. Motion Energy Classifier: Computes raw inter-frame pixel differences across time.
   Physical altercations involve rapid movement, so a simple motion detector
   can separate static scenes from energetic movement without understanding violence.
   If a complex neural network cannot beat motion energy, it has only learned to
   detect motion, not violence.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from mosaic.aggregation import score_to_label


class BaselineError(ValueError):
    """Raised when baseline arguments or states are invalid."""


class MajorityClassClassifier:
    """Predicts the empirical majority class observed during training."""

    def __init__(self) -> None:
        self.prior: Optional[float] = None
        self.majority_label: Optional[str] = None
        self.is_fitted = False

    def fit(self, y_train: Sequence[int]) -> "MajorityClassClassifier":
        """Fit on binary training labels (0 and 1)."""
        if len(y_train) == 0:
            raise BaselineError("cannot fit on empty labels")
        for yt in y_train:
            if yt not in (0, 1):
                raise BaselineError(f"labels must be 0 or 1, got {yt}")
        self.prior = float(np.mean(y_train))
        self.majority_label = "violent" if self.prior >= 0.5 else "non_violent"
        self.is_fitted = True
        return self

    def predict_one(self) -> Tuple[float, str, float]:
        """Return (violence_score, label, confidence) for one sample."""
        if not self.is_fitted or self.prior is None or self.majority_label is None:
            raise BaselineError("classifier must be fitted before predict")
        conf = max(self.prior, 1.0 - self.prior)
        return self.prior, self.majority_label, float(conf)

    def predict(self, n_samples: int) -> Tuple[List[float], List[str]]:
        """Return (scores, labels) for n_samples."""
        if not self.is_fitted or self.prior is None:
            raise BaselineError("classifier must be fitted before predict")
        scores = [self.prior] * n_samples
        labels = ["violent" if self.prior >= 0.5 else "non_violent"] * n_samples
        return scores, labels


class MotionEnergyClassifier:
    """Classifies video clips by the average magnitude of inter-frame motion."""

    def __init__(self, threshold: Optional[float] = None) -> None:
        self.threshold: Optional[float] = threshold
        self.is_fitted = threshold is not None

    @staticmethod
    def extract_motion_energy(
        frames: Union[np.ndarray, torch.Tensor],
        validity_mask: Optional[Union[np.ndarray, torch.Tensor, Sequence[bool]]] = None,
    ) -> float:
        """Compute the average pixel difference between consecutive valid frames.

        Input shape:
        - numpy: (T, H, W, C) in [0, 255] or (T, H, W)
        - torch: (T, C, H, W) in [0, 1] or [0, 255]

        Returns
        -------
        float
            Normalized motion energy score in [0.0, 1.0].
        """
        if isinstance(frames, torch.Tensor):
            arr = frames.detach().cpu().numpy()
            if arr.ndim == 4 and arr.shape[1] in (1, 3):  # (T, C, H, W)
                arr = np.transpose(arr, (0, 2, 3, 1))
        elif isinstance(frames, np.ndarray):
            arr = frames
        else:
            raise BaselineError(f"unsupported frames type {type(frames)}")

        if validity_mask is not None:
            if isinstance(validity_mask, torch.Tensor):
                mask_np = validity_mask.detach().cpu().numpy().astype(bool)
            else:
                mask_np = np.asarray(validity_mask, dtype=bool)
            arr = arr[mask_np]

        if arr.ndim < 3 or arr.shape[0] < 2:
            return 0.0

        # If in [0, 255], normalize to [0, 1]
        if arr.dtype == np.uint8 or np.max(arr) > 1.0:
            arr = arr.astype(np.float32) / 255.0

        # Differences between consecutive frames: |I_t - I_{t-1}|
        diffs = np.abs(arr[1:] - arr[:-1])
        energy = float(np.mean(diffs))
        return float(np.clip(energy, 0.0, 1.0))

    def fit(
        self,
        energies_train: Sequence[float],
        y_train: Sequence[int],
    ) -> "MotionEnergyClassifier":
        """Find the motion threshold that maximizes training accuracy."""
        if len(energies_train) != len(y_train):
            raise BaselineError("length mismatch between energies and labels")
        if len(y_train) == 0:
            raise BaselineError("cannot fit on empty training data")

        unique_vals = sorted(set(float(e) for e in energies_train))
        best_acc = -1.0
        best_th = 0.5

        # Candidate thresholds midway between sorted unique values
        candidates: List[float] = []
        if unique_vals:
            candidates.append(max(0.0, unique_vals[0] * 0.5))
        for a, b in zip(unique_vals, unique_vals[1:]):
            candidates.append((a + b) / 2.0)
        if unique_vals:
            candidates.append(min(1.0, unique_vals[-1] + 1e-6))

        y_t = np.array(y_train, dtype=int)
        e_t = np.array(energies_train, dtype=float)

        for th in candidates:
            preds = (e_t >= th).astype(int)
            acc = float(np.mean(preds == y_t))
            if acc > best_acc:
                best_acc = acc
                best_th = float(th)

        self.threshold = best_th
        self.is_fitted = True
        return self

    def _energy_to_prob(self, energy: float) -> float:
        """Strictly monotonic mapping from energy >= 0 to probability in [0, 1].

        Maps ``energy == self.threshold`` to ``0.5`` without flat-clipping high
        energies, preserving exact rank order for ROC-AUC evaluation.
        """
        assert self.threshold is not None
        e = max(0.0, float(energy))
        if self.threshold <= 0.0:
            return 1.0 if e > 0.0 else 0.5
        return float(np.clip(e / (e + self.threshold), 0.0, 1.0))

    def predict_one(self, frames: Union[np.ndarray, torch.Tensor]) -> Tuple[float, str, float]:
        """Predict (violence_score, label, confidence) for a single clip."""
        if not self.is_fitted or self.threshold is None:
            raise BaselineError("classifier must be fitted before predict")
        energy = self.extract_motion_energy(frames)
        prob = self._energy_to_prob(energy)
        label = score_to_label(prob, threshold=0.5)
        conf = float(abs(prob - 0.5) * 2.0)
        return prob, label, conf

    def predict(
        self,
        energies: Sequence[float],
    ) -> Tuple[List[float], List[str]]:
        """Predict scores and labels for a sequence of precomputed energies."""
        if not self.is_fitted or self.threshold is None:
            raise BaselineError("classifier must be fitted before predict")
        scores = []
        labels = []
        for e in energies:
            prob = self._energy_to_prob(e)
            scores.append(prob)
            labels.append(score_to_label(prob, threshold=0.5))
        return scores, labels
