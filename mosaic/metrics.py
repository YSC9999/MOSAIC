"""Evaluation metrics for binary violence classification.

Zero external ML dependencies (pure NumPy). Computes:
* Accuracy: (TP + TN) / Total
* Precision: TP / (TP + FP) (safely returns 0.0 when TP + FP == 0)
* Recall: TP / (TP + FN) (safely returns 0.0 when TP + FN == 0)
* F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
* ROC-AUC: Trapezoidal integration over the False Positive Rate vs
  True Positive Rate curve across sorted prediction thresholds.
* Confusion Matrix: [[TN, FP], [FN, TP]]
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


class MetricsError(ValueError):
    """Raised when metric inputs are invalid."""


@dataclass(frozen=True)
class ClassificationMetrics:
    """Standardized summary of binary classification performance."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    total_samples: int
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    def to_dict(self) -> Dict[str, Union[float, int]]:
        return {
            "accuracy": float(self.accuracy),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
            "roc_auc": float(self.roc_auc),
            "total_samples": int(self.total_samples),
            "true_positives": int(self.true_positives),
            "true_negatives": int(self.true_negatives),
            "false_positives": int(self.false_positives),
            "false_negatives": int(self.false_negatives),
        }


def compute_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> Tuple[int, int, int, int]:
    """Compute (TP, TN, FP, FN) for binary labels (0 and 1)."""
    if len(y_true) != len(y_pred):
        raise MetricsError(
            f"length mismatch: y_true ({len(y_true)}) != y_pred ({len(y_pred)})"
        )
    tp = tn = fp = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt not in (0, 1) or yp not in (0, 1):
            raise MetricsError(f"labels must be 0 or 1, got y_true={yt}, y_pred={yp}")
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1
    return tp, tn, fp, fn


def compute_roc_auc(
    y_true: Sequence[int],
    y_score: Sequence[float],
) -> float:
    """Compute Receiver Operating Characteristic Area Under the Curve (ROC-AUC).

    Uses the exact Mann-Whitney U / rank-sum formulation which computes the
    normalized probability that a randomly chosen positive sample has a higher
    predicted score than a randomly chosen negative sample:
      AUC = (sum(ranks of positive samples) - P*(P+1)/2) / (P * N)
    Handles ties by assigning average ranks.
    """
    if len(y_true) != len(y_score):
        raise MetricsError(
            f"length mismatch: y_true ({len(y_true)}) != y_score ({len(y_score)})"
        )
    if len(y_true) == 0:
        raise MetricsError("cannot compute ROC-AUC on empty sequences")

    y_t = np.asarray(y_true, dtype=int)
    y_s = np.asarray(y_score, dtype=float)

    pos_mask = y_t == 1
    neg_mask = y_t == 0

    num_pos = int(np.sum(pos_mask))
    num_neg = int(np.sum(neg_mask))

    if num_pos + num_neg != len(y_t):
        raise MetricsError("y_true contains labels other than 0 and 1")

    # If only one class is present in the ground truth, AUC is mathematically undefined.
    # Return 0.5 (random guessing floor).
    if num_pos == 0 or num_neg == 0:
        return 0.5

    # Rank scores in ascending order, breaking ties with average rank
    order = np.argsort(y_s)
    ranked = np.empty_like(order, dtype=float)
    ranked[order] = np.arange(1, len(y_s) + 1)

    # Handle ties: find identical score values and average their ranks
    unique_scores, counts = np.unique(y_s, return_counts=True)
    for score, count in zip(unique_scores, counts):
        if count > 1:
            mask = y_s == score
            ranked[mask] = np.mean(ranked[mask])

    pos_rank_sum = float(np.sum(ranked[pos_mask]))
    u_stat = pos_rank_sum - (num_pos * (num_pos + 1)) / 2.0
    auc = u_stat / (num_pos * num_neg)
    return float(np.clip(auc, 0.0, 1.0))


def evaluate_predictions(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Compute full classification metrics from ground truth and predicted scores.

    Parameters
    ----------
    y_true: Sequence[int]
        Ground truth binary labels (0 or 1).
    y_score: Sequence[float]
        Predicted violence probabilities in [0.0, 1.0].
    threshold: float
        Decision threshold for binary prediction (default 0.5).

    Returns
    -------
    ClassificationMetrics
    """
    if len(y_true) != len(y_score):
        raise MetricsError(
            f"length mismatch: y_true ({len(y_true)}) != y_score ({len(y_score)})"
        )
    if len(y_true) == 0:
        raise MetricsError("cannot evaluate empty predictions")

    y_pred = [1 if s >= threshold else 0 for s in y_score]
    tp, tn, fp, fn = compute_confusion_matrix(y_true, y_pred)
    total = len(y_true)

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )
    roc_auc = compute_roc_auc(y_true, y_score)

    return ClassificationMetrics(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=float(roc_auc),
        total_samples=total,
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
    )
