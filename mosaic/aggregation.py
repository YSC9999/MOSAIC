"""Window-to-clip score aggregation.

RWF-2000 clips are roughly 5 seconds long. In production and evaluation:
* A clip may be processed as a single whole-clip window or a series of sliding windows.
* Each window produces a `WindowOutput` containing a `violence_score` in [0, 1]
  and `temporal_confidence` in [0, 1].
* The clip-level violence score is computed via a configured aggregation rule:
  - "mean" (arithmetic mean, the primary default),
  - "max" (maximum score, used as an ablation).
* The final categorical label ("violent" vs "non_violent") is determined by
  comparing the aggregated violence score against a decision threshold (default 0.5).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

from mosaic.contracts import LABELS, WindowOutput, require_choice, require_unit_interval

AGGREGATION_POLICIES = ("mean", "max")


class AggregationError(ValueError):
    """Raised when score aggregation fails or inputs are invalid."""


def aggregate_window_scores(
    scores: Sequence[float],
    policy: str = "mean",
) -> float:
    """Aggregate a sequence of per-window violence scores into one clip score.

    Parameters
    ----------
    scores: Sequence[float]
        Window violence scores, each in [0.0, 1.0]. Must contain at least one score.
    policy: str
        "mean" or "max".

    Returns
    -------
    float
        Aggregated clip-level score in [0.0, 1.0].
    """
    if policy not in AGGREGATION_POLICIES:
        raise AggregationError(
            f"unknown aggregation policy {policy!r}; supported: {list(AGGREGATION_POLICIES)}"
        )
    if not scores:
        raise AggregationError("cannot aggregate an empty list of window scores")

    for i, s in enumerate(scores):
        if not isinstance(s, (int, float)) or s < 0.0 or s > 1.0:
            raise AggregationError(f"scores[{i}] must be in [0, 1], got {s!r}")

    if policy == "mean":
        return float(sum(scores) / len(scores))
    if policy == "max":
        return float(max(scores))
    raise AggregationError(f"unhandled policy {policy}")  # pragma: no cover


def score_to_label(score: float, threshold: float = 0.5) -> str:
    """Map a violence score in [0, 1] to 'violent' or 'non_violent'."""
    require_unit_interval(score, "score")
    require_unit_interval(threshold, "threshold")
    return "violent" if score >= threshold else "non_violent"


@dataclass(frozen=True)
class ClipOutput:
    """Aggregated result for an entire video clip across its windows."""

    clip_id: str
    violence_score: float
    label: str
    temporal_confidence: float
    window_count: int
    aggregation_policy: str
    windows: List[WindowOutput] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.clip_id.strip():
            raise AggregationError("clip_id must not be empty")
        require_unit_interval(self.violence_score, "clip.violence_score")
        require_choice(self.label, LABELS, "clip.label")
        require_unit_interval(self.temporal_confidence, "clip.temporal_confidence")
        if self.window_count < 1:
            raise AggregationError(f"window_count must be >= 1, got {self.window_count}")
        if self.aggregation_policy not in AGGREGATION_POLICIES:
            raise AggregationError(f"aggregation_policy must be in {list(AGGREGATION_POLICIES)}")

    def to_dict(self) -> Dict[str, object]:
        return {
            "clip_id": self.clip_id,
            "violence_score": self.violence_score,
            "label": self.label,
            "temporal_confidence": self.temporal_confidence,
            "window_count": self.window_count,
            "aggregation_policy": self.aggregation_policy,
            "windows": [w.to_dict() for w in self.windows],
        }


def aggregate_windows_to_clip(
    clip_id: str,
    windows: Sequence[WindowOutput],
    policy: str = "mean",
    threshold: float = 0.5,
) -> ClipOutput:
    """Aggregate a sequence of WindowOutput objects for a clip."""
    if not windows:
        raise AggregationError(f"clip {clip_id!r} has no windows to aggregate")

    scores = [w.violence_score for w in windows]
    confidences = [w.temporal_confidence for w in windows]

    agg_score = aggregate_window_scores(scores, policy=policy)
    # Average confidence across windows
    agg_conf = float(sum(confidences) / len(confidences))
    label = score_to_label(agg_score, threshold=threshold)

    return ClipOutput(
        clip_id=clip_id,
        violence_score=agg_score,
        label=label,
        temporal_confidence=agg_conf,
        window_count=len(windows),
        aggregation_policy=policy,
        windows=list(windows),
    )
