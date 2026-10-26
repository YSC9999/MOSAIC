"""Low-cost motion gating for efficient edge video processing (Block 2).

Filters out redundant static or background-only frames while preserving sudden
violent bursts and motion spikes:
1. Low-resolution frame differencing: Computes absolute pixel differences on downsampled
   grayscale frames (e.g. 64x64) against the last retained observation.
2. Motion thresholding: Selects frames exceeding the configured motion threshold.
3. Burst mode: When motion rises sharply above ``burst_threshold``, locks the gate open
   for ``burst_hold_frames`` to preserve rapid scuffles and multi-frame altercations.
4. Keep-alive guarantee: Enforces that at least one frame is processed every
   ``keep_alive_frames`` so downstream trackers never time out on slow-moving actors.
5. Strict temporal contract: Retained frames preserve their exact original timestamps
   and frame indices without synthetic time-warping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .contracts import MotionGateReport


class MotionGateError(ValueError):
    """Raised when motion gate inputs or configuration are invalid."""


@dataclass(frozen=True)
class GateDecision:
    """Decision and diagnostics for a single inspected frame."""

    frame_index: int
    timestamp: float
    selected: bool
    reason: str  # "first_frame", "motion", "burst", "keep_alive", "no_gate", "skipped"
    motion_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": float(self.timestamp),
            "selected": self.selected,
            "reason": self.reason,
            "motion_score": float(self.motion_score),
        }


@dataclass(frozen=True)
class GatedClipResult:
    """The outcome of passing a sequence of frames through the motion gate."""

    selected_frames: List[np.ndarray]
    selected_indices: List[int]
    selected_timestamps: List[float]
    skipped_indices: List[int]
    motion_scores: List[float]
    decisions: List[GateDecision]

    @property
    def total_frames(self) -> int:
        return len(self.selected_indices) + len(self.skipped_indices)

    @property
    def reduction_fraction(self) -> float:
        total = self.total_frames
        return float(len(self.skipped_indices)) / float(max(1, total))

    def to_report(self) -> MotionGateReport:
        burst_count = sum(1 for d in self.decisions if d.reason == "burst")
        keep_alive_count = sum(1 for d in self.decisions if d.reason == "keep_alive")
        mean_motion = float(np.mean(self.motion_scores)) if self.motion_scores else 0.0
        return MotionGateReport(
            total_frames=self.total_frames,
            selected_frames=len(self.selected_indices),
            skipped_frames=len(self.skipped_indices),
            reduction_fraction=self.reduction_fraction,
            mean_motion_score=mean_motion,
            burst_count=burst_count,
            keep_alive_count=keep_alive_count,
        )



class MotionGate:
    """Evaluates frame-to-frame movement on low-resolution grayscale frames."""

    def __init__(
        self,
        enabled: bool = True,
        threshold: float = 0.02,
        burst_threshold: float = 0.06,
        burst_hold_frames: int = 5,
        keep_alive_frames: int = 15,
        downsample_size: Tuple[int, int] = (64, 64),
    ) -> None:
        self.enabled = bool(enabled)
        if threshold < 0.0:
            raise MotionGateError(f"threshold must be >= 0.0, got {threshold}")
        if burst_threshold < threshold:
            raise MotionGateError(
                f"burst_threshold ({burst_threshold}) cannot be less than threshold ({threshold})"
            )
        if burst_hold_frames < 0:
            raise MotionGateError(f"burst_hold_frames must be >= 0, got {burst_hold_frames}")
        if keep_alive_frames < 1:
            raise MotionGateError(f"keep_alive_frames must be >= 1, got {keep_alive_frames}")
        if len(downsample_size) != 2 or downsample_size[0] < 8 or downsample_size[1] < 8:
            raise MotionGateError(f"downsample_size must be >= (8, 8), got {downsample_size}")

        self.threshold = float(threshold)
        self.burst_threshold = float(burst_threshold)
        self.burst_hold_frames = int(burst_hold_frames)
        self.keep_alive_frames = int(keep_alive_frames)
        self.downsample_size = (int(downsample_size[0]), int(downsample_size[1]))

    def _prepare_thumbnail(self, frame: np.ndarray) -> np.ndarray:
        """Downsample and convert frame to float32 grayscale in [0.0, 1.0]."""
        if not isinstance(frame, np.ndarray):
            raise MotionGateError(f"frame must be numpy array, got {type(frame)}")
        if frame.ndim == 2:
            gray = frame
        elif frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            raise MotionGateError(f"unsupported frame dimensions {frame.ndim}")

        resized = cv2.resize(gray, self.downsample_size, interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    def compute_motion_score(self, current_thumb: np.ndarray, reference_thumb: np.ndarray) -> float:
        """Compute mean absolute pixel difference between downsampled frames."""
        diff = np.abs(current_thumb - reference_thumb)
        return float(np.mean(diff))

    def filter_clip(
        self,
        frames: Sequence[np.ndarray],
        timestamps: Optional[Sequence[float]] = None,
    ) -> GatedClipResult:
        """Filter a sequence of frames, emitting selected and skipped records."""
        n_frames = len(frames)
        if n_frames == 0:
            return GatedClipResult([], [], [], [], [], [])

        if timestamps is None:
            ts_list = [float(i) / 30.0 for i in range(n_frames)]
        else:
            if len(timestamps) != n_frames:
                raise MotionGateError(
                    f"timestamps count ({len(timestamps)}) does not match frames count ({n_frames})"
                )
            ts_list = [float(t) for t in timestamps]

        # Disabled / no-gate mode: all frames are selected
        if not self.enabled:
            decisions = [
                GateDecision(
                    frame_index=i,
                    timestamp=ts_list[i],
                    selected=True,
                    reason="no_gate",
                    motion_score=0.0,
                )
                for i in range(n_frames)
            ]
            return GatedClipResult(
                selected_frames=list(frames),
                selected_indices=list(range(n_frames)),
                selected_timestamps=ts_list,
                skipped_indices=[],
                motion_scores=[0.0] * n_frames,
                decisions=decisions,
            )

        selected_frames: List[np.ndarray] = []
        selected_indices: List[int] = []
        selected_timestamps: List[float] = []
        skipped_indices: List[int] = []
        motion_scores: List[float] = []
        decisions: List[GateDecision] = []

        last_retained_thumb: Optional[np.ndarray] = None
        prev_thumb: Optional[np.ndarray] = None
        burst_countdown = 0
        frames_since_retained = 0

        for i in range(n_frames):
            frame = frames[i]
            ts = ts_list[i]
            thumb = self._prepare_thumbnail(frame)

            # First frame is always retained to anchor temporal sequence
            if i == 0 or last_retained_thumb is None or prev_thumb is None:
                motion_score = 0.0
                selected = True
                reason = "first_frame"
            else:
                instant_score = self.compute_motion_score(thumb, prev_thumb)
                accum_score = self.compute_motion_score(thumb, last_retained_thumb)
                motion_score = accum_score
                selected = False
                reason = "skipped"

                # 1. Instantaneous motion burst trigger or countdown
                if instant_score >= self.burst_threshold:
                    selected = True
                    reason = "burst"
                    burst_countdown = self.burst_hold_frames
                elif burst_countdown > 0:
                    selected = True
                    reason = "burst"
                    burst_countdown -= 1
                # 2. Accumulated scene motion threshold exceedance
                elif accum_score >= self.threshold:
                    selected = True
                    reason = "motion"
                # 3. Keep-alive fallback
                elif (frames_since_retained + 1) >= self.keep_alive_frames:
                    selected = True
                    reason = "keep_alive"

            prev_thumb = thumb
            motion_scores.append(motion_score)
            dec = GateDecision(
                frame_index=i,
                timestamp=ts,
                selected=selected,
                reason=reason,
                motion_score=motion_score,
            )
            decisions.append(dec)

            if selected:
                selected_frames.append(frame)
                selected_indices.append(i)
                selected_timestamps.append(ts)
                last_retained_thumb = thumb
                frames_since_retained = 0
            else:
                skipped_indices.append(i)
                frames_since_retained += 1

        return GatedClipResult(
            selected_frames=selected_frames,
            selected_indices=selected_indices,
            selected_timestamps=selected_timestamps,
            skipped_indices=skipped_indices,
            motion_scores=motion_scores,
            decisions=decisions,
        )
