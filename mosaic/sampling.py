"""Deterministic video frame sampling, padding, and truncation.

This module provides reproducible frame extraction for temporal models:
* Frame sampling calculates exact, deterministic frame indices across a video clip.
* Bounds are strictly enforced: no frame index outside [0, total_frames - 1]
  is ever requested.
* Padding and truncation policies are explicit and deterministic:
  - If a clip has fewer frames than required, it is padded (repeating the edge frame)
    and accompanied by a boolean validity mask where padded frames are False.
  - If a clip has more frames than required, it is sampled to the exact count.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from mosaic.config import WindowConfig
from mosaic.ingestion import FrameSource, VideoReader


class SamplingError(ValueError):
    """Raised when frame sampling arguments are invalid."""


def compute_time_deltas(
    timestamps: Sequence[float],
    validity_mask: Optional[Sequence[bool]] = None,
) -> List[float]:
    """Compute per-step time deltas (in seconds) between consecutive frames.

    Step 0 and any padded frames (where ``validity_mask[t]`` is False) receive
    a time delta of ``0.0``. This makes irregular or motion-gated frame spacing
    explicit to downstream temporal models (Plan Section 2.2).
    """
    n = len(timestamps)
    if n == 0:
        return []
    if validity_mask is not None and len(validity_mask) != n:
        raise SamplingError(
            f"validity_mask length ({len(validity_mask)}) must match timestamps ({n})"
        )
    deltas: List[float] = [0.0]
    for i in range(1, n):
        if validity_mask is not None and not validity_mask[i]:
            deltas.append(0.0)
        else:
            dt = float(timestamps[i]) - float(timestamps[i - 1])
            deltas.append(max(0.0, dt))
    return deltas


def slice_window_frame_ranges(
    timestamps: Sequence[float],
    window_cfg: Optional[WindowConfig] = None,
) -> List[Tuple[int, int, float, float]]:
    """Return ``(start_idx, end_idx_exclusive, start_ts, end_ts)`` per window.

    Follows Plan Section 2.2:
    * Windows are defined on source timestamps, not processed-frame counts.
    * ``whole_clip`` produces a single window covering the entire clip.
    * ``sliding`` steps windows of ``window_seconds`` by ``stride_seconds``.
    """
    if not timestamps:
        raise SamplingError("cannot slice windows from empty timestamps")
    cfg = window_cfg if window_cfg is not None else WindowConfig()
    n = len(timestamps)
    first_ts = float(timestamps[0])
    last_ts = float(timestamps[-1])

    if cfg.policy == "whole_clip" or (last_ts - first_ts) <= cfg.window_seconds:
        return [(0, n, first_ts, last_ts)]

    windows: List[Tuple[int, int, float, float]] = []
    win_start = first_ts
    while win_start <= last_ts + 1e-9:
        win_end = win_start + cfg.window_seconds
        indices = [i for i, ts in enumerate(timestamps) if win_start - 1e-9 <= ts <= win_end + 1e-9]
        if indices:
            s_idx, e_idx = indices[0], indices[-1] + 1
            windows.append((s_idx, e_idx, float(timestamps[s_idx]), float(timestamps[e_idx - 1])))
        if win_end >= last_ts - 1e-9:
            break
        win_start += cfg.stride_seconds

    return windows if windows else [(0, n, first_ts, last_ts)]


def sample_frame_indices(
    total_frames: int,
    num_frames: int,
    strategy: str = "uniform",
) -> List[int]:
    """Return a deterministic list of frame indices of length ``num_frames``.

    Parameters
    ----------
    total_frames: int
        Total number of frames available in the clip (must be >= 1).
    num_frames: int
        Number of frames to sample (must be >= 1).
    strategy: str
        Sampling strategy:
        - "uniform": evenly spaced across [0, total_frames - 1].
        - "stride": step-based sampling starting at index 0.

    Returns
    -------
    List[int]
        Strictly bounded frame indices in [0, total_frames - 1].
    """
    if total_frames < 1:
        raise SamplingError(f"total_frames must be >= 1, got {total_frames}")
    if num_frames < 1:
        raise SamplingError(f"num_frames must be >= 1, got {num_frames}")

    if strategy == "uniform":
        if total_frames == 1:
            return [0] * num_frames
        if num_frames == 1:
            return [total_frames // 2]
        # np.linspace gives evenly spaced floats across the full interval [0, total_frames - 1]
        raw = np.linspace(0, total_frames - 1, num_frames)
        # Round to nearest integer and clip to guarantee bounds
        indices = [int(np.clip(round(v), 0, total_frames - 1)) for v in raw]
        return indices

    if strategy == "stride":
        stride = max(1, total_frames // num_frames)
        indices = []
        for i in range(num_frames):
            idx = min(i * stride, total_frames - 1)
            indices.append(idx)
        return indices

    raise SamplingError(
        f"unknown sampling strategy {strategy!r}; supported strategies: ['uniform', 'stride']"
    )


def pad_or_truncate_frames(
    frames: Sequence[np.ndarray],
    num_frames: int,
    pad_mode: str = "edge",
) -> Tuple[List[np.ndarray], List[bool]]:
    """Ensure a sequence of frames has length ``num_frames``.

    Parameters
    ----------
    frames: Sequence[np.ndarray]
        List or array of frames.
    num_frames: int
        Target sequence length.
    pad_mode: str
        How to pad if len(frames) < num_frames. "edge" repeats the last frame.

    Returns
    -------
    (padded_frames, validity_mask)
        padded_frames: List of length num_frames.
        validity_mask: List of bool of length num_frames, where True indicates
        an original frame and False indicates a padded frame.
    """
    if num_frames < 1:
        raise SamplingError(f"num_frames must be >= 1, got {num_frames}")
    count = len(frames)
    if count == 0:
        raise SamplingError("cannot pad an empty sequence of frames")

    if count >= num_frames:
        # Truncate
        return list(frames[:num_frames]), [True] * num_frames

    # Padding needed
    padded = list(frames)
    mask = [True] * count
    shortfall = num_frames - count

    if pad_mode == "edge":
        last_frame = frames[-1]
        for _ in range(shortfall):
            padded.append(last_frame.copy())
            mask.append(False)
    elif pad_mode == "zero":
        zero_frame = np.zeros_like(frames[0])
        for _ in range(shortfall):
            padded.append(zero_frame.copy())
            mask.append(False)
    else:
        raise SamplingError(f"unknown pad_mode {pad_mode!r}; supported: ['edge', 'zero']")

    return padded, mask


def resize_frame(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize frame to (target_height, target_width).

    Uses cv2.INTER_AREA for downsampling (anti-aliasing) and cv2.INTER_LINEAR
    for upsampling.
    """
    target_h, target_w = target_size
    if target_h < 1 or target_w < 1:
        raise SamplingError(f"target_size must be positive, got {target_size}")
    h, w = frame.shape[:2]
    if (h, w) == (target_h, target_w):
        return frame
    interpolation = cv2.INTER_AREA if (target_w < w or target_h < h) else cv2.INTER_LINEAR
    return cv2.resize(frame, (target_w, target_h), interpolation=interpolation)


def sample_frames_from_source(
    source: FrameSource,
    num_frames: int = 16,
    strategy: str = "uniform",
    target_size: Optional[Tuple[int, int]] = (224, 224),
) -> Tuple[np.ndarray, List[float], np.ndarray]:
    """Read and sample exactly ``num_frames`` from a FrameSource.

    Returns
    -------
    frames: np.ndarray
        Array of shape (num_frames, H, W, C) in uint8 [0, 255].
    timestamps: List[float]
        Monotonic timestamps in seconds for each frame.
    validity_mask: np.ndarray
        Boolean array of shape (num_frames,) indicating real (True) vs padded (False) frames.
    """
    records = list(source.iter_frames())
    total = len(records)
    if total == 0:
        raise SamplingError("cannot sample from empty FrameSource")

    all_frames = [rec.frame for rec in records]
    all_timestamps = [rec.timestamp for rec in records]

    if total < num_frames:
        selected_frames, mask_list = pad_or_truncate_frames(
            all_frames, num_frames, pad_mode="edge"
        )
        selected_timestamps = list(all_timestamps)
        dt = (
            (selected_timestamps[-1] - selected_timestamps[0]) / max(1, len(selected_timestamps) - 1)
            if len(selected_timestamps) > 1
            else (1.0 / 30.0)
        )
        if dt <= 0:
            dt = 1.0 / 30.0
        while len(selected_timestamps) < num_frames:
            selected_timestamps.append(selected_timestamps[-1] + dt)
        validity_mask = np.array(mask_list, dtype=bool)
    else:
        target_indices = sample_frame_indices(total, num_frames, strategy=strategy)
        selected_frames = [all_frames[i] for i in target_indices]
        selected_timestamps = [all_timestamps[i] for i in target_indices]
        validity_mask = np.ones(num_frames, dtype=bool)

    if target_size is not None:
        selected_frames = [resize_frame(f, target_size) for f in selected_frames]

    frames_array = np.stack(selected_frames, axis=0)
    return frames_array, selected_timestamps, validity_mask
