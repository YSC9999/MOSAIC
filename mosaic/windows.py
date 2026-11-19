"""Frame selection and window layout for the v3 pipeline (spec section 1).

Clips with at most ``max_frames`` frames use every native frame (no duplicates,
so motion and kinematic rates never see a zero time step). Longer clips are
subsampled to ``max_frames`` distinct, evenly spaced frames.

Windows of ``window_len`` frames are laid out so that every sampled frame is
covered: W = ceil((T - window_len) / stride) + 1 windows whose starts are spread
evenly from 0 to T - window_len.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np


class WindowError(ValueError):
    """Raised when a clip cannot be split into windows."""


def select_frame_indices(n_frames: int, max_frames: int = 64) -> List[int]:
    """Return the source frame indices to use for a clip with ``n_frames`` frames."""
    if n_frames <= 0:
        raise WindowError(f"clip has no frames (n_frames={n_frames})")
    if max_frames < 1:
        raise WindowError(f"max_frames must be >= 1, got {max_frames}")
    if n_frames <= max_frames:
        return list(range(n_frames))
    idx = [round(i * (n_frames - 1) / (max_frames - 1)) for i in range(max_frames)]
    if len(set(idx)) != len(idx):  # cannot happen when n_frames > max_frames, but guard it
        raise WindowError("frame subsampling produced duplicate indices")
    return idx


def sample_indices(n_frames: int, total: int = 64) -> tuple[List[int], List[bool]]:
    """Exactly ``total`` sample positions for a clip of any length, so every clip gets the same window layout.

    ``n_frames >= total``: evenly spaced distinct frames (nothing padded).
    ``n_frames <  total``: every native frame, then the last frame repeated; ``is_pad`` marks the repeats. Padded positions
    must be excluded from detection, tracking and every rate computation (their time step would be zero).
    """
    idx = select_frame_indices(n_frames, total)
    is_pad = [False] * len(idx)
    while len(idx) < total:
        idx.append(idx[-1])
        is_pad.append(True)
    return idx, is_pad


def window_starts(n_sampled: int, window_len: int = 16, stride: int = 8) -> List[int]:
    """Start positions (into the sampled frame list) of each window.

    Requires ``n_sampled >= window_len``; shorter clips must be padded first
    (see :func:`pad_indices`).
    """
    if window_len < 1 or stride < 1:
        raise WindowError("window_len and stride must be >= 1")
    if n_sampled < window_len:
        raise WindowError(f"need at least {window_len} frames, got {n_sampled}")
    span = n_sampled - window_len
    if span == 0:
        return [0]
    n_windows = math.ceil(span / stride) + 1
    return [int(round(k * span / (n_windows - 1))) for k in range(n_windows)]


def pad_indices(indices: Sequence[int], window_len: int = 16) -> tuple[List[int], List[bool]]:
    """Pad a short index list by repeating its last frame.

    Returns the padded indices and a parallel ``is_pad`` flag list. Padded frames
    must be excluded from every rate computation.
    """
    idx = list(indices)
    if not idx:
        raise WindowError("cannot pad an empty index list")
    is_pad = [False] * len(idx)
    while len(idx) < window_len:
        idx.append(idx[-1])
        is_pad.append(True)
    return idx, is_pad


def top_k_for(n_windows: int, fraction: float = 0.4) -> int:
    """Number of windows averaged for the clip logit: ceil(fraction * W), at least 1."""
    return max(1, math.ceil(fraction * n_windows))


def window_index_matrix(n_sampled: int, window_len: int = 16, stride: int = 8) -> np.ndarray:
    """(W, window_len) array of positions into the sampled frame list."""
    starts = window_starts(n_sampled, window_len, stride)
    return np.asarray([[s + j for j in range(window_len)] for s in starts], dtype=np.int64)
