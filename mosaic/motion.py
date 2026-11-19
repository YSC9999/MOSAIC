"""Window motion score (spec section 2): d_t = mean|g_{t+1} - g_t| on 112x112 grayscale in [0,1];
m_w = mean of the d_t that fall inside window w. A plain feature, nothing is skipped."""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from .ingestion import VideoReader
from .windows import sample_indices, window_index_matrix


def gray_frames(path, max_frames: int = 64, size: int = 112, n_frames: Optional[int] = None) -> np.ndarray:
    def prep(f):
        return cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (size, size), interpolation=cv2.INTER_AREA)

    if n_frames is None:
        recs = VideoReader(path).read_all()
        idx, _ = sample_indices(len(recs), max_frames)
        return np.stack([prep(recs[i].frame) for i in idx]).astype(np.float32) / 255.0
    idx, _ = sample_indices(n_frames, max_frames)
    wanted, last, got = set(idx), max(idx), {}
    for rec in VideoReader(path).iter_frames():
        if rec.frame_index in wanted:
            got[rec.frame_index] = prep(rec.frame)
        if rec.frame_index >= last:
            break
    missing = sorted(wanted - set(got))
    if missing:
        raise RuntimeError(f"{path}: frames {missing[:5]} not decodable (declared {n_frames})")
    return np.stack([got[i] for i in idx]).astype(np.float32) / 255.0


def window_motion(gray: np.ndarray, window_len: int = 16, stride: int = 8) -> np.ndarray:
    """(W,) motion score per window from (T, H, W) grayscale frames."""
    d = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))           # (T-1,)
    mat = window_index_matrix(len(gray), window_len, stride)
    return np.asarray([d[row[:-1]].mean() for row in mat], np.float32)
