"""Spatio-temporal outputs (spec section 9, localization variant).

Output 1  clip class      (computed elsewhere: top-k mean of window logits)
Output 2  time intervals  from window scores: size-3 median filter, threshold, merge consecutive windows
Output 3  where           one *group box* per violent window from the person-branch attention - an area of action,
                          not a judgment about individuals. Attention is the model's focus, not verified involvement.

Pure numpy; no model or video dependencies, so it is unit-testable.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def median_filter3(p: Sequence[float]) -> np.ndarray:
    """Size-3 median filter with edge replication."""
    a = np.asarray(p, dtype=np.float64)
    if len(a) < 3:
        return a.copy()
    pad = np.concatenate([a[:1], a, a[-1:]])
    return np.median(np.stack([pad[:-2], pad[1:-1], pad[2:]]), axis=0)


def window_time_spans(timestamps: Sequence[float], starts: Sequence[int], window_len: int = 16) -> np.ndarray:
    """(W, 2) [t_start, t_end] seconds per window. A window ends one frame interval after its last frame."""
    ts = np.asarray(timestamps, dtype=np.float64)
    dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.0
    spans = []
    for s in starts:
        e = min(s + window_len - 1, len(ts) - 1)
        spans.append((ts[s], ts[e] + dt))
    return np.asarray(spans)


def intervals_from_windows(p: Sequence[float], spans: np.ndarray, threshold: float = 0.5) -> Dict:
    """Violent intervals from per-window probabilities.

    Returns ``{"smoothed", "windows": violent window indices, "intervals": [(t0, t1, first, last)], "primary": idx}``.
    If nothing passes the threshold the single highest-probability window is used.
    """
    p = np.asarray(p, dtype=np.float64)
    sm = median_filter3(p)
    flags = sm >= threshold
    if not flags.any():
        flags = np.zeros_like(flags)
        flags[int(np.argmax(sm))] = True
    runs, k = [], 0
    while k < len(flags):
        if flags[k]:
            j = k
            while j + 1 < len(flags) and flags[j + 1]:
                j += 1
            runs.append((float(spans[k, 0]), float(spans[j, 1]), k, j))
            k = j + 1
        else:
            k += 1
    peak = int(np.argmax(sm))
    primary = next((i for i, r in enumerate(runs) if r[2] <= peak <= r[3]), 0)
    return {"smoothed": sm, "windows": [int(i) for i in np.nonzero(flags)[0]], "intervals": runs, "primary": primary}


def select_people(attn: Sequence[float], cum: float = 0.6, max_people: int = 4) -> List[int]:
    """Indices of the smallest set of people whose attention sums to >= ``cum`` (at most ``max_people``)."""
    a = np.asarray(attn, dtype=np.float64)
    order = [int(i) for i in np.argsort(-a) if a[i] > 0]
    chosen, total = [], 0.0
    for i in order:
        chosen.append(i)
        total += a[i]
        if total >= cum or len(chosen) >= max_people:
            break
    return chosen


def union_box(boxes: Sequence[Sequence[float]]) -> Optional[np.ndarray]:
    if len(boxes) == 0:
        return None
    b = np.asarray(boxes, dtype=np.float64)
    return np.array([b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()])


def group_boxes(
    attn: Sequence[float],
    member_ids: Sequence[int],
    boxes_by_pos: Dict[int, Dict[int, Sequence[float]]],
    positions: Sequence[int],
    frame_wh: Tuple[float, float],
    cum: float = 0.6,
    max_people: int = 4,
    max_area_frac: float = 0.6,
) -> Dict[int, Optional[List[float]]]:
    """One group box per frame position in a window.

    ``boxes_by_pos[pos][track_id]`` holds the tracked person boxes. The union of the selected high-attention people is
    drawn; if it covers more than ``max_area_frac`` of the frame, the selection shrinks to the top two people. Frames with
    none of the selected people borrow the nearest frame's box inside the window (held, not invented).
    """
    W, H = frame_wh
    ids = [int(member_ids[i]) for i in select_people(attn, cum, max_people)]

    def box_at(pos: int, tids: Sequence[int]) -> Optional[np.ndarray]:
        present = boxes_by_pos.get(pos, {})
        return union_box([present[t] for t in tids if t in present])

    out: Dict[int, Optional[np.ndarray]] = {}
    for pos in positions:
        b = box_at(pos, ids)
        if b is not None and len(ids) > 2 and (b[2] - b[0]) * (b[3] - b[1]) > max_area_frac * W * H:
            b = box_at(pos, ids[:2]) if box_at(pos, ids[:2]) is not None else b
        out[pos] = b
    have = [p for p in positions if out[p] is not None]
    for pos in positions:
        if out[pos] is None and have:
            out[pos] = out[min(have, key=lambda q: abs(q - pos))]
    return {pos: (None if b is None else [float(x) for x in b]) for pos, b in out.items()}


def temporal_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0
