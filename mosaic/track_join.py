"""ByteTrack over cached detections + window membership (spec section 4).

Runs in ``.venv`` (``trackers``, ``supervision``); those imports are lazy so the
pure window-membership logic can be tested in the main environment.

Tracker settings: fixed-rate mode (one step per sampled frame, timestamp=None),
``frame_rate=30`` and ``lost_track_buffer=16`` so the patience is exactly 16
sampled frames (``trackers`` scales the buffer by frame_rate / 30).
Detections are joined back to cache rows through a ``det_idx`` data field that
``sv.Detections`` carries through indexing, so the join is exact.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

HIGH_THRESHOLD = 0.25
LOW_THRESHOLD = 0.10
LOST_PATIENCE_FRAMES = 16
MIN_TRACK_FRAMES = 3
MIN_WINDOW_FRAMES = 4
MAX_PEOPLE_PER_WINDOW = 16


def make_tracker(high: float = HIGH_THRESHOLD):
    from trackers import ByteTrackTracker

    return ByteTrackTracker(
        lost_track_buffer=LOST_PATIENCE_FRAMES,
        frame_rate=30.0,
        track_activation_threshold=high,
        high_conf_det_threshold=high,
        minimum_consecutive_frames=2,
    )


def nms_keep(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy NMS; returns a boolean keep mask aligned to the input rows (DETR-style duplicates)."""
    n = len(boxes)
    keep = np.zeros(n, bool)
    if n == 0:
        return keep
    order = np.argsort(-scores, kind="stable")
    suppressed = np.zeros(n, bool)
    for i in order:
        if suppressed[i]:
            continue
        keep[i] = True
        xx1 = np.maximum(boxes[i, 0], boxes[:, 0]); yy1 = np.maximum(boxes[i, 1], boxes[:, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[:, 2]); yy2 = np.minimum(boxes[i, 3], boxes[:, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iou = inter / np.maximum(area[i] + area - inter, 1e-9)
        suppressed |= iou > iou_thr
        suppressed[i] = False
    return keep


def track_clip(frames: Sequence[Dict[str, Any]], high: float = HIGH_THRESHOLD,
               nms_iou: Optional[float] = None) -> List[np.ndarray]:
    """Run a fresh tracker over cached frames; return per-frame track IDs aligned to cache rows.

    ``nms_iou`` (default off) removes duplicate detections before tracking; removed rows get ID -1."""
    import supervision as sv

    tracker = make_tracker(high)
    out: List[np.ndarray] = []
    for f in frames:
        boxes = np.asarray(f["boxes"], np.float32).reshape(-1, 4)
        scores = np.asarray(f["scores"], np.float32)
        n = len(boxes)
        ids = np.full(n, -1, dtype=np.int64)
        keep = scores >= LOW_THRESHOLD
        if nms_iou is not None and n:
            keep &= nms_keep(boxes, scores, nms_iou)
        det = sv.Detections(
            xyxy=boxes[keep], confidence=scores[keep], class_id=np.zeros(int(keep.sum()), int),
            data={"det_idx": np.nonzero(keep)[0]},
        )
        res = tracker.update(det)
        if len(res):
            ids[np.asarray(res.data["det_idx"], int)] = np.asarray(res.tracker_id, np.int64)
        out.append(ids)
    return out


def drop_short_tracks(ids: List[np.ndarray], min_frames: int = MIN_TRACK_FRAMES) -> List[np.ndarray]:
    counts: Dict[int, int] = defaultdict(int)
    for a in ids:
        for t in set(a[a >= 0].tolist()):
            counts[t] += 1
    short = {t for t, c in counts.items() if c < min_frames}
    return [np.where(np.isin(a, list(short)), -1, a) if short else a.copy() for a in ids]


def window_members(
    ids: List[np.ndarray],
    scores: List[np.ndarray],
    window_rows: Sequence[Sequence[int]],
    min_frames: int = MIN_WINDOW_FRAMES,
    cap: int = MAX_PEOPLE_PER_WINDOW,
) -> List[Dict[str, Any]]:
    """Per window: ranked member track IDs (capped) and the uncapped count.

    Rank: observed frames (desc), mean detection score (desc), track ID (asc).
    """
    result = []
    for rows in window_rows:
        obs: Dict[int, List[float]] = defaultdict(list)
        for r in rows:
            for t, s in zip(ids[r].tolist(), np.asarray(scores[r]).tolist()):
                if t >= 0:
                    obs[t].append(s)
        cand = [(len(v), float(np.mean(v)), t) for t, v in obs.items() if len(v) >= min_frames]
        cand.sort(key=lambda c: (-c[0], -c[1], c[2]))
        result.append({"members": [c[2] for c in cand[:cap]], "count_uncapped": len(cand)})
    return result


def gated_windows(kept_pos: Sequence[int], ids: List[np.ndarray], scores: List[np.ndarray], n_sampled: int,
                  min_frames: int = MIN_WINDOW_FRAMES, cap: int = MAX_PEOPLE_PER_WINDOW) -> Dict[str, Any]:
    """Window membership when only some sampled frames were detected.

    ``kept_pos[i]`` is the position (0..n_sampled-1) of the i-th detected row. Windows keep the layout of the
    ungated pipeline; a person belongs to a window if seen in >= ``min_frames`` of its detected frames.
    Returns ``window_slots`` (per window, 16 entries: row index or -1) and ``windows`` (members, count).
    """
    from .windows import window_index_matrix

    row_of = {int(p): r for r, p in enumerate(kept_pos)}
    mat = window_index_matrix(n_sampled)
    slots = [[row_of.get(int(p), -1) for p in row] for row in mat]
    rows_in = [[r for r in s if r >= 0] for s in slots]
    return {"window_slots": slots, "windows": window_members(ids, scores, rows_in, min_frames, cap)}
