"""Multi-object tracking engine (Block 5).

Maintains local temporal continuity of detected persons across video frames
using a 2D Kalman filter, IoU association (with ByteTrack-style two-stage
matching), and a strict lifecycle state machine.

Key Design Principles:
1. Motion & Spatial Association Only: NO heavy appearance Re-ID networks.
2. Irregular Time-Delta Support: Accommodates motion-gated skipped frames.
3. Explicit Predicted-Only Flagging: Distinguishes true detector hits from
   Kalman predictions during occlusions.
4. Deterministic: Given the same sequence of detections and timestamps,
   yields identical track IDs and trajectories.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import TrackerConfig
from .contracts import Box, Detection, TrackSummary


class TrackState(str, enum.Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    DELETED = "deleted"


class LifecycleEventType(str, enum.Enum):
    TRACK_STARTED = "TrackStarted"
    TRACK_UPDATED = "TrackUpdated"
    TRACK_PREDICTED = "TrackPredicted"
    TRACK_RECOVERED = "TrackRecovered"
    TRACK_ENDED = "TrackEnded"


@dataclass(frozen=True)
class LifecycleRecord:
    """Record of a lifecycle event emitted by a track."""

    track_id: int
    event: LifecycleEventType
    frame_index: int
    timestamp: float
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "event": self.event.value,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "details": self.details,
        }


def compute_iou(box1: Box, box2: Box) -> float:
    """Calculate Intersection-over-Union (IoU) between two bounding boxes."""
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    union_area = box1.area + box2.area - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def compute_iou_matrix(tracks: Sequence[Track], detections: Sequence[Detection]) -> np.ndarray:
    """Compute (N, M) matrix where element (i, j) is IoU(tracks[i], detections[j])."""
    n, m = len(tracks), len(detections)
    iou_mat = np.zeros((n, m), dtype=np.float32)
    for i, t in enumerate(tracks):
        for j, d in enumerate(detections):
            iou_mat[i, j] = compute_iou(t.box, d.box)
    return iou_mat


def linear_assignment(
    cost_matrix: np.ndarray,
    max_cost: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Assign tracks to detections using Hungarian matching or greedy fallback."""
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except Exception:
        # Greedy fallback if scipy is unavailable
        row_ind, col_ind = [], []
        used_rows = set()
        used_cols = set()
        flat_indices = np.argsort(cost_matrix.ravel())
        for idx in flat_indices:
            r = int(idx // cost_matrix.shape[1])
            c = int(idx % cost_matrix.shape[1])
            if r not in used_rows and c not in used_cols:
                row_ind.append(r)
                col_ind.append(c)
                used_rows.add(r)
                used_cols.add(c)
        row_ind = np.array(row_ind, dtype=int)
        col_ind = np.array(col_ind, dtype=int)

    matches: List[Tuple[int, int]] = []
    unmatched_rows = set(range(cost_matrix.shape[0]))
    unmatched_cols = set(range(cost_matrix.shape[1]))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= max_cost:
            matches.append((int(r), int(c)))
            unmatched_rows.discard(int(r))
            unmatched_cols.discard(int(c))

    return matches, sorted(unmatched_rows), sorted(unmatched_cols)


class KalmanBoxFilter:
    """2D Kalman filter tracking bounding box coordinates [x1, y1, x2, y2].

    State vector: [x1, y1, x2, y2, vx1, vy1, vx2, vy2]^T.
    Measurement vector: [x1, y1, x2, y2]^T.
    """

    def __init__(self, initial_box: Box):
        self.dim_x = 8
        self.dim_z = 4

        # State mean
        self.x = np.zeros(8, dtype=np.float32)
        self.x[:4] = [initial_box.x1, initial_box.y1, initial_box.x2, initial_box.y2]

        # Covariance P
        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 100.0  # high uncertainty for initial velocity

        # Measurement matrix H
        self.H = np.zeros((4, 8), dtype=np.float32)
        self.H[:4, :4] = np.eye(4, dtype=np.float32)

        # Measurement noise R
        self.R = np.eye(4, dtype=np.float32) * 1.0

        # Process noise base
        self.q_pos = 1.0
        self.q_vel = 0.1

    def predict(self, dt: float = 1.0) -> Box:
        """Advance state by dt seconds."""
        dt = max(0.001, float(dt))

        # Transition matrix F
        F = np.eye(8, dtype=np.float32)
        for i in range(4):
            F[i, i + 4] = dt

        # Process noise Q
        Q = np.eye(8, dtype=np.float32)
        Q[:4, :4] *= self.q_pos * (dt**2)
        Q[4:, 4:] *= self.q_vel * dt

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.get_box()

    def update(self, box: Box) -> Box:
        """Incorporate observed measurement box."""
        z = np.array([box.x1, box.y1, box.x2, box.y2], dtype=np.float32)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(self.dim_x, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        return self.get_box()

    def get_box(self) -> Box:
        """Extract valid Box from current state."""
        x1, y1, x2, y2 = float(self.x[0]), float(self.x[1]), float(self.x[2]), float(self.x[3])
        # Guarantee non-degenerate coordinates
        if x2 <= x1 + 1.0:
            x2 = x1 + 2.0
        if y2 <= y1 + 1.0:
            y2 = y1 + 2.0
        return Box(max(0.0, x1), max(0.0, y1), max(1.0, x2), max(1.0, y2))


@dataclass
class TrackObservation:
    """One historic step of a track."""

    frame_index: int
    timestamp: float
    box: Box
    observed: bool
    confidence: Optional[float]


class Track:
    """Stateful representation of one tracked human actor over time."""

    def __init__(
        self,
        track_id: int,
        initial_detection: Detection,
        min_hits: int = 2,
    ):
        self.track_id = int(track_id)
        self.state = TrackState.TENTATIVE
        self.min_hits = int(min_hits)
        self.kalman = KalmanBoxFilter(initial_detection.box)

        self.box = initial_detection.box
        self.observed = True
        self.confidence: Optional[float] = float(initial_detection.confidence)

        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.start_timestamp = float(initial_detection.timestamp)
        self.end_timestamp = float(initial_detection.timestamp)

        self.history: List[TrackObservation] = [
            TrackObservation(
                frame_index=initial_detection.frame_index,
                timestamp=float(initial_detection.timestamp),
                box=initial_detection.box,
                observed=True,
                confidence=float(initial_detection.confidence),
            )
        ]

        if self.min_hits <= 1:
            self.state = TrackState.CONFIRMED

    def predict_box(self, dt: float) -> Box:
        """Advance Kalman state to predict box for association."""
        dt = max(0.001, float(dt))
        self.box = self.kalman.predict(dt)
        return self.box

    def mark_missed(self, frame_index: int, timestamp: float) -> None:
        """Mark track as missed (unmatched) in this frame."""
        self.observed = False
        self.confidence = None
        self.age += 1
        self.time_since_update += 1
        self.end_timestamp = float(timestamp)
        if self.state == TrackState.CONFIRMED:
            self.state = TrackState.LOST

        self.history.append(
            TrackObservation(
                frame_index=frame_index,
                timestamp=float(timestamp),
                box=self.box,
                observed=False,
                confidence=None,
            )
        )

    def predict(self, frame_index: int, timestamp: float) -> Box:
        """Advance track state via Kalman prediction without observation."""
        dt = timestamp - self.end_timestamp
        if dt <= 0:
            dt = 1.0 / 30.0
        self.predict_box(dt)
        self.mark_missed(frame_index, timestamp)
        return self.box

    def update(self, detection: Detection) -> Box:
        """Update track state with a matched detection observation."""
        self.box = self.kalman.update(detection.box)
        self.observed = True
        self.confidence = float(detection.confidence)
        self.hits += 1
        self.age += 1
        self.time_since_update = 0
        self.end_timestamp = float(detection.timestamp)

        self.history.append(
            TrackObservation(
                frame_index=detection.frame_index,
                timestamp=float(detection.timestamp),
                box=self.box,
                observed=True,
                confidence=float(detection.confidence),
            )
        )

        if self.hits >= self.min_hits:
            self.state = TrackState.CONFIRMED

        return self.box

    def mark_deleted(self) -> None:
        self.state = TrackState.DELETED

    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    @property
    def is_deleted(self) -> bool:
        return self.state == TrackState.DELETED

    def to_summary(self) -> TrackSummary:
        """Produce the standardized TrackSummary contract."""
        observed_confidences = [
            h.confidence for h in self.history if h.observed and h.confidence is not None
        ]
        mean_conf = (
            float(np.mean(observed_confidences)) if observed_confidences else None
        )
        observed_count = sum(1 for h in self.history if h.observed)
        predicted_count = sum(1 for h in self.history if not h.observed)

        return TrackSummary(
            track_id=self.track_id,
            start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp,
            observed_count=observed_count,
            predicted_count=predicted_count,
            mean_detector_confidence=mean_conf,
        )


class Tracker:
    """ByteTrack-style Multi-Object Tracking Engine."""

    def __init__(self, cfg: Optional[TrackerConfig] = None):
        self.cfg = cfg if cfg is not None else TrackerConfig()
        self._next_id = 1
        self.tracks: List[Track] = []
        self.all_tracks: List[Track] = []  # all tracks created, including terminated
        self.lifecycle_records: List[LifecycleRecord] = []
        self._last_timestamp: Optional[float] = None

    def reset(self) -> None:
        """Reset tracker state between clips/videos."""
        self._next_id = 1
        self.tracks = []
        self.all_tracks = []
        self.lifecycle_records = []
        self._last_timestamp = None

    def _emit_event(
        self,
        track_id: int,
        event: LifecycleEventType,
        frame_index: int,
        timestamp: float,
        details: str = "",
    ) -> None:
        rec = LifecycleRecord(
            track_id=track_id,
            event=event,
            frame_index=frame_index,
            timestamp=timestamp,
            details=details,
        )
        self.lifecycle_records.append(rec)

    def update(
        self,
        detections: Sequence[Detection],
        frame_index: int,
        timestamp: float,
    ) -> List[Track]:
        """Update tracker with detections for the current frame.

        Parameters
        ----------
        detections:
            List of detected person boxes for this frame.
        frame_index:
            Original video frame index.
        timestamp:
            Physical timestamp in seconds.

        Returns
        -------
        List of active tracks (confirmed or tentative) for this frame.
        """
        self._last_timestamp = float(timestamp)

        # 1. Split detections into high and low confidence pools
        high_dets: List[Detection] = []
        low_dets: List[Detection] = []
        for det in detections:
            if det.confidence >= self.cfg.high_det_threshold:
                high_dets.append(det)
            elif det.confidence >= self.cfg.low_det_threshold:
                low_dets.append(det)

        # 2. Advance all existing tracks with Kalman prediction
        for t in self.tracks:
            dt = timestamp - t.end_timestamp
            if dt <= 0:
                dt = 1.0 / 30.0
            t.predict_box(dt)

        # 3. First Association: Match active tracks with high-confidence detections
        iou_mat_1 = compute_iou_matrix(self.tracks, high_dets)
        cost_mat_1 = 1.0 - iou_mat_1
        max_cost_1 = 1.0 - self.cfg.iou_threshold

        matches_1, unmatched_tracks_1, unmatched_high_dets = linear_assignment(
            cost_mat_1, max_cost=max_cost_1
        )

        # Process first-stage matches
        for t_idx, d_idx in matches_1:
            track = self.tracks[t_idx]
            det = high_dets[d_idx]
            was_lost = track.state == TrackState.LOST
            track.update(det)
            if was_lost:
                self._emit_event(
                    track.track_id,
                    LifecycleEventType.TRACK_RECOVERED,
                    frame_index,
                    timestamp,
                    f"Recovered via high-conf det (conf={det.confidence:.2f})",
                )
            else:
                self._emit_event(
                    track.track_id,
                    LifecycleEventType.TRACK_UPDATED,
                    frame_index,
                    timestamp,
                    f"Updated via high-conf det (conf={det.confidence:.2f})",
                )

        # 4. Second Association: Match remaining unmatched tracks with low-confidence detections
        rem_tracks = [self.tracks[i] for i in unmatched_tracks_1]
        iou_mat_2 = compute_iou_matrix(rem_tracks, low_dets)
        cost_mat_2 = 1.0 - iou_mat_2
        max_cost_2 = 1.0 - (self.cfg.iou_threshold * 0.75)  # slightly relaxed for low conf

        matches_2, unmatched_rem_tracks, _ = linear_assignment(cost_mat_2, max_cost=max_cost_2)

        for rem_idx, d_idx in matches_2:
            track = rem_tracks[rem_idx]
            det = low_dets[d_idx]
            was_lost = track.state == TrackState.LOST
            track.update(det)
            if was_lost:
                self._emit_event(
                    track.track_id,
                    LifecycleEventType.TRACK_RECOVERED,
                    frame_index,
                    timestamp,
                    f"Recovered via low-conf det (conf={det.confidence:.2f})",
                )
            else:
                self._emit_event(
                    track.track_id,
                    LifecycleEventType.TRACK_UPDATED,
                    frame_index,
                    timestamp,
                    f"Updated via low-conf det (conf={det.confidence:.2f})",
                )

        # 5. Tracks that remained unmatched are purely predicted
        final_unmatched_tracks = [rem_tracks[i] for i in unmatched_rem_tracks]
        for track in final_unmatched_tracks:
            track.mark_missed(frame_index, timestamp)
            self._emit_event(
                track.track_id,
                LifecycleEventType.TRACK_PREDICTED,
                frame_index,
                timestamp,
                f"Lost for {track.time_since_update} frames",
            )

        # 6. Initialize new tracks from unmatched high-confidence detections
        for d_idx in unmatched_high_dets:
            if len(self.tracks) < self.cfg.max_tracks:
                det = high_dets[d_idx]
                new_track = Track(
                    track_id=self._next_id,
                    initial_detection=det,
                    min_hits=self.cfg.min_hits,
                )
                self._next_id += 1
                self.tracks.append(new_track)
                self.all_tracks.append(new_track)
                self._emit_event(
                    new_track.track_id,
                    LifecycleEventType.TRACK_STARTED,
                    frame_index,
                    timestamp,
                    f"Started with det conf={det.confidence:.2f}",
                )

        # 7. Check for stale tracks to terminate
        active_survivors: List[Track] = []
        for track in self.tracks:
            if track.time_since_update > self.cfg.max_lost_frames:
                track.mark_deleted()
                self._emit_event(
                    track.track_id,
                    LifecycleEventType.TRACK_ENDED,
                    frame_index,
                    timestamp,
                    f"Terminated after {track.time_since_update} lost frames",
                )
            else:
                active_survivors.append(track)
        self.tracks = active_survivors

        # Return tracks active in this frame (confirmed, lost-predicted, or tentative)
        return [
            t for t in self.tracks
            if not t.is_deleted and (t.hits >= self.cfg.min_hits or t.state == TrackState.TENTATIVE)
        ]

    def get_track_summaries(self) -> List[TrackSummary]:
        """Return TrackSummary for all tracks observed across this video."""
        # Include confirmed tracks, or tracks with at least min_hits
        valid_tracks = [t for t in self.all_tracks if t.hits >= self.cfg.min_hits]
        return [t.to_summary() for t in valid_tracks]
