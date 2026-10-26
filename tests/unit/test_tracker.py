"""Unit tests for multi-object tracking engine (Block 5).

Verifies:
- Stationary detections maintain a stable track ID.
- Moving detections produce sensible trajectory updates.
- Missing detections create predicted-only observations.
- Stale tracks terminate after configured timeout.
- Track IDs are unique within a clip.
- Empty frames do not corrupt active tracks.
- Lifecycle events occur in valid order (TrackStarted -> TrackUpdated/Predicted -> TrackEnded).
- Identical inputs produce identical track IDs and bounding boxes.
- Scripted occlusion survives and recovers the same track ID.
"""

from __future__ import annotations

import pytest

from mosaic.config import TrackerConfig
from mosaic.contracts import Box, Detection
from mosaic.tracker import (
    LifecycleEventType,
    Track,
    Tracker,
    TrackState,
    compute_iou,
)


def _make_det(x1: float, y1: float, x2: float, y2: float, frame_idx: int, ts: float, conf: float = 0.9) -> Detection:
    return Detection(
        box=Box(x1, y1, x2, y2),
        confidence=conf,
        class_name="person",
        frame_index=frame_idx,
        timestamp=ts,
    )


def test_compute_iou():
    b1 = Box(10, 10, 30, 30)  # area 400
    b2 = Box(10, 10, 30, 30)  # area 400
    assert compute_iou(b1, b2) == pytest.approx(1.0)

    b3 = Box(50, 50, 70, 70)  # no overlap
    assert compute_iou(b1, b3) == 0.0

    b4 = Box(20, 10, 40, 30)  # 50% overlap: intersection 10x20=200, union 400+400-200=600
    assert compute_iou(b1, b4) == pytest.approx(200.0 / 600.0)


def test_stationary_detection_maintains_stable_id():
    cfg = TrackerConfig(min_hits=1)
    tracker = Tracker(cfg)

    # Frame 0
    dets_0 = [_make_det(100, 100, 150, 200, 0, 0.0)]
    active_0 = tracker.update(dets_0, 0, 0.0)
    assert len(active_0) == 1
    t_id = active_0[0].track_id

    # Frame 1 - 5 (same position)
    for f in range(1, 6):
        dets = [_make_det(100, 100, 150, 200, f, f * 0.033)]
        active = tracker.update(dets, f, f * 0.033)
        assert len(active) == 1
        assert active[0].track_id == t_id
        assert active[0].observed is True


def test_moving_detection_updates_trajectory():
    cfg = TrackerConfig(min_hits=1)
    tracker = Tracker(cfg)

    # Moving to the right: x increases by 5 pixels per frame
    for f in range(10):
        x = 100.0 + f * 5.0
        dets = [_make_det(x, 100, x + 50, 200, f, f * 0.1)]
        active = tracker.update(dets, f, f * 0.1)
        assert len(active) == 1
        assert active[0].track_id == 1
        assert active[0].box.x1 == pytest.approx(x, abs=2.0)


def test_missing_detection_creates_predicted_only_observation():
    cfg = TrackerConfig(min_hits=1, max_lost_frames=5)
    tracker = Tracker(cfg)

    # Observed in frame 0
    tracker.update([_make_det(100, 100, 150, 200, 0, 0.0)], 0, 0.0)

    # Missing in frame 1
    active_1 = tracker.update([], 1, 0.033)
    assert len(active_1) == 1
    t = active_1[0]
    assert t.track_id == 1
    assert t.observed is False
    assert t.confidence is None
    assert t.time_since_update == 1


def test_stale_track_terminates_after_timeout():
    cfg = TrackerConfig(min_hits=1, max_lost_frames=3)
    tracker = Tracker(cfg)

    tracker.update([_make_det(100, 100, 150, 200, 0, 0.0)], 0, 0.0)

    # Miss 1, 2, 3 frames
    tracker.update([], 1, 0.033)
    tracker.update([], 2, 0.066)
    active_3 = tracker.update([], 3, 0.099)
    assert len(active_3) == 1

    # Frame 4: lost for 4 frames > max_lost_frames(3) -> terminated
    active_4 = tracker.update([], 4, 0.132)
    assert len(active_4) == 0


def test_unique_track_ids_within_clip():
    cfg = TrackerConfig(min_hits=1)
    tracker = Tracker(cfg)

    # Two distinct people far apart
    dets_0 = [
        _make_det(50, 50, 100, 150, 0, 0.0),
        _make_det(300, 300, 350, 400, 0, 0.0),
    ]
    active_0 = tracker.update(dets_0, 0, 0.0)
    assert len(active_0) == 2
    ids = {t.track_id for t in active_0}
    assert len(ids) == 2


def test_empty_frames_do_not_corrupt_active_tracks():
    cfg = TrackerConfig(min_hits=1, max_lost_frames=10)
    tracker = Tracker(cfg)

    tracker.update([_make_det(100, 100, 150, 200, 0, 0.0)], 0, 0.0)
    # 3 empty frames
    for f in range(1, 4):
        tracker.update([], f, f * 0.033)

    # Person re-appears
    active = tracker.update([_make_det(102, 100, 152, 200, 4, 4 * 0.033)], 4, 4 * 0.033)
    assert len(active) == 1
    assert active[0].track_id == 1
    assert active[0].observed is True


def test_lifecycle_event_ordering():
    cfg = TrackerConfig(min_hits=1, max_lost_frames=2)
    tracker = Tracker(cfg)

    # Start
    tracker.update([_make_det(100, 100, 150, 200, 0, 0.0)], 0, 0.0)
    # Update
    tracker.update([_make_det(102, 100, 152, 200, 1, 0.033)], 1, 0.033)
    # Predict (lost)
    tracker.update([], 2, 0.066)
    # Recover
    tracker.update([_make_det(104, 100, 154, 200, 3, 0.099)], 3, 0.099)
    # Predict until termination
    tracker.update([], 4, 0.132)
    tracker.update([], 5, 0.165)
    tracker.update([], 6, 0.198)

    events = [r.event for r in tracker.lifecycle_records if r.track_id == 1]
    assert events[0] == LifecycleEventType.TRACK_STARTED
    assert LifecycleEventType.TRACK_UPDATED in events
    assert LifecycleEventType.TRACK_PREDICTED in events
    assert LifecycleEventType.TRACK_RECOVERED in events
    assert events[-1] == LifecycleEventType.TRACK_ENDED


def test_identical_input_produces_identical_results():
    cfg = TrackerConfig(min_hits=1)
    tracker1 = Tracker(cfg)
    tracker2 = Tracker(cfg)

    dets_seq = [
        [_make_det(100 + i * 2, 100, 150 + i * 2, 200, i, i * 0.05)]
        for i in range(5)
    ]

    res1 = [tracker1.update(d, i, i * 0.05) for i, d in enumerate(dets_seq)]
    res2 = [tracker2.update(d, i, i * 0.05) for i, d in enumerate(dets_seq)]

    for r1, r2 in zip(res1, res2):
        assert len(r1) == len(r2)
        assert r1[0].track_id == r2[0].track_id
        assert r1[0].box.to_list() == r2[0].box.to_list()
