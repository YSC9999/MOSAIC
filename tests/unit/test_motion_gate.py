"""Unit tests for low-cost motion gating (Block 2)."""

import numpy as np
import pytest

from mosaic.contracts import MotionGateReport
from mosaic.motion_gate import GateDecision, GatedClipResult, MotionGate, MotionGateError


def test_motion_gate_initialization_validation():
    with pytest.raises(MotionGateError, match="threshold"):
        MotionGate(threshold=-0.01)
    with pytest.raises(MotionGateError, match="burst_threshold"):
        MotionGate(threshold=0.05, burst_threshold=0.02)
    with pytest.raises(MotionGateError, match="burst_hold_frames"):
        MotionGate(burst_hold_frames=-1)
    with pytest.raises(MotionGateError, match="keep_alive_frames"):
        MotionGate(keep_alive_frames=0)
    with pytest.raises(MotionGateError, match="downsample_size"):
        MotionGate(downsample_size=(4, 4))


def test_motion_gate_empty_input():
    gate = MotionGate()
    res = gate.filter_clip([])
    assert res.total_frames == 0
    assert len(res.selected_frames) == 0
    assert res.reduction_fraction == 0.0


def test_motion_gate_disabled_mode():
    frames = [np.full((32, 32, 3), i * 10, dtype=np.uint8) for i in range(10)]
    gate = MotionGate(enabled=False)
    res = gate.filter_clip(frames)
    assert res.total_frames == 10
    assert len(res.selected_indices) == 10
    assert len(res.skipped_indices) == 0
    assert res.reduction_fraction == 0.0
    assert all(d.reason == "no_gate" for d in res.decisions)


def test_motion_gate_static_sequence_enforces_keep_alive():
    # 35 identical frames: frame 0 selected, frame 15 selected, frame 30 selected
    frame = np.full((64, 64, 3), 128, dtype=np.uint8)
    frames = [frame.copy() for _ in range(35)]
    gate = MotionGate(
        threshold=0.02,
        keep_alive_frames=15,
        burst_hold_frames=5,
    )
    res = gate.filter_clip(frames)

    assert res.selected_indices == [0, 15, 30]
    assert len(res.skipped_indices) == 32
    assert res.decisions[0].reason == "first_frame"
    assert res.decisions[15].reason == "keep_alive"
    assert res.decisions[30].reason == "keep_alive"
    assert res.reduction_fraction == 32.0 / 35.0


def test_motion_gate_burst_mode_locks_gate_open():
    # 20 frames total:
    # 0: first frame (gray 50)
    # 1-3: static (gray 50) -> skipped
    # 4: sharp spike (gray 200) -> triggers burst (burst_hold_frames=3)
    # 5, 6, 7: hold open even if identical to frame 4
    # 8+: static identical to frame 4 -> skipped
    frames = [np.full((64, 64, 3), 50, dtype=np.uint8) for _ in range(4)]
    spike = [np.full((64, 64, 3), 200, dtype=np.uint8) for _ in range(10)]
    all_frames = frames + spike

    gate = MotionGate(
        threshold=0.02,
        burst_threshold=0.10,
        burst_hold_frames=3,
        keep_alive_frames=10,
    )
    res = gate.filter_clip(all_frames)

    # Frame 0 is first frame
    assert 0 in res.selected_indices
    # Frame 4 triggers burst
    assert res.decisions[4].reason == "burst"
    assert res.decisions[4].selected is True
    # Frames 5, 6, 7 are held open by burst mode
    assert res.decisions[5].reason == "burst" and res.decisions[5].selected is True
    assert res.decisions[6].reason == "burst" and res.decisions[6].selected is True
    assert res.decisions[7].reason == "burst" and res.decisions[7].selected is True
    # Frame 8 is identical to frame 7 and burst countdown expired -> skipped
    assert res.decisions[8].selected is False
    assert res.decisions[8].reason == "skipped"


def test_motion_gate_strict_temporal_contract():
    rng = np.random.RandomState(42)
    frames = [rng.randint(0, 256, (48, 48, 3), dtype=np.uint8) for _ in range(25)]
    custom_timestamps = [10.0 + i * 0.05 for i in range(25)]

    gate = MotionGate(threshold=0.03, burst_threshold=0.08)
    res = gate.filter_clip(frames, timestamps=custom_timestamps)

    # Invariants:
    # 1. Total count
    assert len(res.selected_indices) + len(res.skipped_indices) == 25
    # 2. Strict disjointness
    assert set(res.selected_indices).isdisjoint(set(res.skipped_indices))
    # 3. Union equals all indices
    assert sorted(res.selected_indices + res.skipped_indices) == list(range(25))
    # 4. Strict monotonicity
    assert res.selected_indices == sorted(res.selected_indices)
    assert res.skipped_indices == sorted(res.skipped_indices)
    # 5. Timestamp alignment
    for idx, ts in zip(res.selected_indices, res.selected_timestamps):
        assert ts == pytest.approx(custom_timestamps[idx])


def test_motion_gate_to_report_conversion():
    frames = [np.full((32, 32, 3), (i % 2) * 200, dtype=np.uint8) for i in range(20)]
    gate = MotionGate(threshold=0.02, burst_threshold=0.05)
    res = gate.filter_clip(frames)

    report = res.to_report()
    assert isinstance(report, MotionGateReport)
    assert report.total_frames == 20
    assert report.selected_frames == len(res.selected_indices)
    assert report.skipped_frames == len(res.skipped_indices)
    assert report.reduction_fraction == pytest.approx(res.reduction_fraction)

    # JSON roundtrip
    dict_rep = report.to_dict()
    restored = MotionGateReport.from_dict(dict_rep)
    assert restored == report
