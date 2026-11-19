import numpy as np
import pytest

from mosaic.motion_gate import DEFAULT_GATE, GATE_SETTINGS, MotionGate, MotionGateError, make_gate
from mosaic.track_join import gated_windows


def test_gate_settings_are_valid_and_default_exists():
    assert DEFAULT_GATE in GATE_SETTINGS
    for name in GATE_SETTINGS:
        g = make_gate(name)
        t, b, h, k = GATE_SETTINGS[name]
        assert isinstance(g, MotionGate) and g.threshold == t and g.burst_threshold == b
        assert g.burst_hold_frames == h and g.keep_alive_frames == k
    assert GATE_SETTINGS["g02"][0] > GATE_SETTINGS["gdef"][0]        # the stronger gate has the higher threshold


def test_unknown_gate_is_rejected():
    with pytest.raises(MotionGateError):
        make_gate("nope")


def test_gate_keeps_first_frame_and_stronger_gate_keeps_fewer():
    rng = np.random.default_rng(0)
    frames = [np.clip(30 + 4 * i + rng.normal(0, 6, (64, 64)), 0, 255).astype(np.uint8) for i in range(64)]
    ts = [i / 12.8 for i in range(64)]
    k1 = make_gate("gdef").filter_clip(frames, ts).selected_indices
    k2 = make_gate("g02").filter_clip(frames, ts).selected_indices
    assert k1[0] == 0 and k2[0] == 0
    assert len(k2) <= len(k1)


def test_gated_windows_slots_follow_original_positions():
    kept = [0, 1, 5, 20, 21, 22, 23, 30]                       # positions kept among 64
    ids = [np.array([1]), np.array([1]), np.array([1]), np.array([2]), np.array([2]), np.array([2]), np.array([2]), np.array([-1])]
    scores = [np.array([0.9])] * len(kept)
    gw = gated_windows(kept, ids, scores, 64, min_frames=3)
    slots = gw["window_slots"]
    assert len(slots) == 7 and all(len(s) == 16 for s in slots)
    assert slots[0][0] == 0 and slots[0][1] == 1 and slots[0][5] == 2   # window 0 starts at position 0
    assert slots[0][2] == -1                                           # position 2 was skipped by the gate
    assert gw["windows"][0]["members"] == [1]                          # track 1: 3 detected frames in window 0
    assert 2 in gw["windows"][2]["members"]                            # window 2 covers positions 16..31 -> track 2
    assert gw["windows"][6]["members"] == []                           # nothing detected near the end
