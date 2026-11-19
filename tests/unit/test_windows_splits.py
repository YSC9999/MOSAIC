import pytest

from mosaic.splits import (
    FIT,
    INTERNAL_VAL,
    TEST,
    SplitError,
    build_grouped_split,
    source_video_id,
    validate_split,
)
from mosaic.windows import (
    WindowError,
    pad_indices,
    select_frame_indices,
    top_k_for,
    window_index_matrix,
    window_starts,
)


def test_native_frames_used_when_short():
    assert select_frame_indices(50) == list(range(50))


def test_subsample_distinct_and_endpoints():
    idx = select_frame_indices(150, 64)
    assert len(idx) == 64 and len(set(idx)) == 64
    assert idx[0] == 0 and idx[-1] == 149
    assert idx == sorted(idx)


def test_window_starts_50_frames():
    assert window_starts(50) == [0, 7, 14, 20, 27, 34]


def test_window_starts_64_frames():
    assert window_starts(64) == [0, 8, 16, 24, 32, 40, 48]


def test_windows_cover_every_frame():
    for n in range(16, 130):
        m = window_index_matrix(n)
        assert set(m.ravel().tolist()) == set(range(n))
        assert m.max() == n - 1


def test_exactly_one_window():
    assert window_starts(16) == [0]


def test_too_short_raises_and_pad_flags():
    with pytest.raises(WindowError):
        window_starts(10)
    idx, is_pad = pad_indices(list(range(10)))
    assert len(idx) == 16 and idx[-1] == 9
    assert is_pad == [False] * 10 + [True] * 6


def test_top_k():
    assert top_k_for(6) == 3 and top_k_for(7) == 3 and top_k_for(1) == 1


def test_source_video_id():
    assert source_video_id("train/Fight/0H2s9UJcNJ0_12.avi") == "0H2s9UJcNJ0"
    assert source_video_id("train\\NonFight\\a_b_3.avi") == "a_b"
    assert source_video_id("val/Fight/noclipnum.avi") == "noclipnum"


def _entries():
    out = []
    for s in range(60):
        for c in range(3):
            out.append({"rel_path": f"train/X/src{s}_{c}.avi", "target": (s + c) % 2, "split": "train"})
    out.append({"rel_path": "val/Fight/t_0.avi", "target": 1, "split": "test"})
    return out


def test_grouped_split_no_leak_and_test_kept():
    entries = _entries()
    a = build_grouped_split(entries, holdout_fraction=0.1, seed=42)
    assert a["val/Fight/t_0.avi"] == TEST
    assert {FIT, INTERNAL_VAL} <= set(a.values())
    validate_split(entries, a)
    held = sum(v == INTERNAL_VAL for v in a.values())
    assert 9 <= held <= 27  # ~10% of 180 clips, whole groups of 3


def test_grouped_split_deterministic():
    e = _entries()
    assert build_grouped_split(e, seed=42) == build_grouped_split(e, seed=42)


def test_validate_detects_leak():
    e = _entries()[:2]
    with pytest.raises(SplitError):
        validate_split(e, {e[0]["rel_path"]: FIT, e[1]["rel_path"]: INTERNAL_VAL})


def test_window_motion_static_and_moving():
    import numpy as np
    from mosaic.motion import window_motion
    static = np.zeros((50, 8, 8), np.float32)
    assert np.allclose(window_motion(static), 0)
    moving = np.stack([np.full((8, 8), 0.01 * i, np.float32) for i in range(50)])
    wm = window_motion(moving)
    assert wm.shape == (6,) and np.allclose(wm, 0.01, atol=1e-5)


def test_sample_indices_is_always_64_and_marks_padding():
    from mosaic.windows import sample_indices, window_starts
    for n in (1, 16, 41, 64, 65, 150, 900):
        idx, pad = sample_indices(n)
        assert len(idx) == 64 and len(pad) == 64
        assert len(window_starts(64)) == 7
        real = [i for i, p in zip(idx, pad) if not p]
        assert len(real) == min(n, 64) and real == sorted(real) and len(set(real)) == len(real)
        assert all(i == idx[len(real) - 1] for i, p in zip(idx, pad) if p)     # repeats copy the last real frame
        assert not pad[0]                                                        # position 0 is always a real frame
    idx, pad = sample_indices(150)
    assert not any(pad) and idx[0] == 0 and idx[-1] == 149
