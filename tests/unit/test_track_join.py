import numpy as np

from mosaic.track_join import drop_short_tracks, window_members


def test_drop_short_tracks():
    ids = [np.array([1, 2]), np.array([1, 2]), np.array([1, -1]), np.array([3])]
    out = drop_short_tracks(ids, min_frames=3)
    assert [a.tolist() for a in out] == [[1, -1], [1, -1], [1, -1], [-1]]


def test_window_members_threshold_rank_and_cap():
    # 16-frame window; track 5 seen 6x (score .9), track 7 seen 6x (score .5), track 9 seen 3x
    ids, scores = [], []
    for r in range(16):
        a, s = [], []
        if r < 6:
            a += [5, 7]; s += [0.9, 0.5]
        if r < 3:
            a.append(9); s.append(0.99)
        ids.append(np.array(a, int)); scores.append(np.array(s))
    w = window_members(ids, scores, [list(range(16))], min_frames=4, cap=16)[0]
    assert w["members"] == [5, 7] and w["count_uncapped"] == 2
    w1 = window_members(ids, scores, [list(range(16))], min_frames=4, cap=1)[0]
    assert w1["members"] == [5] and w1["count_uncapped"] == 2


def test_window_members_tie_breaks_on_lowest_id():
    ids = [np.array([8, 4]) for _ in range(5)]
    scores = [np.array([0.6, 0.6]) for _ in range(5)]
    assert window_members(ids, scores, [list(range(5))])[0]["members"] == [4, 8]


def test_nms_keep_removes_duplicates_keeps_best():
    from mosaic.track_join import nms_keep
    boxes = np.array([[0, 0, 100, 200], [2, 2, 102, 202], [300, 0, 400, 200]], np.float32)
    scores = np.array([0.6, 0.9, 0.5], np.float32)
    assert nms_keep(boxes, scores, 0.7).tolist() == [False, True, True]
    assert nms_keep(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), 0.7).shape == (0,)
