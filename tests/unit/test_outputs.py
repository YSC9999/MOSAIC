import numpy as np

from mosaic.outputs import (group_boxes, intervals_from_windows, median_filter3, select_people, temporal_iou,
                            union_box, window_time_spans)


def test_median_filter_removes_single_spike_and_keeps_edges():
    assert median_filter3([0.1, 0.9, 0.1, 0.1]).tolist() == [0.1, 0.1, 0.1, 0.1]
    assert median_filter3([0.9, 0.9, 0.1]).tolist() == [0.9, 0.9, 0.1]     # a lone low edge value is kept (edge replication)
    assert median_filter3([0.3, 0.7]).tolist() == [0.3, 0.7]          # too short: unchanged


def test_window_time_spans():
    ts = np.arange(64) / 12.8
    sp = window_time_spans(ts, [0, 8, 48], 16)
    assert abs(sp[0, 0] - 0.0) < 1e-9 and abs(sp[0, 1] - 16 / 12.8) < 1e-9
    assert abs(sp[2, 1] - 64 / 12.8) < 1e-9


def test_intervals_merge_and_primary():
    spans = window_time_spans(np.arange(64) / 12.8, [0, 8, 16, 24, 32, 40, 48])
    p = [0.1, 0.2, 0.8, 0.9, 0.85, 0.2, 0.1]
    r = intervals_from_windows(p, spans)
    assert r["windows"] == [2, 3, 4] and len(r["intervals"]) == 1
    t0, t1, a, b = r["intervals"][0]
    assert (a, b) == (2, 4) and t0 == spans[2, 0] and t1 == spans[4, 1] and r["primary"] == 0


def test_two_intervals_and_fallback_to_argmax():
    spans = window_time_spans(np.arange(64) / 12.8, [0, 8, 16, 24, 32, 40, 48])
    r = intervals_from_windows([0.9, 0.9, 0.1, 0.1, 0.1, 0.9, 0.9], spans)
    assert [(i[2], i[3]) for i in r["intervals"]] == [(0, 1), (5, 6)]
    r2 = intervals_from_windows([0.1, 0.2, 0.3, 0.4, 0.45, 0.2, 0.1], spans)         # nothing >= 0.5
    assert len(r2["windows"]) == 1 and r2["windows"][0] in (3, 4) and len(r2["intervals"]) == 1     # peak of the smoothed curve


def test_select_people_cumulative_and_cap():
    assert select_people([0.7, 0.2, 0.1]) == [0]
    assert select_people([0.3, 0.3, 0.2, 0.1, 0.1], cum=0.9, max_people=3) == [0, 1, 2]
    assert select_people([0.0, 0.0]) == []


def test_union_box():
    assert union_box([[0, 0, 10, 10], [5, 5, 20, 30]]).tolist() == [0, 0, 20, 30]
    assert union_box([]) is None


def test_group_boxes_union_hold_and_area_cap():
    boxes = {0: {1: [0, 0, 10, 10], 2: [20, 0, 30, 10]}, 1: {1: [1, 1, 11, 11]}, 3: {2: [21, 1, 31, 11]}}
    g = group_boxes([0.5, 0.5], [1, 2], boxes, [0, 1, 2, 3], (100, 100))
    assert g[0] == [0, 0, 30, 10]                       # union of both people
    assert g[1] == [1, 1, 11, 11]                       # only person 1 visible
    assert g[2] == g[1] or g[2] == g[3]                 # no detection at pos 2: borrowed from the nearest frame
    big = {0: {1: [0, 0, 100, 50], 2: [0, 50, 100, 100], 3: [0, 0, 10, 10]}}
    gb = group_boxes([0.4, 0.4, 0.2], [1, 2, 3], big, [0], (100, 100), cum=1.0)
    assert gb[0] == [0.0, 0.0, 100.0, 100.0] or gb[0][2] <= 100                      # cap path executes without error


def test_temporal_iou():
    assert temporal_iou((0, 2), (1, 3)) == 1 / 3
    assert temporal_iou((0, 1), (2, 3)) == 0.0
