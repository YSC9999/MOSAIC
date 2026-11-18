import numpy as np

from mosaic.det_cache import box_iou, clamp_and_filter, mask_stats, pairwise_overlap


def test_clamp_and_filter():
    b, keep = clamp_and_filter(np.array([[-5, -5, 50, 50], [10, 10, 12, 30], [0, 0, 20, 5]]), 40, 40)
    assert b[0].tolist() == [0, 0, 40, 40]
    assert keep.tolist() == [True, False, False]  # 2 px wide, 5 px tall dropped


def test_mask_stats_valid_and_invalid():
    m = np.zeros((40, 40), bool)
    m[10:30, 10:20] = True
    st = mask_stats(m, np.array([10, 10, 20, 30], np.float32))
    assert st["valid"] and st["area"] == 200 and st["bbox"] == (10.0, 10.0, 20.0, 30.0)
    assert not mask_stats(m, np.array([25, 0, 40, 9], np.float32))["valid"]  # mask bbox far from box
    tiny = np.zeros((40, 40), bool)
    tiny[0:3, 0:3] = True
    assert not mask_stats(tiny, np.array([0, 0, 3, 3], np.float32))["valid"]  # < 16 px


def test_pairwise_overlap_smaller_area():
    a = np.zeros((20, 20), bool); a[0:10, 0:10] = True   # area 100
    b = np.zeros((20, 20), bool); b[5:10, 5:10] = True   # area 25, inside a
    boxes = np.array([[0, 0, 10, 10], [5, 5, 10, 10]], np.float32)
    ov = pairwise_overlap([a, b], boxes, np.array([100, 25]), np.array([True, True]))
    assert abs(float(ov[0, 1]) - 1.0) < 1e-3 and ov[0, 1] == ov[1, 0]


def test_box_iou():
    assert box_iou(np.array([0, 0, 10, 10]), np.array([0, 0, 10, 10])) == 1.0
    assert box_iou(np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30])) == 0.0


def test_crop_stats_match_full_frame():
    from mosaic.det_cache import crop_region
    m = np.zeros((100, 100), bool)
    m[20:60, 30:50] = True
    box = np.array([30, 20, 50, 60], np.float32)
    c = crop_region(box, 100, 100)
    full = mask_stats(m, box)
    crop = mask_stats(m[c[1]:c[3], c[0]:c[2]], box, offset=(c[0], c[1]))
    assert full == crop


def test_overlap_with_crops_matches_full():
    from mosaic.det_cache import crop_region
    a = np.zeros((50, 50), bool); a[0:20, 0:20] = True
    b = np.zeros((50, 50), bool); b[10:30, 10:30] = True
    boxes = np.array([[0, 0, 20, 20], [10, 10, 30, 30]], np.float32)
    areas, valid = np.array([400, 400]), np.array([True, True])
    full = pairwise_overlap([a, b], boxes, areas, valid)
    cs = [crop_region(bx, 50, 50) for bx in boxes]
    cm = [m[c[1]:c[3], c[0]:c[2]] for m, c in zip([a, b], cs)]
    assert np.allclose(full, pairwise_overlap(cm, boxes, areas, valid, cs))
