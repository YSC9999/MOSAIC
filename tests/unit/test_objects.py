"""Unit tests for auxiliary object and threat evidence (Block 4)."""

import pytest

from mosaic.contracts import Box, ObjectEvidence
from mosaic.objects import associate_objects_to_tracks, compute_box_center_distance, compute_box_iou


def test_compute_box_iou():
    b1 = Box(0.0, 0.0, 10.0, 10.0)      # area 100
    b2 = Box(5.0, 0.0, 15.0, 10.0)      # area 100, intersection [5, 0, 10, 10] = 50
    # union = 100 + 100 - 50 = 150 -> IoU = 50 / 150 = 1/3
    assert compute_box_iou(b1, b2) == pytest.approx(1.0 / 3.0)

    # Identical boxes
    assert compute_box_iou(b1, b1) == pytest.approx(1.0)

    # Disjoint boxes
    b3 = Box(20.0, 20.0, 30.0, 30.0)
    assert compute_box_iou(b1, b3) == 0.0


def test_associate_objects_to_nearby_person_tracks():
    obj1 = ObjectEvidence(
        object_id=1,
        category="stick",
        box=Box(45.0, 50.0, 55.0, 90.0),
        confidence=0.80,
        frame_index=10,
    )
    obj_distant = ObjectEvidence(
        object_id=2,
        category="backpack",
        box=Box(500.0, 500.0, 550.0, 550.0),
        confidence=0.75,
        frame_index=10,
    )

    person_tracks = [
        (101, Box(40.0, 40.0, 80.0, 150.0)),   # overlaps with obj1
        (102, Box(200.0, 40.0, 240.0, 150.0)), # distant
    ]

    associated = associate_objects_to_tracks([obj1, obj_distant], person_tracks, max_distance_pixels=100.0)
    assert len(associated) == 2

    # obj1 should be associated with track 101
    assert associated[0].associated_track_id == 101
    # obj_distant is too far from both tracks (>100px)
    assert associated[1].associated_track_id is None


def test_empty_objects_or_persons_safe():
    assert associate_objects_to_tracks([], [(1, Box(1, 1, 5, 5))]) == []
    obj = ObjectEvidence(1, "item", Box(1, 1, 3, 3), 0.5, 0)
    assert associate_objects_to_tracks([obj], []) == [obj]
