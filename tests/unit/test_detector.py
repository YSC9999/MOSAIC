"""Unit tests for person and object detection backbones (Block 3)."""

import numpy as np
import pytest

from mosaic.clarity import calculate_person_scale_component, compute_frame_clarity, compute_window_clarity
from mosaic.config import DetectorConfig
from mosaic.contracts import Box, Detection
from mosaic.detector import BaseDetector, DetectorError, MockDetector, TorchvisionPersonDetector, create_detector


def test_mock_detector_satisfies_detection_contract():
    detector = MockDetector(confidence=0.85)
    frame = np.full((120, 160, 3), 128, dtype=np.uint8)
    detections = detector.detect(frame, frame_index=5, timestamp=0.166)

    assert len(detections) == 1
    det = detections[0]
    assert isinstance(det, Detection)
    assert det.class_name == "person"
    assert det.confidence == pytest.approx(0.85)
    assert det.frame_index == 5
    assert det.timestamp == pytest.approx(0.166)
    assert det.box.x1 >= 0.0 and det.box.y1 >= 0.0
    assert det.box.x2 <= 160.0 and det.box.y2 <= 120.0
    assert det.box.width > 0.0 and det.box.height > 0.0


def test_mock_detector_batch_processing():
    detector = MockDetector()
    frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
    indices = [0, 5, 10]
    timestamps = [0.0, 0.25, 0.50]

    batch_res = detector.detect_batch(frames, indices, timestamps)
    assert len(batch_res) == 3
    for i, dets in enumerate(batch_res):
        assert len(dets) == 1
        assert dets[0].frame_index == indices[i]
        assert dets[0].timestamp == timestamps[i]


def test_mock_detector_scripted_detections():
    scripted = {
        0: [],
        1: [
            Detection(
                box=Box(10.0, 15.0, 40.0, 80.0),
                confidence=0.92,
                class_name="person",
                frame_index=1,
                timestamp=0.033,
            )
        ],
    }
    detector = MockDetector(scripted_detections=scripted)
    assert len(detector.detect(np.zeros((100, 100, 3), dtype=np.uint8), 0, 0.0)) == 0
    d1 = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8), 1, 0.033)
    assert len(d1) == 1
    assert d1[0].box.width == 30.0 and d1[0].box.height == 65.0


def test_person_scale_component_calculation():
    # 720p frame (H=720)
    # Person 1: height = 180 (ratio = 180 / 720 = 0.25 -> score 1.0)
    det1 = Detection(Box(50, 50, 100, 230), 0.9, "person", 0, 0.0)
    # Person 2: height = 36 (ratio = 36 / 720 = 0.05 -> score 0.05 / 0.25 = 0.20)
    det2 = Detection(Box(50, 50, 100, 86), 0.9, "person", 0, 0.0)

    score_tall = calculate_person_scale_component([det1], frame_height=720)
    score_small = calculate_person_scale_component([det2], frame_height=720)
    assert score_tall == pytest.approx(1.0)
    assert score_small == pytest.approx(0.20)

    # Empty detections return None
    assert calculate_person_scale_component([], frame_height=720) is None
    # Non-person classes ignored
    car_det = Detection(Box(10, 10, 50, 50), 0.9, "car", 0, 0.0)
    assert calculate_person_scale_component([car_det], frame_height=720) is None


def test_clarity_updates_when_detections_exist():
    frame = np.full((120, 160, 3), 100, dtype=np.uint8)
    rep_no_det = compute_frame_clarity(frame)
    assert rep_no_det.person_scale_component is None

    # Person with height 30 on H=120 (ratio = 30/120 = 0.25 -> 1.0)
    det = Detection(Box(20, 20, 40, 50), 0.9, "person", 0, 0.0)
    rep_with_det = compute_frame_clarity(frame, detections=[det])
    assert rep_with_det.person_scale_component == pytest.approx(1.0)
    assert rep_with_det.score > 0.0


def test_detector_factory():
    cfg_mock = DetectorConfig(architecture="mock")
    det_mock = create_detector(cfg_mock)
    assert isinstance(det_mock, MockDetector)

    cfg_ssdlite = DetectorConfig(architecture="ssdlite320")
    det_ssdlite = create_detector(cfg_ssdlite)
    assert isinstance(det_ssdlite, TorchvisionPersonDetector)
    assert det_ssdlite.architecture == "ssdlite320"


def test_torchvision_detector_empty_frame():
    # Uniform black frame -> zero person detections
    detector = TorchvisionPersonDetector(architecture="ssdlite320", confidence_threshold=0.50)
    blank = np.zeros((120, 160, 3), dtype=np.uint8)
    dets = detector.detect(blank, frame_index=0, timestamp=0.0)
    assert isinstance(dets, list)
    assert len(dets) == 0


def test_torchvision_detector_feature_extraction():
    detector = TorchvisionPersonDetector(
        architecture="ssdlite320",
        confidence_threshold=0.40,
        extract_features=True,
        feature_dim=128,
    )
    frame = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    box = Box(10.0, 10.0, 40.0, 80.0)
    feat = detector._extract_crop_feature(frame, box)
    assert isinstance(feat, list)
    assert len(feat) == 128
    assert all(isinstance(v, float) for v in feat)
    norm = np.linalg.norm(feat)
    assert norm == pytest.approx(1.0, abs=1e-4)
