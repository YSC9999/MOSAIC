"""Integration tests for Phase 4: Ingestion -> Clarity -> Motion Gate -> Person Detection."""

import json
from pathlib import Path

import numpy as np
import pytest

from mosaic.config import config_from_dict
from mosaic.contracts import Detection
from mosaic.pipeline import Pipeline
from mosaic.synthetic import generate_synthetic


@pytest.mark.integration
def test_detector_stage_in_pipeline_populates_diagnostics():
    """When 'detector' is in pipeline.stages, Pipeline.run() produces person detections."""
    cfg = config_from_dict(
        {
            "pipeline": {
                "stages": ["ingestion", "clarity", "motion_gate", "detector"]
            },
            "detector": {
                "architecture": "mock",
                "confidence_threshold": 0.50,
            },
        }
    )
    pipeline = Pipeline.from_config(cfg)
    syn_source = generate_synthetic("moving", seed=1337).frame_source()
    diag = pipeline.run(syn_source)

    assert diag.status == "ok"
    assert diag.detection_count is not None
    assert diag.detection_count > 0
    assert diag.detections is not None
    assert len(diag.detections) == len(diag.selected_frame_indices)

    # Check person scale component is populated in clarity report
    assert diag.clarity is not None
    assert diag.clarity.person_scale_component is not None
    assert 0.0 <= diag.clarity.person_scale_component <= 1.0


@pytest.mark.integration
def test_motion_gated_frames_flow_into_detector_with_timestamps():
    """Only motion-gate-retained frames reach the detector, strictly matching timestamps."""
    cfg = config_from_dict(
        {
            "pipeline": {
                "stages": ["ingestion", "motion_gate", "detector"]
            },
            "motion_gate": {
                "threshold": 0.05,
                "burst_threshold": 0.08,
                "keep_alive_frames": 10,
            },
            "detector": {
                "architecture": "mock",
            },
        }
    )
    pipeline = Pipeline.from_config(cfg)
    syn_source = generate_synthetic("burst", seed=42).frame_source()
    diag = pipeline.run(syn_source)

    assert diag.detections is not None
    assert len(diag.detections) == len(diag.selected_frame_indices)

    for frame_dets, expected_idx in zip(diag.detections, diag.selected_frame_indices):
        for det in frame_dets:
            assert det.frame_index == expected_idx


@pytest.mark.integration
def test_detector_output_serializes_cleanly():
    """Pipeline diagnostics with detections survive JSON serialization."""
    cfg = config_from_dict(
        {
            "pipeline": {
                "stages": ["ingestion", "clarity", "detector"]
            },
            "detector": {
                "architecture": "mock",
            },
        }
    )
    pipeline = Pipeline.from_config(cfg)
    syn_source = generate_synthetic("static", seed=10).frame_source()
    diag = pipeline.run(syn_source)

    dumped = json.dumps(diag.to_dict())
    loaded = json.loads(dumped)
    assert loaded["detection_count"] == diag.detection_count
    assert len(loaded["detections"]) == len(diag.detections)


@pytest.mark.integration
def test_detector_with_camera_degradations():
    """Detections remain valid and within coordinate bounds even under severe camera degradations."""
    cfg = config_from_dict(
        {
            "pipeline": {
                "stages": ["ingestion", "detector"]
            },
            "degradation": {
                "enabled": True,
                "kind": "blur",
                "severity": 0.75,
            },
            "detector": {
                "architecture": "mock",
            },
        }
    )
    pipeline = Pipeline.from_config(cfg)
    syn_source = generate_synthetic("moving", seed=99).frame_source()
    diag = pipeline.run(syn_source)

    assert diag.status == "ok"
    for frame_dets in diag.detections:
        for det in frame_dets:
            assert det.box.x1 >= 0.0 and det.box.y1 >= 0.0
            assert det.box.x2 <= 160.0 and det.box.y2 <= 120.0
