"""Integration tests for Phase 3: Ingestion -> Clarity -> Motion Gating."""

import json
from pathlib import Path

import pytest

from mosaic.config import config_from_dict
from mosaic.contracts import MotionGateReport, WindowDiagnostics, WindowOutput
from mosaic.pipeline import Pipeline
from mosaic.synthetic import generate_synthetic


@pytest.mark.integration
def test_motion_gate_stage_in_pipeline_attaches_report():
    """When 'motion_gate' is in pipeline.stages, Pipeline.run() produces a valid MotionGateReport."""
    cfg = config_from_dict(
        {
            "pipeline": {"stages": ["ingestion", "clarity", "motion_gate"]},
            "motion_gate": {
                "enabled": True,
                "threshold": 0.02,
                "burst_threshold": 0.06,
                "burst_hold_frames": 5,
                "keep_alive_frames": 15,
            },
        }
    )
    pipeline = Pipeline.from_config(cfg)
    syn_source = generate_synthetic("moving", seed=1337).frame_source()
    diag = pipeline.run(syn_source)

    assert diag.status == "ok"
    assert diag.motion_gate is not None
    assert isinstance(diag.motion_gate, MotionGateReport)
    assert diag.selected_frame_indices is not None
    assert diag.skipped_frame_indices is not None
    assert diag.motion_gate.total_frames == 90
    assert (
        diag.motion_gate.selected_frames + diag.motion_gate.skipped_frames
        == diag.motion_gate.total_frames
    )
    assert 0.0 <= diag.motion_gate.reduction_fraction <= 1.0


@pytest.mark.integration
def test_motion_gate_drastically_reduces_static_video_frames():
    """In a static scene, motion gating skips ~85%+ of frames, retaining only keep-alives."""
    cfg = config_from_dict(
        {
            "pipeline": {"stages": ["ingestion", "motion_gate"]},
            "motion_gate": {
                "enabled": True,
                "threshold": 0.02,
                "burst_threshold": 0.06,
                "burst_hold_frames": 5,
                "keep_alive_frames": 15,
            },
        }
    )
    pipeline = Pipeline.from_config(cfg)
    syn_source = generate_synthetic("static", seed=42).frame_source()
    diag = pipeline.run(syn_source)

    assert diag.motion_gate is not None
    # 90 frames / 15 keep-alive = ~6-7 retained frames out of 90
    assert diag.motion_gate.reduction_fraction >= 0.80
    assert diag.motion_gate.keep_alive_count >= 5
    assert diag.motion_gate.burst_count == 0


@pytest.mark.integration
def test_motion_gate_preserves_sudden_bursts():
    """When intense motion occurs, the gate triggers burst mode and retains consecutive frames."""
    cfg = config_from_dict(
        {
            "pipeline": {"stages": ["ingestion", "motion_gate"]},
            "motion_gate": {
                "enabled": True,
                "threshold": 0.0005,
                "burst_threshold": 0.001,
                "burst_hold_frames": 5,
                "keep_alive_frames": 15,
            },
        }
    )
    pipeline = Pipeline.from_config(cfg)
    syn_source = generate_synthetic("burst", seed=42).frame_source()
    diag = pipeline.run(syn_source)

    assert diag.motion_gate is not None
    assert diag.motion_gate.burst_count > 0
    # Burst scenario retains significantly more frames than purely static
    assert diag.motion_gate.selected_frames > 15


@pytest.mark.integration
def test_motion_gate_pipeline_is_deterministic():
    """Running identical synthetic sources with the same configuration produces identical gate indices."""
    cfg = config_from_dict(
        {
            "pipeline": {"stages": ["ingestion", "clarity", "motion_gate"]},
            "motion_gate": {
                "threshold": 0.025,
                "burst_threshold": 0.07,
            },
        }
    )
    pipe1 = Pipeline.from_config(cfg)
    pipe2 = Pipeline.from_config(cfg)

    src1 = generate_synthetic("moving", seed=99).frame_source()
    src2 = generate_synthetic("moving", seed=99).frame_source()

    diag1 = pipe1.run(src1)
    diag2 = pipe2.run(src2)

    assert diag1.selected_frame_indices == diag2.selected_frame_indices
    assert diag1.skipped_frame_indices == diag2.skipped_frame_indices
    assert diag1.motion_gate == diag2.motion_gate


@pytest.mark.integration
def test_pipeline_gate_diagnostics_fit_in_window_diagnostics():
    """Motion gate diagnostics safely populate typed WindowDiagnostics contracts."""
    cfg = config_from_dict(
        {
            "pipeline": {"stages": ["ingestion", "motion_gate"]},
        }
    )
    pipeline = Pipeline.from_config(cfg)
    syn_source = generate_synthetic("burst", seed=10).frame_source()
    diag = pipeline.run(syn_source)

    window_diag = WindowDiagnostics(
        selected_frame_indices=diag.selected_frame_indices,
        skipped_frame_indices=diag.skipped_frame_indices,
        decode=diag.decode,
        notes=["motion_gate_active"],
    )
    assert set(window_diag.selected_frame_indices).isdisjoint(set(window_diag.skipped_frame_indices))

    # Serialization test
    dumped = json.dumps(window_diag.to_dict())
    loaded = WindowDiagnostics.from_dict(json.loads(dumped))
    assert loaded == window_diag
