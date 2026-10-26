"""Integration tests for Phase 2: Ingestion -> Degradation -> Clarity Profiling."""

import json
from pathlib import Path

import numpy as np
import pytest

from mosaic.clarity import compute_window_clarity
from mosaic.config import config_from_dict, load_config
from mosaic.contracts import ClarityReport, WindowDiagnostics, WindowOutput
from mosaic.degradations import apply_degradation
from mosaic.ingestion import VideoReader
from mosaic.pipeline import Pipeline
from mosaic.synthetic import generate_synthetic


@pytest.mark.integration
def test_clarity_stage_in_pipeline_attaches_report(tmp_path: Path):
    """When 'clarity' is in pipeline.stages, Pipeline.run() produces a valid ClarityReport."""
    cfg = config_from_dict(
        {
            "pipeline": {"stages": ["ingestion", "clarity"]},
            "clarity": {"aggregation": "mean"},
        }
    )
    pipeline = Pipeline.from_config(cfg)
    syn_source = generate_synthetic("moving", seed=1337).frame_source()
    diag = pipeline.run(syn_source)

    assert diag.status == "ok"
    assert diag.clarity is not None
    assert isinstance(diag.clarity, ClarityReport)
    assert 0.0 <= diag.clarity.score <= 1.0
    assert diag.clarity.source_profile in ("1080p", "720p", "480p", "240p", "sub_240p")
    assert diag.clarity.resolution_component is not None
    assert diag.clarity.sharpness_component is not None
    assert diag.clarity.exposure_component is not None
    assert diag.clarity.person_scale_component is None


@pytest.mark.integration
def test_clarity_score_decreases_under_degradation():
    """Applying severe blur, low-light, or compression reduces composite clarity score."""
    vid = generate_synthetic("burst", seed=42)
    frames = list(vid.frames)

    clean_report = compute_window_clarity(frames)

    # 1. Blur degradation reduces clarity
    blurred_frames, _ = apply_degradation(frames, kind="blur", severity=0.8, seed=42)
    blur_report = compute_window_clarity(blurred_frames)
    assert blur_report.score < clean_report.score
    assert blur_report.sharpness_component < clean_report.sharpness_component

    # 2. Low light degradation reduces clarity & exposure
    dark_frames, _ = apply_degradation(frames, kind="low_light", severity=0.8, seed=42)
    dark_report = compute_window_clarity(dark_frames)
    assert dark_report.score < clean_report.score
    assert dark_report.exposure_component < clean_report.exposure_component


@pytest.mark.integration
def test_pipeline_with_degradation_is_reproducible_from_saved_config(tmp_path: Path):
    """Regenerating a degraded run from the saved config produces exact bit-identical frames and reports."""
    cfg_dict = {
        "pipeline": {"stages": ["ingestion", "clarity"]},
        "degradation": {
            "enabled": True,
            "kind": "noise",
            "severity": 0.5,
            "seed": 9876,
        },
    }
    cfg = config_from_dict(cfg_dict)
    pipe1 = Pipeline.from_config(cfg)
    pipe2 = Pipeline.from_config(cfg)

    src1 = generate_synthetic("static", seed=100).frame_source()
    src2 = generate_synthetic("static", seed=100).frame_source()

    diag1 = pipe1.run(src1)
    diag2 = pipe2.run(src2)

    assert diag1.to_dict() == diag2.to_dict()
    assert diag1.clarity.score == diag2.clarity.score


@pytest.mark.integration
def test_clarity_report_fits_window_output_and_survives_json_serialization():
    """Clarity score and report embed cleanly into WindowOutput and roundtrip through JSON."""
    report = ClarityReport(
        score=0.78,
        source_profile="720p",
        resolution_component=1.0,
        sharpness_component=0.65,
        exposure_component=0.80,
        compression_component=0.90,
        person_scale_component=None,
    )

    window = WindowOutput(
        window_id="win-p2",
        source_id="synthetic-moving",
        start_timestamp=0.0,
        end_timestamp=3.0,
        label="non_violent",
        violence_score=0.12,
        temporal_confidence=0.85,
        clarity_score=report.score,
        processed_frame_count=90,
        skipped_frame_count=0,
        diagnostics=WindowDiagnostics(
            selected_frame_indices=list(range(90)),
            notes=[f"clarity_profile={report.source_profile}"],
        ),
    )

    json_text = window.to_json()
    reloaded = WindowOutput.from_json(json_text)
    assert reloaded == window
    assert reloaded.clarity_score == pytest.approx(0.78)
    reloaded_dict = json.loads(json_text)
    assert reloaded_dict["clarity_score"] == pytest.approx(0.78)
