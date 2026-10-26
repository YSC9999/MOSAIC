"""Integration tests for Phase 8: Robustness and Ablation Validation."""

import pytest
import torch

from mosaic.ablations import aggregate_seed_metrics
from mosaic.config import config_from_dict
from mosaic.ingestion import ArrayFrameSource
from mosaic.pipeline import Pipeline
from mosaic.synthetic import generate_synthetic


@pytest.mark.integration
def test_multiseed_reproducibility_on_synthetic():
    """Identical random seeds produce identical pipeline outputs."""
    syn = generate_synthetic("moving", seed=42, duration_seconds=1.0, fps=15.0)

    cfg1 = config_from_dict({
        "experiment": {"seed": 1337},
        "pipeline": {"stages": ["ingestion", "motion_gate", "detector", "tracker", "temporal_classifier"]},
        "detector": {"architecture": "mock"},
        "tracker": {"min_hits": 1},
        "model": {"hidden_dim": 16},
    })
    cfg2 = config_from_dict({
        "experiment": {"seed": 1337},
        "pipeline": {"stages": ["ingestion", "motion_gate", "detector", "tracker", "temporal_classifier"]},
        "detector": {"architecture": "mock"},
        "tracker": {"min_hits": 1},
        "model": {"hidden_dim": 16},
    })

    src1 = ArrayFrameSource(syn.frames, fps=syn.fps, source_id="syn_seed_test")
    src2 = ArrayFrameSource(syn.frames, fps=syn.fps, source_id="syn_seed_test")

    diag1 = Pipeline(cfg1).run(src1)
    diag2 = Pipeline(cfg2).run(src2)

    assert diag1.status == "ok"
    assert diag2.status == "ok"
    assert diag1.window_output is not None
    assert diag2.window_output is not None
    assert diag1.window_output.violence_score == pytest.approx(diag2.window_output.violence_score, abs=1e-5)


@pytest.mark.integration
def test_all_ablation_variants_execute_safely():
    """Validates that all key ablation variants execute end-to-end on synthetic footage."""
    syn = generate_synthetic("burst", seed=10, duration_seconds=1.0, fps=15.0)

    ablations = [
        # 1. No Gating
        {"pipeline": {"stages": ["ingestion", "detector", "tracker", "temporal_classifier"]}, "detector": {"architecture": "mock"}},
        # 2. Motion Gated
        {"pipeline": {"stages": ["ingestion", "motion_gate", "detector", "tracker", "temporal_classifier"]}, "detector": {"architecture": "mock"}},
        # 3. Box-only (no kinematics)
        {"pipeline": {"stages": ["ingestion", "detector", "tracker", "temporal_classifier"]}, "detector": {"architecture": "mock"}, "model": {"use_kinematics": False}},
        # 4. No masks
        {"pipeline": {"stages": ["ingestion", "detector", "tracker", "temporal_classifier"]}, "detector": {"architecture": "mock"}, "model": {"use_masks": False}},
        # 5. Tracks only (no scene)
        {"pipeline": {"stages": ["ingestion", "detector", "tracker", "temporal_classifier"]}, "detector": {"architecture": "mock"}, "model": {"use_scene": False}},
    ]

    for i, override in enumerate(ablations):
        cfg = config_from_dict(override)
        src = ArrayFrameSource(syn.frames, fps=syn.fps, source_id=f"syn_abl_{i}")
        diag = Pipeline(cfg).run(src)
        assert diag.status == "ok"
        assert diag.window_output is not None
        assert 0.0 <= diag.window_output.violence_score <= 1.0


@pytest.mark.integration
def test_degradation_pipeline_execution():
    """Pipeline processes frames under progressive degradation without unhandled exceptions."""
    syn = generate_synthetic("moving", seed=99, duration_seconds=1.0, fps=15.0)

    for kind in ("blur", "low_light", "compression", "frame_drop"):
        cfg = config_from_dict({
            "pipeline": {"stages": ["ingestion", "clarity", "detector", "tracker", "temporal_classifier"]},
            "detector": {"architecture": "mock"},
            "degradation": {"enabled": True, "kind": kind, "severity": 0.5},
        })
        src = ArrayFrameSource(syn.frames, fps=syn.fps, source_id=f"syn_deg_{kind}")
        diag = Pipeline(cfg).run(src)
        assert diag.status == "ok"
        assert diag.clarity is not None
        assert diag.window_output is not None
