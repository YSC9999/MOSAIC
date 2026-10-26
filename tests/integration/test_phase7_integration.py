"""Integration tests for Phase 7: Temporal Context Encoding & End-to-End Classification."""

import pytest
import torch

from mosaic.config import config_from_dict
from mosaic.contracts import WindowOutput
from mosaic.ingestion import ArrayFrameSource
from mosaic.pipeline import Pipeline
from mosaic.synthetic import generate_synthetic
from mosaic.temporal import SpatialTemporalFusionModel


@pytest.mark.integration
def test_full_pipeline_end_to_end_to_window_output():
    """Full pipeline execution from synthetic frames to final WindowOutput contract."""
    syn = generate_synthetic("moving", seed=42, duration_seconds=1.0, fps=15.0)
    source = ArrayFrameSource(syn.frames, fps=syn.fps, source_id="syn_full_test")

    cfg = config_from_dict({
        "pipeline": {
            "stages": [
                "ingestion",
                "clarity",
                "motion_gate",
                "detector",
                "tracker",
                "temporal_classifier",
            ]
        },
        "detector": {"architecture": "mock", "enabled": True},
        "motion_gate": {"enabled": True, "threshold": 0.005, "burst_threshold": 0.010},
        "tracker": {"min_hits": 1, "max_lost_frames": 10},
        "model": {"architecture": "spatial_temporal_fusion", "hidden_dim": 32},
    })

    pipe = Pipeline(cfg)
    diag = pipe.run(source)

    assert diag.status == "ok"
    assert diag.clarity is not None
    assert diag.motion_gate is not None
    assert diag.detections is not None
    assert diag.tracks is not None
    assert diag.track_tokens is not None
    assert diag.window_output is not None

    win = diag.window_output
    assert isinstance(win, WindowOutput)
    assert win.source_id == "syn_full_test"
    assert win.label in ("violent", "non_violent")
    assert 0.0 <= win.violence_score <= 1.0
    assert 0.0 <= win.temporal_confidence <= 1.0
    assert 0.0 <= win.clarity_score <= 1.0
    assert win.processed_frame_count > 0
    assert len(win.tracks) >= 1


@pytest.mark.integration
def test_checkpoint_save_and_load_reproduces_predictions(tmp_path):
    """Model state dict save/load preserves identical numerical inference."""
    cfg = config_from_dict({
        "model": {"architecture": "spatial_temporal_fusion", "hidden_dim": 32}
    })
    model1 = SpatialTemporalFusionModel(cfg.model, scene_dim=64)
    model1.eval()

    scene_t = torch.randn(1, 8, 64)
    trk_feats = torch.randn(1, 2, 8, 24)
    trk_mask = torch.ones(1, 2, 8, dtype=torch.bool)

    with torch.no_grad():
        out1 = model1(scene_t, trk_feats, trk_mask).item()

    # Save checkpoint
    ckpt_path = tmp_path / "fusion_model.pt"
    torch.save(model1.state_dict(), ckpt_path)

    # Load into fresh model
    model2 = SpatialTemporalFusionModel(cfg.model, scene_dim=64)
    model2.load_state_dict(torch.load(ckpt_path))
    model2.eval()

    with torch.no_grad():
        out2 = model2(scene_t, trk_feats, trk_mask).item()

    assert out1 == pytest.approx(out2, abs=1e-6)


@pytest.mark.integration
def test_modality_ablation_paths_complete_safely():
    """Pipeline operates without error when tracks or scene branches are ablated."""
    syn = generate_synthetic("burst", seed=12, duration_seconds=1.0, fps=15.0)
    source = ArrayFrameSource(syn.frames, fps=syn.fps, source_id="syn_burst_test")

    # Ablate scene, keep only tracks
    cfg_tracks_only = config_from_dict({
        "pipeline": {"stages": ["ingestion", "detector", "tracker", "temporal_classifier"]},
        "detector": {"architecture": "mock"},
        "tracker": {"min_hits": 1},
        "model": {"use_scene": False, "use_tracks": True, "hidden_dim": 32},
    })
    diag1 = Pipeline(cfg_tracks_only).run(source)
    assert diag1.window_output is not None

    # Ablate tracks, keep only scene
    cfg_scene_only = config_from_dict({
        "pipeline": {"stages": ["ingestion", "temporal_classifier"]},
        "model": {"use_scene": True, "use_tracks": False, "hidden_dim": 32},
    })
    diag2 = Pipeline(cfg_scene_only).run(source)
    assert diag2.window_output is not None
