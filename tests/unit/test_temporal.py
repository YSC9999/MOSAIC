"""Unit tests for Temporal Context Encoding (Block 6).

Verifies:
- TrackToken vectorization produces 24-dim finite feature representations.
- Variable-length track sequences are padded and masked accurately.
- Modality ablations (kinematics, masks, scene, tracks) behave deterministically.
- Empty-track windows still execute safely and produce valid WindowOutput contracts.
- Confidence score is bounded in [0, 1] and reflects distance from threshold.
- No goon, role, or instigator attributes are ever produced.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from mosaic.config import ModelConfig
from mosaic.contracts import Box, TrackToken, WindowOutput
from mosaic.temporal import (
    TRACK_FEATURE_DIM,
    SpatialTemporalFusionModel,
    TemporalModelError,
    batch_track_tokens,
    build_window_output,
    vectorize_track_token,
)


def _make_sample_token(track_id: int, frame_idx: int, ts: float, observed: bool = True) -> TrackToken:
    conf = 0.92 if observed else None
    return TrackToken(
        track_id=track_id,
        timestamp=ts,
        frame_index=frame_idx,
        box=Box(10.0, 10.0, 50.0, 80.0),
        observed=observed,
        detector_confidence=conf,
        visibility_score=1.0 if observed else 0.5,
        mask_available=observed,
        centroid=[0.25, 0.35],
        velocity=[0.05, -0.02],
        acceleration=[0.01, 0.0],
        silhouette_features=[0.5, 0.2, 0.1, 0.8, 0.3, 0.4, 0.15, 0.05] if observed else None,
    )


def test_vectorize_track_token_dimensions():
    tok = _make_sample_token(1, 0, 0.0, observed=True)
    vec = vectorize_track_token(tok, prev_timestamp=None)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (TRACK_FEATURE_DIM,)
    assert vec.dtype == np.float32
    assert np.all(np.isfinite(vec))
    assert vec[0] == pytest.approx(0.25)
    assert vec[1] == pytest.approx(0.35)
    assert vec[11] == 1.0  # observed


def test_vectorize_predicted_token_handles_missing_fields():
    tok = _make_sample_token(1, 1, 0.05, observed=False)
    vec = vectorize_track_token(tok, prev_timestamp=0.0)
    assert vec[11] == 0.0  # observed=False
    assert vec[14] == 0.0  # confidence=0.0
    assert vec[15] == 0.0  # mask_available=0.0
    assert np.all(vec[16:24] == 0.0)


def test_modality_ablation_vectorization():
    tok = _make_sample_token(1, 0, 0.0, observed=True)
    # Ablate kinematics
    vec_no_kin = vectorize_track_token(tok, use_kinematics=False)
    assert vec_no_kin[2] == 0.0  # vx
    assert vec_no_kin[3] == 0.0  # vy
    assert vec_no_kin[4] == 0.0  # ax
    assert vec_no_kin[5] == 0.0  # ay

    # Ablate masks
    vec_no_mask = vectorize_track_token(tok, use_masks=False)
    assert vec_no_mask[15] == 0.0
    assert np.all(vec_no_mask[16:24] == 0.0)


def test_batch_track_tokens_padding_and_masking():
    tokens = [
        _make_sample_token(1, 0, 0.0),
        _make_sample_token(1, 1, 0.05),
        _make_sample_token(2, 0, 0.0),
    ]

    feats, mask = batch_track_tokens(tokens, max_tracks=4, max_steps=8)
    assert feats.shape == (1, 4, 8, TRACK_FEATURE_DIM)
    assert mask.shape == (1, 4, 8)
    # Track 1 has 2 steps
    assert mask[0, 0, 0].item() is True
    assert mask[0, 0, 1].item() is True
    assert mask[0, 0, 2].item() is False
    # Track 2 has 1 step
    assert mask[0, 1, 0].item() is True
    assert mask[0, 1, 1].item() is False
    # Tracks 3 and 4 are empty padding
    assert mask[0, 2].any().item() is False
    assert mask[0, 3].any().item() is False


def test_model_forward_empty_tracks_does_not_crash():
    cfg = ModelConfig(hidden_dim=32, use_scene=True, use_tracks=True)
    model = SpatialTemporalFusionModel(cfg, scene_dim=64)
    model.eval()

    scene_t = torch.randn(2, 10, 64)
    # Empty track tensors (all False in mask)
    trk_feats = torch.zeros(2, 4, 8, TRACK_FEATURE_DIM)
    trk_mask = torch.zeros(2, 4, 8, dtype=torch.bool)

    out = model(scene_features=scene_t, track_features=trk_feats, track_mask=trk_mask)
    assert out.shape == (2, 1)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_calibrated_confidence_calculation():
    cfg = ModelConfig(hidden_dim=16, decision_threshold=0.50)
    model = SpatialTemporalFusionModel(cfg, scene_dim=32)

    score, label, conf = model.predict_window(tokens=[_make_sample_token(1, 0, 0.0)])
    assert 0.0 <= score <= 1.0
    assert label in ("violent", "non_violent")
    assert 0.0 <= conf <= 1.0
    assert conf == pytest.approx(min(1.0, 2.0 * abs(score - 0.50)), abs=1e-5)


def test_build_window_output_contract():
    out = build_window_output(
        window_id="test_win_1",
        source_id="clip_abc",
        start_timestamp=0.0,
        end_timestamp=2.5,
        violence_score=0.82,
        temporal_confidence=0.64,
        clarity_score=0.75,
        processed_frame_count=20,
        skipped_frame_count=55,
        decision_threshold=0.50,
    )

    assert isinstance(out, WindowOutput)
    assert out.label == "violent"
    assert out.violence_score == pytest.approx(0.82)
    assert out.temporal_confidence == pytest.approx(0.64)
    # Verify serializable to valid JSON
    d = out.to_dict()
    assert d["label"] == "violent"
    assert "instigator" not in d
    assert "role" not in d
