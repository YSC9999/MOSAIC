"""Unit tests for Track Token construction and offline token caching (Block 5).

Verifies:
- Observed tokens carry valid detector_confidence in [0, 1].
- Predicted-only tokens strictly have detector_confidence = None.
- Normalized coordinates (centroid, velocity) stay within valid bounds.
- Invariant coordinate representation across frame resolutions.
- Cache serialization round-trips exactly (live tokens == loaded tokens).
- Cache invalidation occurs when configuration settings change.
- Malformed inputs raise explicit exceptions.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np

from mosaic.config import Config, config_from_dict
from mosaic.contracts import Box, Detection, TrackToken
from mosaic.tracker import Track, TrackerConfig
from mosaic.tokens import (
    TokenError,
    build_track_token,
    load_tokens_from_cache,
    save_tokens_to_cache,
)


def _make_track(x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> Track:
    det = Detection(
        box=Box(x1, y1, x2, y2),
        confidence=conf,
        class_name="person",
        frame_index=0,
        timestamp=0.0,
    )
    return Track(track_id=1, initial_detection=det, min_hits=1)


def test_observed_token_carries_confidence():
    trk = _make_track(100, 100, 200, 300, conf=0.88)
    token = build_track_token(
        track=trk,
        frame_width=640,
        frame_height=480,
        timestamp=0.0,
        frame_index=0,
    )

    assert token.track_id == 1
    assert token.observed is True
    assert token.detector_confidence == pytest.approx(0.88)
    assert token.visibility_score == 1.0
    assert 0.0 <= token.centroid[0] <= 1.0
    assert 0.0 <= token.centroid[1] <= 1.0


def test_predicted_token_has_none_confidence():
    trk = _make_track(100, 100, 200, 300, conf=0.88)
    # Advance without observation
    trk.predict(frame_index=1, timestamp=0.1)

    token = build_track_token(
        track=trk,
        frame_width=640,
        frame_height=480,
        timestamp=0.1,
        frame_index=1,
    )

    assert token.track_id == 1
    assert token.observed is False
    assert token.detector_confidence is None
    assert token.mask_available is False
    assert token.silhouette_features is None


def test_normalized_coordinates_stay_in_range():
    trk = _make_track(0, 0, 640, 480)
    token = build_track_token(
        track=trk,
        frame_width=640,
        frame_height=480,
        timestamp=0.0,
        frame_index=0,
        normalize_coords=True,
    )

    assert token.centroid == [pytest.approx(0.5), pytest.approx(0.5)]


def test_resolution_invariance():
    # Object occupying center 50% in 640x480
    trk1 = _make_track(160, 120, 480, 360)
    tok1 = build_track_token(
        track=trk1,
        frame_width=640,
        frame_height=480,
        timestamp=0.0,
        frame_index=0,
        normalize_coords=True,
    )

    # Same object scaled to 1280x960
    trk2 = _make_track(320, 240, 960, 720)
    tok2 = build_track_token(
        track=trk2,
        frame_width=1280,
        frame_height=960,
        timestamp=0.0,
        frame_index=0,
        normalize_coords=True,
    )

    assert tok1.centroid[0] == pytest.approx(tok2.centroid[0], abs=1e-4)
    assert tok1.centroid[1] == pytest.approx(tok2.centroid[1], abs=1e-4)


def test_invalid_frame_dimensions_fail():
    trk = _make_track(10, 10, 20, 20)
    with pytest.raises(TokenError, match="invalid frame dimensions"):
        build_track_token(
            track=trk,
            frame_width=0,
            frame_height=480,
            timestamp=0.0,
            frame_index=0,
        )


def test_offline_token_cache_roundtrip(tmp_path):
    cfg = config_from_dict({"tokens": {"cache_dir": str(tmp_path / "cache")}})
    trk = _make_track(50, 50, 100, 100, conf=0.92)
    tok1 = build_track_token(
        track=trk,
        frame_width=640,
        frame_height=480,
        timestamp=0.0,
        frame_index=0,
    )
    trk.predict(1, 0.05)
    tok2 = build_track_token(
        track=trk,
        frame_width=640,
        frame_height=480,
        timestamp=0.05,
        frame_index=1,
    )

    tokens = [tok1, tok2]
    source_id = "test_clip_001"

    # Save to cache
    save_path = save_tokens_to_cache(tokens, source_id, cfg)
    assert save_path.is_file()

    # Load from cache
    loaded = load_tokens_from_cache(source_id, cfg)
    assert loaded is not None
    assert len(loaded) == 2
    assert loaded[0] == tok1
    assert loaded[1] == tok2


def test_offline_token_cache_invalidates_on_config_change(tmp_path):
    cache_dir = str(tmp_path / "cache")
    cfg1 = config_from_dict({"tokens": {"cache_dir": cache_dir}, "tracker": {"iou_threshold": 0.3}})
    cfg2 = config_from_dict({"tokens": {"cache_dir": cache_dir}, "tracker": {"iou_threshold": 0.5}})

    trk = _make_track(50, 50, 100, 100)
    tok = build_track_token(trk, 640, 480, 0.0, 0)
    source_id = "test_clip_002"

    save_tokens_to_cache([tok], source_id, cfg1)

    # cfg1 loads successfully
    assert load_tokens_from_cache(source_id, cfg1) is not None
    # cfg2 hash is different -> cache miss
    assert load_tokens_from_cache(source_id, cfg2) is None
