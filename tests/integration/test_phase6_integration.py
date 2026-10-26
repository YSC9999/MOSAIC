"""Integration tests for Phase 6: Multi-Object Tracking & Track Tokens."""

import pytest
import numpy as np

from mosaic.config import config_from_dict
from mosaic.contracts import Box, Detection, ObjectEvidence, TrackToken
from mosaic.ingestion import ArrayFrameSource
from mosaic.pipeline import Pipeline
from mosaic.synthetic import generate_synthetic
from mosaic.tokens import (
    build_track_token,
    load_tokens_from_cache,
    save_tokens_to_cache,
)
from mosaic.tracker import Tracker, TrackerConfig


@pytest.mark.integration
def test_pipeline_end_to_end_with_tracker(tmp_path):
    """End-to-end pipeline execution with ingestion -> motion_gate -> detector -> tracker."""
    syn_video = generate_synthetic("moving", seed=101, duration_seconds=1.0, fps=15.0)
    source = ArrayFrameSource(
        syn_video.frames,
        fps=syn_video.fps,
        source_id="syn_moving_clip",
    )

    cfg = config_from_dict({
        "pipeline": {"stages": ["ingestion", "motion_gate", "detector", "tracker"]},
        "detector": {"architecture": "mock", "enabled": True},
        "motion_gate": {"enabled": True, "threshold": 0.005, "burst_threshold": 0.010},
        "tracker": {"min_hits": 1, "max_lost_frames": 10},
        "tokens": {"cache_dir": str(tmp_path / "tokens_cache")},
    })

    pipe = Pipeline(cfg)
    diag = pipe.run(source)

    assert diag.status == "ok"
    assert diag.tracks is not None
    assert len(diag.tracks) >= 1
    assert diag.track_tokens is not None
    assert len(diag.track_tokens) >= 1

    # Verify TrackToken contracts
    for token in diag.track_tokens:
        assert isinstance(token, TrackToken)
        assert 0.0 <= token.centroid[0] <= 1.0
        assert 0.0 <= token.centroid[1] <= 1.0
        if token.observed:
            assert token.detector_confidence is not None
            assert 0.0 <= token.detector_confidence <= 1.0
        else:
            assert token.detector_confidence is None


@pytest.mark.integration
def test_synthetic_occlusion_survival_and_recovery():
    """In scripted occlusion, track survives behind the occluder and recovers with same track_id."""
    # Synthetic video where rectangle passes behind an occluding wall
    syn = generate_synthetic("occlusion", seed=42, fps=30.0, duration_seconds=2.0)
    tracker = Tracker(TrackerConfig(min_hits=1, max_lost_frames=20))

    observed_ids_per_frame = []
    track_ids_seen = set()

    for gt in syn.ground_truth:
        # Simulate detector: detects object only when visible_fraction >= 0.35
        if gt.visible_fraction >= 0.35:
            dets = [
                Detection(
                    box=gt.box,
                    confidence=0.95,
                    class_name="person",
                    frame_index=gt.frame_index,
                    timestamp=gt.timestamp,
                )
            ]
        else:
            dets = []  # Occluded by static wall!

        active = tracker.update(dets, gt.frame_index, gt.timestamp)
        for t in active:
            track_ids_seen.add(t.track_id)
        observed_ids_per_frame.append([t.track_id for t in active])

    # Exactly 1 moving object was scripted in this scenario
    # It must survive occlusion and NOT create a second track ID!
    assert len(track_ids_seen) == 1, f"Expected 1 track ID, got {track_ids_seen}"
    summaries = tracker.get_track_summaries()
    assert len(summaries) == 1
    sum_0 = summaries[0]
    assert sum_0.track_id == 1
    # Both observed hits and predicted (during occlusion) steps should exist
    assert sum_0.observed_count > 0
    assert sum_0.predicted_count > 0


@pytest.mark.integration
def test_track_token_cache_live_replay_identity(tmp_path):
    """Tokens reloaded from the disk cache are byte-identical to live generated tokens."""
    syn = generate_synthetic("moving", seed=77, duration_seconds=1.0, fps=20.0)
    cfg = config_from_dict({
        "tokens": {"cache_dir": str(tmp_path / "tokens_cache")},
        "tracker": {"min_hits": 1},
    })
    tracker = Tracker(cfg.tracker)

    live_tokens = []
    for gt in syn.ground_truth:
        det = Detection(
            box=gt.box,
            confidence=0.90,
            class_name="person",
            frame_index=gt.frame_index,
            timestamp=gt.timestamp,
        )
        active = tracker.update([det], gt.frame_index, gt.timestamp)
        for trk in active:
            tok = build_track_token(
                track=trk,
                frame_width=syn.width,
                frame_height=syn.height,
                timestamp=gt.timestamp,
                frame_index=gt.frame_index,
                frame=syn.frames[gt.frame_index],
                mask_cfg=cfg.masks,
                normalize_coords=True,
            )
            live_tokens.append(tok)

    # Save to disk
    save_path = save_tokens_to_cache(live_tokens, "test_syn_clip", cfg)
    assert save_path.is_file()

    # Reload from disk
    cached_tokens = load_tokens_from_cache("test_syn_clip", cfg)
    assert cached_tokens is not None
    assert len(cached_tokens) == len(live_tokens)
    for live, cached in zip(live_tokens, cached_tokens):
        assert live == cached


@pytest.mark.integration
def test_mask_and_object_evidence_attach_to_track_token():
    """Auxiliary object proximity and silhouette features attach cleanly to TrackToken."""
    box = Box(10.0, 10.0, 50.0, 80.0)
    det = Detection(box=box, confidence=0.92, class_name="person", frame_index=0, timestamp=0.0)
    trk = Tracker(TrackerConfig(min_hits=1)).update([det], 0, 0.0)[0]

    obj = ObjectEvidence(
        object_id=1,
        category="bottle",
        box=Box(45.0, 30.0, 60.0, 60.0),
        confidence=0.80,
        frame_index=0,
        associated_track_id=trk.track_id,
    )

    frame = np.full((120, 160, 3), 100, dtype=np.uint8)
    frame[10:80, 10:50] = (200, 200, 200)

    token = build_track_token(
        track=trk,
        frame_width=160,
        frame_height=120,
        timestamp=0.0,
        frame_index=0,
        frame=frame,
        object_evidences=[obj],
    )

    assert token.object_evidence is not None
    assert len(token.object_evidence) == 1
    assert token.object_evidence[0].category == "bottle"
