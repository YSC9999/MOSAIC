"""Track Token construction and offline token caching (Block 5 / Hub Adapter).

Serves as the dedicated adapter between tracking and temporal modelling:
1. Normalizes feature dimensions (frame-normalized coordinates [0, 1]).
2. Preserves physical observation metadata (observed vs. predicted-only).
3. Connects kinematics, silhouette mask features, and object evidence.
4. Provides deterministic disk-backed token caching per clip, keyed by
   the behavioural config_hash to eliminate redundant computation during
   Phase 7 temporal model training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .config import Config, MaskConfig, TokenConfig, config_hash
from .contracts import Box, ObjectEvidence, TrackToken
from .kinematics import compute_centroid, compute_kinematics
from .masks import extract_person_silhouette
from .tracker import Track


class TokenError(ValueError):
    """Raised when token construction inputs are invalid."""


def build_track_token(
    track: Track,
    frame_width: int,
    frame_height: int,
    timestamp: float,
    frame_index: int,
    frame: Optional[np.ndarray] = None,
    clarity_score: Optional[float] = None,
    mask_cfg: Optional[MaskConfig] = None,
    object_evidences: Optional[Sequence[ObjectEvidence]] = None,
    normalize_coords: bool = True,
    max_lost_frames: int = 15,
) -> TrackToken:
    """Build a validated, immutable TrackToken for a single track observation.

    Parameters
    ----------
    track:
        Track instance (observed or predicted).
    frame_width:
        Original video frame width in pixels.
    frame_height:
        Original video frame height in pixels.
    timestamp:
        Physical timestamp in seconds.
    frame_index:
        Original video frame index.
    frame:
        Optional frame image (BGR/RGB) for extracting silhouette mask.
    clarity_score:
        Optional video clarity score for this frame window.
    mask_cfg:
        Optional MaskConfig.
    object_evidences:
        Optional auxiliary objects detected in this frame.
    normalize_coords:
        Whether to express centroid, velocity, acceleration in [0, 1] frame-normalized units.
    max_lost_frames:
        Tracker timeout to compute visibility score decay during occlusion.
    """
    if frame_width <= 0 or frame_height <= 0:
        raise TokenError(f"invalid frame dimensions: {frame_width}x{frame_height}")

    w_f = float(frame_width)
    h_f = float(frame_height)

    # 1. Clamp bounding box to valid frame coordinates
    clamped_box = Box(
        max(0.0, min(track.box.x1, w_f - 2.0)),
        max(0.0, min(track.box.y1, h_f - 2.0)),
        max(2.0, min(track.box.x2, w_f)),
        max(2.0, min(track.box.y2, h_f)),
    )

    # 2. Compute kinematics with variable delta_t from track history
    prev_box: Optional[Box] = None
    prev_timestamp: Optional[float] = None
    prev_velocity: Optional[List[float]] = None

    if len(track.history) >= 2:
        prev_obs = track.history[-2]
        prev_box = prev_obs.box
        prev_timestamp = prev_obs.timestamp
        if len(track.history) >= 3:
            prev_prev_obs = track.history[-3]
            p_dt = prev_obs.timestamp - prev_prev_obs.timestamp
            if p_dt > 1e-5:
                p_cx, p_cy = compute_centroid(prev_obs.box)
                pp_cx, pp_cy = compute_centroid(prev_prev_obs.box)
                prev_velocity = [(p_cx - pp_cx) / p_dt, (p_cy - pp_cy) / p_dt]

    kin = compute_kinematics(
        current_box=clamped_box,
        current_timestamp=timestamp,
        previous_box=prev_box,
        previous_timestamp=prev_timestamp,
        previous_velocity=prev_velocity,
        frame_width=frame_width,
        frame_height=frame_height,
    )

    if normalize_coords:
        centroid = [
            max(0.0, min(1.0, float(kin.normalized_centroid[0]))),
            max(0.0, min(1.0, float(kin.normalized_centroid[1]))),
        ]
        velocity = [float(kin.normalized_velocity[0]), float(kin.normalized_velocity[1])]
        acceleration = [float(kin.acceleration[0] / w_f), float(kin.acceleration[1] / h_f)]
    else:
        centroid = [float(kin.centroid[0]), float(kin.centroid[1])]
        velocity = [float(kin.velocity[0]), float(kin.velocity[1])]
        acceleration = [float(kin.acceleration[0]), float(kin.acceleration[1])]

    # 3. Observation state and silhouette features
    is_observed = bool(track.observed)
    det_conf: Optional[float] = None
    mask_available = False
    silhouette_features: Optional[List[float]] = None

    if is_observed:
        det_conf = float(track.confidence) if track.confidence is not None else 1.0
        det_conf = max(0.0, min(1.0, det_conf))
        vis_score = 1.0

        # Attempt mask extraction if frame provided
        if (
            frame is not None
            and mask_cfg is not None
            and getattr(mask_cfg, "enabled", True)
        ):
            mask_ev = extract_person_silhouette(
                frame=frame,
                box=clamped_box,
                frame_index=frame_index,
                confidence=mask_cfg.confidence,
                min_visible_fraction=mask_cfg.min_visible_fraction,
            )
            if mask_ev is not None and mask_ev.contour_features is not None:
                mask_available = True
                silhouette_features = [float(v) for v in mask_ev.contour_features]
                vis_score = float(mask_ev.visible_fraction)
    else:
        # Predicted-only token: strictly detector_confidence = None
        det_conf = None
        mask_available = False
        silhouette_features = None
        # Decay visibility score with lost time
        decay = max(0.0, 1.0 - (float(track.time_since_update) / max(1.0, float(max_lost_frames))))
        vis_score = float(decay)

    # 4. Associate object evidence
    associated_objects: Optional[List[ObjectEvidence]] = None
    if object_evidences:
        matched = [obj for obj in object_evidences if obj.associated_track_id == track.track_id]
        if matched:
            associated_objects = list(matched)

    # 5. Clarity score clamp
    c_score = None
    if clarity_score is not None:
        c_score = max(0.0, min(1.0, float(clarity_score)))

    return TrackToken(
        track_id=int(track.track_id),
        timestamp=float(timestamp),
        frame_index=int(frame_index),
        box=clamped_box,
        observed=is_observed,
        detector_confidence=det_conf,
        visibility_score=max(0.0, min(1.0, vis_score)),
        mask_available=mask_available,
        centroid=centroid,
        velocity=velocity,
        acceleration=acceleration,
        clarity_score=c_score,
        silhouette_features=silhouette_features,
        object_evidence=associated_objects,
    )


# --------------------------------------------------------------------------
# Offline Token Cache
# --------------------------------------------------------------------------
def get_cache_path(source_id: str, cfg: Config) -> Path:
    """Determine the deterministic cache file path for a source clip."""
    c_hash = config_hash(cfg)
    # Sanitize source_id to safe filename
    safe_name = "".join(c if (c.isalnum() or c in "._-") else "_" for c in source_id)
    cache_dir = Path(cfg.tokens.cache_dir) / c_hash
    return cache_dir / f"{safe_name}.json"


def save_tokens_to_cache(
    tokens: Sequence[TrackToken],
    source_id: str,
    cfg: Config,
) -> Path:
    """Serialize and save TrackTokens to disk for fast temporal model training."""
    path = get_cache_path(source_id, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "source_id": str(source_id),
        "config_hash": config_hash(cfg),
        "token_count": len(tokens),
        "tokens": [t.to_dict() for t in tokens],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_tokens_from_cache(
    source_id: str,
    cfg: Config,
) -> Optional[List[TrackToken]]:
    """Load cached TrackTokens from disk if available and config_hash matches."""
    path = get_cache_path(source_id, cfg)
    if not path.is_file():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("config_hash") != config_hash(cfg):
            # Outdated cache
            return None
        return [TrackToken.from_dict(item) for item in data.get("tokens", [])]
    except Exception:
        return None
