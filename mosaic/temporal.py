"""Temporal context encoding with spatial evidence (Block 6).

Fuses whole-scene visual features with multi-person TrackToken sequences:
1. Track Token Vectorizer: Converts immutable TrackToken contracts into 24-dim
   continuous feature vectors with resolution-normalized kinematics, irregular
   delta_t temporal spacing, observation certainty, and silhouette contour moments.
2. Modality Ablation Support: Cleanly zeros out scene, track, kinematic, or mask
   features via configuration flags.
3. Track Sequence Aggregator: Permutation-invariant pooling across active human
   tracks with padding masks.
4. Scene + Track Fusion Network: Joint GRU architecture mapping multi-modal temporal
   evidence into a single calibrated violence score and WindowOutput contract.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import Config, ModelConfig
from .contracts import (
    LABELS,
    Box,
    ObjectEvidence,
    TrackSummary,
    TrackToken,
    WindowDiagnostics,
    WindowOutput,
)

TRACK_FEATURE_DIM = 24


class TemporalModelError(ValueError):
    """Raised when temporal modeling inputs or configurations are invalid."""


def vectorize_track_token(
    token: TrackToken,
    prev_timestamp: Optional[float] = None,
    use_kinematics: bool = True,
    use_masks: bool = True,
) -> np.ndarray:
    """Convert a TrackToken into a 24-dimensional continuous feature vector."""
    vec = np.zeros(TRACK_FEATURE_DIM, dtype=np.float32)

    # 0, 1: Centroid [cx, cy] in [0, 1]
    vec[0] = float(np.clip(token.centroid[0], 0.0, 1.0))
    vec[1] = float(np.clip(token.centroid[1], 0.0, 1.0))

    # 2, 3: Physical Velocity [vx, vy] (normalized)
    if use_kinematics:
        vec[2] = float(np.nan_to_num(token.velocity[0], nan=0.0, posinf=10.0, neginf=-10.0))
        vec[3] = float(np.nan_to_num(token.velocity[1], nan=0.0, posinf=10.0, neginf=-10.0))
        # 4, 5: Acceleration [ax, ay]
        vec[4] = float(np.nan_to_num(token.acceleration[0], nan=0.0, posinf=100.0, neginf=-100.0))
        vec[5] = float(np.nan_to_num(token.acceleration[1], nan=0.0, posinf=100.0, neginf=-100.0))
        # 6: Speed magnitude
        vec[6] = float(math.sqrt(vec[2] ** 2 + vec[3] ** 2))

    # 7, 8: Bounding Box Dimensions
    w = max(0.001, token.box.width)
    h = max(0.001, token.box.height)
    vec[7] = float(w)
    vec[8] = float(h)

    # 9: Aspect Ratio
    ar = float(w / h)
    vec[9] = float(np.clip(ar, 0.05, 10.0))

    # 10: Aspect Ratio Fall Dynamics (rate of change)
    if use_kinematics:
        # If speed > 0 and ar is large, signature of falling
        vec[10] = float(np.clip(vec[6] * (vec[9] - 0.4), -5.0, 5.0))

    # 11: Observed flag
    vec[11] = 1.0 if token.observed else 0.0

    # 12: Visibility score
    vec[12] = float(np.clip(token.visibility_score, 0.0, 1.0))

    # 13: Time delta dt
    if prev_timestamp is not None:
        dt = float(token.timestamp - prev_timestamp)
        vec[13] = float(np.clip(dt, 0.001, 2.0))
    else:
        vec[13] = float(1.0 / 30.0)

    # 14: Detector confidence
    if token.observed and token.detector_confidence is not None:
        vec[14] = float(np.clip(token.detector_confidence, 0.0, 1.0))
    else:
        vec[14] = 0.0

    # 15: Mask availability
    if use_masks and token.mask_available:
        vec[15] = 1.0
        if token.silhouette_features:
            for k in range(min(8, len(token.silhouette_features))):
                val = float(token.silhouette_features[k])
                vec[16 + k] = float(np.nan_to_num(val, nan=0.0, posinf=10.0, neginf=-10.0))
    else:
        vec[15] = 0.0

    return vec


def batch_track_tokens(
    tokens: Sequence[TrackToken],
    max_tracks: int = 8,
    max_steps: int = 32,
    use_kinematics: bool = True,
    use_masks: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Group, pad, and vectorize a sequence of TrackTokens into model-ready tensors.

    Returns
    -------
    feature_tensor: (1, max_tracks, max_steps, 24) float32
    mask_tensor: (1, max_tracks, max_steps) bool (True for valid step, False for padding)
    """
    if max_tracks < 1 or max_steps < 1:
        raise TemporalModelError("max_tracks and max_steps must be >= 1")

    # Group tokens by track_id
    by_track: Dict[int, List[TrackToken]] = {}
    for tok in tokens:
        by_track.setdefault(tok.track_id, []).append(tok)

    # Sort tracks by length (descending) to prioritize prominent actors
    sorted_tracks = sorted(by_track.values(), key=len, reverse=True)[:max_tracks]

    feat_arr = np.zeros((1, max_tracks, max_steps, TRACK_FEATURE_DIM), dtype=np.float32)
    mask_arr = np.zeros((1, max_tracks, max_steps), dtype=bool)

    for i, trk_tokens in enumerate(sorted_tracks):
        # Sort tokens chronologically
        trk_tokens_sorted = sorted(trk_tokens, key=lambda t: (t.frame_index, t.timestamp))[:max_steps]
        prev_ts = None
        for j, tok in enumerate(trk_tokens_sorted):
            feat_arr[0, i, j] = vectorize_track_token(
                token=tok,
                prev_timestamp=prev_ts,
                use_kinematics=use_kinematics,
                use_masks=use_masks,
            )
            mask_arr[0, i, j] = True
            prev_ts = tok.timestamp

    return torch.from_numpy(feat_arr), torch.from_numpy(mask_arr)


class SpatialTemporalFusionModel(nn.Module):
    """End-to-end temporal violence classifier fusing scene & track features."""

    def __init__(
        self,
        cfg: Optional[ModelConfig] = None,
        scene_dim: int = 512,
        track_dim: int = TRACK_FEATURE_DIM,
    ):
        super().__init__()
        self.cfg = cfg if cfg is not None else ModelConfig()
        hidden = self.cfg.hidden_dim
        dropout_p = self.cfg.dropout

        self.use_scene = bool(self.cfg.use_scene)
        self.use_tracks = bool(self.cfg.use_tracks)
        self.use_kinematics = bool(self.cfg.use_kinematics)
        self.use_masks = bool(self.cfg.use_masks)
        self.decision_threshold = float(self.cfg.decision_threshold)
        self.mask_predicted = bool(self.cfg.mask_predicted_observations)

        # 1. Scene Branch
        self.scene_proj = nn.Sequential(
            nn.Linear(scene_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
        )
        self.scene_gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=self.cfg.rnn_layers,
            batch_first=True,
            dropout=dropout_p if self.cfg.rnn_layers > 1 else 0.0,
        )

        # 2. Track Branch
        self.track_proj = nn.Sequential(
            nn.Linear(track_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
        )
        self.track_gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        # Track interaction aggregator (pools across K tracks)
        self.track_agg = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
        )

        # 3. Combined Fusion Head
        fusion_dim = hidden * 2
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden, 1),
        )

    def forward_tracks(
        self,
        track_features: torch.Tensor,
        track_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode batched track sequences.

        Parameters
        ----------
        track_features: (B, M, T, D)
        track_mask: (B, M, T) bool

        Returns
        -------
        track_summary: (B, hidden_dim)
        """
        B, M, T, D = track_features.shape
        hidden = self.cfg.hidden_dim

        if not self.use_tracks:
            return torch.zeros(B, hidden, device=track_features.device, dtype=track_features.dtype)

        # Flatten (B*M, T, D)
        flat_feats = track_features.view(B * M, T, D)
        flat_mask = track_mask.view(B * M, T)

        # Optional predicted-only masking
        if self.mask_predicted:
            obs_flags = flat_feats[:, :, 11:12]  # observed flag index
            flat_feats = flat_feats * obs_flags

        proj_feats = self.track_proj(flat_feats)
        gru_out, _ = self.track_gru(proj_feats)  # (B*M, T, hidden)

        # Mean-pool over valid time steps for each track
        valid_steps = flat_mask.unsqueeze(-1).float()  # (B*M, T, 1)
        sum_steps = (gru_out * valid_steps).sum(dim=1)
        count_steps = valid_steps.sum(dim=1).clamp(min=1.0)
        per_track_embed = sum_steps / count_steps  # (B*M, hidden)

        # Reshape back to (B, M, hidden)
        per_track_embed = per_track_embed.view(B, M, hidden)

        # Track active mask: track has at least one valid step
        track_has_steps = track_mask.any(dim=-1).unsqueeze(-1).float()  # (B, M, 1)

        # Aggregate across M tracks with Mean + Max pooling
        masked_tracks = per_track_embed * track_has_steps
        count_tracks = track_has_steps.sum(dim=1).clamp(min=1.0)
        mean_tracks = masked_tracks.sum(dim=1) / count_tracks

        # Max pooling
        masked_for_max = masked_tracks.masked_fill(track_has_steps == 0, -1e4)
        max_tracks = masked_for_max.max(dim=1)[0]
        # Guard if all tracks empty in batch element
        all_empty = (track_has_steps.sum(dim=1) == 0).squeeze(-1)
        if all_empty.any():
            max_tracks = max_tracks.masked_fill(all_empty.unsqueeze(-1), 0.0)

        pooled_tracks = torch.cat([mean_tracks, max_tracks], dim=-1)  # (B, 2*hidden)
        track_summary = self.track_agg(pooled_tracks)  # (B, hidden)

        # If window had zero tracks, explicitly zero representation
        track_summary = track_summary.masked_fill(all_empty.unsqueeze(-1), 0.0)
        return track_summary

    def forward_scene(self, scene_features: Optional[torch.Tensor]) -> torch.Tensor:
        """Encode scene-level frame features."""
        if not self.use_scene or scene_features is None:
            B = scene_features.shape[0] if scene_features is not None else 1
            return torch.zeros(B, self.cfg.hidden_dim, device=self.head[0].weight.device)

        proj = self.scene_proj(scene_features)
        gru_out, _ = self.scene_gru(proj)
        # Take last time step
        scene_summary = gru_out[:, -1, :]
        return scene_summary

    def forward(
        self,
        scene_features: Optional[torch.Tensor] = None,
        track_features: Optional[torch.Tensor] = None,
        track_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Joint forward pass producing a violence probability score in [0.0, 1.0]."""
        # Determine batch size
        if scene_features is not None:
            B = scene_features.shape[0]
            device = scene_features.device
        elif track_features is not None:
            B = track_features.shape[0]
            device = track_features.device
        else:
            raise TemporalModelError("at least one of scene_features or track_features must be provided")

        # Encode branches
        e_scene = (
            self.forward_scene(scene_features)
            if self.use_scene and scene_features is not None
            else torch.zeros(B, self.cfg.hidden_dim, device=device)
        )

        if self.use_tracks and track_features is not None and track_mask is not None:
            e_track = self.forward_tracks(track_features, track_mask)
        else:
            e_track = torch.zeros(B, self.cfg.hidden_dim, device=device)

        # Fuse
        fused = torch.cat([e_scene, e_track], dim=-1)
        logits = self.head(fused)
        violence_score = torch.sigmoid(logits)
        return violence_score

    def predict_window(
        self,
        scene_features: Optional[torch.Tensor] = None,
        tokens: Optional[Sequence[TrackToken]] = None,
    ) -> Tuple[float, str, float]:
        """Inference helper returning (violence_score, label, temporal_confidence)."""
        self.eval()
        with torch.no_grad():
            if tokens and len(tokens) > 0 and self.use_tracks:
                trk_feats, trk_mask = batch_track_tokens(
                    tokens,
                    max_tracks=self.cfg.max_tracks,
                    max_steps=self.cfg.max_steps,
                    use_kinematics=self.use_kinematics,
                    use_masks=self.use_masks,
                )
                device = next(self.parameters()).device
                trk_feats = trk_feats.to(device)
                trk_mask = trk_mask.to(device)
            else:
                trk_feats, trk_mask = None, None

            if scene_features is not None:
                device = next(self.parameters()).device
                scene_features = scene_features.to(device)
                if scene_features.ndim == 2:
                    scene_features = scene_features.unsqueeze(0)

            score_t = self.forward(scene_features, trk_feats, trk_mask)
            score = float(score_t.item())

        label = "violent" if score >= self.decision_threshold else "non_violent"
        # Calibrated distance from decision boundary
        conf = float(np.clip(2.0 * abs(score - self.decision_threshold), 0.0, 1.0))
        return score, label, conf


def build_window_output(
    window_id: str,
    source_id: str,
    start_timestamp: float,
    end_timestamp: float,
    violence_score: float,
    temporal_confidence: float,
    clarity_score: float,
    processed_frame_count: int,
    skipped_frame_count: int,
    tracks: Optional[List[TrackSummary]] = None,
    objects: Optional[List[ObjectEvidence]] = None,
    diagnostics: Optional[WindowDiagnostics] = None,
    decision_threshold: float = 0.50,
) -> WindowOutput:
    """Build the final immutable WindowOutput data contract."""
    label = "violent" if violence_score >= decision_threshold else "non_violent"
    return WindowOutput(
        window_id=str(window_id),
        source_id=str(source_id),
        start_timestamp=float(max(0.0, start_timestamp)),
        end_timestamp=float(max(start_timestamp, end_timestamp)),
        label=label,
        violence_score=float(np.clip(violence_score, 0.0, 1.0)),
        temporal_confidence=float(np.clip(temporal_confidence, 0.0, 1.0)),
        clarity_score=float(np.clip(clarity_score, 0.0, 1.0)),
        processed_frame_count=int(processed_frame_count),
        skipped_frame_count=int(skipped_frame_count),
        tracks=tracks if tracks is not None else [],
        objects=objects if objects is not None else [],
        diagnostics=diagnostics if diagnostics is not None else WindowDiagnostics(),
    )
