"""The pipeline skeleton.

In Phase 0 the pipeline has exactly one stage, ``ingestion``: it reads a video
source and reports diagnostics. It needs no model weights, no dataset and no
network. Later phases add stages (clarity, motion gate, detector, tracker, ...)
by registering their names in ``config.KNOWN_STAGES`` and extending ``run``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .clarity import compute_window_clarity
from .config import Config, config_hash, experiment_id
from .contracts import (
    ClarityReport,
    DecodeDiagnostics,
    Detection,
    MotionGateReport,
    TrackSummary,
    TrackToken,
    WindowOutput,
)
from .degradations import apply_degradation
from .ingestion import DecodeError, FrameSource
from .logging_utils import get_logger, log_event
from .motion_gate import MotionGate
from .seeding import seed_everything

RUN_DIAGNOSTICS_SCHEMA = "run-diagnostics-0.1"
_log = get_logger("pipeline")


@dataclass(frozen=True)
class RunDiagnostics:
    """Machine-readable summary of one pipeline run (same schema for every source)."""

    experiment_id: str
    config_hash: str
    stages: List[str]
    status: str
    processed_frame_count: int
    decode: DecodeDiagnostics
    frame_shape: Optional[List[int]] = None
    clarity: Optional[ClarityReport] = None
    motion_gate: Optional[MotionGateReport] = None
    selected_frame_indices: Optional[List[int]] = None
    skipped_frame_indices: Optional[List[int]] = None
    detection_count: Optional[int] = None
    detections: Optional[List[List[Detection]]] = None
    tracks: Optional[List[TrackSummary]] = None
    track_tokens: Optional[List[TrackToken]] = None
    window_output: Optional[WindowOutput] = None
    schema_version: str = RUN_DIAGNOSTICS_SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "config_hash": self.config_hash,
            "stages": list(self.stages),
            "status": self.status,
            "processed_frame_count": self.processed_frame_count,
            "frame_shape": self.frame_shape,
            "clarity": None if self.clarity is None else self.clarity.to_dict(),
            "motion_gate": None if self.motion_gate is None else self.motion_gate.to_dict(),
            "selected_frame_indices": None if self.selected_frame_indices is None else list(self.selected_frame_indices),
            "skipped_frame_indices": None if self.skipped_frame_indices is None else list(self.skipped_frame_indices),
            "detection_count": self.detection_count,
            "detections": None if self.detections is None else [[d.to_dict() for d in frame_dets] for frame_dets in self.detections],
            "tracks": None if self.tracks is None else [t.to_dict() for t in self.tracks],
            "track_tokens": None if self.track_tokens is None else [tok.to_dict() for tok in self.track_tokens],
            "window_output": None if self.window_output is None else self.window_output.to_dict(),
            "decode": self.decode.to_dict(),
        }


class Pipeline:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.stages: List[str] = list(cfg.pipeline.stages)

    @classmethod
    def from_config(cls, cfg: Config) -> "Pipeline":
        """Instantiate a pipeline from configuration alone (no weights needed)."""
        return cls(cfg)

    def run(self, source: FrameSource) -> RunDiagnostics:
        """Run all configured stages over ``source`` and return diagnostics.

        Decode problems never raise from here: they are reported in the returned
        diagnostics (``status`` is not ``ok``) so the caller decides what to do.
        """
        exp_id = experiment_id(self.cfg)
        seed_everything(self.cfg.experiment.seed)
        log_event(_log, "run_started", experiment=exp_id, source=source.source_id)
        processed = 0
        shape: Optional[List[int]] = None
        frames: List[np.ndarray] = []
        timestamps: List[float] = []
        needs_frames = (
            "clarity" in self.stages
            or "motion_gate" in self.stages
            or "detector" in self.stages
            or "tracker" in self.stages
            or "temporal_classifier" in self.stages
            or self.cfg.degradation.enabled
        )
        try:
            for record in source.iter_frames():
                processed += 1
                if shape is None:
                    shape = [int(v) for v in record.frame.shape]
                if needs_frames:
                    frames.append(record.frame)
                    timestamps.append(record.timestamp)
        except DecodeError as exc:
            _log.error("decode failed for %s: %s", source.source_id, exc)
        decode = source.diagnostics

        proc_frames = frames
        if self.cfg.degradation.enabled and proc_frames:
            proc_frames, _ = apply_degradation(
                proc_frames,
                kind=self.cfg.degradation.kind,
                severity=self.cfg.degradation.severity,
                seed=self.cfg.degradation.seed,
            )

        motion_gate_report: Optional[MotionGateReport] = None
        selected_indices: Optional[List[int]] = None
        skipped_indices: Optional[List[int]] = None
        gated_frames = proc_frames
        gated_indices = list(range(len(proc_frames)))
        gated_timestamps = timestamps

        if "motion_gate" in self.stages and proc_frames:
            gate = MotionGate(
                enabled=self.cfg.motion_gate.enabled,
                threshold=self.cfg.motion_gate.threshold,
                burst_threshold=self.cfg.motion_gate.burst_threshold,
                burst_hold_frames=self.cfg.motion_gate.burst_hold_frames,
                keep_alive_frames=self.cfg.motion_gate.keep_alive_frames,
                downsample_size=(
                    self.cfg.motion_gate.downsample_width,
                    self.cfg.motion_gate.downsample_height,
                ),
            )
            gate_res = gate.filter_clip(proc_frames, timestamps)
            motion_gate_report = gate_res.to_report()
            selected_indices = gate_res.selected_indices
            skipped_indices = gate_res.skipped_indices
            gated_frames = gate_res.selected_frames
            gated_indices = gate_res.selected_indices
            gated_timestamps = gate_res.selected_timestamps

        # Block 3: Person detection
        detections: Optional[List[List[Detection]]] = None
        detection_count: Optional[int] = None
        if "detector" in self.stages and gated_frames:
            from .detector import create_detector

            detector = create_detector(self.cfg.detector)
            detections = detector.detect_batch(
                gated_frames, gated_indices, gated_timestamps
            )
            detection_count = sum(len(d_list) for d_list in detections)

        # Block 1: Clarity estimation (incorporating person scale if detections exist)
        clarity_report: Optional[ClarityReport] = None
        if "clarity" in self.stages and proc_frames:
            flat_dets = (
                [d for f_dets in detections for d in f_dets]
                if detections is not None
                else None
            )
            clarity_report = compute_window_clarity(
                proc_frames,
                aggregation=self.cfg.clarity.aggregation,
                detections=flat_dets,
            )

        # Block 5: Multi-object tracking & Track Tokens
        track_summaries: Optional[List[TrackSummary]] = None
        track_tokens: Optional[List[TrackToken]] = None
        if "tracker" in self.stages and gated_frames:
            from .tokens import build_track_token, save_tokens_to_cache
            from .tracker import Tracker

            tracker = Tracker(self.cfg.tracker)
            track_tokens = []
            h_f, w_f = gated_frames[0].shape[:2]
            c_score = clarity_report.score if clarity_report else None

            frame_detections = detections if detections is not None else [[] for _ in gated_frames]

            for g_frame, g_idx, g_ts, f_dets in zip(
                gated_frames, gated_indices, gated_timestamps, frame_detections
            ):
                active_tracks = tracker.update(f_dets, g_idx, g_ts)
                for trk in active_tracks:
                    token = build_track_token(
                        track=trk,
                        frame_width=w_f,
                        frame_height=h_f,
                        timestamp=g_ts,
                        frame_index=g_idx,
                        frame=g_frame,
                        clarity_score=c_score,
                        mask_cfg=self.cfg.masks,
                        normalize_coords=self.cfg.tokens.normalize_coords,
                        max_lost_frames=self.cfg.tracker.max_lost_frames,
                    )
                    track_tokens.append(token)

            track_summaries = tracker.get_track_summaries()

            if track_tokens:
                try:
                    save_tokens_to_cache(track_tokens, source.source_id, self.cfg)
                except Exception as exc:
                    _log.warning("failed to cache track tokens: %s", exc)

        # Block 6: Temporal context encoding & classification
        window_output: Optional[WindowOutput] = None
        if "temporal_classifier" in self.stages and proc_frames:
            import torch
            from .temporal import SpatialTemporalFusionModel, build_window_output

            temporal_model = SpatialTemporalFusionModel(self.cfg.model)
            scene_t = torch.zeros(1, len(gated_frames), 512)
            c_score = clarity_report.score if clarity_report else 1.0

            v_score, v_label, v_conf = temporal_model.predict_window(
                scene_features=scene_t,
                tokens=track_tokens,
            )

            start_t = gated_timestamps[0] if gated_timestamps else 0.0
            end_t = gated_timestamps[-1] if gated_timestamps else 0.0
            skipped_cnt = len(skipped_indices) if skipped_indices is not None else 0

            window_output = build_window_output(
                window_id=f"win_{source.source_id}",
                source_id=source.source_id,
                start_timestamp=start_t,
                end_timestamp=end_t,
                violence_score=v_score,
                temporal_confidence=v_conf,
                clarity_score=c_score,
                processed_frame_count=len(gated_frames),
                skipped_frame_count=skipped_cnt,
                tracks=track_summaries,
                objects=None,
                decision_threshold=self.cfg.model.decision_threshold,
            )

        log_event(
            _log,
            "run_finished",
            experiment=exp_id,
            frames=processed,
            status=decode.status,
        )
        return RunDiagnostics(
            experiment_id=exp_id,
            config_hash=config_hash(self.cfg),
            stages=list(self.stages),
            status=decode.status,
            processed_frame_count=processed,
            decode=decode,
            frame_shape=shape,
            clarity=clarity_report,
            motion_gate=motion_gate_report,
            selected_frame_indices=selected_indices,
            skipped_frame_indices=skipped_indices,
            detection_count=detection_count,
            detections=detections,
            tracks=track_summaries,
            track_tokens=track_tokens,
            window_output=window_output,
        )
