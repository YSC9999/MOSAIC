"""Empirical tracking and token benchmark on real RWF-2000 CCTV video."""

import json
import time
from pathlib import Path

import cv2
import numpy as np

from mosaic.config import config_from_dict
from mosaic.detector import TorchvisionPersonDetector
from mosaic.ingestion import VideoReader
from mosaic.motion_gate import MotionGate
from mosaic.tokens import build_track_token, save_tokens_to_cache
from mosaic.tracker import Tracker, TrackerConfig


def benchmark_rwf2000(num_clips_per_class: int = 10) -> dict:
    manifest_path = Path("artifacts/manifests/rwf2000_manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Sample from val_carved split
    val_clips = [c for c in manifest["entries"] if c["split"] == "val_carved"]
    fight_clips = [c for c in val_clips if c["label"] == "Fight"][:num_clips_per_class]
    nonfight_clips = [c for c in val_clips if c["label"] == "NonFight"][:num_clips_per_class]
    eval_clips = fight_clips + nonfight_clips

    cfg = config_from_dict({
        "motion_gate": {"enabled": True, "threshold": 0.010, "burst_threshold": 0.020},
        "tracker": {"min_hits": 2, "max_lost_frames": 15},
    })

    detector = TorchvisionPersonDetector(
        architecture="fasterrcnn_mobilenet_320",
        confidence_threshold=0.40,
        device="cpu",
    )
    gate = MotionGate(
        enabled=True,
        threshold=0.010,
        burst_threshold=0.020,
        burst_hold_frames=5,
        keep_alive_frames=15,
    )

    total_video_frames = 0
    total_gated_frames = 0
    total_tracks_formed = 0
    total_observed_tokens = 0
    total_predicted_tokens = 0
    all_track_durations = []

    det_latencies = []
    track_latencies = []
    token_latencies = []

    print(f"Benchmarking {len(eval_clips)} RWF-2000 clips...")

    for clip in eval_clips:
        src = VideoReader(clip["abs_path"], source_id=clip["clip_id"])

        frames = []
        timestamps = []
        for rec in src.iter_frames():
            frames.append(rec.frame)
            timestamps.append(rec.timestamp)

        total_video_frames += len(frames)

        # 1. Motion Gate
        gate_res = gate.filter_clip(frames, timestamps)
        gated_f = gate_res.selected_frames
        gated_ts = gate_res.selected_timestamps
        gated_idx = gate_res.selected_indices
        total_gated_frames += len(gated_f)

        if not gated_f:
            continue

        # 2. Detector
        t0 = time.perf_counter()
        detections = detector.detect_batch(gated_f, gated_idx, gated_ts)
        t_det = time.perf_counter() - t0
        det_latencies.append(t_det / len(gated_f))

        # 3. Tracker
        tracker = Tracker(cfg.tracker)
        clip_tokens = []
        h_f, w_f = gated_f[0].shape[:2]

        t1 = time.perf_counter()
        for f, idx, ts, dets in zip(gated_f, gated_idx, gated_ts, detections):
            active_tracks = tracker.update(dets, idx, ts)

            t_tok_start = time.perf_counter()
            for trk in active_tracks:
                tok = build_track_token(
                    track=trk,
                    frame_width=w_f,
                    frame_height=h_f,
                    timestamp=ts,
                    frame_index=idx,
                    frame=f,
                    mask_cfg=cfg.masks,
                )
                clip_tokens.append(tok)
                if tok.observed:
                    total_observed_tokens += 1
                else:
                    total_predicted_tokens += 1
            t_tok_elapsed = time.perf_counter() - t_tok_start
            token_latencies.append(t_tok_elapsed)

        t_track = time.perf_counter() - t1
        track_latencies.append(t_track / len(gated_f))

        # Cache tokens
        save_tokens_to_cache(clip_tokens, clip["clip_id"], cfg)

        summaries = tracker.get_track_summaries()
        total_tracks_formed += len(summaries)
        for s in summaries:
            all_track_durations.append(s.end_timestamp - s.start_timestamp)

    mean_det_ms = float(np.mean(det_latencies) * 1000.0) if det_latencies else 0.0
    mean_track_ms = float(np.mean(track_latencies) * 1000.0) if track_latencies else 0.0
    mean_combined_ms = mean_det_ms + mean_track_ms
    fps_cpu = float(1000.0 / mean_combined_ms) if mean_combined_ms > 0 else 0.0

    mean_duration_s = float(np.mean(all_track_durations)) if all_track_durations else 0.0
    total_tokens = total_observed_tokens + total_predicted_tokens
    obs_ratio = float(total_observed_tokens / total_tokens) if total_tokens > 0 else 1.0

    report = {
        "benchmark_clips": len(eval_clips),
        "total_video_frames": total_video_frames,
        "total_gated_frames": total_gated_frames,
        "gating_reduction_fraction": float(1.0 - total_gated_frames / total_video_frames),
        "total_tracks_formed": total_tracks_formed,
        "mean_tracks_per_clip": float(total_tracks_formed / len(eval_clips)),
        "mean_track_duration_seconds": mean_duration_s,
        "total_tokens_cached": total_tokens,
        "observed_token_ratio": obs_ratio,
        "predicted_token_ratio": float(1.0 - obs_ratio),
        "detector_latency_ms": mean_det_ms,
        "tracker_latency_ms": mean_track_ms,
        "combined_edge_latency_ms": mean_combined_ms,
        "throughput_fps_cpu": fps_cpu,
    }

    print("\nBenchmark Results:")
    for k, v in report.items():
        print(f"  {k}: {v}")

    out_file = Path("artifacts/reports/tracker_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved benchmark report to: {out_file}")
    return report


if __name__ == "__main__":
    benchmark_rwf2000(num_clips_per_class=10)
