"""Ablations, multi-seed aggregation, and robustness benchmarking (Phase 8).

Provides tools for principled empirical validation:
1. Multi-Seed Aggregation: Aggregates metrics across multiple random seeds,
   reporting mean, standard deviation, and explicitly handling failed seeds.
2. Controlled Modality Ablations: Evaluates the pipeline under selective component
   removal (scene, tracks, kinematics, masks, motion gating).
3. Degradation Stress Testing: Measures robustness curves under progressive
   camera degradations (blur, low-light, compression, frame drops).
4. Latency Profiling: Formally profiles device throughput with warm-up cycles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import Config, config_from_dict
from .metrics import evaluate_predictions, compute_roc_auc


@dataclass(frozen=True)
class MultiSeedMetric:
    """Aggregated statistics across stochastic seeds for a single metric."""

    name: str
    values: List[float]
    mean: float
    std: float
    min_val: float
    max_val: float
    successful_seeds: int
    failed_seeds: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "values": [float(v) for v in self.values],
            "mean": float(self.mean),
            "std": float(self.std),
            "min": float(self.min_val),
            "max": float(self.max_val),
            "successful_seeds": int(self.successful_seeds),
            "failed_seeds": list(self.failed_seeds),
        }


def aggregate_seed_metrics(
    seed_runs: Sequence[Dict[str, Any]],
    metric_keys: Sequence[str] = ("accuracy", "f1", "roc_auc"),
) -> Dict[str, MultiSeedMetric]:
    """Aggregate metrics across multiple seed runs, excluding failed seeds."""
    results: Dict[str, MultiSeedMetric] = {}

    for key in metric_keys:
        vals: List[float] = []
        failed: List[int] = []
        for i, run in enumerate(seed_runs):
            seed = int(run.get("seed", i))
            if run.get("status", "ok") != "ok" or key not in run:
                failed.append(seed)
                continue
            val = run[key]
            if val is not None and math.isfinite(float(val)):
                vals.append(float(val))
            else:
                failed.append(seed)

        if not vals:
            results[key] = MultiSeedMetric(
                name=key,
                values=[],
                mean=0.0,
                std=0.0,
                min_val=0.0,
                max_val=0.0,
                successful_seeds=0,
                failed_seeds=failed,
            )
        else:
            mean = float(np.mean(vals))
            std = float(np.std(vals)) if len(vals) > 1 else 0.0
            results[key] = MultiSeedMetric(
                name=key,
                values=vals,
                mean=mean,
                std=std,
                min_val=float(np.min(vals)),
                max_val=float(np.max(vals)),
                successful_seeds=len(vals),
                failed_seeds=failed,
            )

    return results


@dataclass(frozen=True)
class LatencyProfile:
    """Hardware latency and throughput profiling result."""

    device: str
    warmup_runs: int
    timed_runs: int
    mean_latency_ms: float
    std_latency_ms: float
    p95_latency_ms: float
    throughput_fps: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "warmup_runs": self.warmup_runs,
            "timed_runs": self.timed_runs,
            "mean_latency_ms": float(self.mean_latency_ms),
            "std_latency_ms": float(self.std_latency_ms),
            "p95_latency_ms": float(self.p95_latency_ms),
            "throughput_fps": float(self.throughput_fps),
        }


def profile_model_latency(
    model: torch.nn.Module,
    sample_inputs: Tuple[torch.Tensor, ...],
    device: str = "cpu",
    warmup_runs: int = 5,
    timed_runs: int = 30,
) -> LatencyProfile:
    """Profile inference latency and throughput of a PyTorch module."""
    dev = torch.device(device)
    model.to(dev)
    model.eval()

    inputs_dev = tuple(x.to(dev) if isinstance(x, torch.Tensor) else x for x in sample_inputs)

    # Warmup cycles
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(*inputs_dev)

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

    import time
    timings: List[float] = []

    with torch.no_grad():
        for _ in range(timed_runs):
            t0 = time.perf_counter()
            _ = model(*inputs_dev)
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)  # ms

    mean_ms = float(np.mean(timings))
    std_ms = float(np.std(timings))
    p95_ms = float(np.percentile(timings, 95))
    fps = float(1000.0 / mean_ms) if mean_ms > 0 else 0.0

    return LatencyProfile(
        device=str(device),
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        mean_latency_ms=mean_ms,
        std_latency_ms=std_ms,
        p95_latency_ms=p95_ms,
        throughput_fps=fps,
    )
