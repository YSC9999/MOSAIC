"""Unit tests for Phase 8: Ablations and multi-seed aggregation."""

import math
import pytest
import torch
import torch.nn as nn

from mosaic.ablations import (
    LatencyProfile,
    MultiSeedMetric,
    aggregate_seed_metrics,
    profile_model_latency,
)


def test_aggregate_seed_metrics_clean_runs():
    runs = [
        {"seed": 42, "accuracy": 0.75, "f1": 0.77, "status": "ok"},
        {"seed": 101, "accuracy": 0.77, "f1": 0.79, "status": "ok"},
        {"seed": 1337, "accuracy": 0.76, "f1": 0.78, "status": "ok"},
    ]

    res = aggregate_seed_metrics(runs, metric_keys=("accuracy", "f1"))

    assert "accuracy" in res
    assert "f1" in res

    acc_stat = res["accuracy"]
    assert acc_stat.successful_seeds == 3
    assert len(acc_stat.failed_seeds) == 0
    assert acc_stat.mean == pytest.approx(0.76, abs=1e-4)
    assert acc_stat.min_val == pytest.approx(0.75)
    assert acc_stat.max_val == pytest.approx(0.77)
    assert acc_stat.std > 0.0


def test_aggregate_seed_metrics_excludes_failed_seeds():
    runs = [
        {"seed": 1, "accuracy": 0.80, "status": "ok"},
        {"seed": 2, "accuracy": None, "status": "failed"},
        {"seed": 3, "accuracy": 0.82, "status": "ok"},
    ]

    res = aggregate_seed_metrics(runs, metric_keys=("accuracy",))
    acc_stat = res["accuracy"]

    assert acc_stat.successful_seeds == 2
    assert acc_stat.failed_seeds == [2]
    assert acc_stat.mean == pytest.approx(0.81, abs=1e-4)


def test_aggregate_seed_metrics_all_failed():
    runs = [
        {"seed": 1, "status": "failed"},
        {"seed": 2, "status": "failed"},
    ]

    res = aggregate_seed_metrics(runs, metric_keys=("accuracy",))
    acc_stat = res["accuracy"]

    assert acc_stat.successful_seeds == 0
    assert acc_stat.failed_seeds == [1, 2]
    assert acc_stat.mean == 0.0
    assert acc_stat.std == 0.0


def test_profile_model_latency_cpu():
    class DummyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(32, 16)

        def forward(self, x):
            return self.fc(x)

    model = DummyNet()
    dummy_input = (torch.randn(2, 32),)

    prof = profile_model_latency(model, dummy_input, device="cpu", warmup_runs=2, timed_runs=5)
    assert isinstance(prof, LatencyProfile)
    assert prof.device == "cpu"
    assert prof.warmup_runs == 2
    assert prof.timed_runs == 5
    assert prof.mean_latency_ms > 0.0
    assert math.isfinite(prof.mean_latency_ms)
    assert prof.throughput_fps > 0.0
