"""Unit tests for Phase 1 models, metrics, baselines, and aggregation."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from mosaic.aggregation import (
    AggregationError,
    aggregate_window_scores,
    score_to_label,
)
from mosaic.metrics import (
    MetricsError,
    compute_confusion_matrix,
    compute_roc_auc,
    evaluate_predictions,
)
from mosaic.models.baselines import (
    BaselineError,
    MajorityClassClassifier,
    MotionEnergyClassifier,
)
from mosaic.models.temporal_baseline import (
    BaselineTemporalClassifier,
    LightCNNBackbone,
    ModelError,
    ResNet18Backbone,
)


@pytest.mark.unit
def test_temporal_model_output_shapes():
    """Model output shape must be (B, 1) for batch size 1 and greater."""
    for backbone in ["cnn_light", "resnet18"]:
        model = BaselineTemporalClassifier(
            backbone=backbone,
            pretrained=False,
            freeze_backbone=False,
            hidden_dim=32,
            rnn_type="gru",
            dropout=0.1,
        )
        model.eval()

        for b in [1, 2, 4]:
            # [B, T=8, C=3, H=32, W=32]
            frames = torch.randn(b, 8, 3, 32, 32)
            out = model(frames)
            assert out.shape == (b, 1)
            assert torch.all(torch.isfinite(out))

            scores = model.predict_score(frames)
            assert scores.shape == (b, 1)
            assert torch.all((scores >= 0.0) & (scores <= 1.0))


@pytest.mark.unit
def test_temporal_model_respects_validity_mask_and_time_deltas():
    """Padded tail frames marked False in validity_mask do not affect output; time_deltas are used."""
    torch.manual_seed(42)
    model = BaselineTemporalClassifier(
        backbone="cnn_light",
        pretrained=False,
        freeze_backbone=False,
        hidden_dim=16,
        rnn_type="gru",
        dropout=0.0,
    )
    model.eval()

    frames_a = torch.randn(2, 6, 3, 32, 32)
    frames_b = frames_a.clone()
    # Corrupt the last 2 padded frames with extreme values
    frames_b[:, 4:, :, :, :] = 99.0

    mask = torch.tensor(
        [[True, True, True, True, False, False], [True, True, True, False, False, False]]
    )
    out_a = model(frames_a, validity_mask=mask)
    out_b = model(frames_b, validity_mask=mask)
    assert torch.allclose(out_a, out_b, atol=1e-5)

    # Changing time deltas changes the recurrent trajectory
    dt_uniform = torch.full((2, 6), 1.0 / 30.0)
    dt_gated = torch.full((2, 6), 0.5)
    out_dt1 = model(frames_a, validity_mask=mask, time_deltas=dt_uniform)
    out_dt2 = model(frames_a, validity_mask=mask, time_deltas=dt_gated)
    assert not torch.allclose(out_dt1, out_dt2)

    # Shape mismatch on validity_mask or time_deltas fails explicitly
    with pytest.raises(ModelError):
        model(frames_a, validity_mask=torch.ones(2, 4, dtype=torch.bool))
    with pytest.raises(ModelError):
        model(frames_a, time_deltas=torch.zeros(2, 4))


@pytest.mark.unit
def test_frozen_backbone_keeps_batchnorm_in_eval_mode():
    """When ResNet-18 backbone is frozen, calling model.train() keeps backbone in eval mode."""
    model = BaselineTemporalClassifier(
        backbone="resnet18",
        pretrained=False,
        freeze_backbone=True,
        hidden_dim=16,
    )
    model.train()
    assert model.training is True
    assert model.backbone.training is False
    assert model.backbone.bn1.training is False


@pytest.mark.unit
def test_binary_loss_and_metrics_handle_both_classes():
    """Binary cross-entropy loss and metrics handle positive, negative, and mixed batches."""
    criterion = nn.BCEWithLogitsLoss()
    logits = torch.tensor([[2.0], [-2.0], [1.5], [-1.5]], requires_grad=True)
    targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    loss = criterion(logits, targets)
    assert torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None and torch.all(torch.isfinite(logits.grad))

    # All-zero or all-one ground truth returns finite metrics without division-by-zero
    m_all_neg = evaluate_predictions([0, 0, 0], [0.1, 0.2, 0.3])
    assert m_all_neg.accuracy == 1.0 and m_all_neg.roc_auc == 0.5
    m_all_pos = evaluate_predictions([1, 1, 1], [0.8, 0.9, 0.7])
    assert m_all_pos.accuracy == 1.0 and m_all_pos.recall == 1.0 and m_all_pos.roc_auc == 0.5


@pytest.mark.unit
def test_window_to_clip_aggregation():
    """Window-to-clip aggregation must be deterministic and handle single/multiple windows."""
    # Single window
    assert aggregate_window_scores([0.7], policy="mean") == pytest.approx(0.7)
    assert aggregate_window_scores([0.7], policy="max") == pytest.approx(0.7)

    # Multiple windows
    scores = [0.2, 0.6, 0.4]
    assert aggregate_window_scores(scores, policy="mean") == pytest.approx(0.4)
    assert aggregate_window_scores(scores, policy="max") == pytest.approx(0.6)

    # Invalid policy or inputs
    with pytest.raises(AggregationError):
        aggregate_window_scores([])
    with pytest.raises(AggregationError):
        aggregate_window_scores([0.5], policy="sum")
    with pytest.raises(AggregationError):
        aggregate_window_scores([1.5], policy="mean")

    # Label derivation
    assert score_to_label(0.5, threshold=0.5) == "violent"
    assert score_to_label(0.499, threshold=0.5) == "non_violent"


@pytest.mark.unit
def test_majority_baseline_on_balanced_fixture():
    """Majority baseline gives exact prior and expected accuracy on balanced fixture."""
    clf = MajorityClassClassifier()
    # Balanced dataset: two 0s and two 1s
    clf.fit([0, 1, 0, 1])
    score, label, conf = clf.predict_one()
    assert score == pytest.approx(0.5)
    assert label == "violent"  # 0.5 >= 0.5 threshold
    assert conf == pytest.approx(0.5)

    scores, labels = clf.predict(4)
    assert len(scores) == 4
    metrics = evaluate_predictions([0, 1, 0, 1], scores)
    assert metrics.accuracy == pytest.approx(0.5)

    # Unfitted predict must raise
    with pytest.raises(BaselineError):
        MajorityClassClassifier().predict_one()


@pytest.mark.unit
def test_motion_energy_classifier():
    """Motion energy classifier produces non-negative energy and separates motion from static."""
    # Static video: all identical frames
    static_frames = np.ones((8, 32, 32, 3), dtype=np.uint8) * 128
    energy_static = MotionEnergyClassifier.extract_motion_energy(static_frames)
    assert energy_static == pytest.approx(0.0)

    # Moving video: alternating black and white frames (maximum movement)
    moving_frames = np.zeros((8, 32, 32, 3), dtype=np.uint8)
    moving_frames[1::2] = 255
    energy_moving = MotionEnergyClassifier.extract_motion_energy(moving_frames)
    assert energy_moving > 0.5

    clf = MotionEnergyClassifier().fit([energy_static, energy_moving], [0, 1])
    prob_s, lbl_s, _ = clf.predict_one(static_frames)
    prob_m, lbl_m, _ = clf.predict_one(moving_frames)

    assert lbl_s == "non_violent"
    assert lbl_m == "violent"
    assert prob_m > prob_s


@pytest.mark.unit
def test_metrics_evaluation():
    """Metrics calculation must handle perfect, inverted, and edge cases."""
    # Perfect predictions
    y_true = [0, 0, 1, 1]
    y_score_perfect = [0.1, 0.2, 0.9, 0.8]
    m_perf = evaluate_predictions(y_true, y_score_perfect)
    assert m_perf.accuracy == 1.0
    assert m_perf.precision == 1.0
    assert m_perf.recall == 1.0
    assert m_perf.f1 == 1.0
    assert m_perf.roc_auc == 1.0

    # Inverted predictions
    y_score_inverted = [0.9, 0.8, 0.1, 0.2]
    m_inv = evaluate_predictions(y_true, y_score_inverted)
    assert m_inv.accuracy == 0.0
    assert m_inv.roc_auc == 0.0

    # Mismatched length
    with pytest.raises(MetricsError):
        evaluate_predictions([0, 1], [0.5])

    # Empty sequence
    with pytest.raises(MetricsError):
        evaluate_predictions([], [])
