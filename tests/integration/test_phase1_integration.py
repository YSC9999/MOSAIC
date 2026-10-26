"""Integration tests for Phase 1: End-to-end dataset loading, overfitting, reload, and reproducibility."""

import json
from pathlib import Path

import pytest
import torch

from mosaic.config import config_from_dict
from mosaic.dataset import RWFDataset, make_rwf_dataloader, scan_rwf_directory
from mosaic.models.temporal_baseline import BaselineTemporalClassifier
from mosaic.seeding import seed_everything
from mosaic.synthetic_rwf import create_mock_rwf_dataset
from mosaic.training import load_checkpoint_and_evaluate, train_baseline_classifier


@pytest.mark.integration
def test_loader_batch_passes_through_model(tmp_path: Path):
    """A batch from RWF DataLoader passes cleanly through preprocessing and the model."""
    mock_dir = tmp_path / "mock_rwf"
    create_mock_rwf_dataset(mock_dir, clips_per_class=2, num_frames=8, width=48, height=48, seed=10)
    manifest = scan_rwf_directory(mock_dir, val_carve_ratio=0.5, seed=10)

    train_ds = RWFDataset(
        manifest.entries_for_split("train"),
        num_frames=8,
        target_size=(48, 48),
        is_train=True,
    )
    loader = make_rwf_dataloader(train_ds, batch_size=2, shuffle=False)

    batch = next(iter(loader))
    assert "frames" in batch
    assert "target" in batch
    assert "validity_mask" in batch
    assert "time_deltas" in batch
    assert batch["frames"].shape == (2, 8, 3, 48, 48)
    assert batch["validity_mask"].shape == (2, 8)
    assert batch["time_deltas"].shape == (2, 8)

    model = BaselineTemporalClassifier(
        backbone="cnn_light",
        pretrained=False,
        freeze_backbone=False,
        hidden_dim=32,
    )
    out = model(
        batch["frames"],
        validity_mask=batch["validity_mask"],
        time_deltas=batch["time_deltas"],
    )
    assert out.shape == (2, 1)


@pytest.mark.integration
def test_tiny_overfit_memorization(tmp_path: Path):
    """Confirm the model has sufficient learning capacity to memorize a tiny 2-clip fixture."""
    mock_dir = tmp_path / "mock_rwf"
    create_mock_rwf_dataset(mock_dir, clips_per_class=1, num_frames=8, width=32, height=32, seed=77)
    manifest = scan_rwf_directory(mock_dir, val_carve_ratio=0.5, seed=77)

    # Use the 2 clips in val_carved as our tiny memorization target
    train_entries = manifest.entries_for_split("val_carved")  # Exactly 1 Fight + 1 NonFight
    assert len(train_entries) == 2

    ds = RWFDataset(
        train_entries,
        num_frames=8,
        target_size=(32, 32),
        is_train=False,
        horizontal_flip_prob=0.0,
    )
    loader = make_rwf_dataloader(ds, batch_size=2, shuffle=False)

    seed_everything(42, seed_torch=True)
    model = BaselineTemporalClassifier(
        backbone="cnn_light",
        pretrained=False,
        freeze_backbone=False,
        hidden_dim=32,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    model.train()
    final_loss = 1.0
    for _ in range(40):
        for batch in loader:
            optimizer.zero_grad()
            logits = model(
                batch["frames"],
                validity_mask=batch["validity_mask"],
                time_deltas=batch["time_deltas"],
            )
            loss = criterion(logits, batch["target"].view(-1, 1))
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())

    # After 40 steps on 2 samples, loss should drop near zero (< 0.15) and accuracy be 100%
    assert final_loss < 0.15

    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        scores = (
            torch.sigmoid(
                model(
                    batch["frames"],
                    validity_mask=batch["validity_mask"],
                    time_deltas=batch["time_deltas"],
                )
            )
            .view(-1)
            .numpy()
        )
        targets = batch["target"].numpy()
        preds = (scores >= 0.5).astype(int)
        assert (preds == targets).all()


@pytest.mark.integration
def test_evaluation_output_reload_and_training_reproducibility(tmp_path: Path):
    """Training twice with same seed gives identical metrics, and saved checkpoints reproduce evaluation."""
    mock_dir = tmp_path / "mock_rwf"
    create_mock_rwf_dataset(mock_dir, clips_per_class=2, num_frames=6, width=32, height=32, seed=99)
    manifest = scan_rwf_directory(mock_dir, val_carve_ratio=0.5, seed=99)

    cfg = config_from_dict(
        {
            "experiment": {"seed": 1337},
            "train": {"epochs": 2, "learning_rate": 0.001, "device": "cpu"},
            "model": {"backbone": "cnn_light", "hidden_dim": 16, "dropout": 0.0},
            "data": {"horizontal_flip_prob": 0.0},
        }
    )

    def make_loaders():
        train_ds = RWFDataset(
            manifest.entries_for_split("train"),
            num_frames=6,
            target_size=(32, 32),
            is_train=True,
            horizontal_flip_prob=0.0,
            seed=1337,
        )
        val_ds = RWFDataset(
            manifest.entries_for_split("val_carved"),
            num_frames=6,
            target_size=(32, 32),
            is_train=False,
            horizontal_flip_prob=0.0,
            seed=1337,
        )
        t_loader = make_rwf_dataloader(train_ds, batch_size=2, shuffle=True, seed=1337)
        v_loader = make_rwf_dataloader(val_ds, batch_size=2, shuffle=False, seed=1337)
        return t_loader, v_loader

    def run_trial(run_name: str):
        seed_everything(1337, seed_torch=True)
        t_loader, v_loader = make_loaders()
        model = BaselineTemporalClassifier(
            backbone="cnn_light",
            pretrained=False,
            freeze_backbone=False,
            hidden_dim=16,
            dropout=0.0,
        )
        return train_baseline_classifier(model, t_loader, v_loader, cfg, tmp_path / run_name)

    res1 = run_trial("run1")
    res2 = run_trial("run2")

    assert res1["final_metrics"] == res2["final_metrics"]
    assert res1["motion_energy_baseline_metrics"] == res2["motion_energy_baseline_metrics"]
    assert res1["history"][0]["train_loss"] == pytest.approx(res2["history"][0]["train_loss"], abs=1e-5)
    assert res1["history"][1]["train_loss"] == pytest.approx(res2["history"][1]["train_loss"], abs=1e-5)

    # Reload best_checkpoint.pt into a fresh model and verify evaluation matches saved files
    _, v_loader = make_loaders()
    fresh_model = BaselineTemporalClassifier(
        backbone="cnn_light",
        pretrained=False,
        freeze_backbone=False,
        hidden_dim=16,
        dropout=0.0,
    )
    _, reloaded_metrics, reloaded_preds = load_checkpoint_and_evaluate(
        tmp_path / "run1" / "best_checkpoint.pt",
        fresh_model,
        v_loader,
        device=torch.device("cpu"),
    )
    saved_summary = json.loads((tmp_path / "run1" / "metrics_summary.json").read_text(encoding="utf-8"))
    saved_preds = json.loads((tmp_path / "run1" / "val_predictions.json").read_text(encoding="utf-8"))
    assert reloaded_metrics.to_dict() == saved_summary["final_metrics"]
    assert len(reloaded_preds) == len(saved_preds)
    for rp, sp in zip(reloaded_preds, saved_preds):
        assert rp["clip_id"] == sp["clip_id"]
        assert rp["predicted_label"] == sp["predicted_label"]
        assert rp["violence_score"] == pytest.approx(sp["violence_score"], abs=1e-5)
