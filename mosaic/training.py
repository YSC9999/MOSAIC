"""Training and evaluation engine for the baseline temporal classifier.

Handles:
* Seeded, deterministic training loop with AdamW optimizer and BCEWithLogitsLoss.
* Mask-aware and time-delta-conditioned forward passes (Plan Section 2.2).
* Configured window-to-clip score aggregation (mean or max).
* Per-epoch validation, structured event logging, early stopping, and metric tracking.
* Checkpoint persistence (best_checkpoint.pt, final_checkpoint.pt) and reload verification.
* Per-clip prediction export (clip_id, ground truth, predicted violence score, label).
* Automated sanity comparison against trivial reference baselines (Majority Class and
  Motion Energy) with explicit documentation when a baseline is not beaten.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mosaic.aggregation import aggregate_window_scores, score_to_label
from mosaic.config import Config
from mosaic.logging_utils import get_logger, log_event
from mosaic.metrics import ClassificationMetrics, evaluate_predictions
from mosaic.models.baselines import MajorityClassClassifier, MotionEnergyClassifier
from mosaic.models.temporal_baseline import BaselineTemporalClassifier
from mosaic.seeding import seed_everything

_log = get_logger("training")


class TrainingError(ValueError):
    """Raised when training configuration or data is invalid."""


def get_torch_device(requested: str = "auto") -> torch.device:
    """Select the execution device."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise TrainingError("CUDA requested but not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise TrainingError(f"unknown device {requested!r}; supported: ['auto', 'cuda', 'cpu']")


def _forward_batch(
    model: nn.Module,
    batch: Dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    """Run model forward pass with frames (or cached feat_seq), validity_mask, and time_deltas."""
    validity_mask = batch.get("validity_mask")
    time_deltas = batch.get("time_deltas")
    if validity_mask is not None:
        validity_mask = validity_mask.to(device)
    if time_deltas is not None:
        time_deltas = time_deltas.to(device)

    if isinstance(model, BaselineTemporalClassifier) and "feat_seq" in batch:
        feat_seq = batch["feat_seq"].to(device)
        return model.forward_from_features(
            feat_seq, validity_mask=validity_mask, time_deltas=time_deltas
        )

    frames = batch["frames"].to(device)
    if isinstance(model, BaselineTemporalClassifier):
        return model(frames, validity_mask=validity_mask, time_deltas=time_deltas)
    return model(frames)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader | Sequence[Dict[str, Any]],
    device: torch.device,
    aggregation_policy: str = "mean",
    threshold: float = 0.5,
) -> Tuple[float, ClassificationMetrics, List[Dict[str, Any]]]:
    """Evaluate model on a DataLoader or cached batch sequence, aggregating window scores per clip."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    clip_Window_scores: Dict[str, List[float]] = {}
    clip_targets: Dict[str, int] = {}
    clip_order: List[str] = []
    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch in loader:
            targets = batch["target"].to(device).view(-1, 1)
            clip_ids = batch["clip_id"]

            logits = _forward_batch(model, batch, device)
            loss = criterion(logits, targets)

            probs = torch.sigmoid(logits).view(-1).cpu().numpy()
            target_vals = targets.view(-1).cpu().numpy().astype(int)

            total_loss += float(loss.item())
            batch_count += 1

            for cid, yt, score in zip(clip_ids, target_vals.tolist(), probs.tolist()):
                if cid not in clip_Window_scores:
                    clip_Window_scores[cid] = []
                    clip_targets[cid] = int(yt)
                    clip_order.append(cid)
                clip_Window_scores[cid].append(float(score))

    mean_loss = total_loss / max(1, batch_count)

    all_targets: List[int] = []
    all_scores: List[float] = []
    predictions: List[Dict[str, Any]] = []

    for cid in clip_order:
        agg_score = aggregate_window_scores(clip_Window_scores[cid], policy=aggregation_policy)
        yt = clip_targets[cid]
        label = score_to_label(agg_score, threshold=threshold)
        all_targets.append(yt)
        all_scores.append(agg_score)
        predictions.append(
            {
                "clip_id": cid,
                "ground_truth": yt,
                "violence_score": float(agg_score),
                "predicted_label": label,
                "window_count": len(clip_Window_scores[cid]),
                "aggregation_policy": aggregation_policy,
            }
        )

    metrics = evaluate_predictions(all_targets, all_scores, threshold=threshold)
    return mean_loss, metrics, predictions


def load_checkpoint_and_evaluate(
    checkpoint_path: Path | str,
    model: nn.Module,
    loader: DataLoader | Sequence[Dict[str, Any]],
    device: torch.device,
    aggregation_policy: str = "mean",
    threshold: float = 0.5,
) -> Tuple[float, ClassificationMetrics, List[Dict[str, Any]]]:
    """Reload model state from a saved checkpoint and evaluate on ``loader``."""
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.is_file():
        raise TrainingError(f"checkpoint file not found: {ckpt_file}")
    payload = torch.load(ckpt_file, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    return evaluate_model(
        model,
        loader,
        device=device,
        aggregation_policy=aggregation_policy,
        threshold=threshold,
    )


def train_baseline_classifier(
    model: BaselineTemporalClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Config,
    run_dir: Path | str,
    test_loader: Optional[DataLoader] = None,
) -> Dict[str, Any]:
    """Train the baseline classifier and evaluate against trivial baselines."""
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config.experiment.seed, seed_torch=True)
    device = get_torch_device(config.train.device)
    model.to(device)
    agg_policy = config.window.clip_aggregation

    log_event(
        _log,
        "training_started",
        backbone=config.model.backbone,
        device=str(device),
        epochs=config.train.epochs,
        lr=config.train.learning_rate,
        seed=config.experiment.seed,
    )

    # Only train parameters that require grad (respects frozen backbone)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.train.epochs), eta_min=config.train.learning_rate * 0.1
    )
    criterion = nn.BCEWithLogitsLoss()

    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_val_score = -1.0
    best_epoch = 1
    epochs_without_improvement = 0
    best_checkpoint_path = output_dir / "best_checkpoint.pt"
    final_checkpoint_path = output_dir / "final_checkpoint.pt"

    train_targets: List[int] = []
    train_energies: List[float] = []
    val_targets: List[int] = []
    val_energies: List[float] = []

    use_feature_cache = bool(getattr(model.backbone, "frozen", False)) and config.train.epochs >= 1
    cached_train_samples: List[Dict[str, Any]] = []
    cached_val_batches: List[Dict[str, Any]] = []
    cached_test_batches: List[Dict[str, Any]] = []

    disk_cache_path: Optional[Path] = None
    n_train = len(train_loader.dataset) if hasattr(train_loader, "dataset") else 0
    n_val = len(val_loader.dataset) if hasattr(val_loader, "dataset") else 0
    if use_feature_cache and n_train >= 100:
        cache_dir = Path(config.paths.artifacts_dir) / "token_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = (
            f"phase1_{model.backbone_name}_N{n_train}_{n_val}_"
            f"T{config.data.num_frames}_{config.data.target_height}x{config.data.target_width}.pt"
        )
        disk_cache_path = cache_dir / cache_key
        if disk_cache_path.is_file():
            try:
                payload = torch.load(disk_cache_path, map_location="cpu", weights_only=False)
                cached_train_samples = payload["cached_train_samples"]
                cached_val_batches = payload["cached_val_batches"]
                cached_test_batches = payload.get("cached_test_batches", [])
                train_targets = payload["train_targets"]
                train_energies = payload["train_energies"]
                val_targets = payload["val_targets"]
                val_energies = payload["val_energies"]
                print(f"  [Cache] Loaded frozen-backbone features from {disk_cache_path}", flush=True)
            except Exception:
                cached_train_samples = []
                cached_val_batches = []
                cached_test_batches = []

    for epoch in range(1, config.train.epochs + 1):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        train_batches = 0

        if not use_feature_cache or not cached_train_samples:
            for batch in train_loader:
                if epoch == 1:
                    batch_t = batch["target"].view(-1).cpu().numpy().astype(int).tolist()
                    frames_batch = batch["frames"]
                    masks_batch = batch.get("validity_mask")
                    for idx, f in enumerate(frames_batch):
                        m = masks_batch[idx] if masks_batch is not None else None
                        train_targets.append(batch_t[idx])
                        train_energies.append(MotionEnergyClassifier.extract_motion_energy(f, validity_mask=m))

                targets = batch["target"].to(device).view(-1, 1)
                optimizer.zero_grad()

                if use_feature_cache and epoch == 1:
                    frames_gpu = batch["frames"].to(device)
                    feat_seq = model.extract_spatial_features(frames_gpu)
                    with torch.no_grad():
                        feat_flip = (
                            model.extract_spatial_features(torch.flip(frames_gpu, dims=[-1])).cpu()
                            if config.data.horizontal_flip_prob > 0.0
                            else feat_seq.detach().cpu()
                        )
                    feat_cpu = feat_seq.detach().cpu()
                    bsz = feat_cpu.size(0)
                    for i in range(bsz):
                        cached_train_samples.append(
                            {
                                "clip_id": batch["clip_id"][i],
                                "feat_orig": feat_cpu[i],
                                "feat_flip": feat_flip[i],
                                "target": batch["target"][i],
                                "time_deltas": batch["time_deltas"][i],
                                "validity_mask": batch["validity_mask"][i],
                            }
                        )
                    cached_batch = {
                        "feat_seq": feat_seq,
                        "target": batch["target"],
                        "time_deltas": batch.get("time_deltas"),
                        "validity_mask": batch.get("validity_mask"),
                    }
                    logits = _forward_batch(model, cached_batch, device)
                else:
                    logits = _forward_batch(model, batch, device)

                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                train_loss += float(loss.item())
                train_batches += 1
        else:
            # Fast epoch over cached frozen-backbone features
            ep_rng = torch.Generator()
            ep_rng.manual_seed(config.experiment.seed + epoch * 1009)
            perm = torch.randperm(len(cached_train_samples), generator=ep_rng).tolist()
            flip_probs = torch.rand(len(cached_train_samples), generator=ep_rng).tolist()
            bsz = max(1, config.data.batch_size)
            for start in range(0, len(perm), bsz):
                chunk_idx = perm[start : start + bsz]
                feats = []
                targs = []
                dts = []
                masks = []
                for k in chunk_idx:
                    samp = cached_train_samples[k]
                    do_flip = flip_probs[k] < config.data.horizontal_flip_prob
                    feats.append(samp["feat_flip"] if do_flip else samp["feat_orig"])
                    targs.append(samp["target"])
                    dts.append(samp["time_deltas"])
                    masks.append(samp["validity_mask"])
                c_batch = {
                    "feat_seq": torch.stack(feats, dim=0),
                    "target": torch.stack(targs, dim=0),
                    "time_deltas": torch.stack(dts, dim=0),
                    "validity_mask": torch.stack(masks, dim=0),
                }
                targets = c_batch["target"].to(device).view(-1, 1)
                optimizer.zero_grad()
                logits = _forward_batch(model, c_batch, device)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())
                train_batches += 1

        scheduler.step()
        mean_train_loss = train_loss / max(1, train_batches)

        # Validation phase (cache frozen-backbone features on Epoch 1)
        if use_feature_cache and epoch == 1 and not cached_val_batches:
            for vbatch in val_loader:
                batch_t = vbatch["target"].view(-1).cpu().numpy().astype(int).tolist()
                frames_batch = vbatch["frames"]
                masks_batch = vbatch.get("validity_mask")
                for idx, f in enumerate(frames_batch):
                    m = masks_batch[idx] if masks_batch is not None else None
                    val_targets.append(batch_t[idx])
                    val_energies.append(MotionEnergyClassifier.extract_motion_energy(f, validity_mask=m))
                with torch.no_grad():
                    v_feat = model.extract_spatial_features(vbatch["frames"].to(device)).cpu()
                cached_val_batches.append(
                    {
                        "clip_id": vbatch["clip_id"],
                        "feat_seq": v_feat,
                        "target": vbatch["target"],
                        "time_deltas": vbatch.get("time_deltas"),
                        "validity_mask": vbatch.get("validity_mask"),
                    }
                )
            if test_loader is not None and not cached_test_batches:
                for tbatch in test_loader:
                    with torch.no_grad():
                        t_feat = model.extract_spatial_features(tbatch["frames"].to(device)).cpu()
                    cached_test_batches.append(
                        {
                            "clip_id": tbatch["clip_id"],
                            "feat_seq": t_feat,
                            "target": tbatch["target"],
                            "time_deltas": tbatch.get("time_deltas"),
                            "validity_mask": tbatch.get("validity_mask"),
                        }
                    )
            if disk_cache_path is not None and not disk_cache_path.is_file():
                torch.save(
                    {
                        "cached_train_samples": cached_train_samples,
                        "cached_val_batches": cached_val_batches,
                        "cached_test_batches": cached_test_batches,
                        "train_targets": train_targets,
                        "train_energies": train_energies,
                        "val_targets": val_targets,
                        "val_energies": val_energies,
                    },
                    disk_cache_path,
                )
                print(f"  [Cache] Saved frozen-backbone features to {disk_cache_path}", flush=True)

        val_source: DataLoader | Sequence[Dict[str, Any]] = (
            cached_val_batches if (use_feature_cache and cached_val_batches) else val_loader
        )
        val_loss, val_metrics, _ = evaluate_model(
            model, val_source, device, aggregation_policy=agg_policy
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": mean_train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics.accuracy,
            "val_f1": val_metrics.f1,
            "val_roc_auc": val_metrics.roc_auc,
        }
        history.append(epoch_record)
        print(
            f"  [Epoch {epoch}/{config.train.epochs}] "
            f"train_loss={mean_train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_metrics.accuracy*100:.1f}% | val_f1={val_metrics.f1:.3f} | "
            f"val_auc={val_metrics.roc_auc:.3f}",
            flush=True,
        )
        log_event(
            _log,
            "epoch_finished",
            epoch=epoch,
            train_loss=f"{mean_train_loss:.4f}",
            val_acc=f"{val_metrics.accuracy:.4f}",
            val_f1=f"{val_metrics.f1:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_roc_auc=f"{val_metrics.roc_auc:.4f}",
        )

        # Save best checkpoint (prioritize validation accuracy + ROC-AUC, tie-break with val_loss)
        val_score = float(val_metrics.accuracy) + 0.5 * float(val_metrics.roc_auc)
        improved = (val_score > best_val_score + 1e-6) or (
            abs(val_score - best_val_score) <= 1e-6 and val_loss < best_val_loss - 1e-6
        )
        if improved or not best_checkpoint_path.is_file():
            best_val_score = max(best_val_score, val_score)
            best_val_loss = min(best_val_loss, val_loss)
            best_val_acc = max(best_val_acc, val_metrics.accuracy)
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_metrics": val_metrics.to_dict(),
                    "config_seed": config.experiment.seed,
                },
                best_checkpoint_path,
            )
            log_event(_log, "best_checkpoint_saved", epoch=epoch, val_loss=f"{val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.train.early_stopping_patience:
                log_event(_log, "early_stopping_triggered", epoch=epoch)
                break

    # Save final checkpoint
    torch.save(
        {
            "epoch": history[-1]["epoch"],
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_metrics": history[-1],
        },
        final_checkpoint_path,
    )

    # Reload best checkpoint for final evaluation so reported predictions reflect best model
    val_eval_source: DataLoader | Sequence[Dict[str, Any]] = (
        cached_val_batches if (use_feature_cache and cached_val_batches) else val_loader
    )
    final_val_loss, final_metrics, final_preds = load_checkpoint_and_evaluate(
        best_checkpoint_path,
        model,
        val_eval_source,
        device=device,
        aggregation_policy=agg_policy,
    )

    test_metrics_dict: Optional[Dict[str, Any]] = None
    if test_loader is not None:
        test_eval_source: DataLoader | Sequence[Dict[str, Any]] = (
            cached_test_batches if (use_feature_cache and cached_test_batches) else test_loader
        )
        _, test_metrics, test_preds = evaluate_model(
            model,
            test_eval_source,
            device=device,
            aggregation_policy=agg_policy,
        )
        test_metrics_dict = test_metrics.to_dict()
        (output_dir / "test_predictions.json").write_text(
            json.dumps(test_preds, indent=2), encoding="utf-8"
        )

    # Use train targets and motion energies collected during Epoch 1
    if not train_targets:
        for batch in train_loader:
            batch_t = batch["target"].view(-1).cpu().numpy().astype(int).tolist()
            frames_batch = batch["frames"]
            masks_batch = batch.get("validity_mask")
            for idx, f in enumerate(frames_batch):
                m = masks_batch[idx] if masks_batch is not None else None
                train_targets.append(batch_t[idx])
                train_energies.append(MotionEnergyClassifier.extract_motion_energy(f, validity_mask=m))

    if not val_targets:
        for batch in val_loader:
            batch_t = batch["target"].view(-1).cpu().numpy().astype(int).tolist()
            frames_batch = batch["frames"]
            masks_batch = batch.get("validity_mask")
            for idx, f in enumerate(frames_batch):
                m = masks_batch[idx] if masks_batch is not None else None
                val_targets.append(batch_t[idx])
                val_energies.append(MotionEnergyClassifier.extract_motion_energy(f, validity_mask=m))

    # Evaluate Trivial Baseline 1: Majority Class
    maj_clf = MajorityClassClassifier().fit(train_targets)
    maj_scores, _ = maj_clf.predict(len(val_targets))
    majority_metrics = evaluate_predictions(val_targets, maj_scores)

    # Evaluate Trivial Baseline 2: Motion Energy
    motion_clf = MotionEnergyClassifier().fit(train_energies, train_targets)
    motion_scores, _ = motion_clf.predict(val_energies)
    motion_metrics = evaluate_predictions(val_targets, motion_scores)

    # Honest comparison and explicit documentation (Plan Phase 1 Exit Gate)
    beats_majority = bool(
        final_metrics.accuracy > majority_metrics.accuracy
        or (
            final_metrics.accuracy == majority_metrics.accuracy
            and final_metrics.roc_auc > majority_metrics.roc_auc
        )
    )
    beats_motion = bool(
        final_metrics.accuracy > motion_metrics.accuracy
        or (
            final_metrics.accuracy == motion_metrics.accuracy
            and final_metrics.roc_auc > motion_metrics.roc_auc
        )
    )

    comparison_notes: List[str] = []
    if not beats_majority:
        comparison_notes.append(
            f"Learned model did not strictly beat Majority Class baseline "
            f"(model acc={final_metrics.accuracy:.3f}, auc={final_metrics.roc_auc:.3f} vs "
            f"majority acc={majority_metrics.accuracy:.3f}, auc={majority_metrics.roc_auc:.3f}). "
            f"This is expected on tiny smoke subsets or very few training epochs."
        )
    if not beats_motion:
        comparison_notes.append(
            f"Learned model did not strictly beat Motion Energy baseline "
            f"(model acc={final_metrics.accuracy:.3f}, auc={final_metrics.roc_auc:.3f} vs "
            f"motion acc={motion_metrics.accuracy:.3f}, auc={motion_metrics.roc_auc:.3f}). "
            f"On synthetic motion-defined fixtures, raw pixel difference is an oracle separator."
        )

    # Save outputs to run directory
    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (output_dir / "val_predictions.json").write_text(
        json.dumps(final_preds, indent=2), encoding="utf-8"
    )

    summary = {
        "best_epoch": best_epoch,
        "final_metrics": final_metrics.to_dict(),
        "test_metrics": test_metrics_dict,
        "majority_baseline_metrics": majority_metrics.to_dict(),
        "motion_energy_baseline_metrics": motion_metrics.to_dict(),
        "beats_majority": beats_majority,
        "beats_motion_energy": beats_motion,
        "comparison_notes": comparison_notes,
        "history": history,
        "best_checkpoint": str(best_checkpoint_path),
        "final_checkpoint": str(final_checkpoint_path),
    }

    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    log_event(
        _log,
        "training_finished",
        beats_majority=beats_majority,
        beats_motion=beats_motion,
        best_epoch=best_epoch,
        val_acc=f"{final_metrics.accuracy:.4f}",
        val_f1=f"{final_metrics.f1:.4f}",
        val_roc_auc=f"{final_metrics.roc_auc:.4f}",
    )

    return summary
