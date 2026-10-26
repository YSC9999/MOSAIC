"""Command-line interface.

    python -m mosaic smoke --synthetic burst            # no files needed
    python -m mosaic smoke --video path/to/clip.avi     # any video file
    python -m mosaic make-fixtures                      # write sample videos
    python -m mosaic make-mock-rwf                      # generate mock RWF-2000 clips
    python -m mosaic scan-dataset                       # build & verify RWF manifest
    python -m mosaic train-baseline                     # train & evaluate Phase 1 baseline

``smoke`` is the Phase 0 "does everything work" command: it loads the config,
reads a video, and writes diagnostics (plus a config snapshot, the environment
record and a log) into ``artifacts/runs/<run id>/``.

Heavy ML modules (PyTorch, torchvision) are imported lazily inside Phase 1+
commands so Phase 0 commands (``smoke``, ``make-fixtures``) start instantaneously
and run in clean environments without PyTorch installed.

Exit codes: 0 = success, 1 = bad configuration/arguments, 2 = the video could
not be decoded cleanly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .artifacts import ArtifactLayout, make_run_id
from .config import ConfigError, dump_config, experiment_id, load_config
from .environment import write_environment_record
from .ingestion import FrameSource, VideoReader
from .logging_utils import get_logger, setup_logging, shutdown_logging
from .pipeline import Pipeline
from .seeding import seed_everything
from .synthetic import SCENARIOS, generate_synthetic

EXIT_OK = 0
EXIT_CONFIG = 1
EXIT_DECODE = 2

DEFAULT_CONFIG = Path("configs") / "default.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mosaic", description="MOSAIC-FL bounded pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="load config, read a video, write diagnostics")
    smoke.add_argument("--config", default=None, help="path to a YAML config (default: configs/default.yaml if present)")
    smoke.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a setting, e.g. --set experiment.seed=7 (repeatable)",
    )
    source = smoke.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="path to a video file")
    source.add_argument("--synthetic", choices=SCENARIOS, help="use a generated synthetic video")
    smoke.set_defaults(func=cmd_smoke)

    fixtures = sub.add_parser("make-fixtures", help="write sample synthetic videos and their ground truth")
    fixtures.add_argument("--out", default=None, help="output folder (default: <artifacts_dir>/fixtures)")
    fixtures.add_argument("--config", default=None)
    fixtures.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    fixtures.set_defaults(func=cmd_make_fixtures)

    mock_rwf = sub.add_parser("make-mock-rwf", help="generate a self-contained mock RWF-2000 dataset")
    mock_rwf.add_argument("--out", default="data/rwf2000_mock", help="output folder")
    mock_rwf.add_argument("--clips-per-class", type=int, default=2, help="clips per class per split")
    mock_rwf.add_argument("--num-frames", type=int, default=16, help="frames per clip")
    mock_rwf.add_argument("--config", default=None)
    mock_rwf.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    mock_rwf.set_defaults(func=cmd_make_mock_rwf)

    scan = sub.add_parser("scan-dataset", help="scan RWF-2000 folder, carve split, verify, and write manifest")
    scan.add_argument("--data-dir", default=None, help="path to RWF-2000 root (default from config)")
    scan.add_argument("--out", default=None, help="manifest output path (default from config)")
    scan.add_argument("--config", default=None)
    scan.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    scan.set_defaults(func=cmd_scan_dataset)

    train = sub.add_parser("train-baseline", help="train baseline temporal classifier on RWF dataset/mock")
    train.add_argument("--config", default=None)
    train.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    train.add_argument("--manifest", default=None, help="path to manifest json")
    train.set_defaults(func=cmd_train_baseline)

    dl = sub.add_parser("download-rwf", help="download and extract the RWF-2000 dataset from Hugging Face")
    dl.add_argument("--out", default="data/RWF-2000", help="target directory (default: data/RWF-2000)")
    dl.add_argument("--keep-archive", action="store_true", help="keep the .tar.gz archive after extraction")
    dl.set_defaults(func=cmd_download_rwf)

    prof = sub.add_parser("profile-clarity", help="compute clarity distribution across dataset clips and degradations")
    prof.add_argument("--manifest", default=None, help="path to manifest json (default from config)")
    prof.add_argument("--split", default="test", choices=["train", "val_carved", "test", "all"], help="split to profile")
    prof.add_argument("--max-clips", type=int, default=50, help="maximum clips to evaluate (default: 50)")
    prof.add_argument("--out", default=None, help="output JSON report path")
    prof.add_argument("--config", default=None)
    prof.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    prof.set_defaults(func=cmd_profile_clarity)

    gate_parser = sub.add_parser("evaluate-gate", help="evaluate motion gating reduction and recall on video clips")
    gate_parser.add_argument("--manifest", default=None, help="path to manifest json (default from config)")
    gate_parser.add_argument("--split", default="val_carved", choices=["train", "val_carved", "test", "all"], help="split to evaluate")
    gate_parser.add_argument("--max-clips", type=int, default=50, help="maximum clips to evaluate (default: 50)")
    gate_parser.add_argument("--out", default=None, help="output JSON report path")
    gate_parser.add_argument("--config", default=None)
    gate_parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    gate_parser.set_defaults(func=cmd_evaluate_gate)

    det_parser = sub.add_parser("detect-persons", help="run person detector on video clips and report detections and latency")
    det_parser.add_argument("--manifest", default=None, help="path to manifest json (default from config)")
    det_parser.add_argument("--split", default="val_carved", choices=["train", "val_carved", "test", "all"], help="split to evaluate")
    det_parser.add_argument("--max-clips", type=int, default=20, help="maximum clips to evaluate (default: 20)")
    det_parser.add_argument("--out", default=None, help="output JSON report path")
    det_parser.add_argument("--config", default=None)
    det_parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    det_parser.set_defaults(func=cmd_detect_persons)

    return parser


def _load(args: argparse.Namespace):
    config_path: Optional[Path] = Path(args.config) if args.config else None
    if config_path is None and DEFAULT_CONFIG.is_file():
        config_path = DEFAULT_CONFIG
    return load_config(config_path, args.overrides)


def cmd_smoke(args: argparse.Namespace) -> int:
    try:
        cfg = _load(args)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    seed_everything(cfg.experiment.seed)
    layout = ArtifactLayout(Path(cfg.paths.artifacts_dir)).ensure()
    exp_id = experiment_id(cfg)
    run_id = make_run_id(exp_id)
    run_dir = layout.run_dir(run_id)
    log_file = run_dir / "run.log" if cfg.logging.to_file else None
    setup_logging(cfg.logging.level, log_file)
    log = get_logger("cli")
    try:
        dump_config(cfg, run_dir / "config.yaml")
        write_environment_record(run_dir / "environment.json")

        source: FrameSource
        if args.video:
            source = VideoReader(args.video, decode=cfg.decode)
        else:
            video = generate_synthetic(args.synthetic, seed=cfg.experiment.seed)
            source = video.frame_source()

        diagnostics = Pipeline.from_config(cfg).run(source)
        out_path = run_dir / "diagnostics.json"
        out_path.write_text(json.dumps(diagnostics.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        log.info("wrote %s", out_path)

        print(
            f"status={diagnostics.status} frames={diagnostics.processed_frame_count} "
            f"experiment={exp_id}\nrun folder: {run_dir}"
        )
        if diagnostics.status != "ok":
            for message in diagnostics.decode.errors:
                print(f"DECODE PROBLEM: {message}", file=sys.stderr)
            return EXIT_DECODE
        return EXIT_OK
    finally:
        shutdown_logging()


def cmd_make_fixtures(args: argparse.Namespace) -> int:
    try:
        cfg = _load(args)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    out_dir = Path(args.out) if args.out else Path(cfg.paths.artifacts_dir) / "fixtures"
    for scenario in SCENARIOS:
        video = generate_synthetic(scenario, seed=cfg.experiment.seed)
        avi = video.write(out_dir / f"{scenario}.avi")
        video.write_ground_truth(out_dir / f"{scenario}_ground_truth.json")
        print(f"wrote {avi} ({video.num_frames} frames)")
    return EXIT_OK


def cmd_make_mock_rwf(args: argparse.Namespace) -> int:
    try:
        cfg = _load(args)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    from .synthetic_rwf import create_mock_rwf_dataset

    out_dir = Path(args.out)
    created = create_mock_rwf_dataset(
        target_dir=out_dir,
        clips_per_class=args.clips_per_class,
        num_frames=args.num_frames,
        seed=cfg.experiment.seed,
    )
    print(f"created mock RWF-2000 dataset at: {created.resolve()}")
    return EXIT_OK


def cmd_scan_dataset(args: argparse.Namespace) -> int:
    try:
        cfg = _load(args)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    from .dataset import scan_rwf_directory

    data_dir = Path(args.data_dir) if args.data_dir else Path(cfg.data.raw_dir)
    out_path = Path(args.out) if args.out else Path(cfg.data.manifest_path)

    if not data_dir.is_dir():
        print(f"ERROR: dataset directory not found: {data_dir}", file=sys.stderr)
        return EXIT_CONFIG

    print(f"Scanning dataset in {data_dir}...")
    manifest = scan_rwf_directory(
        raw_dir=data_dir,
        val_carve_ratio=cfg.data.val_carve_ratio,
        seed=cfg.experiment.seed,
        window_cfg=cfg.window,
    )
    manifest.save(out_path)
    train_entries = manifest.entries_for_split("train")
    val_entries = manifest.entries_for_split("val_carved")
    test_entries = manifest.entries_for_split("test")
    train_f = sum(1 for e in train_entries if e.target == 1)
    val_f = sum(1 for e in val_entries if e.target == 1)
    test_f = sum(1 for e in test_entries if e.target == 1)
    exact_dups = int(manifest.meta.get("removed_exact_duplicate_count", 0))
    src_ov = int(manifest.meta.get("train_val_source_video_overlap", 0))
    print(
        f"Manifest built and verified: {len(manifest)} valid clips.\n"
        f"  train:       {len(train_entries)} clips ({train_f} Fight, {len(train_entries)-train_f} NonFight)\n"
        f"  val_carved:  {len(val_entries)} clips ({val_f} Fight, {len(val_entries)-val_f} NonFight)\n"
        f"  test:        {len(test_entries)} clips ({test_f} Fight, {len(test_entries)-test_f} NonFight)\n"
        f"  corrupt:     {len(manifest.corrupt_entries)} clips\n"
        f"  exact-dups:  {exact_dups} removed\n"
        f"  src-overlap: {src_ov} shared cameras between train & val_carved\n"
        f"  near-dups:   {len(manifest.near_duplicates)} pairs\n"
        f"Saved manifest to {out_path}"
    )
    return EXIT_OK


def cmd_train_baseline(args: argparse.Namespace) -> int:
    try:
        cfg = _load(args)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    from .dataset import RWFDataset, RWFManifest, make_rwf_dataloader, scan_rwf_directory
    from .models.temporal_baseline import BaselineTemporalClassifier
    from .synthetic_rwf import create_mock_rwf_dataset
    from .training import train_baseline_classifier

    seed_everything(cfg.experiment.seed, seed_torch=True)
    manifest_path = Path(args.manifest) if args.manifest else Path(cfg.data.manifest_path)

    if not manifest_path.is_file():
        raw_dir = Path(cfg.data.raw_dir)
        if not raw_dir.is_dir():
            print(f"data_dir {raw_dir} not found; generating mock RWF dataset...")
            create_mock_rwf_dataset(raw_dir, clips_per_class=4, seed=cfg.experiment.seed)
        manifest = scan_rwf_directory(
            raw_dir,
            val_carve_ratio=cfg.data.val_carve_ratio,
            seed=cfg.experiment.seed,
            window_cfg=cfg.window,
        )
        manifest.save(manifest_path)
    else:
        manifest = RWFManifest.load(manifest_path)

    layout = ArtifactLayout(Path(cfg.paths.artifacts_dir)).ensure()
    exp_id = experiment_id(cfg)
    run_id = make_run_id(f"{exp_id}_baseline")
    run_dir = layout.run_dir(run_id)

    log_file = run_dir / "run.log" if cfg.logging.to_file else None
    setup_logging(cfg.logging.level, log_file)
    try:
        dump_config(cfg, run_dir / "config.yaml")
        write_environment_record(run_dir / "environment.json")

        train_entries = manifest.entries_for_split("train")
        val_entries = manifest.entries_for_split("val_carved")
        test_entries = manifest.entries_for_split("test")

        train_ds = RWFDataset(
            train_entries,
            num_frames=cfg.data.num_frames,
            target_size=(cfg.data.target_height, cfg.data.target_width),
            sampling_strategy=cfg.data.sampling_strategy,
            is_train=True,
            horizontal_flip_prob=cfg.data.horizontal_flip_prob,
            seed=cfg.experiment.seed,
        )
        val_ds = RWFDataset(
            val_entries,
            num_frames=cfg.data.num_frames,
            target_size=(cfg.data.target_height, cfg.data.target_width),
            sampling_strategy=cfg.data.sampling_strategy,
            is_train=False,
            horizontal_flip_prob=0.0,
            seed=cfg.experiment.seed,
        )

        train_loader = make_rwf_dataloader(
            train_ds,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            seed=cfg.experiment.seed,
        )
        val_loader = make_rwf_dataloader(
            val_ds,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            seed=cfg.experiment.seed,
        )
        test_loader = None
        if test_entries:
            test_ds = RWFDataset(
                test_entries,
                num_frames=cfg.data.num_frames,
                target_size=(cfg.data.target_height, cfg.data.target_width),
                sampling_strategy=cfg.data.sampling_strategy,
                is_train=False,
                horizontal_flip_prob=0.0,
                seed=cfg.experiment.seed,
            )
            test_loader = make_rwf_dataloader(
                test_ds,
                batch_size=cfg.data.batch_size,
                shuffle=False,
                num_workers=cfg.data.num_workers,
                pin_memory=cfg.data.pin_memory,
                seed=cfg.experiment.seed,
            )

        model = BaselineTemporalClassifier(
            backbone=cfg.model.backbone,
            pretrained=cfg.model.pretrained,
            freeze_backbone=cfg.model.freeze_backbone,
            hidden_dim=cfg.model.hidden_dim,
            rnn_type=cfg.model.rnn_type,
            rnn_layers=cfg.model.rnn_layers,
            dropout=cfg.model.dropout,
        )

        print(f"Starting baseline training for {cfg.train.epochs} epochs on {run_dir}...")
        results = train_baseline_classifier(
            model, train_loader, val_loader, cfg, run_dir, test_loader=test_loader
        )
        acc = results["final_metrics"]["accuracy"]
        f1 = results["final_metrics"]["f1"]
        auc = results["final_metrics"]["roc_auc"]
        maj_acc = results["majority_baseline_metrics"]["accuracy"]
        mot_acc = results["motion_energy_baseline_metrics"]["accuracy"]
        test_line = ""
        if results.get("test_metrics"):
            tm = results["test_metrics"]
            test_line = (
                f"  Test Accuracy: {tm['accuracy']*100:.1f}% "
                f"(F1: {tm['f1']:.3f}, ROC-AUC: {tm['roc_auc']:.3f})\n"
            )
        print(
            f"Baseline Training Complete!\n"
            f"  Val Accuracy:  {acc*100:.1f}% (vs Majority: {maj_acc*100:.1f}%, Motion: {mot_acc*100:.1f}%)\n"
            f"  Val F1:        {f1:.3f}\n"
            f"  Val ROC-AUC:   {auc:.3f}\n"
            f"{test_line}"
            f"Artifacts written to: {run_dir}"
        )
        return EXIT_OK
    finally:
        shutdown_logging()


def cmd_download_rwf(args: argparse.Namespace) -> int:
    from scripts.download_rwf2000 import download_rwf2000

    download_rwf2000(target_dir=Path(args.out), keep_archive=bool(args.keep_archive))
    return EXIT_OK


def cmd_profile_clarity(args: argparse.Namespace) -> int:
    try:
        cfg = _load(args)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    import cv2
    import numpy as np

    from .clarity import compute_window_clarity
    from .dataset import RWFManifest
    from .degradations import DEGRADATION_KINDS, apply_degradation
    from .sampling import sample_frame_indices

    manifest_path = Path(args.manifest) if args.manifest else Path(cfg.data.manifest_path)
    if not manifest_path.is_file():
        print(f"ERROR: manifest not found at {manifest_path}", file=sys.stderr)
        return EXIT_CONFIG

    manifest = RWFManifest.load(manifest_path)
    entries = manifest.entries if args.split == "all" else manifest.entries_for_split(args.split)
    if not entries:
        print(f"ERROR: no clips found for split {args.split!r}", file=sys.stderr)
        return EXIT_CONFIG

    entries = entries[: max(1, args.max_clips)]
    print(f"Profiling clarity across {len(entries)} clips ({args.split} split)...")

    results_by_condition: dict[str, list[float]] = {
        "clean": [],
        **{k: [] for k in DEGRADATION_KINDS},
    }

    for idx, entry in enumerate(entries):
        cap = cv2.VideoCapture(entry.abs_path)
        raw_frames = []
        try:
            while True:
                ok, frm = cap.read()
                if not ok or frm is None:
                    break
                raw_frames.append(frm)
        finally:
            cap.release()

        if len(raw_frames) < 2:
            continue

        idxs = sample_frame_indices(len(raw_frames), min(16, len(raw_frames)), strategy="uniform")
        sampled = [raw_frames[i] for i in idxs]

        # Clean score
        clean_rep = compute_window_clarity(sampled, aggregation=cfg.clarity.aggregation)
        results_by_condition["clean"].append(clean_rep.score)

        # Degraded scores at severity 0.6
        for kind in DEGRADATION_KINDS:
            deg_frames, _ = apply_degradation(
                sampled, kind=kind, severity=0.6, seed=cfg.experiment.seed + idx
            )
            deg_rep = compute_window_clarity(deg_frames, aggregation=cfg.clarity.aggregation)
            results_by_condition[kind].append(deg_rep.score)

    summary_stats = {}
    clean_mean = float(np.mean(results_by_condition["clean"]))
    print(f"\nClarity Profiling Results (N={len(results_by_condition['clean'])} clips):")
    print(f"  Clean Input: mean_clarity = {clean_mean:.3f}")
    summary_stats["clean"] = {
        "mean_clarity": clean_mean,
        "std_clarity": float(np.std(results_by_condition["clean"])),
    }

    print("\nImpact of Camera Degradations (severity=0.6):")
    for kind in DEGRADATION_KINDS:
        deg_scores = results_by_condition[kind]
        m_score = float(np.mean(deg_scores))
        delta = m_score - clean_mean
        summary_stats[kind] = {
            "mean_clarity": m_score,
            "std_clarity": float(np.std(deg_scores)),
            "delta_from_clean": delta,
        }
        print(f"  {kind:<14} mean = {m_score:.3f} (delta = {delta:+.3f})")

    out_file = (
        Path(args.out)
        if args.out
        else Path(cfg.paths.artifacts_dir) / "reports" / "clarity_profile_report.json"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary_stats, indent=2), encoding="utf-8")
    print(f"\nSaved report to: {out_file}")
    return EXIT_OK


def cmd_evaluate_gate(args: argparse.Namespace) -> int:
    try:
        cfg = _load(args)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    import cv2
    import numpy as np

    from .dataset import RWFManifest
    from .motion_gate import MotionGate

    manifest_path = Path(args.manifest) if args.manifest else Path(cfg.data.manifest_path)
    if not manifest_path.is_file():
        print(f"ERROR: manifest not found at {manifest_path}", file=sys.stderr)
        return EXIT_CONFIG

    manifest = RWFManifest.load(manifest_path)
    entries = manifest.entries if args.split == "all" else manifest.entries_for_split(args.split)
    if not entries:
        print(f"ERROR: no clips found for split {args.split!r}", file=sys.stderr)
        return EXIT_CONFIG

    fights = [e for e in entries if e.label.lower() in ("fight", "violent")]
    non_fights = [e for e in entries if e.label.lower() not in ("fight", "violent")]
    half = max(1, args.max_clips // 2)
    selected_entries = fights[:half] + non_fights[:half]
    if not selected_entries:
        selected_entries = entries[: max(1, args.max_clips)]

    print(
        f"Evaluating motion gate across {len(selected_entries)} clips "
        f"({len(fights[:half])} violent, {len(non_fights[:half])} non-violent, {args.split} split)..."
    )

    gate = MotionGate(
        enabled=cfg.motion_gate.enabled,
        threshold=cfg.motion_gate.threshold,
        burst_threshold=cfg.motion_gate.burst_threshold,
        burst_hold_frames=cfg.motion_gate.burst_hold_frames,
        keep_alive_frames=cfg.motion_gate.keep_alive_frames,
        downsample_size=(
            cfg.motion_gate.downsample_width,
            cfg.motion_gate.downsample_height,
        ),
    )

    by_class: dict[str, list[dict[str, Any]]] = {"violent": [], "non_violent": []}

    for entry in selected_entries:
        cap = cv2.VideoCapture(entry.abs_path)
        frames = []
        try:
            while True:
                ok, frm = cap.read()
                if not ok or frm is None:
                    break
                frames.append(frm)
        finally:
            cap.release()

        if len(frames) < 2:
            continue

        res = gate.filter_clip(frames)
        rep = res.to_report()

        # Compute adjacent frame motion energy for recall proxy
        adj_energies = []
        for i in range(1, len(frames)):
            f_prev = gate._prepare_thumbnail(frames[i - 1])
            f_curr = gate._prepare_thumbnail(frames[i])
            adj_energies.append(float(np.mean(np.abs(f_curr - f_prev))))

        # High-motion recall proxy:
        # A frame transition is considered a physical motion event if adjacent motion energy
        # exceeds the noise threshold (0.8 * gate.threshold). If a clip is quiescent (no motion
        # above noise), zero critical events occurred, so recall is 1.0.
        noise_floor = gate.threshold * 0.80
        sig_indices = {i + 1 for i, v in enumerate(adj_energies) if v >= noise_floor}
        if sig_indices:
            selected_set = set(res.selected_indices)
            recalled = len(sig_indices & selected_set)
            high_motion_recall = float(recalled) / float(len(sig_indices))
        else:
            high_motion_recall = 1.0

        clip_stat = {
            "clip_id": entry.clip_id,
            "label": entry.label,
            "total_frames": rep.total_frames,
            "selected_frames": rep.selected_frames,
            "skipped_frames": rep.skipped_frames,
            "reduction_fraction": rep.reduction_fraction,
            "mean_motion_score": rep.mean_motion_score,
            "burst_count": rep.burst_count,
            "keep_alive_count": rep.keep_alive_count,
            "high_motion_recall": high_motion_recall,
        }
        label_key = "violent" if entry.label.lower() in ("fight", "violent") else "non_violent"
        by_class[label_key].append(clip_stat)

    summary: dict[str, Any] = {
        "parameters": {
            "threshold": cfg.motion_gate.threshold,
            "burst_threshold": cfg.motion_gate.burst_threshold,
            "burst_hold_frames": cfg.motion_gate.burst_hold_frames,
            "keep_alive_frames": cfg.motion_gate.keep_alive_frames,
            "downsample_size": [
                cfg.motion_gate.downsample_width,
                cfg.motion_gate.downsample_height,
            ],
        },
        "total_clips_evaluated": len(by_class["violent"]) + len(by_class["non_violent"]),
        "by_class": {},
    }

    print("\nMotion Gating Evaluation Summary:")
    print("=" * 65)

    all_reductions = []
    all_recalls = []

    for label in ("violent", "non_violent"):
        stats = by_class[label]
        if not stats:
            continue
        reductions = [s["reduction_fraction"] for s in stats]
        selected = [s["selected_frames"] for s in stats]
        recalls = [s["high_motion_recall"] for s in stats]
        bursts = [s["burst_count"] for s in stats]
        keep_alives = [s["keep_alive_count"] for s in stats]
        motions = [s["mean_motion_score"] for s in stats]

        all_reductions.extend(reductions)
        all_recalls.extend(recalls)

        class_summary = {
            "num_clips": len(stats),
            "mean_reduction_fraction": float(np.mean(reductions)),
            "mean_selected_frames": float(np.mean(selected)),
            "mean_high_motion_recall": float(np.mean(recalls)),
            "mean_burst_events": float(np.mean(bursts)),
            "mean_keep_alive_events": float(np.mean(keep_alives)),
            "mean_motion_score": float(np.mean(motions)),
        }
        summary["by_class"][label] = class_summary

        print(f"Class: {label.upper():<12} (N={len(stats)})")
        print(f"  Frame Reduction:    {class_summary['mean_reduction_fraction']:.1%}")
        print(f"  Frames Retained:    {class_summary['mean_selected_frames']:.1f} frames/clip")
        print(f"  High-Motion Recall: {class_summary['mean_high_motion_recall']:.1%}")
        print(f"  Burst Triggers:     {class_summary['mean_burst_events']:.1f} per clip")
        print(f"  Keep-Alive Events:  {class_summary['mean_keep_alive_events']:.1f} per clip")
        print(f"  Mean Motion Score:  {class_summary['mean_motion_score']:.4f}\n")

    overall_reduction = float(np.mean(all_reductions)) if all_reductions else 0.0
    overall_recall = float(np.mean(all_recalls)) if all_recalls else 1.0
    summary["overall"] = {
        "mean_reduction_fraction": overall_reduction,
        "mean_high_motion_recall": overall_recall,
    }
    print("=" * 65)
    print(f"OVERALL Frame Reduction:    {overall_reduction:.1%} compute saved")
    print(f"OVERALL High-Motion Recall: {overall_recall:.1%} critical frames preserved")

    out_file = (
        Path(args.out)
        if args.out
        else Path(cfg.paths.artifacts_dir) / "reports" / "motion_gate_report.json"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved report to: {out_file}")
    return EXIT_OK


def cmd_detect_persons(args: argparse.Namespace) -> int:
    try:
        cfg = _load(args)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG

    import time
    import cv2
    import numpy as np

    from .dataset import RWFManifest
    from .detector import create_detector
    from .motion_gate import MotionGate
    from .clarity import compute_window_clarity

    manifest_path = Path(args.manifest) if args.manifest else Path(cfg.data.manifest_path)
    if not manifest_path.is_file():
        print(f"ERROR: manifest not found at {manifest_path}", file=sys.stderr)
        return EXIT_CONFIG

    manifest = RWFManifest.load(manifest_path)
    entries = manifest.entries if args.split == "all" else manifest.entries_for_split(args.split)
    if not entries:
        print(f"ERROR: no clips found for split {args.split!r}", file=sys.stderr)
        return EXIT_CONFIG

    fights = [e for e in entries if e.label.lower() in ("fight", "violent")]
    non_fights = [e for e in entries if e.label.lower() not in ("fight", "violent")]
    half = max(1, args.max_clips // 2)
    selected_entries = fights[:half] + non_fights[:half]
    if not selected_entries:
        selected_entries = entries[: max(1, args.max_clips)]

    print(
        f"Running person detector ({cfg.detector.architecture}) across {len(selected_entries)} clips "
        f"({len(fights[:half])} violent, {len(non_fights[:half])} non-violent, {args.split} split)..."
    )

    gate = MotionGate(
        enabled=cfg.motion_gate.enabled,
        threshold=cfg.motion_gate.threshold,
        burst_threshold=cfg.motion_gate.burst_threshold,
        burst_hold_frames=cfg.motion_gate.burst_hold_frames,
        keep_alive_frames=cfg.motion_gate.keep_alive_frames,
    )
    detector = create_detector(cfg.detector)

    clip_results = []
    latencies = []

    for entry in selected_entries:
        cap = cv2.VideoCapture(entry.abs_path)
        frames = []
        try:
            while True:
                ok, frm = cap.read()
                if not ok or frm is None:
                    break
                frames.append(frm)
        finally:
            cap.release()

        if len(frames) < 2:
            continue

        gate_res = gate.filter_clip(frames)
        retained_frames = gate_res.selected_frames
        retained_indices = gate_res.selected_indices
        retained_ts = gate_res.selected_timestamps

        # Sample a subset of retained frames (up to 16) for detector benchmarking
        sample_step = max(1, len(retained_frames) // 16)
        bench_frames = retained_frames[::sample_step]
        bench_indices = retained_indices[::sample_step]
        bench_ts = retained_ts[::sample_step]

        t0 = time.perf_counter()
        detections = detector.detect_batch(bench_frames, bench_indices, bench_ts)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        lat_per_frame = elapsed_ms / max(1, len(bench_frames))
        latencies.append(lat_per_frame)

        det_counts = [len(dets) for dets in detections]
        flat_dets = [d for frame_dets in detections for d in frame_dets]

        # Compute clarity with person scale
        clarity_rep = compute_window_clarity(
            bench_frames,
            aggregation=cfg.clarity.aggregation,
            detections=flat_dets,
        )

        person_heights = [d.box.height for d in flat_dets]
        h_frame = bench_frames[0].shape[0]
        scale_ratios = [h / h_frame for h in person_heights]

        clip_results.append({
            "clip_id": entry.clip_id,
            "label": entry.label,
            "total_frames": len(frames),
            "retained_frames": len(retained_frames),
            "tested_frames": len(bench_frames),
            "total_detections": len(flat_dets),
            "mean_dets_per_frame": float(np.mean(det_counts)),
            "mean_latency_ms": lat_per_frame,
            "clarity_score": clarity_rep.score,
            "person_scale_component": clarity_rep.person_scale_component,
            "median_person_scale_ratio": float(np.median(scale_ratios)) if scale_ratios else None,
        })

    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    all_det_means = [r["mean_dets_per_frame"] for r in clip_results]
    all_scale_comps = [r["person_scale_component"] for r in clip_results if r["person_scale_component"] is not None]
    all_clarity = [r["clarity_score"] for r in clip_results]

    summary = {
        "architecture": cfg.detector.architecture,
        "confidence_threshold": cfg.detector.confidence_threshold,
        "total_clips": len(clip_results),
        "mean_latency_per_frame_ms": mean_lat,
        "fps_equivalent": 1000.0 / max(1e-3, mean_lat),
        "mean_persons_per_frame": float(np.mean(all_det_means)) if all_det_means else 0.0,
        "mean_person_scale_component": float(np.mean(all_scale_comps)) if all_scale_comps else None,
        "mean_window_clarity_score": float(np.mean(all_clarity)) if all_clarity else 0.0,
        "clips": clip_results,
    }

    print("\nPerson Detection Evaluation Summary:")
    print("=" * 65)
    print(f"Architecture:          {cfg.detector.architecture}")
    print(f"Inference Latency:     {mean_lat:.1f} ms / frame ({summary['fps_equivalent']:.1f} FPS on CPU)")
    print(f"Persons Per Frame:     {summary['mean_persons_per_frame']:.2f}")
    if summary["mean_person_scale_component"] is not None:
        print(f"Person Scale Score:    {summary['mean_person_scale_component']:.3f} (median height vs frame)")
    print(f"Window Clarity Score:  {summary['mean_window_clarity_score']:.3f}")
    print("=" * 65)

    out_file = (
        Path(args.out)
        if args.out
        else Path(cfg.paths.artifacts_dir) / "reports" / "detector_report.json"
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved detector report to: {out_file}")
    return EXIT_OK




def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
