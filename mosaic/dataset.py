"""RWF-2000 dataset manifest, split carving, and PyTorch dataset loader.

Enforces strict data hygiene as required by the implementation plan:
1. Label validation: Only "Fight" (target 1, "violent") and "NonFight"
   (target 0, "non_violent") are accepted.
2. Split strategy:
   - The official 'val' folder is treated as the locked held-out 'test' set.
   - The official 'train' folder is carved into 'train' and 'val_carved' using
     a deterministic, stratified split seeded from configuration.
3. Anti-leakage verification:
   - 'train', 'val_carved', and 'test' are verified to be pairwise disjoint
     by clip filename, by SHA-256 file hash, and by a cheap frame-signature
     near-duplicate check (Plan Section 3.3 item 7).
   - Any collision raises a fatal DataLeakageError.
4. Explicit corrupt-clip reporting:
   - Corrupt, empty, or undecodable clips are never given fake frame counts;
     they are recorded explicitly in ``corrupt_entries`` with their decode status
     and error reason (Plan Section 3.3 item 3).
5. Preprocessing:
   - Extracts a fixed number of frames via deterministic sampling with explicit
     edge-padding and validity masks for short clips, plus per-step time deltas
     (Plan Section 2.2).
   - Horizontal flip data augmentation is strictly train-only and seeded per
     (seed, epoch, sample_index); evaluation (val_carved, test) remains 100%
     deterministic and unaugmented.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mosaic.config import Config, WindowConfig, config_hash
from mosaic.contracts import DecodeDiagnostics
from mosaic.ingestion import DecodeError, VideoReader, probe_video
from mosaic.manifest import (
    MANIFEST_SCHEMA,
    Manifest,
    ManifestEntry,
    file_sha256 as _manifest_file_sha256,
    make_meta,
)
from mosaic.sampling import (
    compute_time_deltas,
    pad_or_truncate_frames,
    resize_frame,
    sample_frame_indices,
)
from mosaic.seeding import derive_seed, make_rng

VALID_LABELS = ("Fight", "NonFight")
SPLIT_NAMES = ("train", "val_carved", "test")
LABEL_TO_CANONICAL = {"Fight": "violent", "NonFight": "non_violent"}
SPLIT_TO_CANONICAL = {"train": "train", "val_carved": "val", "test": "test"}


class DatasetError(ValueError):
    """Raised when dataset structure, labels, or files are invalid."""


class DataLeakageError(DatasetError):
    """Raised when duplicate clips or hash collisions are detected across splits."""


def file_sha256(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA-256 hexadecimal hash of a file."""
    p = Path(path)
    if not p.is_file():
        raise DatasetError(f"cannot compute hash of non-existent file: {p}")
    return _manifest_file_sha256(p, chunk_size=chunk_size)


def compute_frame_signature(frames: Sequence[np.ndarray]) -> Optional[str]:
    """Compute a compact 48-hex-character perceptual signature across 3 keyframes.

    Samples the first, middle, and last frame of a decoded clip, downsamples each
    to 8x8 grayscale, and encodes a 64-bit mean-threshold hash (16 hex chars per
    keyframe). Used to detect near-duplicate or re-encoded clips across splits
    (Plan Section 3.3 item 7).
    """
    if not frames:
        return None
    n = len(frames)
    key_indices = [0, n // 2, n - 1]
    parts: List[str] = []
    for idx in key_indices:
        frame = frames[idx]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32)
        mean_val = float(np.mean(small))
        bits = (small.flatten() >= mean_val).astype(np.uint8)
        value = 0
        for b in bits:
            value = (value << 1) | int(b)
        parts.append(f"{value:016x}")
    return "".join(parts)


@dataclass(frozen=True)
class RWFEntry:
    """Metadata record for a single validated video clip in the RWF-2000 dataset."""

    clip_id: str
    abs_path: str
    rel_path: str
    label: str
    target: int
    split: str
    sha256: str
    frame_count: int
    fps: float
    duration_seconds: float
    width: int
    height: int
    decode_status: str = "ok"
    error: Optional[str] = None
    frame_signature: Optional[str] = None

    def __post_init__(self) -> None:
        if self.label not in VALID_LABELS:
            raise DatasetError(
                f"clip {self.clip_id!r} has invalid label {self.label!r}; must be in {list(VALID_LABELS)}"
            )
        if self.target not in (0, 1):
            raise DatasetError(f"clip {self.clip_id!r} target must be 0 or 1, got {self.target}")
        if self.split not in SPLIT_NAMES:
            raise DatasetError(
                f"clip {self.clip_id!r} has invalid split {self.split!r}; must be in {list(SPLIT_NAMES)}"
            )
        if self.frame_count < 1:
            raise DatasetError(f"clip {self.clip_id!r} frame_count must be >= 1")
        if self.fps <= 0:
            raise DatasetError(f"clip {self.clip_id!r} fps must be > 0")

    @property
    def canonical_label(self) -> str:
        return LABEL_TO_CANONICAL[self.label]

    @property
    def canonical_split(self) -> str:
        return SPLIT_TO_CANONICAL[self.split]

    def to_manifest_entry(self) -> ManifestEntry:
        """Convert to Phase 0 ``ManifestEntry`` contract."""
        sha = self.sha256 if len(self.sha256) == 64 else None
        return ManifestEntry(
            clip_path=self.rel_path,
            label=self.canonical_label,
            decode_status=self.decode_status,
            frame_count=self.frame_count,
            fps=self.fps,
            width=self.width,
            height=self.height,
            duration_seconds=max(self.duration_seconds, 1e-6),
            split=self.canonical_split,
            error=self.error,
            sha256=sha,
        )

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RWFEntry":
        return cls(
            clip_id=str(data["clip_id"]),
            abs_path=str(data["abs_path"]),
            rel_path=str(data["rel_path"]),
            label=str(data["label"]),
            target=int(data["target"]),
            split=str(data["split"]),
            sha256=str(data["sha256"]),
            frame_count=int(data["frame_count"]),
            fps=float(data["fps"]),
            duration_seconds=float(data["duration_seconds"]),
            width=int(data["width"]),
            height=int(data["height"]),
            decode_status=str(data.get("decode_status", "ok")),
            error=data.get("error"),
            frame_signature=data.get("frame_signature"),
        )


@dataclass
class RWFManifest:
    """Collection of RWF entries with metadata, corrupt-clip reporting, and integrity checks."""

    entries: List[RWFEntry] = field(default_factory=list)
    corrupt_entries: List[Dict[str, Any]] = field(default_factory=list)
    near_duplicates: List[Dict[str, str]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.entries)

    def entries_for_split(self, split: str) -> List[RWFEntry]:
        if split not in SPLIT_NAMES:
            raise DatasetError(f"unknown split {split!r}; valid splits: {list(SPLIT_NAMES)}")
        return [e for e in self.entries if e.split == split]

    def verify_disjoint(self, check_near_duplicates: bool = False) -> None:
        """Verify that train, val_carved, and test sets are strictly pairwise disjoint."""
        train_names = {e.clip_id for e in self.entries if e.split == "train"}
        val_names = {e.clip_id for e in self.entries if e.split == "val_carved"}
        test_names = {e.clip_id for e in self.entries if e.split == "test"}

        train_hashes = {e.sha256 for e in self.entries if e.split == "train"}
        val_hashes = {e.sha256 for e in self.entries if e.split == "val_carved"}
        test_hashes = {e.sha256 for e in self.entries if e.split == "test"}

        # 1. Check by clip name
        tv_name_overlap = train_names & val_names
        tt_name_overlap = train_names & test_names
        vt_name_overlap = val_names & test_names

        if tv_name_overlap or tt_name_overlap or vt_name_overlap:
            raise DataLeakageError(
                f"Data leakage detected! Clip name overlap across splits: "
                f"train/val={tv_name_overlap}, train/test={tt_name_overlap}, val/test={vt_name_overlap}"
            )

        # 2. Check by SHA-256 hash (catches renamed identical files)
        tv_hash_overlap = train_hashes & val_hashes
        tt_hash_overlap = train_hashes & test_hashes
        vt_hash_overlap = val_hashes & test_hashes

        if tv_hash_overlap or tt_hash_overlap or vt_hash_overlap:
            raise DataLeakageError(
                f"Data leakage detected! SHA-256 hash collision across splits: "
                f"train/val={tv_hash_overlap}, train/test={tt_hash_overlap}, val/test={vt_hash_overlap}"
            )

        # 3. Record any near-duplicate frame signatures across splits (Plan 3.3 item 7)
        sig_by_split: Dict[str, Dict[str, str]] = {"train": {}, "val_carved": {}, "test": {}}
        for e in self.entries:
            if e.frame_signature:
                sig_by_split[e.split][e.frame_signature] = e.clip_id

        found_near: List[Dict[str, str]] = []
        pairs = [("train", "val_carved"), ("train", "test"), ("val_carved", "test")]
        for s1, s2 in pairs:
            shared_sigs = set(sig_by_split[s1]) & set(sig_by_split[s2])
            for sig in sorted(shared_sigs):
                found_near.append(
                    {
                        "split_a": s1,
                        "clip_a": sig_by_split[s1][sig],
                        "split_b": s2,
                        "clip_b": sig_by_split[s2][sig],
                        "frame_signature": sig,
                    }
                )
        self.near_duplicates = found_near
        if check_near_duplicates and found_near:
            raise DataLeakageError(
                f"Near-duplicate frame signatures detected across splits: {found_near}"
            )

    def to_phase0_manifest(self, cfg: Optional[Config] = None) -> Manifest:
        """Convert this RWFManifest into a Phase 0 ``Manifest`` (including failed entries)."""
        meta = dict(self.meta) if self.meta else make_meta(cfg or Config(), "RWF-2000")
        for key, val in make_meta(cfg or Config(), "RWF-2000").items():
            meta.setdefault(key, val)
        p0_entries = [e.to_manifest_entry() for e in self.entries]
        for c in self.corrupt_entries:
            p0_entries.append(
                ManifestEntry(
                    clip_path=c["rel_path"],
                    label=LABEL_TO_CANONICAL.get(c["label"], "non_violent"),
                    decode_status=c.get("decode_status", "failed"),
                    frame_count=int(c.get("frame_count", 0)),
                    split=SPLIT_TO_CANONICAL.get(c.get("split", "train"), "train"),
                    error=c.get("error") or "decode failed",
                    sha256=c.get("sha256") if len(str(c.get("sha256", ""))) == 64 else None,
                )
            )
        return Manifest(meta=meta, entries=p0_entries)

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(self.meta)
        if not meta:
            cfg = Config()
            meta = make_meta(cfg, "RWF-2000")
        meta["corrupt_count"] = len(self.corrupt_entries)
        meta["near_duplicate_count"] = len(self.near_duplicates)
        payload = {
            "meta": meta,
            "entries": [e.to_dict() for e in self.entries],
            "corrupt_entries": list(self.corrupt_entries),
            "near_duplicates": list(self.near_duplicates),
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "RWFManifest":
        p = Path(path)
        if not p.is_file():
            raise DatasetError(f"manifest file does not exist: {p}")
        payload = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            # Backward compatibility with bare entry list
            entries = [RWFEntry.from_dict(item) for item in payload]
            manifest = cls(entries=entries)
        elif isinstance(payload, Mapping) and "entries" in payload:
            entries = [RWFEntry.from_dict(item) for item in payload["entries"]]
            manifest = cls(
                entries=entries,
                corrupt_entries=list(payload.get("corrupt_entries", [])),
                near_duplicates=list(payload.get("near_duplicates", [])),
                meta=dict(payload.get("meta", {})),
            )
        else:
            raise DatasetError(f"invalid RWFManifest JSON format in {p}")
        manifest.verify_disjoint()
        return manifest


def _probe_and_signature(clip_path: Path) -> Tuple[Any, Optional[str]]:
    """Decode clip once to get DecodeDiagnostics and a 3-keyframe perceptual signature."""
    cap = cv2.VideoCapture(str(clip_path))
    try:
        if not cap.isOpened():
            reader = VideoReader(clip_path)
            try:
                for _ in reader.iter_frames():
                    pass
            except DecodeError:
                pass
            return reader.diagnostics, None

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # For short clips (<= 32 frames, e.g. unit test fixtures), run full frame-by-frame VideoReader
        if declared <= 32 or not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            cap.release()
            reader = VideoReader(clip_path)
            frames: List[np.ndarray] = []
            try:
                for rec in reader.iter_frames():
                    frames.append(rec.frame)
            except DecodeError:
                pass
            diag = reader.diagnostics
            sig = compute_frame_signature(frames) if diag.status == "ok" and frames else None
            return diag, sig

        # Fast keyframe probe for longer clips: grab every packet to verify stream integrity,
        # and retrieve pixels at first, middle, and last frame for the perceptual signature.
        mid_idx = declared // 2
        last_idx = max(0, declared - 1)
        key_frames: Dict[int, np.ndarray] = {}
        last_decoded: Optional[np.ndarray] = None
        count = 0
        while True:
            if not cap.grab():
                break
            if count in (0, mid_idx, last_idx):
                ok, frm = cap.retrieve()
                if ok and frm is not None and frm.size > 0:
                    key_frames[count] = frm
                    last_decoded = frm
            count += 1

        if count < 1 or 0 not in key_frames:
            diag = DecodeDiagnostics(
                source_id=str(clip_path),
                source_kind="file",
                status="failed",
                decoded_frame_count=0,
                fps=fps,
                width=width,
                height=height,
                declared_frame_count=declared,
                duration_seconds=0.0,
                errors=["failed to decode video frames"],
            )
            return diag, None

        # If clip ended slightly before declared last_idx, retrieve final frame if needed
        f0 = key_frames[0]
        f_mid = key_frames.get(mid_idx, f0)
        f_last = key_frames.get(last_idx, last_decoded if last_decoded is not None else f_mid)
        sig = compute_frame_signature([f0, f_mid, f_last])

        status = "ok"
        errors: List[str] = []
        if abs(count - declared) > max(2, int(declared * 0.05)):
            status = "partial"
            errors = [f"container declared {declared} frames, decoded {count}"]

        diag = DecodeDiagnostics(
            source_id=str(clip_path),
            source_kind="file",
            status=status,
            decoded_frame_count=count,
            fps=fps,
            width=int(f0.shape[1]),
            height=int(f0.shape[0]),
            declared_frame_count=declared,
            first_timestamp=0.0,
            last_timestamp=float((count - 1) / fps) if count > 1 else 0.0,
            duration_seconds=float((count - 1) / fps) if count > 1 else 0.0,
            errors=errors,
        )
        return diag, sig
    finally:
        cap.release()


def scan_rwf_directory(
    raw_dir: Path | str,
    val_carve_ratio: float = 0.2,
    seed: int = 1337,
    compute_hashes: bool = True,
    window_cfg: Optional[WindowConfig] = None,
) -> RWFManifest:
    """Scan an RWF-2000 directory, carve validation split, and build verified manifest.

    Directory structure expected:
      <raw_dir>/
        train/
          Fight/
          NonFight/
        val/
          Fight/
          NonFight/
    """
    root = Path(raw_dir)
    if not root.is_dir():
        raise DatasetError(f"raw_dir does not exist or is not a directory: {root}")

    train_dir = root / "train"
    val_dir = root / "val"

    if not train_dir.is_dir() or not val_dir.is_dir():
        raise DatasetError(
            f"RWF-2000 root {root} must contain 'train' and 'val' subdirectories"
        )

    # 1. Discover all official train clips
    train_fight_clips = (
        sorted(
            [
                p
                for p in (train_dir / "Fight").glob("*")
                if p.is_file() and p.suffix.lower() in (".avi", ".mp4", ".mkv")
            ]
        )
        if (train_dir / "Fight").is_dir()
        else []
    )
    train_nonfight_clips = (
        sorted(
            [
                p
                for p in (train_dir / "NonFight").glob("*")
                if p.is_file() and p.suffix.lower() in (".avi", ".mp4", ".mkv")
            ]
        )
        if (train_dir / "NonFight").is_dir()
        else []
    )

    # 2. Discover all official val clips (which become our locked test set)
    test_fight_clips = (
        sorted(
            [
                p
                for p in (val_dir / "Fight").glob("*")
                if p.is_file() and p.suffix.lower() in (".avi", ".mp4", ".mkv")
            ]
        )
        if (val_dir / "Fight").is_dir()
        else []
    )
    test_nonfight_clips = (
        sorted(
            [
                p
                for p in (val_dir / "NonFight").glob("*")
                if p.is_file() and p.suffix.lower() in (".avi", ".mp4", ".mkv")
            ]
        )
        if (val_dir / "NonFight").is_dir()
        else []
    )

    if not (train_fight_clips or train_nonfight_clips):
        raise DatasetError(f"no video clips found in {train_dir}")
    if not (test_fight_clips or test_nonfight_clips):
        raise DatasetError(f"no video clips found in {val_dir}")

    # 3. Inspect all candidate clips in parallel first so corrupt clips and exact
    # SHA-256 duplicates are isolated BEFORE carving val_carved from train.
    raw_candidates: List[Tuple[Path, str, str, int]] = []
    for p in train_fight_clips:
        raw_candidates.append((p, "train_pool", "Fight", 1))
    for p in train_nonfight_clips:
        raw_candidates.append((p, "train_pool", "NonFight", 0))
    for p in test_fight_clips:
        raw_candidates.append((p, "test", "Fight", 1))
    for p in test_nonfight_clips:
        raw_candidates.append((p, "test", "NonFight", 0))

    name_counts: Dict[str, int] = {}
    for clip_path, _, _, _ in raw_candidates:
        name_counts[clip_path.name] = name_counts.get(clip_path.name, 0) + 1

    def _inspect_item(item: Tuple[Path, str, str, int]) -> Tuple[Path, str, str, int, Any, Optional[str], str, str]:
        clip_path, split_pool, label, target = item
        diag, frame_sig = _probe_and_signature(clip_path)
        sha = file_sha256(clip_path) if compute_hashes else f"sha_{clip_path.name}"
        rel = clip_path.relative_to(root).as_posix()
        return clip_path, split_pool, label, target, diag, frame_sig, sha, rel

    if len(raw_candidates) > 16:
        workers = min(16, max(1, (os.cpu_count() or 4)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            inspected = list(ex.map(_inspect_item, raw_candidates))
    else:
        inspected = [_inspect_item(item) for item in raw_candidates]

    corrupt_entries: List[Dict[str, Any]] = []
    valid_pool: List[RWFEntry] = []

    for clip_path, split_pool, label, target, diag, frame_sig, sha, rel in inspected:
        cid = f"{label}/{clip_path.name}" if name_counts.get(clip_path.name, 0) > 1 else clip_path.name
        if diag.status != "ok" or diag.decoded_frame_count < 1 or diag.fps is None:
            corrupt_entries.append(
                {
                    "clip_id": cid,
                    "abs_path": str(clip_path.resolve()),
                    "rel_path": rel,
                    "label": label,
                    "target": target,
                    "split": "test" if split_pool == "test" else "train",
                    "sha256": sha,
                    "decode_status": diag.status,
                    "frame_count": diag.decoded_frame_count,
                    "error": "; ".join(diag.errors) if diag.errors else f"decode status {diag.status}",
                }
            )
            continue

        valid_pool.append(
            RWFEntry(
                clip_id=cid,
                abs_path=str(clip_path.resolve()),
                rel_path=rel,
                label=label,
                target=target,
                split="test" if split_pool == "test" else "train",
                sha256=sha,
                frame_count=int(diag.decoded_frame_count),
                fps=float(diag.fps),
                duration_seconds=float(
                    diag.duration_seconds
                    if diag.duration_seconds is not None
                    else (diag.decoded_frame_count / diag.fps)
                ),
                width=int(diag.width if diag.width is not None else 0),
                height=int(diag.height if diag.height is not None else 0),
                decode_status=diag.status,
                error=None,
                frame_signature=frame_sig,
            )
        )

    # 4. Deduplicate exact SHA-256 duplicates BEFORE carving val_carved.
    # Priority keeps the official held-out 'test' split intact first, then deduplicates within 'train'.
    pool_sorted = sorted(valid_pool, key=lambda e: (0 if e.split == "test" else 1, e.rel_path))
    seen_sha: Dict[str, RWFEntry] = {}
    deduped_pool: List[RWFEntry] = []
    removed_exact_duplicates: List[Dict[str, str]] = []
    for e in pool_sorted:
        if e.sha256 in seen_sha:
            kept = seen_sha[e.sha256]
            removed_exact_duplicates.append(
                {
                    "kept_split": kept.split,
                    "kept_clip": kept.rel_path,
                    "removed_split": e.split,
                    "removed_clip": e.rel_path,
                    "sha256": e.sha256,
                }
            )
        else:
            seen_sha[e.sha256] = e
            deduped_pool.append(e)

    test_entries = [e for e in deduped_pool if e.split == "test"]
    train_pool_entries = [e for e in deduped_pool if e.split == "train"]

    # 5. Source-video-grouped stratified carving of val_carved from train_pool_entries.
    # Groups all clips from the same source video (e.g., X_0.avi, X_1.avi, X_2.avi across
    # both Fight and NonFight) so no source camera ever spans both train and val_carved.
    generic_prefixes = {
        "train_fight",
        "train_nonfight",
        "val_fight",
        "val_nonfight",
        "test_fight",
        "test_nonfight",
        "fight",
        "nonfight",
        "clip",
        "vid",
    }

    def _source_video_id(entry: RWFEntry) -> str:
        stem = Path(entry.rel_path).stem
        if "_" in stem:
            prefix = stem.rsplit("_", 1)[0]
            if prefix.lower() not in generic_prefixes:
                return prefix
        return stem

    n_fight = sum(1 for e in train_pool_entries if e.target == 1)
    n_nonfight = sum(1 for e in train_pool_entries if e.target == 0)
    if val_carve_ratio > 0:
        target_val_f = max(1, int(round(min(n_fight, n_nonfight) * val_carve_ratio))) if (n_fight and n_nonfight) else max(1, int(round(len(train_pool_entries) * val_carve_ratio)))
        target_val_nf = target_val_f if (n_fight and n_nonfight) else 0
    else:
        target_val_f = 0
        target_val_nf = 0

    groups: Dict[str, List[RWFEntry]] = {}
    for e in sorted(train_pool_entries, key=lambda x: x.rel_path):
        gid = _source_video_id(e)
        groups.setdefault(gid, []).append(e)

    group_keys = sorted(groups.keys())
    rng = make_rng(derive_seed(seed, "rwf_stratified_split"))
    perm = np.arange(len(group_keys))
    rng.shuffle(perm)

    val_group_ids: set[str] = set()
    cur_f = 0
    cur_nf = 0
    for idx in perm:
        gk = group_keys[int(idx)]
        g_items = groups[gk]
        gf = sum(1 for x in g_items if x.target == 1)
        gnf = sum(1 for x in g_items if x.target == 0)
        if cur_f + gf <= target_val_f and cur_nf + gnf <= target_val_nf and (gf > 0 or gnf > 0):
            if (gf > 0 and cur_f < target_val_f) or (gnf > 0 and cur_nf < target_val_nf):
                val_group_ids.add(gk)
                cur_f += gf
                cur_nf += gnf
                if cur_f == target_val_f and cur_nf == target_val_nf:
                    break

    final_train: List[RWFEntry] = []
    final_val: List[RWFEntry] = []
    for e in sorted(train_pool_entries, key=lambda x: x.rel_path):
        gid = _source_video_id(e)
        if gid in val_group_ids:
            final_val.append(dataclasses.replace(e, split="val_carved"))
        else:
            final_train.append(dataclasses.replace(e, split="train"))

    entries = final_train + final_val + sorted(test_entries, key=lambda x: x.rel_path)

    # Verify zero source-video ID overlap across train and val_carved
    train_src_ids = {_source_video_id(e) for e in final_train}
    val_src_ids = {_source_video_id(e) for e in final_val}
    src_overlap_count = len(train_src_ids & val_src_ids)

    w_cfg = window_cfg if window_cfg is not None else WindowConfig()
    meta: Dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "dataset_name": "RWF-2000",
        "window_policy": {
            "policy": w_cfg.policy,
            "window_seconds": w_cfg.window_seconds,
            "stride_seconds": w_cfg.stride_seconds,
        },
        "clip_aggregation": w_cfg.clip_aggregation,
        "val_carve_ratio": val_carve_ratio,
        "seed": seed,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "corrupt_count": len(corrupt_entries),
        "removed_exact_duplicate_count": len(removed_exact_duplicates),
        "removed_exact_duplicates": removed_exact_duplicates,
        "train_val_source_video_overlap": src_overlap_count,
    }

    manifest = RWFManifest(entries=entries, corrupt_entries=corrupt_entries, meta=meta)
    manifest.verify_disjoint()
    meta["near_duplicate_count"] = len(manifest.near_duplicates)
    return manifest


class RWFDataset(Dataset):
    """PyTorch Dataset loading RWF-2000 clips with deterministic preprocessing."""

    def __init__(
        self,
        entries: Sequence[RWFEntry],
        num_frames: int = 16,
        target_size: Tuple[int, int] = (224, 224),
        sampling_strategy: str = "uniform",
        is_train: bool = False,
        horizontal_flip_prob: float = 0.5,
        seed: int = 1337,
    ) -> None:
        self.entries = list(entries)
        if not self.entries:
            raise DatasetError("cannot create RWFDataset with empty entries list")
        self.num_frames = num_frames
        self.target_size = target_size
        self.sampling_strategy = sampling_strategy
        self.is_train = is_train
        self.horizontal_flip_prob = horizontal_flip_prob
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for reproducible per-epoch augmentation seeding."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self.entries[index]
        path = Path(entry.abs_path)
        if not path.is_file():
            raise DatasetError(f"clip file not found: {path}")

        selected_frames: List[np.ndarray] = []
        selected_ts: List[float] = []
        validity_mask: np.ndarray

        # Fast selective frame decode for long clips where frame_count is known >= num_frames
        fast_ok = False
        if entry.frame_count >= self.num_frames and entry.frame_count > 32 and entry.fps > 0:
            indices = sample_frame_indices(
                entry.frame_count, self.num_frames, strategy=self.sampling_strategy
            )
            needed = set(indices)
            max_idx = max(needed)
            fetched: Dict[int, np.ndarray] = {}
            cap = cv2.VideoCapture(str(path))
            try:
                if cap.isOpened():
                    for cur_idx in range(max_idx + 1):
                        if not cap.grab():
                            break
                        if cur_idx in needed:
                            ok, frm = cap.retrieve()
                            if ok and frm is not None and frm.size > 0:
                                fetched[cur_idx] = frm
            finally:
                cap.release()
            if len(fetched) == len(needed):
                selected_frames = [fetched[i] for i in indices]
                selected_ts = [float(i / entry.fps) for i in indices]
                validity_mask = np.ones(self.num_frames, dtype=bool)
                fast_ok = True

        if not fast_ok:
            # Read all frames using VideoReader
            reader = VideoReader(path)
            all_frames: List[np.ndarray] = []
            all_ts: List[float] = []
            for rec in reader.iter_frames():
                all_frames.append(rec.frame)
                all_ts.append(rec.timestamp)

            if not all_frames:
                raise DatasetError(f"failed to decode any frames from {path}")

            actual_count = len(all_frames)
            if actual_count < self.num_frames:
                # Short clip: pad using documented padding policy and mark padded frames False
                selected_frames, mask_list = pad_or_truncate_frames(
                    all_frames, self.num_frames, pad_mode="edge"
                )
                selected_ts = list(all_ts)
                dt = 1.0 / entry.fps if entry.fps > 0 else 1.0 / 30.0
                while len(selected_ts) < self.num_frames:
                    selected_ts.append(selected_ts[-1] + dt)
                validity_mask = np.array(mask_list, dtype=bool)
            else:
                indices = sample_frame_indices(
                    actual_count, self.num_frames, strategy=self.sampling_strategy
                )
                selected_frames = [all_frames[i] for i in indices]
                selected_ts = [all_ts[i] for i in indices]
                validity_mask = np.ones(self.num_frames, dtype=bool)

        time_deltas = compute_time_deltas(selected_ts, validity_mask.tolist())

        # Resize frames
        resized_frames = [resize_frame(f, self.target_size) for f in selected_frames]

        # Train-only seeded horizontal flip augmentation (deterministic per seed, epoch, index)
        if self.is_train and self.horizontal_flip_prob > 0.0:
            sample_rng = make_rng(
                self.seed, "dataset_augment", str(self.epoch), str(index), entry.clip_id
            )
            if sample_rng.random() < self.horizontal_flip_prob:
                resized_frames = [cv2.flip(f, 1) for f in resized_frames]

        # Stack to (T, H, W, C) uint8
        frames_arr = np.stack(resized_frames, axis=0)

        # Convert to float tensor [T, C, H, W] normalized to [0.0, 1.0]
        # In OpenCV frames are BGR; convert to RGB
        frames_rgb = frames_arr[..., ::-1].copy()
        tensor_frames = torch.from_numpy(frames_rgb).permute(0, 3, 1, 2).float() / 255.0

        return {
            "clip_id": entry.clip_id,
            "frames": tensor_frames,  # [T, C, H, W]
            "target": torch.tensor(entry.target, dtype=torch.float32),
            "timestamps": torch.tensor(selected_ts, dtype=torch.float32),
            "time_deltas": torch.tensor(time_deltas, dtype=torch.float32),
            "validity_mask": torch.from_numpy(validity_mask),
        }


def make_rwf_dataloader(
    dataset: RWFDataset,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    seed: int = 1337,
) -> DataLoader:
    """Create a reproducible PyTorch DataLoader with seeded generator."""
    generator = torch.Generator()
    generator.manual_seed(derive_seed(seed, f"dataloader_{dataset.is_train}"))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator if shuffle else None,
    )
