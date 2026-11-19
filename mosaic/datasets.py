"""Dataset-independent clip discovery and manifest building.

A dataset is described by a small YAML file (see ``configs/datasets/``); nothing else in the pipeline needs to know
which dataset it is working on.

    name: rwf2000_30fps
    root: data/RWF-2000_30fps/RWF-2000        # relative to the repository root, or absolute
    layout: split_class                       # split_class: root/<split>/<class>/clip.ext
                                              # class_only:  root/<class>/clip.ext  (a seeded split is made)
    split_dirs: {train: [train], test: [val]} # split_class only: folder names for each role
    classes: {positive: [Fight], negative: [NonFight]}
    group: suffix_number                      # suffix_number: <source>_<n>.ext shares <source>; none: every clip alone
    test_fraction: 0.2                        # class_only only
    seed: 42
    extensions: [.avi, .mp4]

Manifest entries carry ``split`` ("train" or "test"), ``target`` (1 = positive class), ``group`` (source-video key used to keep
clips of one video on the same side of every split) and file facts (frame count, fps, size, sha256).
Exact duplicates (same sha256) are dropped, keeping the test copy so the test set stays intact.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
_SUFFIX = re.compile(r"_\d+$")


class DatasetError(ValueError):
    """Raised for an invalid dataset config or a folder that does not match it."""


@dataclass
class DatasetConfig:
    name: str
    root: Path
    layout: str = "split_class"
    positive: List[str] = field(default_factory=list)
    negative: List[str] = field(default_factory=list)
    split_dirs: Dict[str, List[str]] = field(default_factory=lambda: {"train": ["train"], "test": ["val", "test"]})
    group: str = "suffix_number"
    test_fraction: float = 0.2
    seed: int = 42
    extensions: List[str] = field(default_factory=lambda: [".avi", ".mp4", ".mov", ".mkv"])

    @property
    def manifest_path(self) -> Path:
        return ROOT / "artifacts" / "manifests" / f"{self.name}_manifest.json"

    @property
    def split_path(self) -> Path:
        return ROOT / "artifacts" / "manifests" / f"{self.name}_v3_split.json"


def load_dataset_config(path: Path | str) -> DatasetConfig:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.is_file():
        raise DatasetError(f"dataset config not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    unknown = set(raw) - {"name", "root", "layout", "split_dirs", "classes", "group", "test_fraction", "seed", "extensions"}
    if unknown:
        raise DatasetError(f"{p.name}: unknown keys {sorted(unknown)}")
    for key in ("name", "root", "classes"):
        if key not in raw:
            raise DatasetError(f"{p.name}: missing required key {key!r}")
    classes = raw["classes"] or {}
    if not classes.get("positive") or not classes.get("negative"):
        raise DatasetError(f"{p.name}: classes needs non-empty 'positive' and 'negative' lists")
    layout = raw.get("layout", "split_class")
    if layout not in ("split_class", "class_only"):
        raise DatasetError(f"{p.name}: layout must be split_class or class_only, got {layout!r}")
    group = raw.get("group", "suffix_number")
    if group not in ("suffix_number", "none"):
        raise DatasetError(f"{p.name}: group must be suffix_number or none, got {group!r}")
    root = Path(raw["root"])
    kwargs: Dict[str, Any] = {}
    if "split_dirs" in raw:
        kwargs["split_dirs"] = {k: list(v) for k, v in raw["split_dirs"].items()}
    if "extensions" in raw:
        kwargs["extensions"] = [e.lower() for e in raw["extensions"]]
    return DatasetConfig(
        name=str(raw["name"]), root=root if root.is_absolute() else ROOT / root, layout=layout,
        positive=list(classes["positive"]), negative=list(classes["negative"]), group=group,
        test_fraction=float(raw.get("test_fraction", 0.2)), seed=int(raw.get("seed", 42)), **kwargs)


def group_key(rel_path: str, mode: str) -> str:
    """Source-video key of a clip."""
    stem = rel_path.replace("\\", "/").split("/")[-1].rsplit(".", 1)[0]
    return _SUFFIX.sub("", stem) if mode == "suffix_number" else rel_path


def class_target(cfg: DatasetConfig, class_name: str) -> int:
    low = class_name.lower()
    if low in {c.lower() for c in cfg.positive}:
        return 1
    if low in {c.lower() for c in cfg.negative}:
        return 0
    raise DatasetError(f"folder {class_name!r} is in neither the positive {cfg.positive} nor the negative {cfg.negative} class list")


def discover_clips(cfg: DatasetConfig) -> List[Dict[str, Any]]:
    """Clips on disk as dicts with path, rel_path, class name and (split_class only) split role, in a deterministic order."""
    if not cfg.root.is_dir():
        raise DatasetError(f"dataset root not found: {cfg.root}")
    role_of = {d.lower(): role for role, dirs in cfg.split_dirs.items() for d in dirs}
    found: List[Dict[str, Any]] = []
    for p in sorted(cfg.root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in cfg.extensions:
            continue
        parts = p.relative_to(cfg.root).parts
        if cfg.layout == "split_class":
            if len(parts) != 3:
                raise DatasetError(f"expected <split>/<class>/<clip>, got {'/'.join(parts)}")
            if parts[0].lower() not in role_of:
                raise DatasetError(f"split folder {parts[0]!r} is not listed in split_dirs {cfg.split_dirs}")
            role, cls = role_of[parts[0].lower()], parts[1]
        else:
            if len(parts) != 2:
                raise DatasetError(f"expected <class>/<clip>, got {'/'.join(parts)}")
            role, cls = None, parts[0]
        found.append({"path": p, "rel_path": "/".join(parts), "class": cls, "role": role, "target": class_target(cfg, cls)})
    if not found:
        raise DatasetError(f"no clips with extensions {cfg.extensions} under {cfg.root}")
    return found


def file_sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def probe_clip(path: Path) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened()
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if ok else 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if ok else 0.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first = ok and cap.read()[0]
    cap.release()
    good = bool(first and n > 0 and fps > 0)
    return {"frame_count": n, "fps": fps, "duration_seconds": round(n / fps, 2) if fps else 0.0, "width": w, "height": h,
            "decode_status": "ok" if good else "failed", "error": None if good else "could not decode the first frame or read fps/frame count"}


def build_manifest(cfg: DatasetConfig, progress=None) -> Dict[str, Any]:
    """Manifest dict for a dataset (does not write it)."""
    from .splits import build_grouped_split

    clips = discover_clips(cfg)
    # test copies first, so exact duplicates are removed from the training side
    clips.sort(key=lambda c: (c["role"] != "test", c["rel_path"]))
    entries, corrupt = [], []
    for i, c in enumerate(clips):
        info = probe_clip(c["path"])
        e = {"clip_id": f"{c['class']}/{c['path'].name}", "abs_path": str(c["path"]), "rel_path": c["rel_path"], "label": c["class"],
             "target": c["target"], "split": c["role"] or "train", "sha256": file_sha256(c["path"]), **info,
             "group": group_key(c["rel_path"], cfg.group)}
        (entries if e["decode_status"] == "ok" else corrupt).append(e)
        if progress and (i + 1) % 250 == 0:
            progress(i + 1, len(clips))
    seen: Dict[str, Dict[str, Any]] = {}
    kept, removed = [], []
    for e in entries:
        if e["sha256"] in seen:
            k = seen[e["sha256"]]
            removed.append({"kept_split": k["split"], "kept_clip": k["rel_path"], "removed_split": e["split"],
                            "removed_clip": e["rel_path"], "sha256": e["sha256"]})
        else:
            seen[e["sha256"]] = e
            kept.append(e)
    if cfg.layout == "class_only":      # no official split: seeded, stratified, grouped by source video
        part = build_grouped_split(kept, holdout_fraction=cfg.test_fraction, seed=cfg.seed)
        for e in kept:
            e["split"] = "test" if part[e["rel_path"]] == "internal_val" else "train"
    counts: Dict[str, int] = {}
    for e in kept:
        key = f"{e['split']}/{'positive' if e['target'] else 'negative'}"
        counts[key] = counts.get(key, 0) + 1
    meta = {"schema_version": "manifest-0.2", "dataset_name": cfg.name, "created_utc": datetime.now(timezone.utc).isoformat(),
            "layout": cfg.layout, "group": cfg.group, "corrupt_count": len(corrupt),
            "removed_exact_duplicate_count": len(removed), "removed_exact_duplicates": removed, "counts": counts,
            "fps_histogram": _hist(round(e["fps"]) for e in kept), "frame_count_histogram": _hist(e["frame_count"] for e in kept)}
    return {"meta": meta, "entries": kept, "corrupt_entries": corrupt, "near_duplicates": []}


def _hist(values, top: int = 8) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1])[:top])
