"""The dataset manifest: a checked, machine-readable list of video clips.

A manifest records, for every clip: where it is, its label, and what we learned
by actually decoding it (frame count, FPS, resolution, duration, decode status).
Building it up-front means corrupt or empty clips are found immediately and
listed explicitly - not discovered halfway through a training run.

The manifest also stores *metadata* about the experiment, including the window
policy (see plan section 2.2), so the settings used to prepare the data are
never separated from the data.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .config import Config, DecodeConfig, config_hash
from .contracts import (
    DECODE_STATUSES,
    ContractError,
    require_choice,
    require_finite,
    require_int,
    require_text,
)
from .ingestion import probe_video

MANIFEST_SCHEMA = "manifest-0.1"
CLASS_LABELS = ("violent", "non_violent")
SPLITS = ("train", "val", "test")
REQUIRED_META = ("schema_version", "dataset_name", "window_policy", "clip_aggregation")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    """Raised when a manifest or one of its entries is invalid."""


@dataclass(frozen=True)
class ManifestEntry:
    clip_path: str
    label: str
    decode_status: str
    frame_count: int
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    duration_seconds: Optional[float] = None
    split: Optional[str] = None
    error: Optional[str] = None
    sha256: Optional[str] = None

    def __post_init__(self) -> None:
        try:
            require_text(self.clip_path, "clip_path")
            require_choice(self.label, CLASS_LABELS, "label")
            if self.split is not None:
                require_choice(self.split, SPLITS, "split")
            require_choice(self.decode_status, DECODE_STATUSES, "decode_status")
            require_int(self.frame_count, "frame_count")
            if self.sha256 is not None and not (
                isinstance(self.sha256, str) and _SHA256_RE.match(self.sha256)
            ):
                raise ContractError("sha256 must be 64 lowercase hex characters")
            if self.decode_status == "ok":
                if self.frame_count < 1:
                    raise ContractError("an 'ok' entry needs frame_count >= 1")
                if self.fps is None or not require_finite(self.fps, "fps") > 0:
                    raise ContractError("an 'ok' entry needs fps > 0")
                for name in ("width", "height"):
                    value = getattr(self, name)
                    if value is None:
                        raise ContractError(f"an 'ok' entry needs {name}")
                    require_int(value, name, minimum=1)
                if self.duration_seconds is None or not require_finite(self.duration_seconds, "duration_seconds") > 0:
                    raise ContractError("an 'ok' entry needs duration_seconds > 0")
                if self.error is not None:
                    raise ContractError("an 'ok' entry must not carry an error message")
            else:
                if not self.error or not str(self.error).strip():
                    raise ContractError(
                        f"a '{self.decode_status}' entry must explain itself in 'error'"
                    )
        except ContractError as exc:
            raise ManifestError(f"{exc} (clip: {self.clip_path!r})") from exc

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_path": self.clip_path,
            "label": self.label,
            "split": self.split,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.duration_seconds,
            "decode_status": self.decode_status,
            "error": self.error,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ManifestEntry":
        if not isinstance(data, Mapping):
            raise ManifestError(f"entry must be a mapping, got {type(data).__name__}")
        allowed = {
            "clip_path", "label", "split", "frame_count", "fps", "width", "height",
            "duration_seconds", "decode_status", "error", "sha256",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ManifestError(f"entry has unknown field(s) {unknown}")
        required = ("clip_path", "label", "frame_count", "decode_status")
        missing = [key for key in required if key not in data]
        if missing:
            raise ManifestError(f"entry is missing required field(s) {missing}")
        return cls(**{key: data[key] for key in data})


def file_sha256(path: Path | str, chunk_size: int = 1 << 20) -> str:
    """SHA-256 fingerprint of a file (used later to detect duplicate clips)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def make_meta(cfg: Config, dataset_name: str, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Manifest metadata, including the window policy chosen in the config."""
    meta: Dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "dataset_name": dataset_name,
        "window_policy": {
            "policy": cfg.window.policy,
            "window_seconds": cfg.window.window_seconds,
            "stride_seconds": cfg.window.stride_seconds,
        },
        "clip_aggregation": cfg.window.clip_aggregation,
        "config_hash": config_hash(cfg),
        "seed": cfg.experiment.seed,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        meta.update(extra)
    return meta


@dataclass
class Manifest:
    meta: Dict[str, Any]
    entries: List[ManifestEntry]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        missing = [key for key in REQUIRED_META if key not in self.meta]
        if missing:
            raise ManifestError(f"manifest metadata is missing required key(s) {missing}")
        if self.meta["schema_version"] != MANIFEST_SCHEMA:
            raise ManifestError(
                f"unsupported manifest schema {self.meta['schema_version']!r}; expected {MANIFEST_SCHEMA!r}"
            )
        seen: Dict[str, int] = {}
        for index, entry in enumerate(self.entries):
            if entry.clip_path in seen:
                raise ManifestError(
                    f"duplicate clip_path {entry.clip_path!r} (entries {seen[entry.clip_path]} and {index})"
                )
            seen[entry.clip_path] = index

    def summary(self) -> Dict[str, Any]:
        by_status: Dict[str, int] = {}
        by_label: Dict[str, int] = {}
        for entry in self.entries:
            by_status[entry.decode_status] = by_status.get(entry.decode_status, 0) + 1
            by_label[entry.label] = by_label.get(entry.label, 0) + 1
        return {"total": len(self.entries), "by_status": by_status, "by_label": by_label}

    def failed_entries(self) -> List[ManifestEntry]:
        return [e for e in self.entries if e.decode_status != "ok"]

    def to_dict(self) -> Dict[str, Any]:
        return {"meta": dict(self.meta), "entries": [e.to_dict() for e in self.entries]}

    def write(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Manifest":
        if not isinstance(data, Mapping) or "meta" not in data or "entries" not in data:
            raise ManifestError("manifest must be a mapping with 'meta' and 'entries'")
        if not isinstance(data["entries"], list):
            raise ManifestError("manifest 'entries' must be a list")
        problems: List[str] = []
        entries: List[ManifestEntry] = []
        for index, raw in enumerate(data["entries"]):
            try:
                entries.append(ManifestEntry.from_dict(raw))
            except ManifestError as exc:
                problems.append(f"entries[{index}]: {exc}")
        if problems:
            raise ManifestError("invalid manifest entries:\n" + "\n".join(problems))
        return cls(meta=dict(data["meta"]), entries=entries)

    @classmethod
    def read(cls, path: Path | str) -> "Manifest":
        file = Path(path)
        if not file.is_file():
            raise ManifestError(f"manifest file not found: {file}")
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ManifestError(f"manifest file {file} is not valid JSON: {exc}") from exc
        return cls.from_dict(data)


def build_entry(
    clip_path: str,
    label: str,
    decode: Optional[DecodeConfig] = None,
    split: Optional[str] = None,
    compute_hash: bool = False,
    root: Optional[Path | str] = None,
) -> ManifestEntry:
    """Decode one clip and describe it. Unreadable clips become 'failed' entries."""
    location = Path(root) / clip_path if root is not None else Path(clip_path)
    diagnostics = probe_video(location, decode=decode, source_id=Path(clip_path).name)
    digest = file_sha256(location) if (compute_hash and location.is_file()) else None
    if diagnostics.status == "ok":
        return ManifestEntry(
            clip_path=Path(clip_path).as_posix(),
            label=label,
            split=split,
            frame_count=diagnostics.decoded_frame_count,
            fps=diagnostics.fps,
            width=diagnostics.width,
            height=diagnostics.height,
            duration_seconds=diagnostics.duration_seconds,
            decode_status="ok",
            sha256=digest,
        )
    return ManifestEntry(
        clip_path=Path(clip_path).as_posix(),
        label=label,
        split=split,
        frame_count=diagnostics.decoded_frame_count,
        fps=diagnostics.fps,
        width=diagnostics.width,
        height=diagnostics.height,
        duration_seconds=diagnostics.duration_seconds,
        decode_status=diagnostics.status,
        error="; ".join(diagnostics.errors),
        sha256=digest,
    )


def build_manifest(
    clips: Sequence[Tuple[str, str]],
    meta: Mapping[str, Any],
    decode: Optional[DecodeConfig] = None,
    root: Optional[Path | str] = None,
    compute_hash: bool = False,
) -> Manifest:
    """Build a manifest from ``(clip_path, label)`` pairs."""
    entries = [
        build_entry(path, label, decode=decode, compute_hash=compute_hash, root=root)
        for path, label in clips
    ]
    return Manifest(meta=dict(meta), entries=entries)
