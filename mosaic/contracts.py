"""Typed data contracts for the MOSAIC-FL bounded pipeline.

A *contract* is an agreement about what a piece of data must look like when it
moves from one module to the next (for example from the tracker to the temporal
model). Every contract here is a small, immutable (frozen) Python dataclass that
validates itself the moment it is created. Malformed data therefore fails
loudly at the module boundary instead of silently corrupting later stages.

Design rules (taken from the implementation plan):

* Boxes must be finite, non-negative and have positive area.
* Scores and confidences must be finite numbers inside [0, 1].
* Unavailable information is represented by ``None`` (explicitly missing),
  never by a "success-shaped" default such as ``0.0`` or an empty mask.
* A Kalman-predicted (not actually seen) observation must say
  ``observed = False`` and must NOT carry a detector confidence.
* ``track_id`` is a local temporal label only - it is not a person identity.
"""

from __future__ import annotations

import json
import math
import numbers
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, NoReturn, Optional, Sequence

SCHEMA_VERSION = "window-output-0.1"
LABELS = ("violent", "non_violent")
DECODE_STATUSES = ("ok", "partial", "empty", "failed")
SOURCE_KINDS = ("file", "synthetic", "array")


class ContractError(ValueError):
    """Raised when data violates a contract."""


def _fail(message: str) -> NoReturn:
    raise ContractError(message)


# --------------------------------------------------------------------------
# Small reusable validators
# --------------------------------------------------------------------------
def require_finite(value: Any, name: str) -> float:
    """Return ``value`` as a float, or raise if it is not a finite number."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        _fail(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        _fail(f"{name} must be finite, got {value!r}")
    return number


def require_non_negative(value: Any, name: str) -> float:
    number = require_finite(value, name)
    if number < 0.0:
        _fail(f"{name} must be >= 0, got {value!r}")
    return number


def require_unit_interval(value: Any, name: str) -> float:
    """Scores / confidences / fractions must be finite and within [0, 1]."""
    number = require_finite(value, name)
    if not 0.0 <= number <= 1.0:
        _fail(f"{name} must be within [0, 1], got {value!r}")
    return number


def require_int(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        _fail(f"{name} must be an integer, got {value!r}")
    number = int(value)
    if number < minimum:
        _fail(f"{name} must be >= {minimum}, got {value!r}")
    return number


def require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{name} must be a bool, got {value!r}")
    return value


def require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{name} must be a non-empty string, got {value!r}")
    return value


def require_vector(value: Any, length: int, name: str) -> List[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        _fail(f"{name} must be a list of {length} numbers, got {value!r}")
    if len(value) != length:
        _fail(f"{name} must have exactly {length} numbers, got {len(value)}")
    return [require_finite(item, f"{name}[{i}]") for i, item in enumerate(value)]


def require_float_list(value: Any, name: str) -> List[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        _fail(f"{name} must be a list of numbers, got {value!r}")
    if len(value) == 0:
        _fail(f"{name} must not be empty when provided")
    return [require_finite(item, f"{name}[{i}]") for i, item in enumerate(value)]


def require_choice(value: Any, choices: Sequence[str], name: str) -> str:
    if value not in choices:
        _fail(f"{name} must be one of {list(choices)}, got {value!r}")
    return value


def _get(data: Mapping[str, Any], key: str, where: str) -> Any:
    if not isinstance(data, Mapping):
        _fail(f"{where} must be a mapping, got {type(data).__name__}")
    if key not in data:
        _fail(f"{where} is missing required key '{key}'")
    return data[key]


def _opt(data: Mapping[str, Any], key: str) -> Any:
    return data.get(key) if isinstance(data, Mapping) else None


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Box:
    """Axis-aligned box in pixel coordinates: ``[x1, y1, x2, y2]``.

    ``(x1, y1)`` is the top-left corner and ``(x2, y2)`` the bottom-right
    corner. The box must have positive width and height and be non-negative.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        for name in ("x1", "y1", "x2", "y2"):
            require_non_negative(getattr(self, name), f"box.{name}")
        if not self.x2 > self.x1:
            _fail(f"box must have x2 > x1, got x1={self.x1}, x2={self.x2}")
        if not self.y2 > self.y1:
            _fail(f"box must have y2 > y1, got y1={self.y1}, y2={self.y2}")

    @property
    def width(self) -> float:
        return float(self.x2 - self.x1)

    @property
    def height(self) -> float:
        return float(self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centroid(self) -> List[float]:
        return [(self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0]

    def check_within(self, frame_width: int, frame_height: int) -> None:
        """Raise if any part of the box lies outside the frame."""
        if self.x2 > frame_width or self.y2 > frame_height:
            _fail(
                f"box [{self.x1}, {self.y1}, {self.x2}, {self.y2}] exceeds the "
                f"{frame_width}x{frame_height} frame"
            )

    def to_list(self) -> List[float]:
        return [float(self.x1), float(self.y1), float(self.x2), float(self.y2)]

    @classmethod
    def from_list(cls, values: Any) -> "Box":
        x1, y1, x2, y2 = require_vector(values, 4, "box")
        return cls(x1, y1, x2, y2)


# --------------------------------------------------------------------------
# Ingestion / decoding
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DecodeDiagnostics:
    """What happened while a video source was being read.

    ``status`` is one of: ``ok``, ``partial`` (some frames decoded, but
    something went wrong), ``empty`` (no frames) or ``failed`` (could not be
    opened). Anything other than ``ok`` must carry at least one error message.
    """

    source_id: str
    source_kind: str
    status: str
    decoded_frame_count: int
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    declared_frame_count: Optional[int] = None
    first_timestamp: Optional[float] = None
    last_timestamp: Optional[float] = None
    duration_seconds: Optional[float] = None
    timestamps_monotonic: bool = True
    truncated: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        require_text(self.source_id, "decode.source_id")
        require_choice(self.source_kind, SOURCE_KINDS, "decode.source_kind")
        require_choice(self.status, DECODE_STATUSES, "decode.status")
        require_int(self.decoded_frame_count, "decode.decoded_frame_count")
        require_bool(self.timestamps_monotonic, "decode.timestamps_monotonic")
        require_bool(self.truncated, "decode.truncated")
        if self.status in ("ok", "partial"):
            if self.decoded_frame_count < 1:
                _fail(f"decode status '{self.status}' requires at least one decoded frame")
            for name in ("fps", "width", "height"):
                if getattr(self, name) is None:
                    _fail(f"decode status '{self.status}' requires '{name}' to be known")
        if self.fps is not None and not require_finite(self.fps, "decode.fps") > 0:
            _fail(f"decode.fps must be > 0, got {self.fps!r}")
        for name in ("width", "height"):
            value = getattr(self, name)
            if value is not None:
                require_int(value, f"decode.{name}", minimum=1)
        if self.declared_frame_count is not None:
            require_int(self.declared_frame_count, "decode.declared_frame_count")
        if self.status == "ok" and self.errors:
            _fail("decode status 'ok' must not carry error messages")
        if self.status != "ok" and not self.errors:
            _fail(f"decode status '{self.status}' must explain itself with an error message")
        if self.status == "empty" and self.decoded_frame_count != 0:
            _fail("decode status 'empty' requires decoded_frame_count == 0")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_kind": self.source_kind,
            "status": self.status,
            "decoded_frame_count": self.decoded_frame_count,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "declared_frame_count": self.declared_frame_count,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "duration_seconds": self.duration_seconds,
            "timestamps_monotonic": self.timestamps_monotonic,
            "truncated": self.truncated,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DecodeDiagnostics":
        where = "DecodeDiagnostics"
        return cls(
            source_id=_get(data, "source_id", where),
            source_kind=_get(data, "source_kind", where),
            status=_get(data, "status", where),
            decoded_frame_count=_get(data, "decoded_frame_count", where),
            fps=_opt(data, "fps"),
            width=_opt(data, "width"),
            height=_opt(data, "height"),
            declared_frame_count=_opt(data, "declared_frame_count"),
            first_timestamp=_opt(data, "first_timestamp"),
            last_timestamp=_opt(data, "last_timestamp"),
            duration_seconds=_opt(data, "duration_seconds"),
            timestamps_monotonic=data.get("timestamps_monotonic", True),
            truncated=data.get("truncated", False),
            errors=list(data.get("errors", [])),
            warnings=list(data.get("warnings", [])),
        )


# --------------------------------------------------------------------------
# Detection / evidence
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Detection:
    """One detected object (normally a person) in one frame."""

    box: Box
    confidence: float
    class_name: str
    frame_index: int
    timestamp: float
    pooled_feature: Optional[List[float]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.box, Box):
            _fail("detection.box must be a Box")
        require_unit_interval(self.confidence, "detection.confidence")
        require_text(self.class_name, "detection.class_name")
        require_int(self.frame_index, "detection.frame_index")
        require_non_negative(self.timestamp, "detection.timestamp")
        if self.pooled_feature is not None:
            require_float_list(self.pooled_feature, "detection.pooled_feature")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "box": self.box.to_list(),
            "confidence": float(self.confidence),
            "class_name": self.class_name,
            "frame_index": self.frame_index,
            "timestamp": float(self.timestamp),
            "pooled_feature": None if self.pooled_feature is None else list(self.pooled_feature),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Detection":
        where = "Detection"
        return cls(
            box=Box.from_list(_get(data, "box", where)),
            confidence=_get(data, "confidence", where),
            class_name=_get(data, "class_name", where),
            frame_index=_get(data, "frame_index", where),
            timestamp=_get(data, "timestamp", where),
            pooled_feature=_opt(data, "pooled_feature"),
        )


@dataclass(frozen=True)
class ClarityReport:
    """Input-quality report. A component is ``None`` when it is unavailable."""

    score: float
    source_profile: str
    resolution_component: Optional[float] = None
    sharpness_component: Optional[float] = None
    exposure_component: Optional[float] = None
    compression_component: Optional[float] = None
    person_scale_component: Optional[float] = None

    _COMPONENTS = (
        "resolution_component",
        "sharpness_component",
        "exposure_component",
        "compression_component",
        "person_scale_component",
    )

    def __post_init__(self) -> None:
        require_unit_interval(self.score, "clarity.score")
        require_text(self.source_profile, "clarity.source_profile")
        for name in self._COMPONENTS:
            value = getattr(self, name)
            if value is not None:
                require_unit_interval(value, f"clarity.{name}")

    @property
    def unavailable_components(self) -> List[str]:
        return [name for name in self._COMPONENTS if getattr(self, name) is None]

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"score": float(self.score), "source_profile": self.source_profile}
        for name in self._COMPONENTS:
            out[name] = getattr(self, name)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClarityReport":
        where = "ClarityReport"
        kwargs = {name: _opt(data, name) for name in cls._COMPONENTS}
        return cls(
            score=_get(data, "score", where),
            source_profile=_get(data, "source_profile", where),
            **kwargs,
        )


@dataclass(frozen=True)
class MotionGateReport:
    """Report on motion gating decisions across a processed sequence."""

    total_frames: int
    selected_frames: int
    skipped_frames: int
    reduction_fraction: float
    mean_motion_score: float
    burst_count: int
    keep_alive_count: int

    def __post_init__(self) -> None:
        require_int(self.total_frames, "motion_gate.total_frames")
        require_int(self.selected_frames, "motion_gate.selected_frames")
        require_int(self.skipped_frames, "motion_gate.skipped_frames")
        require_unit_interval(self.reduction_fraction, "motion_gate.reduction_fraction")
        require_non_negative(self.mean_motion_score, "motion_gate.mean_motion_score")
        require_int(self.burst_count, "motion_gate.burst_count")
        require_int(self.keep_alive_count, "motion_gate.keep_alive_count")
        if self.selected_frames + self.skipped_frames != self.total_frames:
            _fail(
                f"motion_gate: selected ({self.selected_frames}) + skipped ({self.skipped_frames}) "
                f"!= total ({self.total_frames})"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_frames": self.total_frames,
            "selected_frames": self.selected_frames,
            "skipped_frames": self.skipped_frames,
            "reduction_fraction": float(self.reduction_fraction),
            "mean_motion_score": float(self.mean_motion_score),
            "burst_count": self.burst_count,
            "keep_alive_count": self.keep_alive_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MotionGateReport":
        where = "MotionGateReport"
        return cls(
            total_frames=_get(data, "total_frames", where),
            selected_frames=_get(data, "selected_frames", where),
            skipped_frames=_get(data, "skipped_frames", where),
            reduction_fraction=_get(data, "reduction_fraction", where),
            mean_motion_score=_get(data, "mean_motion_score", where),
            burst_count=_get(data, "burst_count", where),
            keep_alive_count=_get(data, "keep_alive_count", where),
        )



@dataclass(frozen=True)
class ObjectEvidence:
    """Auxiliary object evidence. It never sets the violence label by itself."""

    object_id: int
    category: str
    box: Box
    confidence: float
    frame_index: int
    associated_track_id: Optional[int] = None

    def __post_init__(self) -> None:
        require_int(self.object_id, "object.object_id")
        require_text(self.category, "object.category")
        if not isinstance(self.box, Box):
            _fail("object.box must be a Box")
        require_unit_interval(self.confidence, "object.confidence")
        require_int(self.frame_index, "object.frame_index")
        if self.associated_track_id is not None:
            require_int(self.associated_track_id, "object.associated_track_id")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "category": self.category,
            "box": self.box.to_list(),
            "confidence": float(self.confidence),
            "frame_index": self.frame_index,
            "associated_track_id": self.associated_track_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObjectEvidence":
        where = "ObjectEvidence"
        return cls(
            object_id=_get(data, "object_id", where),
            category=_get(data, "category", where),
            box=Box.from_list(_get(data, "box", where)),
            confidence=_get(data, "confidence", where),
            frame_index=_get(data, "frame_index", where),
            associated_track_id=_opt(data, "associated_track_id"),
        )


@dataclass(frozen=True)
class MaskEvidence:
    """Silhouette evidence. ``mask_ref`` points to the mask rather than embedding it."""

    mask_ref: str
    confidence: float
    visible_fraction: float
    frame_index: int
    contour_features: Optional[List[float]] = None

    def __post_init__(self) -> None:
        require_text(self.mask_ref, "mask.mask_ref")
        require_unit_interval(self.confidence, "mask.confidence")
        require_unit_interval(self.visible_fraction, "mask.visible_fraction")
        require_int(self.frame_index, "mask.frame_index")
        if self.contour_features is not None:
            require_float_list(self.contour_features, "mask.contour_features")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mask_ref": self.mask_ref,
            "confidence": float(self.confidence),
            "visible_fraction": float(self.visible_fraction),
            "frame_index": self.frame_index,
            "contour_features": None if self.contour_features is None else list(self.contour_features),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MaskEvidence":
        where = "MaskEvidence"
        return cls(
            mask_ref=_get(data, "mask_ref", where),
            confidence=_get(data, "confidence", where),
            visible_fraction=_get(data, "visible_fraction", where),
            frame_index=_get(data, "frame_index", where),
            contour_features=_opt(data, "contour_features"),
        )


# --------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TrackToken:
    """One person-observation on one local track at one time.

    This is the hub interface between tracking and temporal modelling.
    ``detector_confidence`` must be ``None`` for predicted-only tokens
    (``observed = False``) and a real number for observed tokens.
    """

    track_id: int
    timestamp: float
    frame_index: int
    box: Box
    observed: bool
    detector_confidence: Optional[float]
    visibility_score: float
    mask_available: bool
    centroid: List[float]
    velocity: List[float]
    acceleration: List[float]
    clarity_score: Optional[float] = None
    silhouette_features: Optional[List[float]] = None
    object_evidence: Optional[List[ObjectEvidence]] = None

    def __post_init__(self) -> None:
        require_int(self.track_id, "token.track_id")
        require_non_negative(self.timestamp, "token.timestamp")
        require_int(self.frame_index, "token.frame_index")
        if not isinstance(self.box, Box):
            _fail("token.box must be a Box")
        require_bool(self.observed, "token.observed")
        if self.observed:
            if self.detector_confidence is None:
                _fail("an observed token must carry a detector_confidence")
            require_unit_interval(self.detector_confidence, "token.detector_confidence")
        elif self.detector_confidence is not None:
            _fail("a predicted-only token (observed=False) must not carry a detector_confidence")
        require_unit_interval(self.visibility_score, "token.visibility_score")
        require_bool(self.mask_available, "token.mask_available")
        require_vector(self.centroid, 2, "token.centroid")
        require_vector(self.velocity, 2, "token.velocity")
        require_vector(self.acceleration, 2, "token.acceleration")
        if self.clarity_score is not None:
            require_unit_interval(self.clarity_score, "token.clarity_score")
        if self.silhouette_features is not None:
            require_float_list(self.silhouette_features, "token.silhouette_features")
            if not self.mask_available:
                _fail("silhouette_features were given but mask_available is False")
        if self.object_evidence is not None:
            for item in self.object_evidence:
                if not isinstance(item, ObjectEvidence):
                    _fail("token.object_evidence must contain ObjectEvidence items")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "timestamp": float(self.timestamp),
            "frame_index": self.frame_index,
            "box": self.box.to_list(),
            "observed": self.observed,
            "detector_confidence": self.detector_confidence,
            "visibility_score": float(self.visibility_score),
            "mask_available": self.mask_available,
            "centroid": list(self.centroid),
            "velocity": list(self.velocity),
            "acceleration": list(self.acceleration),
            "clarity_score": self.clarity_score,
            "silhouette_features": None
            if self.silhouette_features is None
            else list(self.silhouette_features),
            "object_evidence": None
            if self.object_evidence is None
            else [item.to_dict() for item in self.object_evidence],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrackToken":
        where = "TrackToken"
        evidence = _opt(data, "object_evidence")
        return cls(
            track_id=_get(data, "track_id", where),
            timestamp=_get(data, "timestamp", where),
            frame_index=_get(data, "frame_index", where),
            box=Box.from_list(_get(data, "box", where)),
            observed=_get(data, "observed", where),
            detector_confidence=_get(data, "detector_confidence", where),
            visibility_score=_get(data, "visibility_score", where),
            mask_available=_get(data, "mask_available", where),
            centroid=_get(data, "centroid", where),
            velocity=_get(data, "velocity", where),
            acceleration=_get(data, "acceleration", where),
            clarity_score=_opt(data, "clarity_score"),
            silhouette_features=_opt(data, "silhouette_features"),
            object_evidence=None
            if evidence is None
            else [ObjectEvidence.from_dict(item) for item in evidence],
        )


@dataclass(frozen=True)
class TrackSummary:
    """Compact per-track summary stored in a window output."""

    track_id: int
    start_timestamp: float
    end_timestamp: float
    observed_count: int
    predicted_count: int
    mean_detector_confidence: Optional[float] = None

    def __post_init__(self) -> None:
        require_int(self.track_id, "track.track_id")
        require_non_negative(self.start_timestamp, "track.start_timestamp")
        require_non_negative(self.end_timestamp, "track.end_timestamp")
        if self.end_timestamp < self.start_timestamp:
            _fail("a track cannot end before it starts")
        require_int(self.observed_count, "track.observed_count")
        require_int(self.predicted_count, "track.predicted_count")
        if self.mean_detector_confidence is not None:
            require_unit_interval(self.mean_detector_confidence, "track.mean_detector_confidence")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "start_timestamp": float(self.start_timestamp),
            "end_timestamp": float(self.end_timestamp),
            "observed_count": self.observed_count,
            "predicted_count": self.predicted_count,
            "mean_detector_confidence": self.mean_detector_confidence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrackSummary":
        where = "TrackSummary"
        return cls(
            track_id=_get(data, "track_id", where),
            start_timestamp=_get(data, "start_timestamp", where),
            end_timestamp=_get(data, "end_timestamp", where),
            observed_count=_get(data, "observed_count", where),
            predicted_count=_get(data, "predicted_count", where),
            mean_detector_confidence=_opt(data, "mean_detector_confidence"),
        )


# --------------------------------------------------------------------------
# Window-level output
# --------------------------------------------------------------------------
def _check_index_list(values: Any, name: str) -> List[int]:
    if not isinstance(values, (list, tuple)):
        _fail(f"{name} must be a list of frame indices")
    checked = [require_int(v, f"{name}[{i}]") for i, v in enumerate(values)]
    for a, b in zip(checked, checked[1:]):
        if b <= a:
            _fail(f"{name} must be strictly increasing (ordered, no duplicates)")
    return checked


@dataclass(frozen=True)
class WindowDiagnostics:
    """Debugging information that travels with a window output."""

    selected_frame_indices: List[int] = field(default_factory=list)
    skipped_frame_indices: List[int] = field(default_factory=list)
    decode: Optional[DecodeDiagnostics] = None
    motion_scores: Optional[List[float]] = None
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        selected = _check_index_list(self.selected_frame_indices, "diagnostics.selected_frame_indices")
        skipped = _check_index_list(self.skipped_frame_indices, "diagnostics.skipped_frame_indices")
        if set(selected) & set(skipped):
            _fail("a frame cannot be both selected and skipped")
        if self.decode is not None and not isinstance(self.decode, DecodeDiagnostics):
            _fail("diagnostics.decode must be DecodeDiagnostics")
        if self.motion_scores is not None:
            for i, v in enumerate(self.motion_scores):
                require_non_negative(v, f"diagnostics.motion_scores[{i}]")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_frame_indices": list(self.selected_frame_indices),
            "skipped_frame_indices": list(self.skipped_frame_indices),
            "decode": None if self.decode is None else self.decode.to_dict(),
            "motion_scores": None if self.motion_scores is None else list(self.motion_scores),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WindowDiagnostics":
        decode = _opt(data, "decode")
        return cls(
            selected_frame_indices=list(data.get("selected_frame_indices", [])),
            skipped_frame_indices=list(data.get("skipped_frame_indices", [])),
            decode=None if decode is None else DecodeDiagnostics.from_dict(decode),
            motion_scores=_opt(data, "motion_scores"),
            notes=list(data.get("notes", [])),
        )


@dataclass(frozen=True)
class WindowOutput:
    """The final answer for one processed window (see plan section 2.1)."""

    window_id: str
    source_id: str
    start_timestamp: float
    end_timestamp: float
    label: str
    violence_score: float
    temporal_confidence: float
    clarity_score: float
    processed_frame_count: int
    skipped_frame_count: int
    tracks: List[TrackSummary] = field(default_factory=list)
    objects: List[ObjectEvidence] = field(default_factory=list)
    diagnostics: WindowDiagnostics = field(default_factory=WindowDiagnostics)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_text(self.window_id, "window.window_id")
        require_text(self.source_id, "window.source_id")
        require_non_negative(self.start_timestamp, "window.start_timestamp")
        require_non_negative(self.end_timestamp, "window.end_timestamp")
        if self.end_timestamp < self.start_timestamp:
            _fail("a window cannot end before it starts")
        require_choice(self.label, LABELS, "window.label")
        require_unit_interval(self.violence_score, "window.violence_score")
        require_unit_interval(self.temporal_confidence, "window.temporal_confidence")
        require_unit_interval(self.clarity_score, "window.clarity_score")
        require_int(self.processed_frame_count, "window.processed_frame_count")
        require_int(self.skipped_frame_count, "window.skipped_frame_count")
        for track in self.tracks:
            if not isinstance(track, TrackSummary):
                _fail("window.tracks must contain TrackSummary items")
        for item in self.objects:
            if not isinstance(item, ObjectEvidence):
                _fail("window.objects must contain ObjectEvidence items")
        if not isinstance(self.diagnostics, WindowDiagnostics):
            _fail("window.diagnostics must be WindowDiagnostics")
        require_text(self.schema_version, "window.schema_version")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "window_id": self.window_id,
            "source_id": self.source_id,
            "start_timestamp": float(self.start_timestamp),
            "end_timestamp": float(self.end_timestamp),
            "label": self.label,
            "violence_score": float(self.violence_score),
            "temporal_confidence": float(self.temporal_confidence),
            "clarity_score": float(self.clarity_score),
            "processed_frame_count": self.processed_frame_count,
            "skipped_frame_count": self.skipped_frame_count,
            "tracks": [t.to_dict() for t in self.tracks],
            "objects": [o.to_dict() for o in self.objects],
            "diagnostics": self.diagnostics.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WindowOutput":
        where = "WindowOutput"
        return cls(
            window_id=_get(data, "window_id", where),
            source_id=_get(data, "source_id", where),
            start_timestamp=_get(data, "start_timestamp", where),
            end_timestamp=_get(data, "end_timestamp", where),
            label=_get(data, "label", where),
            violence_score=_get(data, "violence_score", where),
            temporal_confidence=_get(data, "temporal_confidence", where),
            clarity_score=_get(data, "clarity_score", where),
            processed_frame_count=_get(data, "processed_frame_count", where),
            skipped_frame_count=_get(data, "skipped_frame_count", where),
            tracks=[TrackSummary.from_dict(t) for t in data.get("tracks", [])],
            objects=[ObjectEvidence.from_dict(o) for o in data.get("objects", [])],
            diagnostics=WindowDiagnostics.from_dict(_get(data, "diagnostics", where)),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )

    @classmethod
    def from_json(cls, text: str) -> "WindowOutput":
        return cls.from_dict(json.loads(text))
