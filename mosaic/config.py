"""Configuration loading and validation.

All tunable settings live in one typed object (``Config``) so that:

* defaults are written down in exactly one place,
* a typo in a YAML file (``windw:`` instead of ``window:``) is an error instead
  of being silently ignored,
* wrong types (``seed: "abc"``) and impossible values (negative window length)
  fail explicitly with a message that says which setting is wrong,
* a short *hash* of the behaviour-relevant settings identifies an experiment,
  so cached results can be matched to the exact settings that produced them.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import re
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

# Stages that exist in the pipeline. New phases add their stage name here.
KNOWN_STAGES = ("ingestion", "clarity", "motion_gate", "detector", "tracker", "temporal_classifier")
WINDOW_POLICIES = ("whole_clip", "sliding")
AGGREGATIONS = ("mean", "max")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
SAMPLING_STRATEGIES = ("uniform", "stride")
MODEL_ARCHITECTURES = ("baseline_temporal", "spatial_temporal_fusion", "majority", "motion_energy")
BACKBONES = ("resnet18", "cnn_light")
RNN_TYPES = ("gru", "lstm")
DEVICES = ("auto", "cuda", "cpu")
DETECTOR_ARCHITECTURES = ("fasterrcnn_mobilenet_320", "ssdlite320", "mock")
CLARITY_AGGREGATIONS = ("mean", "min")
DEGRADATION_KINDS = (
    "resolution",
    "blur",
    "compression",
    "low_light",
    "noise",
    "frame_drop",
    "occlusion",
)

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ConfigError(ValueError):
    """Raised when configuration is missing, malformed or invalid."""


# --------------------------------------------------------------------------
# Config sections
# --------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    name: str = "phase0"
    seed: int = 1337

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name):
            raise ConfigError(
                f"experiment.name must use only letters, digits, '_', '-', '.' "
                f"(and start with a letter or digit), got {self.name!r}"
            )
        if not 0 <= self.seed < 2**32:
            raise ConfigError(f"experiment.seed must be in [0, 2**32), got {self.seed}")


@dataclass
class PathsConfig:
    artifacts_dir: str = "artifacts"
    data_dir: str = "data"

    def __post_init__(self) -> None:
        for key in ("artifacts_dir", "data_dir"):
            if not getattr(self, key).strip():
                raise ConfigError(f"paths.{key} must not be empty")


@dataclass
class DecodeConfig:
    # Stop after this many frames (None = read the whole video).
    max_frames: Optional[int] = None
    # Some containers report a frame count that is off by one; allow this much.
    frame_count_tolerance: int = 1

    def __post_init__(self) -> None:
        if self.max_frames is not None and self.max_frames < 1:
            raise ConfigError(f"decode.max_frames must be >= 1 or null, got {self.max_frames}")
        if self.frame_count_tolerance < 0:
            raise ConfigError(
                f"decode.frame_count_tolerance must be >= 0, got {self.frame_count_tolerance}"
            )


@dataclass
class WindowConfig:
    # "whole_clip": one window covers the entire clip (the default for RWF-2000).
    # "sliding": windows of window_seconds, advanced by stride_seconds.
    policy: str = "whole_clip"
    window_seconds: float = 5.0
    stride_seconds: float = 5.0
    # How per-window scores are combined into one clip-level score.
    clip_aggregation: str = "mean"

    def __post_init__(self) -> None:
        if self.policy not in WINDOW_POLICIES:
            raise ConfigError(f"window.policy must be one of {list(WINDOW_POLICIES)}, got {self.policy!r}")
        if self.clip_aggregation not in AGGREGATIONS:
            raise ConfigError(
                f"window.clip_aggregation must be one of {list(AGGREGATIONS)}, got {self.clip_aggregation!r}"
            )
        if not self.window_seconds > 0:
            raise ConfigError(f"window.window_seconds must be > 0, got {self.window_seconds}")
        if not self.stride_seconds > 0:
            raise ConfigError(f"window.stride_seconds must be > 0, got {self.stride_seconds}")
        if self.stride_seconds > self.window_seconds:
            raise ConfigError(
                "window.stride_seconds must not exceed window.window_seconds "
                f"(got stride={self.stride_seconds}, window={self.window_seconds}); "
                "that would leave unseen gaps between windows"
            )


@dataclass
class LoggingConfig:
    level: str = "INFO"
    to_file: bool = True

    def __post_init__(self) -> None:
        if self.level not in LOG_LEVELS:
            raise ConfigError(f"logging.level must be one of {list(LOG_LEVELS)}, got {self.level!r}")


@dataclass
class PipelineConfig:
    stages: List[str] = field(default_factory=lambda: ["ingestion"])

    def __post_init__(self) -> None:
        if not self.stages:
            raise ConfigError("pipeline.stages must not be empty")
        unknown = [s for s in self.stages if s not in KNOWN_STAGES]
        if unknown:
            raise ConfigError(
                f"pipeline.stages contains unknown stage(s) {unknown}; known stages: {list(KNOWN_STAGES)}"
            )
        if len(set(self.stages)) != len(self.stages):
            raise ConfigError("pipeline.stages must not contain duplicates")
        if self.stages[0] != "ingestion":
            raise ConfigError("pipeline.stages must start with 'ingestion'")


@dataclass
class DataConfig:
    raw_dir: str = "data/RWF-2000"
    manifest_path: str = "artifacts/manifests/rwf2000_manifest.json"
    val_carve_ratio: float = 0.2
    num_frames: int = 16
    sampling_strategy: str = "uniform"
    target_height: int = 224
    target_width: int = 224
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = True
    horizontal_flip_prob: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.val_carve_ratio < 1.0:
            raise ConfigError(f"data.val_carve_ratio must be in (0, 1), got {self.val_carve_ratio}")
        if self.num_frames < 1:
            raise ConfigError(f"data.num_frames must be >= 1, got {self.num_frames}")
        if self.sampling_strategy not in SAMPLING_STRATEGIES:
            raise ConfigError(
                f"data.sampling_strategy must be in {list(SAMPLING_STRATEGIES)}, got {self.sampling_strategy!r}"
            )
        if self.target_height < 16 or self.target_width < 16:
            raise ConfigError(
                f"data target dimensions must be >= 16, got {self.target_height}x{self.target_width}"
            )
        if self.batch_size < 1:
            raise ConfigError(f"data.batch_size must be >= 1, got {self.batch_size}")
        if self.num_workers < 0:
            raise ConfigError(f"data.num_workers must be >= 0, got {self.num_workers}")
        if not 0.0 <= self.horizontal_flip_prob <= 1.0:
            raise ConfigError(
                f"data.horizontal_flip_prob must be in [0, 1], got {self.horizontal_flip_prob}"
            )


@dataclass
class ModelConfig:
    architecture: str = "baseline_temporal"
    backbone: str = "resnet18"
    pretrained: bool = False
    freeze_backbone: bool = True
    hidden_dim: int = 128
    rnn_type: str = "gru"
    rnn_layers: int = 1
    dropout: float = 0.2
    use_scene: bool = True
    use_tracks: bool = True
    use_kinematics: bool = True
    use_masks: bool = True
    use_objects: bool = False
    decision_threshold: float = 0.50
    mask_predicted_observations: bool = False
    max_tracks: int = 8
    max_steps: int = 32

    def __post_init__(self) -> None:
        if self.architecture not in MODEL_ARCHITECTURES:
            raise ConfigError(
                f"model.architecture must be in {list(MODEL_ARCHITECTURES)}, got {self.architecture!r}"
            )
        if self.backbone not in BACKBONES:
            raise ConfigError(
                f"model.backbone must be in {list(BACKBONES)}, got {self.backbone!r}"
            )
        if self.hidden_dim < 4:
            raise ConfigError(f"model.hidden_dim must be >= 4, got {self.hidden_dim}")
        if self.rnn_type not in RNN_TYPES:
            raise ConfigError(
                f"model.rnn_type must be in {list(RNN_TYPES)}, got {self.rnn_type!r}"
            )
        if self.rnn_layers < 1:
            raise ConfigError(f"model.rnn_layers must be >= 1, got {self.rnn_layers}")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"model.dropout must be in [0, 1), got {self.dropout}")
        if not 0.0 <= self.decision_threshold <= 1.0:
            raise ConfigError(
                f"model.decision_threshold must be in [0, 1], got {self.decision_threshold}"
            )
        if self.max_tracks < 1:
            raise ConfigError(f"model.max_tracks must be >= 1, got {self.max_tracks}")
        if self.max_steps < 1:
            raise ConfigError(f"model.max_steps must be >= 1, got {self.max_steps}")


@dataclass
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "auto"
    early_stopping_patience: int = 3

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ConfigError(f"train.epochs must be >= 1, got {self.epochs}")
        if not self.learning_rate > 0:
            raise ConfigError(f"train.learning_rate must be > 0, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ConfigError(f"train.weight_decay must be >= 0, got {self.weight_decay}")
        if self.device not in DEVICES:
            raise ConfigError(f"train.device must be in {list(DEVICES)}, got {self.device!r}")
        if self.early_stopping_patience < 1:
            raise ConfigError(
                f"train.early_stopping_patience must be >= 1, got {self.early_stopping_patience}"
            )


@dataclass
class ClarityConfig:
    min_laplacian_variance: float = 10.0
    max_laplacian_variance: float = 600.0
    target_width: int = 1280
    target_height: int = 720
    aggregation: str = "mean"

    def __post_init__(self) -> None:
        if self.min_laplacian_variance <= 0:
            raise ConfigError(
                f"clarity.min_laplacian_variance must be > 0, got {self.min_laplacian_variance}"
            )
        if self.max_laplacian_variance <= self.min_laplacian_variance:
            raise ConfigError("clarity.max_laplacian_variance must exceed min_laplacian_variance")
        if self.target_width < 16 or self.target_height < 16:
            raise ConfigError("clarity target dimensions must be >= 16")
        if self.aggregation not in CLARITY_AGGREGATIONS:
            raise ConfigError(
                f"clarity.aggregation must be in {list(CLARITY_AGGREGATIONS)}, got {self.aggregation!r}"
            )


@dataclass
class DegradationConfig:
    enabled: bool = False
    kind: str = "blur"
    severity: float = 0.0
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.kind not in DEGRADATION_KINDS:
            raise ConfigError(
                f"degradation.kind must be in {list(DEGRADATION_KINDS)}, got {self.kind!r}"
            )
        if not 0.0 <= self.severity <= 1.0:
            raise ConfigError(f"degradation.severity must be in [0.0, 1.0], got {self.severity}")


@dataclass
class MotionGateConfig:
    enabled: bool = True
    threshold: float = 0.010
    burst_threshold: float = 0.020
    burst_hold_frames: int = 5
    keep_alive_frames: int = 15
    downsample_width: int = 64
    downsample_height: int = 64

    def __post_init__(self) -> None:
        if self.threshold < 0.0:
            raise ConfigError(f"motion_gate.threshold must be >= 0.0, got {self.threshold}")
        if self.burst_threshold < self.threshold:
            raise ConfigError(
                f"motion_gate.burst_threshold ({self.burst_threshold}) cannot be less than threshold ({self.threshold})"
            )
        if self.burst_hold_frames < 0:
            raise ConfigError(
                f"motion_gate.burst_hold_frames must be >= 0, got {self.burst_hold_frames}"
            )
        if self.keep_alive_frames < 1:
            raise ConfigError(
                f"motion_gate.keep_alive_frames must be >= 1, got {self.keep_alive_frames}"
            )
        if self.downsample_width < 8 or self.downsample_height < 8:
            raise ConfigError("motion_gate downsample dimensions must be >= 8")


@dataclass
class DetectorConfig:
    enabled: bool = True
    architecture: str = "fasterrcnn_mobilenet_320"
    confidence_threshold: float = 0.40
    min_box_area: float = 100.0
    min_box_height: float = 16.0
    extract_features: bool = False
    feature_dim: int = 128
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.architecture not in DETECTOR_ARCHITECTURES:
            raise ConfigError(
                f"detector.architecture must be in {list(DETECTOR_ARCHITECTURES)}, got {self.architecture!r}"
            )
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ConfigError(
                f"detector.confidence_threshold must be in [0.0, 1.0], got {self.confidence_threshold}"
            )
        if self.min_box_area < 0.0:
            raise ConfigError(f"detector.min_box_area must be >= 0.0, got {self.min_box_area}")
        if self.min_box_height < 0.0:
            raise ConfigError(f"detector.min_box_height must be >= 0.0, got {self.min_box_height}")
        if self.feature_dim < 4:
            raise ConfigError(f"detector.feature_dim must be >= 4, got {self.feature_dim}")
        if self.device not in DEVICES:
            raise ConfigError(f"detector.device must be in {list(DEVICES)}, got {self.device!r}")


@dataclass
class MaskConfig:
    enabled: bool = True
    min_visible_fraction: float = 0.05
    confidence: float = 0.85

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_visible_fraction <= 1.0:
            raise ConfigError(
                f"masks.min_visible_fraction must be in [0, 1], got {self.min_visible_fraction}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ConfigError(f"masks.confidence must be in [0, 1], got {self.confidence}")


@dataclass
class ObjectConfig:
    enabled: bool = False
    max_association_distance: float = 80.0

    def __post_init__(self) -> None:
        if self.max_association_distance < 0.0:
            raise ConfigError(
                f"objects.max_association_distance must be >= 0, got {self.max_association_distance}"
            )


@dataclass
class TrackerConfig:
    iou_threshold: float = 0.3
    high_det_threshold: float = 0.5
    low_det_threshold: float = 0.1
    max_lost_frames: int = 15
    min_hits: int = 2
    max_tracks: int = 16

    def __post_init__(self) -> None:
        if not 0.0 < self.iou_threshold < 1.0:
            raise ConfigError(f"tracker.iou_threshold must be in (0, 1), got {self.iou_threshold}")
        if not 0.0 <= self.low_det_threshold <= self.high_det_threshold <= 1.0:
            raise ConfigError("tracker thresholds must satisfy 0 <= low <= high <= 1")
        if self.max_lost_frames < 1:
            raise ConfigError(f"tracker.max_lost_frames must be >= 1, got {self.max_lost_frames}")
        if self.min_hits < 1:
            raise ConfigError(f"tracker.min_hits must be >= 1, got {self.min_hits}")
        if self.max_tracks < 1:
            raise ConfigError(f"tracker.max_tracks must be >= 1, got {self.max_tracks}")


@dataclass
class TokenConfig:
    cache_dir: str = "artifacts/token_cache/tracks"
    normalize_coords: bool = True
    max_tokens_per_window: int = 128

    def __post_init__(self) -> None:
        if not self.cache_dir.strip():
            raise ConfigError("tokens.cache_dir must not be empty or whitespace")
        if self.max_tokens_per_window < 1:
            raise ConfigError(f"tokens.max_tokens_per_window must be >= 1, got {self.max_tokens_per_window}")


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    clarity: ClarityConfig = field(default_factory=ClarityConfig)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)
    motion_gate: MotionGateConfig = field(default_factory=MotionGateConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    masks: MaskConfig = field(default_factory=MaskConfig)
    objects: ObjectConfig = field(default_factory=ObjectConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    tokens: TokenConfig = field(default_factory=TokenConfig)


# --------------------------------------------------------------------------
# Building a Config from plain data (usually parsed YAML)
# --------------------------------------------------------------------------
def _check_type(value: Any, hint: Any, where: str) -> Any:
    """Check ``value`` against a type hint and return it (ints widened to float)."""
    origin = typing.get_origin(hint)
    if origin is typing.Union:
        args = typing.get_args(hint)
        if value is None and type(None) in args:
            return None
        inner = [a for a in args if a is not type(None)]
        return _check_type(value, inner[0], where)
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where} must be true/false, got {value!r}")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where} must be an integer, got {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where} must be a number, got {value!r}")
        if not math.isfinite(float(value)):
            raise ConfigError(f"{where} must be finite, got {value!r}")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where} must be a string, got {value!r}")
        return value
    if origin in (list, typing.List):
        if not isinstance(value, list):
            raise ConfigError(f"{where} must be a list, got {value!r}")
        (item_hint,) = typing.get_args(hint)
        return [_check_type(item, item_hint, f"{where}[{i}]") for i, item in enumerate(value)]
    raise ConfigError(f"{where}: unsupported configuration type {hint!r}")  # pragma: no cover


def _from_mapping(cls: Any, data: Any, where: str) -> Any:
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"'{where}' must be a mapping of settings, got {type(data).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ConfigError(
            f"unknown setting(s) {unknown} in '{where}'; valid settings: {sorted(fields)}"
        )
    hints = typing.get_type_hints(cls)
    kwargs: Dict[str, Any] = {}
    for name, value in data.items():
        hint = hints[name]
        path = f"{where}.{name}" if where else name
        if dataclasses.is_dataclass(hint):
            kwargs[name] = _from_mapping(hint, value, path)
        else:
            kwargs[name] = _check_type(value, hint, path)
    return cls(**kwargs)  # section __post_init__ validates the values


def config_from_dict(data: Optional[Mapping[str, Any]] = None) -> Config:
    """Build a validated ``Config`` from a plain mapping (missing keys use defaults)."""
    return _from_mapping(Config, data, "")


def deep_merge(base: Dict[str, Any], extra: Mapping[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_override(text: str) -> Dict[str, Any]:
    """Turn ``"window.policy=sliding"`` into ``{"window": {"policy": "sliding"}}``."""
    if "=" not in text:
        raise ConfigError(f"override must look like 'section.key=value', got {text!r}")
    key, raw = text.split("=", 1)
    parts = [p for p in key.strip().split(".") if p]
    if not parts:
        raise ConfigError(f"override has an empty key: {text!r}")
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"override value is not valid YAML: {text!r} ({exc})") from exc
    nested: Any = value
    for part in reversed(parts):
        nested = {part: nested}
    return nested


def load_config(path: Optional[Path | str] = None, overrides: Sequence[str] = ()) -> Config:
    """Load a YAML config (or defaults when ``path`` is None) and apply overrides."""
    data: Dict[str, Any] = {}
    if path is not None:
        file = Path(path)
        if not file.is_file():
            raise ConfigError(f"config file not found: {file}")
        try:
            loaded = yaml.safe_load(file.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"config file {file} is not valid YAML: {exc}") from exc
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"config file {file} must contain a mapping at the top level")
        data = loaded
    for text in overrides:
        data = deep_merge(data, parse_override(text))
    return config_from_dict(data)


# --------------------------------------------------------------------------
# Serialization, hash and identifiers
# --------------------------------------------------------------------------
def config_to_dict(cfg: Config) -> Dict[str, Any]:
    return dataclasses.asdict(cfg)


def dump_config(cfg: Config, path: Path | str) -> None:
    """Write the config as YAML (a snapshot stored next to each run's results)."""
    Path(path).write_text(
        yaml.safe_dump(config_to_dict(cfg), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def behavioural_dict(cfg: Config) -> Dict[str, Any]:
    """Settings that can change results. Paths, logging and the run label are excluded."""
    data = config_to_dict(cfg)
    data.pop("paths", None)
    data.pop("logging", None)
    data["experiment"].pop("name", None)
    if "data" in data and isinstance(data["data"], dict):
        data["data"].pop("raw_dir", None)
        data["data"].pop("manifest_path", None)
    return data


def config_hash(cfg: Config) -> str:
    """12-character fingerprint of the behaviour-relevant settings."""
    canonical = json.dumps(behavioural_dict(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def experiment_id(cfg: Config) -> str:
    """Deterministic experiment identifier: ``<name>-<confighash>-s<seed>``."""
    return f"{cfg.experiment.name}-{config_hash(cfg)}-s{cfg.experiment.seed}"
