"""Seeded degradation toolkit for simulated camera conditions (Section 3.4).

Provides reproducible edge camera degradations:
1. Resolution: Downscaling and upscaling (720p, 480p, 240p equivalent).
2. Blur: Gaussian optical and motion defocus.
3. Compression: JPEG/DCT blockiness and quantization.
4. Low Light: Luminance attenuation and non-linear gamma curve.
5. Sensor Noise: Gaussian thermal sensor noise.
6. Frame Dropping: Temporal frame skipping with preserved timestamps.
7. Synthetic Occlusion: Partial obstacle mask covering a bounded region.

Every degradation is deterministic for a fixed (seed, severity), is applied identically
to every frame in a clip (except temporal frame dropping), and operates on numpy arrays.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


class DegradationError(ValueError):
    """Raised when degradation configuration or input is invalid."""


DEGRADATION_KINDS = (
    "resolution",
    "blur",
    "compression",
    "low_light",
    "noise",
    "frame_drop",
    "occlusion",
)


def _validate_severity(severity: float) -> float:
    if isinstance(severity, bool) or not isinstance(severity, (int, float)):
        raise DegradationError(f"severity must be a number, got {severity!r}")
    val = float(severity)
    if not math.isfinite(val) or not (0.0 <= val <= 1.0):
        raise DegradationError(f"severity must be in [0.0, 1.0], got {val}")
    return val


def _validate_frame(frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise DegradationError(f"frame must be numpy array, got {type(frame)}")
    if frame.ndim not in (2, 3):
        raise DegradationError(f"frame must have 2 or 3 dims, got {frame.ndim}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


class BaseDegradation(ABC):
    """Abstract base class for all reproducible video degradations."""

    def __init__(self, severity: float = 0.0, seed: int = 1337) -> None:
        self.severity = _validate_severity(severity)
        self.seed = int(seed)

    @property
    @abstractmethod
    def kind(self) -> str:
        """Name of the degradation."""

    @abstractmethod
    def apply_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply degradation to a single 2D/3D frame."""

    def apply_clip(
        self,
        frames: Sequence[np.ndarray],
        timestamps: Optional[Sequence[float]] = None,
    ) -> Tuple[List[np.ndarray], Optional[List[float]]]:
        """Apply degradation across a clip.

        Non-temporal degradations apply ``apply_frame`` identically to every frame.
        """
        if len(frames) == 0:
            return [], None if timestamps is None else []
        out_frames = [self.apply_frame(f) for f in frames]
        out_ts = list(timestamps) if timestamps is not None else None
        return out_frames, out_ts


class ResolutionDegradation(BaseDegradation):
    """Simulates lower-tier cameras by downscaling and upsampling back."""

    @property
    def kind(self) -> str:
        return "resolution"

    def apply_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = _validate_frame(frame)
        if self.severity <= 0.0:
            return frame.copy()
        h, w = frame.shape[:2]
        # Downscale factor drops from 1.0 down to 0.20
        scale = max(0.15, 1.0 - 0.80 * self.severity)
        down_w = max(16, int(round(w * scale)))
        down_h = max(16, int(round(h * scale)))
        downscaled = cv2.resize(frame, (down_w, down_h), interpolation=cv2.INTER_AREA)
        upscaled = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_LINEAR)
        return upscaled


class BlurDegradation(BaseDegradation):
    """Simulates optical defocus or camera motion blur using Gaussian filtering."""

    @property
    def kind(self) -> str:
        return "blur"

    def apply_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = _validate_frame(frame)
        if self.severity <= 0.0:
            return frame.copy()
        # Kernel size must be odd and increases with severity (3 to 27)
        k_radius = int(round(self.severity * 12))
        k_size = 2 * k_radius + 1
        sigma = 0.5 + 4.5 * self.severity
        return cv2.GaussianBlur(frame, (k_size, k_size), sigmaX=sigma, sigmaY=sigma)


class CompressionDegradation(BaseDegradation):
    """Simulates low-bitrate H.264/JPEG streaming artifacts via DCT quantization."""

    @property
    def kind(self) -> str:
        return "compression"

    def apply_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = _validate_frame(frame)
        if self.severity <= 0.0:
            return frame.copy()
        # Quality drops from 95 down to 5
        quality = int(round(95.0 - 90.0 * self.severity))
        quality = max(5, min(95, quality))
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok or encoded is None:
            return frame.copy()
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR if frame.ndim == 3 else cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            return frame.copy()
        return decoded


class LowLightDegradation(BaseDegradation):
    """Simulates underexposed or nighttime conditions via non-linear gamma and scaling."""

    @property
    def kind(self) -> str:
        return "low_light"

    def apply_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = _validate_frame(frame)
        if self.severity <= 0.0:
            return frame.copy()
        # Scale brightness down to as low as 15%
        scale = max(0.15, 1.0 - 0.85 * self.severity)
        # Gamma exponent increases from 1.0 up to 3.0 (crushes shadow detail)
        gamma = 1.0 + 2.0 * self.severity
        norm = frame.astype(np.float32) / 255.0
        darkened = scale * (norm ** gamma)
        return np.clip(darkened * 255.0, 0, 255).astype(np.uint8)


class NoiseDegradation(BaseDegradation):
    """Simulates sensor thermal and photon noise via seeded Gaussian perturbation."""

    def __init__(self, severity: float = 0.0, seed: int = 1337) -> None:
        super().__init__(severity=severity, seed=seed)
        # Pre-generate deterministic noise pattern for frame identity consistency
        self._rng = np.random.RandomState(self.seed)

    @property
    def kind(self) -> str:
        return "noise"

    def apply_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = _validate_frame(frame)
        if self.severity <= 0.0:
            return frame.copy()
        sigma = 45.0 * self.severity
        rng = np.random.RandomState(self.seed)
        noise = rng.normal(0.0, sigma, frame.shape)
        noisy = frame.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)


class OcclusionDegradation(BaseDegradation):
    """Simulates partial camera occlusion (e.g., foliage, pillar, insect) with a fixed patch."""

    def __init__(self, severity: float = 0.0, seed: int = 1337) -> None:
        super().__init__(severity=severity, seed=seed)

    @property
    def kind(self) -> str:
        return "occlusion"

    def apply_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = _validate_frame(frame)
        if self.severity <= 0.0:
            return frame.copy()
        h, w = frame.shape[:2]
        # Occlusion box covers up to 40% of the frame area
        occ_w = int(round(w * (0.10 + 0.35 * self.severity)))
        occ_h = int(round(h * (0.10 + 0.35 * self.severity)))

        # Deterministic placement from seed
        rng = np.random.RandomState(self.seed)
        x0 = int(rng.randint(0, max(1, w - occ_w)))
        y0 = int(rng.randint(0, max(1, h - occ_h)))

        out = frame.copy()
        # Dark gray occlusion patch with soft border
        color = (20, 20, 20) if out.ndim == 3 else 20
        out[y0 : y0 + occ_h, x0 : x0 + occ_w] = color
        return out


class FrameDropDegradation(BaseDegradation):
    """Simulates transmission packet loss or dropped frames while keeping true timestamps."""

    @property
    def kind(self) -> str:
        return "frame_drop"

    def apply_frame(self, frame: np.ndarray) -> np.ndarray:
        # Frame dropping is an explicitly temporal operation
        return frame.copy()

    def apply_clip(
        self,
        frames: Sequence[np.ndarray],
        timestamps: Optional[Sequence[float]] = None,
    ) -> Tuple[List[np.ndarray], Optional[List[float]]]:
        if len(frames) <= 2 or self.severity <= 0.0:
            return list(frames), list(timestamps) if timestamps is not None else None

        rng = np.random.RandomState(self.seed)
        n = len(frames)
        # Drop probability up to 0.70
        drop_p = 0.70 * self.severity

        # Always retain the first and last frame for temporal bounding
        keep_indices = [0]
        for i in range(1, n - 1):
            if rng.uniform(0.0, 1.0) >= drop_p:
                keep_indices.append(i)
        keep_indices.append(n - 1)

        retained_frames = [frames[i] for i in keep_indices]
        retained_ts = [timestamps[i] for i in keep_indices] if timestamps is not None else None
        return retained_frames, retained_ts


def get_degradation(
    kind: str,
    severity: float = 0.0,
    seed: int = 1337,
) -> BaseDegradation:
    """Factory creating a concrete degradation by kind."""
    if kind == "resolution":
        return ResolutionDegradation(severity=severity, seed=seed)
    if kind == "blur":
        return BlurDegradation(severity=severity, seed=seed)
    if kind == "compression":
        return CompressionDegradation(severity=severity, seed=seed)
    if kind == "low_light":
        return LowLightDegradation(severity=severity, seed=seed)
    if kind == "noise":
        return NoiseDegradation(severity=severity, seed=seed)
    if kind == "frame_drop":
        return FrameDropDegradation(severity=severity, seed=seed)
    if kind == "occlusion":
        return OcclusionDegradation(severity=severity, seed=seed)
    raise DegradationError(
        f"unknown degradation kind {kind!r}; supported: {list(DEGRADATION_KINDS)}"
    )


def apply_degradation(
    frames: Sequence[np.ndarray],
    kind: str,
    severity: float,
    seed: int = 1337,
    timestamps: Optional[Sequence[float]] = None,
) -> Tuple[List[np.ndarray], Optional[List[float]]]:
    """Convenience function to apply a named degradation to a clip."""
    degrader = get_degradation(kind, severity=severity, seed=seed)
    return degrader.apply_clip(frames, timestamps=timestamps)
