"""Input clarity profiling and quality estimation (Block 1).

Quantifies the optical and resolution reliability of video frames without coupling
quality estimation to downstream predictions:
1. Resolution Component: Normalized resolution against a 720p baseline.
2. Sharpness Component: Edge energy via Laplacian variance (identifies optical/motion blur).
3. Exposure Component: Mean luminance and clipping penalty (identifies underexposure & washed-out scenes).
4. Compression Component: Blockiness detection along 8x8 DCT grid boundaries.
5. Person Scale Component: Explicitly None until person detections exist in Phase 4.

Produces a typed, validated ``ClarityReport`` satisfying ``mosaic.contracts.ClarityReport``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from mosaic.contracts import ClarityReport, ContractError, Detection


class ClarityError(ValueError):
    """Raised when clarity inputs or parameters are invalid."""


def infer_source_profile(width: int, height: int) -> str:
    """Infer resolution category based on diagonal pixel count."""
    pixels = width * height
    if pixels >= 1920 * 1080 * 0.85:
        return "1080p"
    if pixels >= 1280 * 720 * 0.75:
        return "720p"
    if pixels >= 640 * 480 * 0.75:
        return "480p"
    if pixels >= 320 * 240 * 0.75:
        return "240p"
    return "sub_240p"


def calculate_resolution_component(
    width: int,
    height: int,
    target_width: int = 1280,
    target_height: int = 720,
) -> float:
    """Calculate normalized resolution component in [0.0, 1.0]."""
    if width <= 0 or height <= 0:
        raise ClarityError(f"invalid dimensions {width}x{height}")
    diag = math.sqrt(width * width + height * height)
    target_diag = math.sqrt(target_width * target_width + target_height * target_height)
    return float(np.clip(diag / target_diag, 0.0, 1.0))


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Safely convert 2D or 3D frame to uint8 grayscale."""
    if not isinstance(frame, np.ndarray):
        raise ClarityError(f"frame must be numpy array, got {type(frame)}")
    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3:
        if frame.shape[2] == 1:
            gray = frame[:, :, 0]
        elif frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.shape[2] == 4:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        else:
            raise ClarityError(f"unsupported channel count {frame.shape[2]}")
    else:
        raise ClarityError(f"unsupported frame dimensions {frame.ndim}")

    if not np.all(np.isfinite(gray)):
        raise ClarityError("frame contains NaN or infinite values")

    if gray.dtype != np.uint8:
        if np.max(gray) <= 1.0 and np.min(gray) >= 0.0:
            gray = (gray * 255.0).astype(np.uint8)
        else:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


_DONOHO_KERNEL = (
    np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32) / 6.0
)


def estimate_noise_level(gray: np.ndarray) -> float:
    """Estimate additive Gaussian sensor noise std via Donoho & Johnstone MAD estimator."""
    res = cv2.filter2D(gray.astype(np.float32), -1, _DONOHO_KERNEL)
    return float(np.median(np.abs(res)) / 0.6745)


def calculate_sharpness_component(
    frame: np.ndarray,
    min_variance: float = 10.0,
    max_variance: float = 600.0,
) -> float:
    """Calculate sharpness component in [0.0, 1.0] using noise-attenuated Laplacian variance."""
    gray = _to_gray(frame)
    if gray.size < 4:
        return 0.0

    # Attenuate single-pixel uncorrelated noise while preserving coherent optical edges
    smoothed = cv2.GaussianBlur(gray, (3, 3), 0.75)
    lap = cv2.Laplacian(smoothed, cv2.CV_64F)
    variance = float(lap.var())
    if not math.isfinite(variance) or variance <= 0.0:
        return 0.0

    # Scale calibrated for Gaussian-smoothed 3x3 Laplacian (clean CCTV typically 20 to 120)
    calibrated_max = max(20.0, max_variance * 0.15)
    calibrated_min = max(0.5, min_variance * 0.15)

    log_v = math.log1p(variance)
    log_min = math.log1p(calibrated_min)
    log_max = math.log1p(calibrated_max)
    if log_v <= log_min:
        norm = float(np.clip(variance / (2.0 * calibrated_min), 0.0, 0.2))
    else:
        norm = 0.2 + 0.8 * ((log_v - log_min) / max(1e-6, log_max - log_min))

    # Penalize excessive sensor noise so high-frequency thermal noise never inflates sharpness
    noise_std = estimate_noise_level(gray)
    noise_cleanliness = float(np.clip(1.0 - max(0.0, noise_std - 6.0) / 25.0, 0.2, 1.0))

    return float(np.clip(norm * noise_cleanliness, 0.0, 1.0))


def calculate_exposure_component(frame: np.ndarray) -> float:
    """Calculate exposure score in [0.0, 1.0] penalizing underexposure, overexposure & clipping."""
    gray = _to_gray(frame)
    if gray.size == 0:
        return 0.0
    mean_lum = float(np.mean(gray))
    # Distance from ideal center (128.0)
    norm_mean = mean_lum / 255.0
    balance = 1.0 - 2.0 * abs(norm_mean - 0.5)
    balance = max(0.0, balance)

    # Clipping penalty: fraction of deep saturated black (<= 8) or saturated white (>= 248) pixels
    clip_low = np.mean(gray <= 8)
    clip_high = np.mean(gray >= 248)
    clip_fraction = float(clip_low + clip_high)
    clipping_penalty = max(0.0, 1.0 - clip_fraction * 1.5)

    return float(np.clip(balance * clipping_penalty, 0.0, 1.0))


def calculate_compression_component(frame: np.ndarray) -> float:
    """Calculate compression quality score in [0.0, 1.0] by measuring 8x8 DCT grid blockiness."""
    gray = _to_gray(frame)
    h, w = gray.shape
    if h < 24 or w < 24:
        return 0.5

    # Differences across 8-pixel block boundaries vs intra-block differences
    # Horizontal block edges: columns 7 to 8, 15 to 16, etc.
    col_diffs = np.abs(gray[:, 1:].astype(np.float32) - gray[:, :-1].astype(np.float32))
    row_diffs = np.abs(gray[1:, :].astype(np.float32) - gray[:-1, :].astype(np.float32))

    block_cols = np.arange(7, w - 1, 8)
    other_cols = np.array([c for c in range(w - 1) if c % 8 != 7], dtype=int)
    block_rows = np.arange(7, h - 1, 8)
    other_rows = np.array([r for r in range(h - 1) if r % 8 != 7], dtype=int)

    if len(block_cols) == 0 or len(other_cols) == 0 or len(block_rows) == 0 or len(other_rows) == 0:
        return 0.5

    boundary_diff = (
        float(np.mean(col_diffs[:, block_cols])) + float(np.mean(row_diffs[block_rows, :]))
    ) / 2.0
    interior_diff = (
        float(np.mean(col_diffs[:, other_cols])) + float(np.mean(row_diffs[other_rows, :]))
    ) / 2.0

    if interior_diff < 1e-4:
        # Uniform or near-uniform frame: no blocking artifacts
        return 1.0 if boundary_diff < 1e-3 else 0.0

    ratio = boundary_diff / interior_diff
    # In uncompressed or clean video, ratio is ~1.0. Heavy blockiness pushes ratio > 1.8.
    if ratio <= 1.05:
        score = 1.0
    else:
        # Monotonically decay as boundary discontinuity exceeds interior gradient
        score = 1.0 - (ratio - 1.05) / 1.5
    return float(np.clip(score, 0.0, 1.0))


def calculate_person_scale_component(
    detections: Sequence[Detection],
    frame_height: int,
) -> Optional[float]:
    """Calculate person scale quality score in [0.0, 1.0] from median detected person height.

    Returns None if detections is empty or no valid person detections exist.
    """
    if frame_height <= 0:
        return None
    person_heights = [
        d.box.height for d in detections if d.class_name == "person" and d.box.height > 0
    ]
    if not person_heights:
        return None

    median_h = float(np.median(person_heights))
    ratio = median_h / float(frame_height)
    # A person height of >= 25% of frame height (ratio >= 0.25) is fully resolvable for limb motion (1.0).
    # Below 5%, individual limbs cannot be reliably resolved.
    score = float(np.clip(ratio / 0.25, 0.0, 1.0))
    return score


def combine_clarity_components(
    resolution: Optional[float] = None,
    sharpness: Optional[float] = None,
    exposure: Optional[float] = None,
    compression: Optional[float] = None,
    person_scale: Optional[float] = None,
) -> float:
    """Combine available clarity components with documented weights into a score in [0.0, 1.0]."""
    weights = {
        "resolution": 0.25,
        "sharpness": 0.35,
        "exposure": 0.25,
        "compression": 0.15,
        "person_scale": 0.20,
    }
    values = {
        "resolution": resolution,
        "sharpness": sharpness,
        "exposure": exposure,
        "compression": compression,
        "person_scale": person_scale,
    }
    active_weight = 0.0
    weighted_sum = 0.0
    for name, val in values.items():
        if val is not None:
            w = weights[name]
            active_weight += w
            weighted_sum += w * float(val)

    if active_weight <= 0.0:
        return 0.0
    return float(np.clip(weighted_sum / active_weight, 0.0, 1.0))


def compute_frame_clarity(
    frame: np.ndarray,
    source_profile: Optional[str] = None,
    person_scale: Optional[float] = None,
    detections: Optional[Sequence[Detection]] = None,
) -> ClarityReport:
    """Compute all clarity components and composite score for a single frame."""
    gray = _to_gray(frame)
    h, w = gray.shape[:2]
    profile = source_profile or infer_source_profile(w, h)

    res_comp = calculate_resolution_component(w, h)
    sharp_comp = calculate_sharpness_component(gray)
    exp_comp = calculate_exposure_component(gray)
    comp_comp = calculate_compression_component(gray)

    if person_scale is None and detections is not None:
        ps_comp = calculate_person_scale_component(detections, h)
    else:
        ps_comp = None if person_scale is None else float(np.clip(person_scale, 0.0, 1.0))

    score = combine_clarity_components(
        resolution=res_comp,
        sharpness=sharp_comp,
        exposure=exp_comp,
        compression=comp_comp,
        person_scale=ps_comp,
    )

    return ClarityReport(
        score=score,
        source_profile=profile,
        resolution_component=res_comp,
        sharpness_component=sharp_comp,
        exposure_component=exp_comp,
        compression_component=comp_comp,
        person_scale_component=ps_comp,
    )


def compute_window_clarity(
    frames: Sequence[np.ndarray],
    source_profile: Optional[str] = None,
    aggregation: str = "mean",
    detections: Optional[Sequence[Detection]] = None,
) -> ClarityReport:
    """Compute aggregate clarity across a sequence of frames."""
    if len(frames) == 0:
        raise ClarityError("cannot compute clarity on empty frame sequence")

    reports = [compute_frame_clarity(f, source_profile=source_profile) for f in frames]
    if aggregation == "mean":
        agg_fn = np.mean
    elif aggregation == "min":
        agg_fn = np.min
    else:
        raise ClarityError(f"unknown aggregation {aggregation!r}; supported: ['mean', 'min']")

    scores = [r.score for r in reports]
    res_vals = [r.resolution_component for r in reports if r.resolution_component is not None]
    sharp_vals = [r.sharpness_component for r in reports if r.sharpness_component is not None]
    exp_vals = [r.exposure_component for r in reports if r.exposure_component is not None]
    comp_vals = [r.compression_component for r in reports if r.compression_component is not None]

    person_scale_comp: Optional[float] = None
    if detections is not None:
        h = frames[0].shape[0] if frames[0].ndim >= 2 else 0
        person_scale_comp = calculate_person_scale_component(detections, h)

    if person_scale_comp is not None:
        window_score = combine_clarity_components(
            resolution=float(agg_fn(res_vals)) if res_vals else None,
            sharpness=float(agg_fn(sharp_vals)) if sharp_vals else None,
            exposure=float(agg_fn(exp_vals)) if exp_vals else None,
            compression=float(agg_fn(comp_vals)) if comp_vals else None,
            person_scale=person_scale_comp,
        )
    else:
        window_score = float(agg_fn(scores))

    profile = reports[0].source_profile
    return ClarityReport(
        score=window_score,
        source_profile=profile,
        resolution_component=float(agg_fn(res_vals)) if res_vals else None,
        sharpness_component=float(agg_fn(sharp_vals)) if sharp_vals else None,
        exposure_component=float(agg_fn(exp_vals)) if exp_vals else None,
        compression_component=float(agg_fn(comp_vals)) if comp_vals else None,
        person_scale_component=person_scale_comp,
    )
