"""Silhouette and mask evidence extraction (Block 4).

Extracts low-cost foreground silhouette masks and contour descriptors from person crops:
1. Low-Cost Silhouette Extraction: Computes adaptive foreground/background segmentation
   within each person bounding box crop.
2. Contour Descriptors: Extracts shape descriptors (compactness, solidity, extent,
   Hu invariant shape moments) capturing body deformation and limb flailing.
3. Visible Fraction: Measures the fraction of the bounding box occupied by the foreground
   actor, detecting partial occlusions.
4. Fallback Hierarchy: If silhouette extraction fails or is disabled, returns None
   with explicit missing-modality metadata rather than fabricating fake masks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .contracts import Box, MaskEvidence


class MaskError(ValueError):
    """Raised when mask extraction inputs are invalid."""


def extract_person_silhouette(
    frame: np.ndarray,
    box: Box,
    frame_index: int,
    confidence: float = 0.85,
    min_visible_fraction: float = 0.05,
    mask_ref_prefix: str = "mem://masks",
) -> Optional[MaskEvidence]:
    """Extract a silhouette mask and contour feature descriptor from a detected person crop.

    Returns None if the crop is degenerate or foreground area is below min_visible_fraction,
    strictly invoking the fallback path.
    """
    if not isinstance(frame, np.ndarray) or frame.ndim != 3:
        raise MaskError(f"expected 3D numpy frame array, got {type(frame)}")

    h_img, w_img = frame.shape[:2]
    x1 = max(0, min(int(round(box.x1)), w_img - 2))
    y1 = max(0, min(int(round(box.y1)), h_img - 2))
    x2 = max(x1 + 2, min(int(round(box.x2)), w_img))
    y2 = max(y1 + 2, min(int(round(box.y2)), h_img))

    crop = frame[y1:y2, x1:x2]
    crop_h, crop_w = crop.shape[:2]
    box_area = float(crop_h * crop_w)
    if box_area <= 16.0:
        return None

    # Convert crop to grayscale and estimate foreground silhouette
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive Otsu thresholding for foreground separation
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Clean up small noise with morphological opening and closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # Calculate visible fraction (foreground pixel ratio)
    fg_pixels = int(np.count_nonzero(cleaned))
    visible_fraction = float(fg_pixels / box_area)

    # If foreground is essentially empty or 100% saturated (failed separation), reject
    if visible_fraction < min_visible_fraction or visible_fraction > 0.98:
        return None

    # Find external contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Get largest contour representing the human silhouette
    c = max(contours, key=cv2.contourArea)
    c_area = float(cv2.contourArea(c))
    c_perimeter = float(cv2.arcLength(c, closed=True))

    if c_area < 8.0 or c_perimeter < 4.0:
        return None

    # Geometric contour descriptors:
    # 1. Area ratio (contour area vs box area)
    area_ratio = float(c_area / box_area)
    # 2. Compactness / isoperimetric quotient: 4 * pi * A / P^2 (1.0 = circle, lower = jagged limbs)
    compactness = float(min(1.0, (4.0 * math.pi * c_area) / max(1.0, c_perimeter * c_perimeter)))
    # 3. Solidity: area / convex_hull_area (punches/kicks produce deep concavities, lowering solidity)
    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(c_area / max(1.0, hull_area))
    # 4. Aspect ratio of the silhouette
    _, _, cw, ch = cv2.boundingRect(c)
    c_ar = float(cw / max(1.0, ch))

    # 5. Hu Moments (shape invariants)
    moments = cv2.moments(c)
    hu = cv2.HuMoments(moments).flatten()
    # Log-transform Hu moments for numerical stability: -sign(h) * log10(|h|)
    norm_hu = []
    for h_val in hu[:4]:
        if abs(h_val) > 1e-12:
            val = float(-np.sign(h_val) * np.log10(abs(h_val)))
            # Scale to ~[0, 1] range (Hu moments typically in [1, 15])
            norm_hu.append(float(np.clip(val / 12.0, 0.0, 1.0)))
        else:
            norm_hu.append(0.0)

    # Combine into an 8-dimensional contour descriptor vector
    contour_features = [
        float(np.clip(area_ratio, 0.0, 1.0)),
        float(np.clip(compactness, 0.0, 1.0)),
        float(np.clip(solidity, 0.0, 1.0)),
        float(np.clip(c_ar / 2.0, 0.0, 1.0)),
        *norm_hu,
    ]

    mask_ref = f"{mask_ref_prefix}/f{frame_index}_b{int(box.x1)}_{int(box.y1)}"

    return MaskEvidence(
        mask_ref=mask_ref,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        visible_fraction=float(np.clip(visible_fraction, 0.0, 1.0)),
        frame_index=int(frame_index),
        contour_features=contour_features,
    )
