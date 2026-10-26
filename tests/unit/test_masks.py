"""Unit tests for silhouette and mask evidence extraction (Block 4)."""

import numpy as np
import pytest

from mosaic.contracts import Box, MaskEvidence
from mosaic.masks import MaskError, extract_person_silhouette


def test_extract_silhouette_from_valid_person_crop():
    # Create a 200x200 background with a bright person-shaped rectangle in the center
    frame = np.full((200, 200, 3), 40, dtype=np.uint8)  # dark background
    # Person box: [50, 20, 150, 180] (width 100, height 160)
    # Foreground body: [70, 40, 130, 160] (bright foreground inside)
    frame[40:160, 70:130] = (220, 210, 200)

    box = Box(50.0, 20.0, 150.0, 180.0)
    evidence = extract_person_silhouette(frame, box, frame_index=12)

    assert evidence is not None
    assert isinstance(evidence, MaskEvidence)
    assert evidence.frame_index == 12
    assert 0.0 < evidence.visible_fraction < 1.0
    assert 0.0 <= evidence.confidence <= 1.0
    assert evidence.contour_features is not None
    assert len(evidence.contour_features) == 8
    assert all(0.0 <= f <= 1.0 for f in evidence.contour_features)
    assert "mem://masks" in evidence.mask_ref


def test_extract_silhouette_uniform_blank_returns_none_for_fallback():
    # Completely uniform frame -> no contrast / no separable foreground -> fallback
    frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    box = Box(10.0, 10.0, 80.0, 80.0)
    evidence = extract_person_silhouette(frame, box, frame_index=0)
    assert evidence is None


def test_extract_silhouette_tiny_degenerate_box_returns_none():
    frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    tiny_box = Box(10.0, 10.0, 12.0, 13.0)  # area = 6
    evidence = extract_person_silhouette(frame, tiny_box, frame_index=0)
    assert evidence is None


def test_mask_evidence_contract_validation():
    with pytest.raises(MaskError):
        extract_person_silhouette(np.zeros((10, 10), dtype=np.uint8), Box(1, 1, 5, 5), 0)
