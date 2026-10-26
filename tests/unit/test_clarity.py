"""Unit tests for Phase 2: Clarity estimation and component calculation."""

import math
import numpy as np
import pytest

from mosaic.clarity import (
    ClarityError,
    calculate_compression_component,
    calculate_exposure_component,
    calculate_resolution_component,
    calculate_sharpness_component,
    combine_clarity_components,
    compute_frame_clarity,
    compute_window_clarity,
    infer_source_profile,
)
from mosaic.contracts import ClarityReport, ContractError


@pytest.mark.unit
def test_blank_and_uniform_frames_receive_low_sharpness():
    """Completely uniform black, white, and gray frames have 0.0 sharpness."""
    black = np.zeros((120, 160, 3), dtype=np.uint8)
    white = np.ones((120, 160, 3), dtype=np.uint8) * 255
    gray = np.ones((120, 160), dtype=np.uint8) * 128

    assert calculate_sharpness_component(black) == 0.0
    assert calculate_sharpness_component(white) == 0.0
    assert calculate_sharpness_component(gray) == 0.0


@pytest.mark.unit
def test_sharpness_increases_with_edge_detail():
    """Adding high-contrast edges and texture monotonically increases sharpness."""
    rng = np.random.RandomState(42)
    smooth = np.ones((120, 160, 3), dtype=np.uint8) * 120
    # Add gentle gradient
    grad = smooth.copy()
    grad[:, :, 0] = np.tile(np.linspace(100, 140, 160, dtype=np.uint8), (120, 1))

    # Add strong high-frequency checkerboard texture
    textured = smooth.copy()
    textured[::4, ::4] = 240
    textured[1::4, 1::4] = 20

    s_smooth = calculate_sharpness_component(smooth)
    s_grad = calculate_sharpness_component(grad)
    s_text = calculate_sharpness_component(textured)

    assert s_smooth == 0.0
    assert s_grad > s_smooth
    assert s_text > s_grad
    assert 0.0 <= s_text <= 1.0


@pytest.mark.unit
def test_resolution_component_behavior():
    """Resolution component normalizes against 720p and scales monotonically."""
    r_1080 = calculate_resolution_component(1920, 1080)
    r_720 = calculate_resolution_component(1280, 720)
    r_480 = calculate_resolution_component(640, 480)
    r_240 = calculate_resolution_component(320, 240)

    assert r_1080 == 1.0  # Capped at target
    assert r_720 == pytest.approx(1.0)
    assert r_720 > r_480 > r_240 > 0.0
    assert r_480 == pytest.approx(0.54, abs=0.03)

    # Invalid dimensions fail explicitly
    with pytest.raises(ClarityError):
        calculate_resolution_component(-100, 720)
    with pytest.raises(ClarityError):
        calculate_resolution_component(1280, 0)


@pytest.mark.unit
def test_exposure_component_penalizes_extremes():
    """Balanced midtones achieve high exposure; pitch-black and washed-out receive low scores."""
    balanced = np.ones((100, 100, 3), dtype=np.uint8) * 128
    dark = np.ones((100, 100, 3), dtype=np.uint8) * 10
    pitch_black = np.zeros((100, 100, 3), dtype=np.uint8)
    washed_out = np.ones((100, 100, 3), dtype=np.uint8) * 255

    e_bal = calculate_exposure_component(balanced)
    e_dark = calculate_exposure_component(dark)
    e_pitch = calculate_exposure_component(pitch_black)
    e_wash = calculate_exposure_component(washed_out)

    assert e_bal > 0.95
    assert e_bal > e_dark > e_pitch
    assert e_pitch == pytest.approx(0.0, abs=1e-3)
    assert e_wash == pytest.approx(0.0, abs=1e-3)


@pytest.mark.unit
def test_compression_component_penalizes_blockiness():
    """Frames with prominent 8x8 block grid discontinuities receive lower scores."""
    from mosaic.synthetic import generate_synthetic

    clean = generate_synthetic("moving", seed=42).frames[0]
    blocked = clean.copy()
    # Artificially inject severe 8x8 block edge discontinuities
    blocked[:, 7::8] = 255
    blocked[:, 8::8] = 0

    c_clean = calculate_compression_component(clean)
    c_blocked = calculate_compression_component(blocked)

    assert c_clean > 0.90
    assert c_clean > c_blocked
    assert 0.0 <= c_blocked <= 1.0


@pytest.mark.unit
def test_nan_and_infinite_values_fail_explicitly():
    """Frames containing NaN or infinity must raise ClarityError."""
    bad_frame = np.ones((64, 64), dtype=np.float32)
    bad_frame[10, 10] = float("nan")

    with pytest.raises(ClarityError, match="NaN or infinite"):
        calculate_sharpness_component(bad_frame)
    with pytest.raises(ClarityError, match="NaN or infinite"):
        compute_frame_clarity(bad_frame)


@pytest.mark.unit
def test_clarity_score_combines_available_components_and_normalizes():
    """Score remains strictly in [0.0, 1.0] and renormalizes over active weights."""
    score_full = combine_clarity_components(
        resolution=1.0, sharpness=0.8, exposure=0.9, compression=0.95
    )
    assert 0.0 <= score_full <= 1.0
    assert score_full > 0.85

    # Missing person scale does not deflate score: weights are renormalized
    score_no_ps = combine_clarity_components(
        resolution=0.5, sharpness=0.5, exposure=0.5, compression=0.5, person_scale=None
    )
    assert score_no_ps == pytest.approx(0.5)

    # Empty components return 0.0
    assert combine_clarity_components() == 0.0


@pytest.mark.unit
def test_compute_frame_and_window_clarity_roundtrip():
    """Full frame and window evaluation produce valid ClarityReport contracts."""
    f1 = np.ones((240, 320, 3), dtype=np.uint8) * 110
    f1[::8, ::8] = 220
    f2 = f1.copy()

    rep1 = compute_frame_clarity(f1)
    assert isinstance(rep1, ClarityReport)
    assert rep1.source_profile == "240p"
    assert rep1.person_scale_component is None
    assert "person_scale_component" in rep1.unavailable_components
    assert 0.0 <= rep1.score <= 1.0

    win_rep = compute_window_clarity([f1, f2], aggregation="mean")
    assert isinstance(win_rep, ClarityReport)
    assert win_rep.score == pytest.approx(rep1.score)
    assert win_rep.to_dict()["source_profile"] == "240p"


@pytest.mark.unit
def test_source_profile_inference():
    """Profile labels map expected resolution categories."""
    assert infer_source_profile(1920, 1080) == "1080p"
    assert infer_source_profile(1280, 720) == "720p"
    assert infer_source_profile(640, 480) == "480p"
    assert infer_source_profile(320, 240) == "240p"
    assert infer_source_profile(120, 90) == "sub_240p"
