"""Unit tests for Phase 2: Seeded camera degradation toolkit."""

import numpy as np
import pytest

from mosaic.clarity import calculate_exposure_component, calculate_sharpness_component
from mosaic.degradations import (
    DEGRADATION_KINDS,
    DegradationError,
    apply_degradation,
    get_degradation,
)


@pytest.mark.unit
def test_all_degradations_are_identity_at_zero_severity():
    """At severity 0.0, every degradation returns an identical array."""
    rng = np.random.RandomState(1337)
    frame = rng.randint(0, 256, (120, 160, 3), dtype=np.uint8)

    for kind in DEGRADATION_KINDS:
        deg = get_degradation(kind, severity=0.0, seed=1337)
        out_frames, _ = deg.apply_clip([frame])
        assert np.array_equal(out_frames[0], frame), f"{kind} altered frame at severity 0.0"


@pytest.mark.unit
def test_degradations_are_deterministic_for_fixed_seed_and_severity():
    """Applying the same degradation twice with the same seed yields bit-exact identical output."""
    rng = np.random.RandomState(42)
    frames = [rng.randint(0, 256, (80, 80, 3), dtype=np.uint8) for _ in range(3)]
    timestamps = [0.0, 0.1, 0.2]

    for kind in DEGRADATION_KINDS:
        out1, ts1 = apply_degradation(frames, kind=kind, severity=0.6, seed=999, timestamps=timestamps)
        out2, ts2 = apply_degradation(frames, kind=kind, severity=0.6, seed=999, timestamps=timestamps)
        assert len(out1) == len(out2), f"{kind} produced different lengths"
        for f1, f2 in zip(out1, out2):
            assert np.array_equal(f1, f2), f"{kind} is non-deterministic"
        assert ts1 == ts2


@pytest.mark.unit
def test_blur_degradation_monotonically_lowers_sharpness():
    """Higher blur severity monotonically lowers Laplacian sharpness on a natural scene."""
    from mosaic.synthetic import generate_synthetic

    base = generate_synthetic("moving", seed=1337).frames[0]

    sharpnesses = []
    for sev in [0.0, 0.2, 0.5, 0.8, 1.0]:
        deg = get_degradation("blur", severity=sev, seed=1337)
        blurred = deg.apply_frame(base)
        assert blurred.shape == base.shape
        assert blurred.dtype == np.uint8
        sharpnesses.append(calculate_sharpness_component(blurred))

    for a, b in zip(sharpnesses, sharpnesses[1:]):
        assert a >= b, f"sharpness did not decrease: {sharpnesses}"
    assert sharpnesses[0] > sharpnesses[-1]


@pytest.mark.unit
def test_low_light_monotonically_lowers_luminance():
    """Higher low_light severity monotonically decreases mean luminance."""
    base = np.ones((80, 80, 3), dtype=np.uint8) * 180
    means = []
    for sev in [0.0, 0.25, 0.5, 0.75, 1.0]:
        deg = get_degradation("low_light", severity=sev, seed=1337)
        dark = deg.apply_frame(base)
        assert dark.shape == base.shape
        assert dark.dtype == np.uint8
        means.append(float(np.mean(dark)))

    for a, b in zip(means, means[1:]):
        assert a > b, f"luminance did not strictly decrease: {means}"


@pytest.mark.unit
def test_noise_monotonically_increases_variance():
    """Higher noise severity monotonically increases pixel standard deviation on uniform frame."""
    base = np.ones((100, 100, 3), dtype=np.uint8) * 128
    stds = []
    for sev in [0.0, 0.25, 0.5, 0.75, 1.0]:
        deg = get_degradation("noise", severity=sev, seed=1337)
        noisy = deg.apply_frame(base)
        assert noisy.shape == base.shape
        assert noisy.dtype == np.uint8
        stds.append(float(np.std(noisy)))

    for a, b in zip(stds, stds[1:]):
        assert b > a, f"noise standard deviation did not increase: {stds}"


@pytest.mark.unit
def test_occlusion_monotonically_increases_occluded_area():
    """Higher occlusion severity covers a larger fraction of pixels."""
    base = np.ones((100, 100, 3), dtype=np.uint8) * 200
    occluded_counts = []
    for sev in [0.1, 0.4, 0.7, 1.0]:
        deg = get_degradation("occlusion", severity=sev, seed=1337)
        occ = deg.apply_frame(base)
        # Count pixels that were changed to dark gray (20)
        changed = int(np.sum(occ[:, :, 0] < 50))
        occluded_counts.append(changed)

    for a, b in zip(occluded_counts, occluded_counts[1:]):
        assert b >= a, f"occluded area did not increase: {occluded_counts}"
    assert occluded_counts[-1] > occluded_counts[0]


@pytest.mark.unit
def test_frame_dropping_preserves_monotonic_timestamps():
    """Frame dropping removes intermediate frames while retaining timestamps of kept frames."""
    frames = [np.ones((40, 40, 3), dtype=np.uint8) * i for i in range(20)]
    timestamps = [i * 0.033 for i in range(20)]

    deg = get_degradation("frame_drop", severity=0.8, seed=42)
    dropped_frames, dropped_ts = deg.apply_clip(frames, timestamps=timestamps)

    assert len(dropped_frames) < len(frames)
    assert len(dropped_frames) >= 2
    assert len(dropped_frames) == len(dropped_ts)

    # First and last frames/timestamps are strictly preserved
    assert dropped_ts[0] == timestamps[0]
    assert dropped_ts[-1] == timestamps[-1]

    # Timestamps remain strictly increasing and match original values
    for t1, t2 in zip(dropped_ts, dropped_ts[1:]):
        assert t2 > t1
        assert t1 in timestamps and t2 in timestamps


@pytest.mark.unit
def test_invalid_severities_and_names_fail_explicitly():
    """Invalid severity (<0, >1, nan) and unknown names raise DegradationError."""
    for bad_sev in [-0.1, 1.1, float("nan"), "0.5"]:
        with pytest.raises(DegradationError):
            get_degradation("blur", severity=bad_sev)

    with pytest.raises(DegradationError, match="unknown degradation kind"):
        get_degradation("teleportation", severity=0.5)
