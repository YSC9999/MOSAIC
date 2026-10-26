"""Unit tests for Phase 1 dataset ingestion, split carver, and frame sampling."""

from pathlib import Path

import numpy as np
import pytest
import torch

from mosaic.dataset import (
    DataLeakageError,
    DatasetError,
    RWFDataset,
    RWFEntry,
    RWFManifest,
    compute_frame_signature,
    scan_rwf_directory,
)
from mosaic.sampling import (
    SamplingError,
    compute_time_deltas,
    pad_or_truncate_frames,
    sample_frame_indices,
    slice_window_frame_ranges,
)
from mosaic.config import WindowConfig
from mosaic.synthetic_rwf import create_mock_rwf_dataset


@pytest.mark.unit
def test_sample_frame_indices_bounds():
    """Sampling must never request indices outside [0, total_frames - 1]."""
    for total in [1, 2, 5, 16, 50, 150]:
        for num in [1, 4, 16, 32]:
            indices = sample_frame_indices(total, num, strategy="uniform")
            assert len(indices) == num
            assert all(0 <= idx < total for idx in indices)
            # Monotonicity check
            for a, b in zip(indices, indices[1:]):
                assert b >= a

            stride_indices = sample_frame_indices(total, num, strategy="stride")
            assert len(stride_indices) == num
            assert all(0 <= idx < total for idx in stride_indices)


@pytest.mark.unit
def test_sample_frame_indices_invalid():
    """Negative or zero frames must fail explicitly."""
    with pytest.raises(SamplingError):
        sample_frame_indices(0, 16)
    with pytest.raises(SamplingError):
        sample_frame_indices(10, 0)
    with pytest.raises(SamplingError):
        sample_frame_indices(10, 16, strategy="unknown_strategy")


@pytest.mark.unit
def test_pad_or_truncate_determinism():
    """Padding and truncation must be deterministic and report accurate validity masks."""
    frames = [np.ones((10, 10, 3), dtype=np.uint8) * i for i in range(5)]

    # Truncate
    trunc, mask = pad_or_truncate_frames(frames, 3)
    assert len(trunc) == 3
    assert len(mask) == 3
    assert all(mask)
    assert np.array_equal(trunc[0], frames[0])
    assert np.array_equal(trunc[2], frames[2])

    # Pad
    padded, mask_p = pad_or_truncate_frames(frames, 8, pad_mode="edge")
    assert len(padded) == 8
    assert len(mask_p) == 8
    assert mask_p == [True, True, True, True, True, False, False, False]
    assert np.array_equal(padded[4], frames[4])
    assert np.array_equal(padded[5], frames[4])  # edge repetition
    assert np.array_equal(padded[7], frames[4])


@pytest.mark.unit
def test_time_deltas_and_window_slicing():
    """Time deltas mark step 0 and padded frames as 0.0; window slicing uses timestamps."""
    ts = [0.0, 0.1, 0.25, 0.35, 0.45]
    mask = [True, True, True, False, False]
    deltas = compute_time_deltas(ts, mask)
    assert deltas == pytest.approx([0.0, 0.1, 0.15, 0.0, 0.0])

    # Sliding window on timestamps (0.0 to 1.0s at 10 fps)
    long_ts = [i * 0.1 for i in range(11)]
    whole = slice_window_frame_ranges(long_ts, WindowConfig(policy="whole_clip"))
    assert len(whole) == 1 and whole[0][:2] == (0, 11)

    sliding = slice_window_frame_ranges(
        long_ts,
        WindowConfig(policy="sliding", window_seconds=0.5, stride_seconds=0.5),
    )
    assert len(sliding) == 2
    assert sliding[0][2] == pytest.approx(0.0) and sliding[0][3] == pytest.approx(0.5)
    assert sliding[1][2] == pytest.approx(0.5) and sliding[1][3] == pytest.approx(1.0)


@pytest.mark.unit
def test_rwf_entry_label_validation():
    """Dataset labels map only to the expected two classes ('Fight', 'NonFight')."""
    e1 = RWFEntry("c1", "abs/1", "rel/1", "Fight", 1, "train", "a" * 64, 100, 30.0, 3.33, 224, 224)
    e2 = RWFEntry("c2", "abs/2", "rel/2", "NonFight", 0, "val_carved", "b" * 64, 100, 30.0, 3.33, 224, 224)
    assert e1.target == 1 and e1.canonical_label == "violent"
    assert e2.target == 0 and e2.canonical_label == "non_violent"
    assert e1.to_manifest_entry().label == "violent"
    assert e2.to_manifest_entry().split == "val"

    # Invalid label
    with pytest.raises(DatasetError, match="invalid label"):
        RWFEntry("c3", "abs/3", "rel/3", "Assault", 1, "train", "sha3", 100, 30.0, 3.33, 224, 224)

    # Invalid target
    with pytest.raises(DatasetError, match="target must be 0 or 1"):
        RWFEntry("c4", "abs/4", "rel/4", "Fight", 2, "train", "sha4", 100, 30.0, 3.33, 224, 224)

    # Invalid split
    with pytest.raises(DatasetError, match="invalid split"):
        RWFEntry("c5", "abs/5", "rel/5", "Fight", 1, "custom_split", "sha5", 100, 30.0, 3.33, 224, 224)


@pytest.mark.unit
def test_pairwise_disjoint_check_catches_leakage():
    """Anti-leakage check raises DataLeakageError if clip names, hashes, or signatures collide."""
    e_train = RWFEntry("clip_01.avi", "p1", "r1", "Fight", 1, "train", "hash_A", 10, 30.0, 0.3, 64, 64, frame_signature="sig_1")
    e_val_ok = RWFEntry("clip_02.avi", "p2", "r2", "NonFight", 0, "val_carved", "hash_B", 10, 30.0, 0.3, 64, 64, frame_signature="sig_2")
    e_test_ok = RWFEntry("clip_03.avi", "p3", "r3", "Fight", 1, "test", "hash_C", 10, 30.0, 0.3, 64, 64, frame_signature="sig_3")

    valid_manifest = RWFManifest(entries=[e_train, e_val_ok, e_test_ok])
    valid_manifest.verify_disjoint(check_near_duplicates=True)
    assert valid_manifest.near_duplicates == []

    # Name collision between train and test
    e_test_collision = RWFEntry("clip_01.avi", "p3", "r3", "Fight", 1, "test", "hash_X", 10, 30.0, 0.3, 64, 64)
    bad_manifest = RWFManifest(entries=[e_train, e_val_ok, e_test_collision])
    with pytest.raises(DataLeakageError, match="Clip name overlap"):
        bad_manifest.verify_disjoint()

    # Hash collision between train and val (different name, identical file hash)
    e_val_hash_collision = RWFEntry("clip_99.avi", "p4", "r4", "Fight", 1, "val_carved", "hash_A", 10, 30.0, 0.3, 64, 64)
    bad_hash_manifest = RWFManifest(entries=[e_train, e_val_hash_collision, e_test_ok])
    with pytest.raises(DataLeakageError, match="SHA-256 hash collision"):
        bad_hash_manifest.verify_disjoint()

    # Near-duplicate frame signature collision between train and test
    e_test_sig_collision = RWFEntry("clip_04.avi", "p5", "r5", "Fight", 1, "test", "hash_D", 10, 30.0, 0.3, 64, 64, frame_signature="sig_1")
    near_manifest = RWFManifest(entries=[e_train, e_val_ok, e_test_sig_collision])
    near_manifest.verify_disjoint(check_near_duplicates=False)
    assert len(near_manifest.near_duplicates) == 1
    with pytest.raises(DataLeakageError, match="Near-duplicate frame signatures"):
        near_manifest.verify_disjoint(check_near_duplicates=True)


@pytest.mark.unit
def test_scan_rwf_directory_stratification_and_corrupt_reporting(tmp_path: Path):
    """Scan creates a disjoint manifest with window metadata and reports corrupt clips explicitly."""
    mock_dir = tmp_path / "mock_rwf"
    create_mock_rwf_dataset(mock_dir, clips_per_class=4, num_frames=10, seed=42)

    # Inject a corrupt clip into train/Fight
    corrupt_file = mock_dir / "train" / "Fight" / "corrupt_99.avi"
    corrupt_file.write_bytes(b"not a valid video file")

    manifest = scan_rwf_directory(mock_dir, val_carve_ratio=0.25, seed=42)
    assert len(manifest) == 16  # 16 valid clips; corrupt clip is NOT fabricated as frame_count=1
    assert len(manifest.corrupt_entries) == 1
    assert manifest.corrupt_entries[0]["clip_id"] == "corrupt_99.avi"
    assert manifest.corrupt_entries[0]["decode_status"] == "failed"
    assert manifest.meta["window_policy"]["policy"] == "whole_clip"
    assert manifest.meta["corrupt_count"] == 1

    # Roundtrip save/load preserves metadata and corrupt entries
    saved_path = tmp_path / "manifest.json"
    manifest.save(saved_path)
    reloaded = RWFManifest.load(saved_path)
    assert len(reloaded) == 16
    assert len(reloaded.corrupt_entries) == 1
    assert reloaded.meta["window_policy"]["policy"] == "whole_clip"

    # Converts cleanly to Phase 0 Manifest contract (16 ok + 1 failed = 17 total)
    p0_manifest = reloaded.to_phase0_manifest()
    assert p0_manifest.summary()["by_status"] == {"ok": 16, "failed": 1}


@pytest.mark.unit
def test_short_clip_padding_and_train_transform_isolation(tmp_path: Path):
    """Short clips are padded with accurate validity_mask; eval never applies train transforms."""
    mock_dir = tmp_path / "mock_rwf"
    create_mock_rwf_dataset(mock_dir, clips_per_class=1, num_frames=6, seed=123)
    manifest = scan_rwf_directory(mock_dir, val_carve_ratio=0.5, seed=123)

    val_entries = manifest.entries_for_split("val_carved")
    ds_eval = RWFDataset(
        val_entries,
        num_frames=10,  # request 10 frames from a 6-frame clip -> 4 frames padded
        target_size=(32, 32),
        is_train=False,
        horizontal_flip_prob=0.0,
    )
    sample1 = ds_eval[0]
    sample2 = ds_eval[0]
    assert torch.equal(sample1["frames"], sample2["frames"])
    assert sample1["validity_mask"].tolist() == [True] * 6 + [False] * 4
    # Padded frames have 0.0 time delta; valid steps > 0 have positive time delta
    deltas = sample1["time_deltas"].tolist()
    assert deltas[0] == 0.0
    assert all(d > 0.0 for d in deltas[1:6])
    assert all(d == 0.0 for d in deltas[6:])
