import json

import cv2
import numpy as np
import pytest
import yaml

from mosaic.datasets import (DatasetError, build_manifest, class_target, discover_clips, group_key, load_dataset_config)
from mosaic.windows import sample_indices, window_starts


def _video(path, n_frames, seed=0, fps=25.0, size=(64, 48)):
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    rng = np.random.default_rng(seed)
    for _ in range(n_frames):
        vw.write(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    vw.release()


def _cfg(tmp_path, **over):
    raw = {"name": "toy", "root": str(tmp_path / "data"), "classes": {"positive": ["Violence"], "negative": ["Normal"]}}
    raw.update(over)
    f = tmp_path / "toy.yaml"
    f.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_dataset_config(f)


def test_class_only_layout_labels_groups_and_seeded_split(tmp_path):
    n = 0
    for cls in ("Violence", "Normal"):
        for src in range(10):
            for part in range(2):
                _video(tmp_path / "data" / cls / f"{cls}{src}_{part}.avi", 20 + 3 * part, seed=n)
                n += 1
    cfg = _cfg(tmp_path, layout="class_only", group="suffix_number", test_fraction=0.25)
    man = build_manifest(cfg)
    es = man["entries"]
    assert len(es) == 40 and {e["target"] for e in es} == {0, 1}
    assert all(e["label"] in ("Violence", "Normal") for e in es)
    assert {e["split"] for e in es} == {"train", "test"}
    by = {}
    for e in es:
        by.setdefault(e["group"], set()).add(e["split"])
    assert all(len(v) == 1 for v in by.values())                     # clips of one source never straddle the split
    assert man["meta"]["counts"]["test/positive"] > 0 and man["meta"]["counts"]["test/negative"] > 0
    assert build_manifest(cfg)["entries"][0]["split"] == es[0]["split"]       # seeded: deterministic


def test_split_class_layout_uses_official_folders_and_drops_duplicates(tmp_path):
    for i in range(3):
        _video(tmp_path / "data" / "train" / "Violence" / f"a{i}.avi", 20, seed=i)
        _video(tmp_path / "data" / "train" / "Normal" / f"b{i}.avi", 20, seed=10 + i)
    _video(tmp_path / "data" / "val" / "Violence" / "t0.avi", 20, seed=0)             # identical to train/Violence/a0.avi
    _video(tmp_path / "data" / "val" / "Normal" / "t1.avi", 20, seed=99)
    cfg = _cfg(tmp_path, layout="split_class", split_dirs={"train": ["train"], "test": ["val"]}, group="none")
    man = build_manifest(cfg)
    split_of = {e["rel_path"]: e["split"] for e in man["entries"]}
    assert split_of["val/Violence/t0.avi"] == "test"                     # the test copy is kept ...
    assert "train/Violence/a0.avi" not in split_of                       # ... and the duplicate is removed from train
    assert man["meta"]["removed_exact_duplicate_count"] == 1


def test_clip_lengths_and_rates_do_not_matter_for_the_window_layout(tmp_path):
    for k, (n, fps) in enumerate([(10, 10.0), (41, 25.0), (64, 30.0), (200, 15.0)]):
        _video(tmp_path / "data" / "Violence" / f"v{k}.avi", n, seed=k, fps=fps)
        _video(tmp_path / "data" / "Normal" / f"n{k}.avi", n, seed=50 + k, fps=fps)
    man = build_manifest(_cfg(tmp_path, layout="class_only", group="none", test_fraction=0.25))
    assert len(man["entries"]) == 8
    for e in man["entries"]:
        idx, pad = sample_indices(e["frame_count"])
        assert len(idx) == 64 and len(window_starts(64)) == 7
        assert sum(not p for p in pad) == min(e["frame_count"], 64)


def test_bad_configs_and_folders_give_clear_errors(tmp_path):
    with pytest.raises(DatasetError, match="missing required key"):
        f = tmp_path / "bad.yaml"
        f.write_text(yaml.safe_dump({"name": "x"}), encoding="utf-8")
        load_dataset_config(f)
    with pytest.raises(DatasetError, match="unknown keys"):
        _cfg(tmp_path, colour="red")
    with pytest.raises(DatasetError, match="layout"):
        _cfg(tmp_path, layout="weird")
    with pytest.raises(DatasetError, match="root not found"):
        discover_clips(_cfg(tmp_path))
    _video(tmp_path / "data" / "Mystery" / "x.avi", 20)
    with pytest.raises(DatasetError, match="neither the positive"):
        discover_clips(_cfg(tmp_path, layout="class_only"))


def test_helpers():
    assert group_key("train/Fight/abc_12.avi", "suffix_number") == "abc"
    assert group_key("train/Fight/abc_12.avi", "none") == "train/Fight/abc_12.avi"
    cfg_pos = type("C", (), {"positive": ["Fight"], "negative": ["NonFight"]})
    assert class_target(cfg_pos, "fight") == 1 and class_target(cfg_pos, "NONFIGHT") == 0


def test_short_and_long_clips_give_the_same_frame_and_window_shapes(tmp_path):
    from mosaic.motion import gray_frames, window_motion
    from mosaic.scene_cache import frames_to_windows, read_clip_frames

    for n in (12, 41, 64, 120):
        v = tmp_path / f"c{n}.avi"
        _video(v, n, seed=n, size=(96, 72))
        for kw in ({}, {"n_frames": n}):                                   # streaming and read-all paths agree
            frames, pad = read_clip_frames(v, **kw)
            assert frames.shape == (64, 224, 224, 3) and len(pad) == 64 and sum(not p for p in pad) == min(n, 64)
            assert frames_to_windows(frames).shape == (7, 3, 16, 224, 224)
            g = gray_frames(v, **kw)
            assert g.shape == (64, 112, 112) and window_motion(g).shape == (7,)
        if n < 64:                                                          # padded frames are exact copies of the last real one
            f, p = read_clip_frames(v, n_frames=n)
            assert all(np.array_equal(f[i], f[n - 1]) for i in range(n, 64))
