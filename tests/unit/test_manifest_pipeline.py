"""Unit tests for the manifest and the pipeline skeleton."""

import hashlib
import json

import numpy as np
import pytest

from mosaic.config import Config, config_from_dict, experiment_id
from mosaic.ingestion import ArrayFrameSource, VideoReader
from mosaic.manifest import (
    REQUIRED_META,
    Manifest,
    ManifestEntry,
    ManifestError,
    build_entry,
    build_manifest,
    file_sha256,
    make_meta,
)
from mosaic.pipeline import Pipeline


def ok_entry(**overrides):
    values = dict(
        clip_path="a/clip.avi", label="violent", decode_status="ok", frame_count=150,
        fps=30.0, width=320, height=240, duration_seconds=5.0,
    )
    values.update(overrides)
    return ManifestEntry(**values)


def valid_meta():
    return make_meta(Config(), "unit-test")


# ------------------------------------------------------------ building
def test_build_entry_for_a_real_encoded_clip(encoded_videos):
    entry = build_entry(str(encoded_videos["moving"]), "non_violent")
    assert entry.decode_status == "ok" and entry.error is None
    assert (entry.frame_count, entry.fps, entry.width, entry.height) == (90, 30.0, 160, 120)
    assert entry.duration_seconds == pytest.approx(3.0)
    assert entry.label == "non_violent"


def test_corrupt_clip_becomes_a_failed_entry_not_an_exception(tmp_path):
    bad = tmp_path / "bad.avi"
    bad.write_bytes(b"junk" * 100)
    entry = build_entry(str(bad), "violent")
    assert entry.decode_status == "failed" and entry.error
    assert entry.frame_count == 0 and entry.fps is None


def test_missing_clip_becomes_a_failed_entry(tmp_path):
    entry = build_entry(str(tmp_path / "ghost.avi"), "violent")
    assert entry.decode_status == "failed" and "not found" in entry.error


def test_root_folder_is_used_to_find_clips(encoded_videos):
    path = encoded_videos["static"]
    entry = build_entry(path.name, "violent", root=path.parent)
    assert entry.decode_status == "ok" and entry.clip_path == "static.avi"


def test_hash_is_recorded_on_request(encoded_videos):
    path = encoded_videos["static"]
    entry = build_entry(str(path), "violent", compute_hash=True)
    assert entry.sha256 == hashlib.sha256(path.read_bytes()).hexdigest() == file_sha256(path)
    assert build_entry(str(path), "violent").sha256 is None


def test_build_manifest_summary_and_failed_list(encoded_videos, tmp_path):
    bad = tmp_path / "bad.avi"
    bad.write_bytes(b"x")
    manifest = build_manifest(
        [(str(encoded_videos["static"]), "violent"), (str(encoded_videos["burst"]), "non_violent"), (str(bad), "violent")],
        valid_meta(),
    )
    assert manifest.summary() == {
        "total": 3,
        "by_status": {"ok": 2, "failed": 1},
        "by_label": {"violent": 2, "non_violent": 1},
    }
    assert [e.clip_path for e in manifest.failed_entries()] == [bad.as_posix()]


# ------------------------------------------------------ entry validation
@pytest.mark.parametrize(
    "overrides",
    [
        {"label": "goon"},
        {"label": ""},
        {"split": "holdout"},
        {"decode_status": "great"},
        {"frame_count": 0},  # ok entry needs frames
        {"frame_count": -3},
        {"fps": 0},
        {"fps": None},
        {"fps": float("nan")},
        {"width": 0},
        {"height": None},
        {"duration_seconds": 0},
        {"duration_seconds": None},
        {"error": "but it is ok?"},
        {"sha256": "XYZ"},
        {"clip_path": " "},
    ],
)
def test_invalid_ok_entries_are_rejected(overrides):
    with pytest.raises(ManifestError):
        ok_entry(**overrides)


def test_failed_entry_must_explain_itself():
    with pytest.raises(ManifestError):
        ManifestEntry("c.avi", "violent", decode_status="failed", frame_count=0)
    entry = ManifestEntry("c.avi", "violent", decode_status="failed", frame_count=0, error="cannot open")
    assert entry.fps is None


def test_valid_split_and_hash_accepted():
    entry = ok_entry(split="train", sha256="a" * 64)
    assert entry.split == "train"


# ------------------------------------------------------ manifest validation
def test_meta_records_window_policy_from_config():
    cfg = config_from_dict({"window": {"policy": "sliding", "window_seconds": 4, "stride_seconds": 2, "clip_aggregation": "max"}})
    meta = make_meta(cfg, "RWF-2000")
    assert meta["window_policy"] == {"policy": "sliding", "window_seconds": 4.0, "stride_seconds": 2.0}
    assert meta["clip_aggregation"] == "max" and meta["dataset_name"] == "RWF-2000"
    assert all(key in meta for key in REQUIRED_META)


@pytest.mark.parametrize("missing", REQUIRED_META)
def test_manifest_requires_each_metadata_key(missing):
    meta = valid_meta()
    del meta[missing]
    with pytest.raises(ManifestError, match=missing):
        Manifest(meta=meta, entries=[])


def test_manifest_rejects_unknown_schema_version():
    meta = valid_meta()
    meta["schema_version"] = "manifest-99"
    with pytest.raises(ManifestError, match="schema"):
        Manifest(meta=meta, entries=[])


def test_duplicate_clip_paths_are_rejected():
    with pytest.raises(ManifestError, match="duplicate"):
        Manifest(valid_meta(), [ok_entry(), ok_entry()])


def test_manifest_roundtrip_through_disk(tmp_path):
    manifest = Manifest(valid_meta(), [ok_entry(), ok_entry(clip_path="b.avi", label="non_violent", split="val")])
    path = manifest.write(tmp_path / "m" / "manifest.json")
    assert Manifest.read(path).to_dict() == manifest.to_dict()


def test_parsing_reports_every_bad_entry_with_its_index():
    good = ok_entry().to_dict()
    bad_label = ok_entry(clip_path="b.avi").to_dict()
    bad_label["label"] = "goon"
    bad_fps = ok_entry(clip_path="c.avi").to_dict()
    bad_fps["fps"] = 0
    data = {"meta": valid_meta(), "entries": [bad_label, good, bad_fps]}
    with pytest.raises(ManifestError) as info:
        Manifest.from_dict(data)
    message = str(info.value)
    assert "entries[0]" in message and "entries[2]" in message and "entries[1]" not in message


def test_parsing_rejects_unknown_and_missing_fields():
    entry = ok_entry().to_dict()
    with pytest.raises(ManifestError, match="unknown"):
        Manifest.from_dict({"meta": valid_meta(), "entries": [{**entry, "extra": 1}]})
    del entry["label"]
    with pytest.raises(ManifestError, match="missing"):
        Manifest.from_dict({"meta": valid_meta(), "entries": [entry]})


@pytest.mark.parametrize("data", [[], {"meta": {}}, {"entries": []}, {"meta": {}, "entries": "no"}])
def test_parsing_rejects_wrong_structure(data):
    with pytest.raises(ManifestError):
        Manifest.from_dict(data)


def test_parsing_rejects_non_mapping_entries():
    with pytest.raises(ManifestError):
        Manifest.from_dict({"meta": valid_meta(), "entries": [42]})


def test_reading_missing_or_invalid_json_fails_explicitly(tmp_path):
    with pytest.raises(ManifestError, match="not found"):
        Manifest.read(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ManifestError, match="not valid JSON"):
        Manifest.read(bad)


def test_manifest_can_hold_ok_and_failed_entries_together(tmp_path):
    failed = ManifestEntry("x.avi", "violent", decode_status="empty", frame_count=0, error="no frames")
    manifest = Manifest(valid_meta(), [ok_entry(), failed])
    reread = Manifest.read(manifest.write(tmp_path / "m.json"))
    assert [e.decode_status for e in reread.entries] == ["ok", "empty"]


# ---------------------------------------------------------------- pipeline
def test_pipeline_instantiates_from_default_config():
    pipeline = Pipeline.from_config(Config())
    assert pipeline.stages == ["ingestion"]


def test_pipeline_run_reports_ok_diagnostics(synthetic_videos):
    cfg = Config()
    diag = Pipeline.from_config(cfg).run(synthetic_videos["burst"].frame_source())
    assert diag.status == "ok" and diag.processed_frame_count == 90
    assert diag.frame_shape == [120, 160, 3]
    assert diag.experiment_id == experiment_id(cfg)
    assert diag.decode.decoded_frame_count == 90
    json.dumps(diag.to_dict())  # must be JSON-serialisable


def test_pipeline_reports_decode_failure_instead_of_raising(tmp_path):
    diag = Pipeline.from_config(Config()).run(VideoReader(tmp_path / "missing.avi"))
    assert diag.status == "failed" and diag.processed_frame_count == 0
    assert diag.decode.errors


def test_pipeline_reports_empty_source():
    diag = Pipeline.from_config(Config()).run(ArrayFrameSource([], fps=30.0))
    assert diag.status == "empty" and diag.processed_frame_count == 0 and diag.frame_shape is None


def test_pipeline_reports_partial_decode():
    frames = [np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8, 3), np.float32)]
    diag = Pipeline.from_config(Config()).run(ArrayFrameSource(frames, fps=30.0))
    assert diag.status == "partial" and diag.processed_frame_count == 1
