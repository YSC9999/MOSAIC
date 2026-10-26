"""Unit tests for seeding, environment recording, artifacts layout and logging."""

import json
import logging
import random
import subprocess
from datetime import datetime, timezone

import numpy as np
import pytest

from mosaic import environment
from mosaic.artifacts import SUBDIRS, ArtifactLayout, make_run_id
from mosaic.logging_utils import get_logger, log_event, setup_logging, shutdown_logging
from mosaic.seeding import derive_seed, make_rng, seed_everything


# ----------------------------------------------------------------- seeding
def test_seed_everything_makes_python_and_numpy_reproducible():
    seed_everything(123)
    first = ([random.random() for _ in range(5)], np.random.rand(5).tolist())
    seed_everything(123)
    second = ([random.random() for _ in range(5)], np.random.rand(5).tolist())
    assert first == second


def test_different_seeds_give_different_sequences():
    seed_everything(1)
    a = np.random.rand(5).tolist()
    seed_everything(2)
    b = np.random.rand(5).tolist()
    assert a != b


def test_derive_seed_is_stable_and_32_bit():
    value = derive_seed(42, "sampler", "train")
    assert value == derive_seed(42, "sampler", "train")
    assert 0 <= value < 2**32


def test_derive_seed_depends_on_every_input():
    assert derive_seed(1, "a") != derive_seed(2, "a")
    assert derive_seed(1, "a") != derive_seed(1, "b")
    assert derive_seed(1, "a", "b") != derive_seed(1, "ab")


def test_make_rng_reproducible_sampling():
    first = make_rng(7, "sampling").integers(0, 1000, size=10).tolist()
    second = make_rng(7, "sampling").integers(0, 1000, size=10).tolist()
    assert first == second


def test_named_rngs_are_independent():
    a = make_rng(7, "sampling").integers(0, 10**6, size=10).tolist()
    b = make_rng(7, "augmentation").integers(0, 10**6, size=10).tolist()
    assert a != b


# ------------------------------------------------------------- environment
def test_environment_record_contains_all_required_fields():
    record = environment.collect_environment()
    for name in environment.REQUIRED_FIELDS:
        assert name in record
    environment.validate_environment_record(record)
    assert record["packages"]["numpy"] is not None
    assert set(record["gpus"]) == {"available", "devices", "reason"}


def test_environment_validation_detects_missing_field():
    record = environment.collect_environment()
    del record["cpu"]
    with pytest.raises(environment.EnvironmentRecordError, match="cpu"):
        environment.validate_environment_record(record)


def test_environment_record_is_written_as_json(tmp_path):
    path = tmp_path / "sub" / "environment.json"
    written = environment.write_environment_record(path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["python_version"] == written["python_version"]
    environment.validate_environment_record(loaded)


def test_missing_gpu_tool_is_reported_not_fatal(monkeypatch):
    def boom(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(environment.subprocess, "run", boom)
    info = environment._gpu_info()
    assert info == {"available": False, "devices": [], "reason": "nvidia-smi not found"}


def test_failing_gpu_tool_is_reported(monkeypatch):
    fake = subprocess.CompletedProcess(args=[], returncode=9, stdout="", stderr="")
    monkeypatch.setattr(environment.subprocess, "run", lambda *a, **k: fake)
    info = environment._gpu_info()
    assert info["available"] is False and "9" in info["reason"]


def test_gpu_output_is_parsed(monkeypatch):
    fake = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="NVIDIA GeForce RTX 4070 Laptop GPU, 8188\n", stderr=""
    )
    monkeypatch.setattr(environment.subprocess, "run", lambda *a, **k: fake)
    info = environment._gpu_info()
    assert info["available"] is True
    assert info["devices"] == [{"name": "NVIDIA GeForce RTX 4070 Laptop GPU", "memory_total_mib": 8188}]


def test_environment_recording_does_not_import_torch():
    import sys

    before = "torch" in sys.modules
    environment.collect_environment()
    assert ("torch" in sys.modules) == before


# ---------------------------------------------------------------- artifacts
def test_layout_creates_all_standard_folders(tmp_path):
    layout = ArtifactLayout(tmp_path / "artifacts").ensure()
    for name in SUBDIRS:
        assert (tmp_path / "artifacts" / name).is_dir()
    layout.ensure()  # idempotent


def test_layout_rejects_unknown_folder(tmp_path):
    with pytest.raises(KeyError):
        ArtifactLayout(tmp_path).path("somewhere_else")


@pytest.mark.parametrize("bad", ["../escape", "a/b", "..", "", "a\\b", ".hidden"])
def test_run_dir_rejects_unsafe_ids(tmp_path, bad):
    with pytest.raises(ValueError):
        ArtifactLayout(tmp_path).ensure().run_dir(bad)


def test_run_dir_is_created_inside_runs(tmp_path):
    layout = ArtifactLayout(tmp_path).ensure()
    folder = layout.run_dir("exp-1_20260101T000000000000Z")
    assert folder.is_dir() and folder.parent == tmp_path / "runs"


def test_run_id_format_and_uniqueness():
    now = datetime(2026, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc)
    assert make_run_id("exp", now) == "exp_20260102T030405123456Z"
    later = datetime(2026, 1, 2, 3, 4, 5, 123457, tzinfo=timezone.utc)
    assert make_run_id("exp", now) != make_run_id("exp", later)


# ------------------------------------------------------------------ logging
def _our_handlers():
    return [h for h in logging.getLogger("mosaic").handlers if getattr(h, "_mosaic_handler", False)]


def test_setup_logging_is_idempotent(tmp_path):
    try:
        setup_logging("INFO", tmp_path / "a.log")
        setup_logging("INFO", tmp_path / "a.log")
        assert len(_our_handlers()) == 2  # one console + one file, never doubled
    finally:
        shutdown_logging()
    assert _our_handlers() == []


def test_log_event_writes_sorted_key_values_to_file(tmp_path):
    log_file = tmp_path / "run.log"
    try:
        setup_logging("INFO", log_file)
        log_event(get_logger("unit"), "decode_finished", frames=90, status="ok")
    finally:
        shutdown_logging()
    text = log_file.read_text(encoding="utf-8")
    assert "event=decode_finished frames=90 status=ok" in text
    assert "mosaic.unit" in text and "INFO" in text


def test_log_level_is_respected(tmp_path):
    log_file = tmp_path / "run.log"
    try:
        setup_logging("WARNING", log_file)
        get_logger("unit").info("hidden")
        get_logger("unit").warning("shown")
    finally:
        shutdown_logging()
    text = log_file.read_text(encoding="utf-8")
    assert "shown" in text and "hidden" not in text


def test_get_logger_namespaces_names():
    assert get_logger("x").name == "mosaic.x"
    assert get_logger("mosaic.y").name == "mosaic.y"
