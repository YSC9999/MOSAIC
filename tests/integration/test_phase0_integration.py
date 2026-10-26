"""Integration tests: several modules working together (Phase 0 exit gate).

These use only fixtures (synthetic videos / small encoded files) - no dataset,
no model weights, no network.
"""

import json
import os
import subprocess
import sys

import pytest

from mosaic.config import load_config
from mosaic.contracts import DecodeDiagnostics, WindowDiagnostics, WindowOutput
from mosaic.environment import validate_environment_record
from mosaic.ingestion import VideoReader
from mosaic.manifest import Manifest, build_manifest, make_meta
from mosaic.pipeline import Pipeline
from mosaic.synthetic import SCENARIOS


def run_cli(project_root, *args):
    return subprocess.run(
        [sys.executable, "-m", "mosaic", *args],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )


def run_folders(artifacts_dir):
    return sorted((artifacts_dir / "runs").iterdir())


def key_structure(value):
    """The shape of a JSON-like value: keys and nesting, ignoring the actual values."""
    if isinstance(value, dict):
        return {k: key_structure(v) for k, v in value.items()}
    if isinstance(value, list):
        return "list"
    return "scalar"


# ------------------------------------------------- tiny video -> manifest
def test_encoded_fixture_is_decoded_and_represented_in_a_manifest(tmp_path, encoded_videos):
    clips = [(str(encoded_videos[name]), "violent" if i % 2 else "non_violent") for i, name in enumerate(SCENARIOS)]
    manifest = build_manifest(clips, make_meta(load_config(None), "fixtures"), compute_hash=True)
    assert manifest.summary()["by_status"] == {"ok": len(SCENARIOS)}
    assert all(e.frame_count == 90 and e.fps == 30.0 for e in manifest.entries)
    assert len({e.sha256 for e in manifest.entries}) == len(SCENARIOS)  # clips differ

    reloaded = Manifest.read(manifest.write(tmp_path / "manifest.json"))
    assert reloaded.to_dict() == manifest.to_dict()
    assert reloaded.meta["window_policy"]["policy"] == "whole_clip"


def test_manifest_frame_count_agrees_with_what_the_pipeline_processes(encoded_videos):
    path = encoded_videos["occlusion"]
    manifest = build_manifest([(str(path), "violent")], make_meta(load_config(None), "fixtures"))
    diag = Pipeline.from_config(load_config(None)).run(VideoReader(path))
    assert manifest.entries[0].frame_count == diag.processed_frame_count == 90


# ------------------------------------------- minimal pipeline, no weights
def test_minimal_pipeline_needs_no_model_libraries(project_root):
    code = (
        "import sys\n"
        "from mosaic.config import load_config\n"
        "from mosaic.pipeline import Pipeline\n"
        "from mosaic.synthetic import generate_synthetic\n"
        "cfg = load_config('configs/default.yaml')\n"
        "diag = Pipeline.from_config(cfg).run(generate_synthetic('static').frame_source())\n"
        "assert diag.status == 'ok', diag\n"
        "assert 'torch' not in sys.modules, 'PyTorch was imported!'\n"
        "assert 'torchvision' not in sys.modules\n"
        "print('NO-TORCH-OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=project_root, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert "NO-TORCH-OK" in result.stdout


# ---------------------------------------- synthetic vs file: same schema
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_synthetic_and_file_sources_produce_the_same_diagnostics_schema(scenario, synthetic_videos, encoded_videos):
    pipeline = Pipeline.from_config(load_config(None))
    from_array = pipeline.run(synthetic_videos[scenario].frame_source()).to_dict()
    from_file = pipeline.run(VideoReader(encoded_videos[scenario])).to_dict()
    assert key_structure(from_array) == key_structure(from_file)
    assert from_array["status"] == from_file["status"] == "ok"
    assert from_array["processed_frame_count"] == from_file["processed_frame_count"]
    assert from_array["decode"]["source_kind"] == "synthetic"
    assert from_file["decode"]["source_kind"] == "file"


# --------------------------------------------- contracts <-> ingestion
def test_real_decode_diagnostics_fit_inside_a_window_output(encoded_videos):
    reader = VideoReader(encoded_videos["burst"])
    reader.read_all()
    window = WindowOutput(
        window_id="burst-w0",
        source_id=reader.diagnostics.source_id,
        start_timestamp=0.0,
        end_timestamp=reader.diagnostics.duration_seconds,
        label="non_violent",
        violence_score=0.1,
        temporal_confidence=0.5,
        clarity_score=0.9,
        processed_frame_count=90,
        skipped_frame_count=0,
        diagnostics=WindowDiagnostics(selected_frame_indices=list(range(90)), decode=reader.diagnostics),
    )
    again = WindowOutput.from_json(window.to_json())
    assert again == window
    assert isinstance(again.diagnostics.decode, DecodeDiagnostics)


# ------------------------------------------------------ smoke command (CLI)
def test_smoke_command_on_a_synthetic_video(tmp_path, project_root):
    art = tmp_path / "art"
    result = run_cli(project_root, "smoke", "--synthetic", "burst", "--set", f"paths.artifacts_dir={art}")
    assert result.returncode == 0, result.stderr
    assert "status=ok" in result.stdout
    (run,) = run_folders(art)
    for name in ("diagnostics.json", "environment.json", "config.yaml", "run.log"):
        assert (run / name).is_file(), name
    diagnostics = json.loads((run / "diagnostics.json").read_text())
    assert diagnostics["status"] == "ok" and diagnostics["processed_frame_count"] == 90
    validate_environment_record(json.loads((run / "environment.json").read_text()))
    assert load_config(run / "config.yaml") == load_config(None, [f"paths.artifacts_dir={art}"])
    assert "event=run_finished" in (run / "run.log").read_text(encoding="utf-8")


def test_smoke_command_on_a_video_file(tmp_path, project_root, encoded_videos):
    art = tmp_path / "art"
    result = run_cli(project_root, "smoke", "--video", str(encoded_videos["moving"]), "--set", f"paths.artifacts_dir={art}")
    assert result.returncode == 0, result.stderr
    (run,) = run_folders(art)
    diagnostics = json.loads((run / "diagnostics.json").read_text())
    assert diagnostics["decode"]["source_kind"] == "file" and diagnostics["decode"]["fps"] == 30.0


def test_smoke_command_reports_a_missing_video_with_exit_code_2(tmp_path, project_root):
    art = tmp_path / "art"
    result = run_cli(project_root, "smoke", "--video", str(tmp_path / "ghost.avi"), "--set", f"paths.artifacts_dir={art}")
    assert result.returncode == 2
    assert "DECODE PROBLEM" in result.stderr
    (run,) = run_folders(art)
    assert json.loads((run / "diagnostics.json").read_text())["status"] == "failed"


def test_smoke_command_reports_bad_config_with_exit_code_1(tmp_path, project_root):
    art = tmp_path / "art"
    result = run_cli(project_root, "smoke", "--synthetic", "static", "--set", "window.policy=bogus", "--set", f"paths.artifacts_dir={art}")
    assert result.returncode == 1
    assert "CONFIG ERROR" in result.stderr and "window.policy" in result.stderr
    assert not art.exists()  # nothing is written for an invalid configuration


def test_smoke_runs_are_deterministic(tmp_path, project_root):
    outputs = []
    for label in ("first", "second"):
        art = tmp_path / label
        assert run_cli(project_root, "smoke", "--synthetic", "occlusion", "--set", f"paths.artifacts_dir={art}").returncode == 0
        (run,) = run_folders(art)
        outputs.append(json.loads((run / "diagnostics.json").read_text()))
    assert outputs[0] == outputs[1]


def test_make_fixtures_then_smoke_every_fixture(tmp_path, project_root):
    out = tmp_path / "fx"
    assert run_cli(project_root, "make-fixtures", "--out", str(out)).returncode == 0
    for scenario in SCENARIOS:
        assert (out / f"{scenario}.avi").is_file()
        truth = json.loads((out / f"{scenario}_ground_truth.json").read_text())
        assert truth["num_frames"] == 90 and len(truth["objects"]) == 90
        result = run_cli(project_root, "smoke", "--video", str(out / f"{scenario}.avi"), "--set", f"paths.artifacts_dir={tmp_path / 'art'}")
        assert result.returncode == 0, result.stderr


def test_default_config_file_drives_the_cli(tmp_path, project_root):
    art = tmp_path / "art"
    result = run_cli(
        project_root, "smoke", "--synthetic", "static", "--config", "configs/default.yaml",
        "--set", f"paths.artifacts_dir={art}", "--set", "experiment.seed=99",
    )
    assert result.returncode == 0, result.stderr
    (run,) = run_folders(art)
    assert load_config(run / "config.yaml").experiment.seed == 99
    assert "-s99_" in run.name
