"""Unit tests for configuration loading, validation and hashing."""

import math
import re

import pytest

from mosaic.config import (
    Config,
    ConfigError,
    config_from_dict,
    config_hash,
    config_to_dict,
    dump_config,
    experiment_id,
    load_config,
    parse_override,
)


# --------------------------------------------------------------- defaults
def test_defaults_load():
    cfg = config_from_dict({})
    assert cfg == Config()
    assert cfg.window.policy == "whole_clip"
    assert cfg.window.clip_aggregation == "mean"
    assert cfg.pipeline.stages == ["ingestion"]


def test_load_config_without_path_gives_defaults():
    assert load_config(None) == Config()


def test_default_yaml_file_matches_code_defaults(project_root):
    # The documented YAML must never drift away from the real defaults.
    assert load_config(project_root / "configs" / "default.yaml") == Config()


def test_yaml_values_override_defaults(tmp_path):
    file = tmp_path / "c.yaml"
    file.write_text("experiment:\n  seed: 7\nwindow:\n  policy: sliding\n  window_seconds: 2\n  stride_seconds: 1\n")
    cfg = load_config(file)
    assert cfg.experiment.seed == 7
    assert cfg.window.policy == "sliding"
    assert cfg.window.window_seconds == 2.0  # int widened to float
    assert cfg.decode == Config().decode  # untouched sections keep defaults


def test_empty_yaml_file_gives_defaults(tmp_path):
    file = tmp_path / "empty.yaml"
    file.write_text("")
    assert load_config(file) == Config()


# ------------------------------------------------------- invalid structure
def test_missing_file_is_an_explicit_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml_is_an_explicit_error(tmp_path):
    file = tmp_path / "bad.yaml"
    file.write_text("experiment: [unclosed")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(file)


def test_yaml_top_level_must_be_a_mapping(tmp_path):
    file = tmp_path / "list.yaml"
    file.write_text("- a\n- b\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(file)


def test_unknown_section_is_rejected():
    with pytest.raises(ConfigError, match="windw"):
        config_from_dict({"windw": {"policy": "sliding"}})


def test_unknown_setting_lists_valid_names():
    with pytest.raises(ConfigError) as info:
        config_from_dict({"window": {"polcy": "sliding"}})
    assert "polcy" in str(info.value) and "policy" in str(info.value)


def test_section_must_be_a_mapping():
    with pytest.raises(ConfigError):
        config_from_dict({"window": 5})


# ------------------------------------------------------------ wrong types
@pytest.mark.parametrize(
    "data",
    [
        {"experiment": {"seed": "abc"}},
        {"experiment": {"seed": True}},
        {"experiment": {"seed": 1.5}},
        {"experiment": {"name": 5}},
        {"window": {"window_seconds": "5"}},
        {"window": {"window_seconds": True}},
        {"logging": {"level": 3}},
        {"logging": {"to_file": "yes"}},
        {"decode": {"max_frames": "ten"}},
        {"pipeline": {"stages": "ingestion"}},
        {"pipeline": {"stages": [1, 2]}},
    ],
)
def test_wrong_types_fail_explicitly(data):
    with pytest.raises(ConfigError):
        config_from_dict(data)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_numbers_are_rejected(value):
    with pytest.raises(ConfigError):
        config_from_dict({"window": {"window_seconds": value}})


# -------------------------------------------------------- invalid values
@pytest.mark.parametrize(
    "data",
    [
        {"experiment": {"seed": -1}},
        {"experiment": {"seed": 2**32}},
        {"experiment": {"name": "has space"}},
        {"experiment": {"name": "../escape"}},
        {"experiment": {"name": ""}},
        {"window": {"window_seconds": 0}},
        {"window": {"window_seconds": -1}},
        {"window": {"stride_seconds": 0}},
        {"window": {"window_seconds": 2, "stride_seconds": 3}},
        {"window": {"policy": "weird"}},
        {"window": {"clip_aggregation": "median"}},
        {"decode": {"max_frames": 0}},
        {"decode": {"frame_count_tolerance": -1}},
        {"logging": {"level": "LOUD"}},
        {"paths": {"artifacts_dir": "  "}},
        {"pipeline": {"stages": []}},
        {"pipeline": {"stages": ["ingestion", "ingestion"]}},
        {"pipeline": {"stages": ["ingestion", "magic_stage"]}},
        {"pipeline": {"stages": ["clarity"]}},
    ],
)
def test_invalid_values_fail_explicitly(data):
    with pytest.raises(ConfigError):
        config_from_dict(data)


def test_error_message_names_the_bad_setting():
    with pytest.raises(ConfigError, match="window.window_seconds"):
        config_from_dict({"window": {"window_seconds": -3}})


def test_programmatic_construction_is_validated_too():
    from mosaic.config import WindowConfig

    with pytest.raises(ConfigError):
        WindowConfig(window_seconds=-1)


# -------------------------------------------------------------- overrides
def test_parse_override_builds_nested_mapping():
    assert parse_override("window.policy=sliding") == {"window": {"policy": "sliding"}}
    assert parse_override("experiment.seed=7") == {"experiment": {"seed": 7}}
    assert parse_override("decode.max_frames=null") == {"decode": {"max_frames": None}}
    assert parse_override("logging.to_file=false") == {"logging": {"to_file": False}}


@pytest.mark.parametrize("text", ["nothing", "=5", "..=5"])
def test_bad_overrides_are_rejected(text):
    with pytest.raises(ConfigError):
        parse_override(text)


def test_overrides_apply_on_top_of_file(tmp_path):
    file = tmp_path / "c.yaml"
    file.write_text("experiment:\n  seed: 7\n")
    cfg = load_config(file, ["experiment.seed=9", "window.clip_aggregation=max"])
    assert cfg.experiment.seed == 9 and cfg.window.clip_aggregation == "max"


def test_override_with_wrong_type_is_rejected():
    with pytest.raises(ConfigError):
        load_config(None, ["experiment.seed=abc"])


# ---------------------------------------------------- hash and identifiers
def test_config_hash_is_stable_and_short():
    a, b = Config(), Config()
    assert config_hash(a) == config_hash(b)
    assert re.fullmatch(r"[0-9a-f]{12}", config_hash(a))


def test_hash_changes_when_behaviour_changes():
    base = config_hash(Config())
    assert config_hash(config_from_dict({"experiment": {"seed": 2}})) != base
    assert config_hash(config_from_dict({"window": {"clip_aggregation": "max"}})) != base
    assert config_hash(config_from_dict({"decode": {"max_frames": 10}})) != base


def test_hash_ignores_paths_logging_and_run_label():
    base = config_hash(Config())
    assert config_hash(config_from_dict({"paths": {"artifacts_dir": "elsewhere"}})) == base
    assert config_hash(config_from_dict({"logging": {"level": "DEBUG", "to_file": False}})) == base
    assert config_hash(config_from_dict({"experiment": {"name": "other_label"}})) == base


def test_experiment_id_is_deterministic_and_readable():
    cfg = config_from_dict({"experiment": {"name": "demo", "seed": 5}})
    assert experiment_id(cfg) == experiment_id(cfg)
    assert re.fullmatch(r"demo-[0-9a-f]{12}-s5", experiment_id(cfg))


def test_dump_and_reload_roundtrip(tmp_path):
    cfg = config_from_dict({"experiment": {"name": "rt", "seed": 11}, "window": {"policy": "sliding", "window_seconds": 4, "stride_seconds": 2}})
    file = tmp_path / "snapshot.yaml"
    dump_config(cfg, file)
    assert load_config(file) == cfg
    assert config_to_dict(load_config(file)) == config_to_dict(cfg)
