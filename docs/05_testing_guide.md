# 5. Testing Guide

## What is a test?

A test is a tiny program that runs part of our code with known input and checks the output. If the output is wrong, the test **fails** and tells us what. We write hundreds of them so that when we change something later, we find out immediately if we broke something that used to work.

## The four tiers

| Tier | Needs the dataset? | Needs model weights? | Needs network? | Purpose | Status |
|---|---|---|---|---|---|
| **Unit** | No | No | **No** (blocked) | Check one small piece in isolation: numbers, edge cases, bad input | ✅ 276 tests |
| **Integration** | No (fixtures only) | No | **No** (blocked) | Check that pieces **work together** across their boundaries | ✅ 15 tests |
| **Model-smoke** | No | Yes | — | One real pretrained model works | empty until Phase 4 |
| **Slow / full** | Yes | Yes | — | Real RWF-2000 training, evaluation, ablations | empty until Phase 1+ |

*Unit vs integration, by analogy:* testing that a car's brake pedal, engine and steering wheel each work on their own = **unit** tests. Driving the car round the block = **integration** test.

The tier of a test is decided by the folder it's in (`tests/unit/`, `tests/integration/`, …) — pytest then labels it automatically.

## How to run (one command per tier)

From the project folder, in PowerShell:

```powershell
python scripts/run_tests.py unit          # fast; run constantly
python scripts/run_tests.py integration   # slightly slower; runs real command-line processes
python scripts/run_tests.py fast          # unit + integration — run this after every change
python scripts/run_tests.py model_smoke   # prints "no tests yet" until Phase 4
python scripts/run_tests.py slow          # prints "no tests yet" until Phase 1+
```

**Reading the output:** each `.` is a passing test; `F` is a failure; `s` is skipped. The last line summarises, e.g. `276 passed`. Add `-q` for terse output, `-v` to list every test name, `-k keyword` to run only tests whose name contains a word:

```powershell
python scripts/run_tests.py unit -k occlusion -v
```

## How the test suite protects the rules of the project

- **Network blocked:** `tests/conftest.py` makes any network connection raise an error during unit and integration tests. They *cannot* secretly depend on a download.
- **Auto-marking:** tests get their tier label from the folder they live in.
- **Shared fixtures:** the four synthetic videos are generated once per session; they are also encoded to real `.avi` files in a temporary folder to act as a "tiny video file".
- **Every test is isolated:** each uses its own temporary folders and leaves no files behind in your project.

## What the tests check, mapped to the plan

### Phase 0 unit tests required by the plan

| Plan requirement | Test(s) |
|---|---|
| Configuration defaults load correctly | `test_config.py`: `test_defaults_load`, `test_default_yaml_file_matches_code_defaults` |
| Invalid configuration values fail explicitly | `test_config.py`: `test_invalid_values_fail_explicitly`, `test_wrong_types_fail_explicitly`, `test_unknown_*`, `test_non_finite_numbers_are_rejected` |
| Data contracts reject malformed boxes and scores | `test_contracts.py`: `test_malformed_boxes_are_rejected`, `test_detection_confidence_is_range_checked`, `test_window_scores_are_range_checked`, `test_malformed_tokens_are_rejected` |
| Seeds produce reproducible sampling | `test_foundation.py`: `test_seed_everything_makes_python_and_numpy_reproducible`, `test_make_rng_reproducible_sampling`, `test_named_rngs_are_independent` |
| Manifest parsing handles valid and invalid entries | `test_manifest_pipeline.py`: `test_manifest_roundtrip_through_disk`, `test_invalid_ok_entries_are_rejected`, `test_parsing_reports_every_bad_entry_with_its_index`, `test_corrupt_clip_becomes_a_failed_entry_not_an_exception` |
| Synthetic generator is deterministic and reports ground truth | `test_synthetic.py`: `test_same_seed_gives_identical_frames_and_truth`, `test_ground_truth_matches_the_drawn_pixels`, scenario tests |
| Environment record is written with all required fields | `test_foundation.py`: `test_environment_record_contains_all_required_fields`, `test_environment_record_is_written_as_json`, `test_environment_validation_detects_missing_field` |
| Timestamps are monotonic and match the declared FPS | `test_ingestion.py`: `test_timestamps_are_monotonic_and_match_declared_fps`, `test_reads_every_frame_with_exact_timestamps` |

### Phase 0 integration tests required by the plan

| Plan requirement | Test |
|---|---|
| A tiny fixture video can be opened, decoded and represented in the manifest | `test_encoded_fixture_is_decoded_and_represented_in_a_manifest`, `test_manifest_frame_count_agrees_with_what_the_pipeline_processes` |
| Configuration can instantiate a minimal pipeline without model weights | `test_minimal_pipeline_needs_no_model_libraries` (runs in a separate process and asserts PyTorch was never imported) |
| A synthetic video and a real file both pass through ingestion and give the same diagnostics schema | `test_synthetic_and_file_sources_produce_the_same_diagnostics_schema` (all four scenarios) |

### Extra checks we added

- **Failure paths are tested as hard as success paths:** missing file, garbage file, zero-byte file, truncated file, invalid FPS, wrong frame type/size, empty source, partial decode.
- **No invented frames:** a corrupt frame stops decoding; frames before it are kept; nothing is duplicated or fabricated (`test_bad_frame_stops_decoding_instead_of_being_replaced`).
- **Command-line behaviour:** exit codes `0`/`1`/`2`, output files written, and an invalid configuration writes **nothing**.
- **Determinism end-to-end:** running the smoke command twice gives identical diagnostics.
- **No goon/role fields:** `test_window_output_has_no_role_or_instigator_fields` fails if anyone ever adds one by accident, and `test_window_label_is_only_violent_or_non_violent` rejects any other label.

## What a failing test looks like

```
FAILED tests/unit/test_contracts.py::test_malformed_boxes_are_rejected[coords0]
E   Failed: DID NOT RAISE <class 'mosaic.contracts.ContractError'>
```

This would mean a malformed box was *accepted*. The file name and test name tell you where to look; the `E` lines tell you what went wrong.

## Rules we follow for future phases

1. Every new module gets unit tests **before** the phase is declared complete.
2. Every boundary between modules gets an integration test.
3. Unit and integration tests must pass on **every** change (`python scripts/run_tests.py fast`).
4. Anything that needs the dataset or model weights goes in the `slow` or `model_smoke` tiers, never in the fast tiers.
5. A test that is skipped or removed must be explained, never hidden.
