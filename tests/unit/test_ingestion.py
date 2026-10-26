"""Unit tests for video ingestion."""

import numpy as np
import pytest

from mosaic.config import DecodeConfig
from mosaic.ingestion import (
    ArrayFrameSource,
    DecodeError,
    VideoReader,
    _State,
    probe_video,
)


def good_frames(n=5, h=24, w=32):
    return [np.full((h, w, 3), i, dtype=np.uint8) for i in range(n)]


# ------------------------------------------------------------ file reading
def test_reads_every_frame_with_exact_timestamps(encoded_videos):
    reader = VideoReader(encoded_videos["moving"])
    records = reader.read_all()
    assert [r.frame_index for r in records] == list(range(90))
    assert [r.timestamp for r in records] == [i / 30.0 for i in range(90)]
    diag = reader.diagnostics
    assert diag.status == "ok"
    assert (diag.decoded_frame_count, diag.fps, diag.width, diag.height) == (90, 30.0, 160, 120)
    assert diag.source_kind == "file"
    assert diag.duration_seconds == pytest.approx(3.0)


def test_timestamps_are_monotonic_and_match_declared_fps(encoded_videos):
    for name in ("static", "burst", "occlusion"):
        reader = VideoReader(encoded_videos[name])
        records = reader.read_all()
        stamps = [r.timestamp for r in records]
        gaps = np.diff(stamps)
        assert (gaps > 0).all()
        assert gaps == pytest.approx(1.0 / reader.diagnostics.fps)
        assert reader.diagnostics.timestamps_monotonic is True


def test_replay_is_deterministic(encoded_videos):
    first = VideoReader(encoded_videos["burst"]).read_all()
    second = VideoReader(encoded_videos["burst"]).read_all()
    assert [r.frame_index for r in first] == [r.frame_index for r in second]
    assert all(np.array_equal(a.frame, b.frame) for a, b in zip(first, second))


def test_frame_indices_are_unique_and_ordered(encoded_videos):
    indices = [r.frame_index for r in VideoReader(encoded_videos["static"]).read_all()]
    assert indices == sorted(set(indices))


def test_frames_are_bgr_uint8(encoded_videos):
    record = VideoReader(encoded_videos["static"]).read_all()[0]
    assert record.frame.dtype == np.uint8 and record.frame.shape == (120, 160, 3)


def test_max_frames_truncates_and_is_reported(encoded_videos):
    reader = VideoReader(encoded_videos["moving"], decode=DecodeConfig(max_frames=10))
    assert len(reader.read_all()) == 10
    diag = reader.diagnostics
    assert diag.status == "ok" and diag.truncated is True and diag.decoded_frame_count == 10


def test_source_id_defaults_to_file_name(encoded_videos):
    reader = VideoReader(encoded_videos["static"])
    reader.read_all()
    assert reader.diagnostics.source_id == "static.avi"
    named = VideoReader(encoded_videos["static"], source_id="custom")
    named.read_all()
    assert named.diagnostics.source_id == "custom"


# ----------------------------------------------------------- failure cases
def test_missing_file_fails_explicitly(tmp_path):
    reader = VideoReader(tmp_path / "missing.avi")
    with pytest.raises(DecodeError, match="not found"):
        reader.read_all()
    diag = reader.diagnostics
    assert diag.status == "failed" and diag.decoded_frame_count == 0
    assert any("not found" in e for e in diag.errors)


def test_garbage_file_fails_explicitly(tmp_path):
    bad = tmp_path / "garbage.avi"
    bad.write_bytes(b"this is definitely not a video" * 50)
    reader = VideoReader(bad)
    with pytest.raises(DecodeError):
        reader.read_all()
    assert reader.diagnostics.status == "failed"


def test_zero_byte_file_fails_explicitly(tmp_path):
    empty = tmp_path / "empty.avi"
    empty.write_bytes(b"")
    reader = VideoReader(empty)
    with pytest.raises(DecodeError):
        reader.read_all()
    assert reader.diagnostics.status == "failed"


def test_truncated_video_never_invents_frames(tmp_path, encoded_videos):
    data = encoded_videos["moving"].read_bytes()
    cut = tmp_path / "cut.avi"
    cut.write_bytes(data[: len(data) // 2])
    reader = VideoReader(cut)
    try:
        records = reader.read_all()
    except DecodeError:
        records = []
    diag = reader.diagnostics
    assert len(records) == diag.decoded_frame_count
    assert diag.decoded_frame_count < 90  # we got fewer frames, and said so
    assert diag.status in ("partial", "empty", "failed", "ok")
    if diag.status == "ok":  # an honest "ok" is only possible if the container agreed
        assert diag.declared_frame_count in (None, diag.decoded_frame_count)


def test_probe_never_raises_for_bad_files(tmp_path, encoded_videos):
    assert probe_video(encoded_videos["static"]).status == "ok"
    assert probe_video(tmp_path / "nope.avi").status == "failed"


def test_diagnostics_need_iteration_first(encoded_videos):
    with pytest.raises(RuntimeError):
        _ = VideoReader(encoded_videos["static"]).diagnostics


def test_container_frame_count_rule():
    state = _State("x", "file", tolerance=1)
    state.fps, state.width, state.height, state.declared = 30.0, 8, 8, 100
    state.reached_end = True
    state.decoded = 99  # within tolerance
    assert state.finalize().status == "ok"
    state.decoded = 97  # three frames missing -> partial, with an explanation
    diag = state.finalize()
    assert diag.status == "partial" and "declared 100" in diag.errors[0]


def test_early_stop_by_consumer_is_flagged(encoded_videos):
    reader = VideoReader(encoded_videos["static"])
    iterator = reader.iter_frames()
    next(iterator)
    iterator.close()
    diag = reader.diagnostics
    assert diag.status == "ok" and diag.decoded_frame_count == 1
    assert diag.warnings and "stopped before the end" in diag.warnings[0]


# -------------------------------------------------------- in-memory sources
def test_array_source_matches_file_diagnostics_shape(synthetic_videos):
    source = synthetic_videos["static"].frame_source()
    records = source.read_all()
    diag = source.diagnostics
    assert len(records) == 90 and diag.status == "ok" and diag.source_kind == "synthetic"
    assert diag.fps == 30.0 and (diag.width, diag.height) == (160, 120)
    assert records[-1].timestamp == pytest.approx(89 / 30.0)


def test_empty_source_is_reported_as_empty_with_no_substitute_frames():
    source = ArrayFrameSource([], fps=30.0)
    assert source.read_all() == []
    diag = source.diagnostics
    assert diag.status == "empty" and diag.decoded_frame_count == 0 and diag.errors


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf"), "30", None, True])
def test_invalid_fps_is_refused(fps):
    source = ArrayFrameSource(good_frames(), fps=fps)
    with pytest.raises(DecodeError):
        source.read_all()
    assert source.diagnostics.status == "failed"


def test_bad_frame_stops_decoding_instead_of_being_replaced():
    frames = good_frames(2) + [np.zeros((24, 32, 3), dtype=np.float32)] + good_frames(2)
    source = ArrayFrameSource(frames, fps=30.0)
    records = source.read_all()
    assert [r.frame_index for r in records] == [0, 1]  # nothing invented, nothing skipped over
    diag = source.diagnostics
    assert diag.status == "partial" and "frame 2" in diag.errors[0]


def test_size_change_midstream_is_reported():
    frames = good_frames(2) + [np.zeros((10, 10, 3), dtype=np.uint8)]
    source = ArrayFrameSource(frames, fps=30.0)
    assert len(source.read_all()) == 2
    assert source.diagnostics.status == "partial"


@pytest.mark.parametrize(
    "bad",
    [np.zeros((10, 10), dtype=np.uint8), np.zeros((10, 10, 4), dtype=np.uint8), "frame", None],
)
def test_wrongly_shaped_frames_are_reported(bad):
    source = ArrayFrameSource([bad], fps=30.0)
    assert source.read_all() == []
    assert source.diagnostics.status == "empty" or source.diagnostics.status == "partial"
    assert source.diagnostics.errors
