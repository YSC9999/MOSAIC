"""Unit tests for the synthetic video generator."""

import json

import numpy as np
import pytest

from mosaic.contracts import Box
from mosaic.ingestion import VideoReader
from mosaic.synthetic import (
    OBJECT_COLOR,
    OCCLUDER_COLOR,
    SCENARIOS,
    generate_synthetic,
)


def region(frame, box):
    x1, y1, x2, y2 = (int(v) for v in box.to_list())
    return frame[y1:y2, x1:x2]


def frame_diffs(video):
    frames = video.frames.astype(np.int32)
    return [int(np.abs(frames[i] - frames[i - 1]).sum()) for i in range(1, video.num_frames)]


# ------------------------------------------------------------ determinism
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_same_seed_gives_identical_frames_and_truth(scenario):
    a = generate_synthetic(scenario, seed=5)
    b = generate_synthetic(scenario, seed=5)
    assert a.frames.tobytes() == b.frames.tobytes()
    assert a.ground_truth_dict() == b.ground_truth_dict()


def test_different_seed_changes_the_background():
    a = generate_synthetic("static", seed=1)
    b = generate_synthetic("static", seed=2)
    assert a.frames.tobytes() != b.frames.tobytes()
    assert a.ground_truth_dict()["objects"] == b.ground_truth_dict()["objects"]  # same script


# -------------------------------------------------------- shape and times
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_shape_dtype_and_timestamps(scenario, synthetic_videos):
    video = synthetic_videos[scenario]
    assert video.frames.dtype == np.uint8
    assert video.frames.shape == (video.num_frames, video.height, video.width, 3)
    assert video.num_frames == 90  # 3 s * 30 fps
    assert video.timestamps == [i / 30.0 for i in range(90)]
    assert [g.frame_index for g in video.ground_truth] == list(range(90))
    assert [g.timestamp for g in video.ground_truth] == video.timestamps


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_ground_truth_boxes_are_valid_and_inside_frame(scenario, synthetic_videos):
    video = synthetic_videos[scenario]
    for item in video.ground_truth:
        assert isinstance(item.box, Box)
        item.box.check_within(video.width, video.height)
        assert 0.0 <= item.visible_fraction <= 1.0


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_ground_truth_is_json_serialisable(scenario, synthetic_videos):
    text = json.dumps(synthetic_videos[scenario].ground_truth_dict())
    assert json.loads(text)["scenario"] == scenario


def test_ground_truth_matches_the_drawn_pixels(synthetic_videos):
    for scenario in SCENARIOS:
        video = synthetic_videos[scenario]
        for item in video.ground_truth:
            if item.visible_fraction == 1.0:
                patch = region(video.frames[item.frame_index], item.box)
                assert (patch == np.array(OBJECT_COLOR, dtype=np.uint8)).all()


# --------------------------------------------------------------- scenarios
def test_static_scene_frames_are_all_identical(synthetic_videos):
    video = synthetic_videos["static"]
    assert all(d == 0 for d in frame_diffs(video))
    assert len({tuple(g.box.to_list()) for g in video.ground_truth}) == 1


def test_moving_scene_moves_steadily_to_the_right(synthetic_videos):
    centroids_x = [g.box.centroid[0] for g in synthetic_videos["moving"].ground_truth]
    assert all(b >= a for a, b in zip(centroids_x, centroids_x[1:]))
    assert centroids_x[-1] - centroids_x[0] > 100
    assert all(d > 0 for d in frame_diffs(synthetic_videos["moving"]))


def test_burst_scene_is_still_then_moves_then_still(synthetic_videos):
    video = synthetic_videos["burst"]
    start, end = video.events["motion_start_frame"], video.events["motion_end_frame"]
    assert 0 < start < end < video.num_frames - 1
    diffs = frame_diffs(video)  # diffs[i-1] compares frame i with frame i-1
    for i in range(1, video.num_frames):
        if i <= start or i > end:
            assert diffs[i - 1] == 0, f"frame {i} should be identical to the previous frame"
        else:
            assert diffs[i - 1] > 0, f"frame {i} should differ from the previous frame"


def test_occlusion_scene_hides_the_object_completely_for_a_while(synthetic_videos):
    video = synthetic_videos["occlusion"]
    first, last = video.events["fully_occluded_start"], video.events["fully_occluded_end"]
    assert video.occluder is not None
    hidden = [g.frame_index for g in video.ground_truth if g.visible_fraction == 0.0]
    assert hidden == list(range(first, last + 1))  # one contiguous run
    assert last - first + 1 >= 5
    for index in hidden:
        patch = region(video.frames[index], video.ground_truth[index].box)
        assert (patch == np.array(OCCLUDER_COLOR, dtype=np.uint8)).all()  # only wall pixels
    assert video.ground_truth[0].visible_fraction == 1.0
    assert video.ground_truth[-1].visible_fraction == 1.0
    assert video.events["partially_occluded_start"] <= first
    assert video.events["partially_occluded_end"] >= last


def test_occlusion_object_keeps_moving_while_hidden(synthetic_videos):
    xs = [g.box.centroid[0] for g in synthetic_videos["occlusion"].ground_truth]
    assert all(b > a for a, b in zip(xs, xs[1:]))  # truth keeps advancing behind the wall


# -------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "kwargs",
    [
        {"scenario": "teleport"},
        {"scenario": "static", "width": 50},
        {"scenario": "static", "height": 30},
        {"scenario": "static", "fps": 0},
        {"scenario": "static", "fps": float("nan")},
        {"scenario": "static", "duration_seconds": -1},
        {"scenario": "static", "duration_seconds": 0.1},  # fewer than 12 frames
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        generate_synthetic(**kwargs)


def test_custom_size_and_rate_are_respected():
    video = generate_synthetic("moving", seed=1, width=200, height=100, fps=10.0, duration_seconds=2.0)
    assert video.frames.shape == (20, 100, 200, 3)
    assert video.fps == 10.0
    for item in video.ground_truth:
        item.box.check_within(200, 100)


# ---------------------------------------------------------- writing to disk
def test_written_video_can_be_read_back(tmp_path, synthetic_videos):
    video = synthetic_videos["moving"]
    path = video.write(tmp_path / "moving.avi")
    records = VideoReader(path).read_all()
    assert len(records) == video.num_frames
    assert records[0].frame.shape == (video.height, video.width, 3)


def test_writing_requires_avi_extension(tmp_path, synthetic_videos):
    with pytest.raises(ValueError):
        synthetic_videos["static"].write(tmp_path / "static.mp4")


def test_ground_truth_file_is_written(tmp_path, synthetic_videos):
    path = synthetic_videos["burst"].write_ground_truth(tmp_path / "gt" / "burst.json")
    assert json.loads(path.read_text())["events"]["motion_start_frame"] == 30
