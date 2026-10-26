"""Unit tests for physical kinematics calculations (Block 4)."""

import math
import pytest

from mosaic.contracts import Box
from mosaic.kinematics import KinematicState, compute_centroid, compute_kinematics


def test_centroid_calculation():
    box = Box(10.0, 20.0, 50.0, 100.0)
    cx, cy = compute_centroid(box)
    assert cx == pytest.approx(30.0)
    assert cy == pytest.approx(60.0)


def test_kinematics_initial_observation():
    box = Box(100.0, 100.0, 150.0, 250.0)
    state = compute_kinematics(box, current_timestamp=0.0, frame_width=1000, frame_height=1000)

    assert state.centroid == [125.0, 175.0]
    assert state.velocity == [0.0, 0.0]
    assert state.acceleration == [0.0, 0.0]
    assert state.speed == 0.0
    assert state.acceleration_magnitude == 0.0
    assert state.aspect_ratio == pytest.approx(50.0 / 150.0)
    assert state.aspect_ratio_velocity == 0.0
    assert state.normalized_centroid == [0.125, 0.175]
    assert state.normalized_velocity == [0.0, 0.0]


def test_kinematics_velocity_and_speed():
    # Displacement from (100, 100) to (130, 140) in 0.5s:
    # dx = 30, dy = 40, dt = 0.5 -> vx = 60, vy = 80, speed = 100
    b0 = Box(80.0, 80.0, 120.0, 120.0)      # center (100, 100)
    b1 = Box(110.0, 120.0, 150.0, 160.0)    # center (130, 140)

    state = compute_kinematics(
        current_box=b1,
        current_timestamp=0.5,
        previous_box=b0,
        previous_timestamp=0.0,
        frame_width=1000,
        frame_height=1000,
    )

    assert state.velocity[0] == pytest.approx(60.0)
    assert state.velocity[1] == pytest.approx(80.0)
    assert state.speed == pytest.approx(100.0)
    assert state.normalized_velocity[0] == pytest.approx(0.06)
    assert state.normalized_velocity[1] == pytest.approx(0.08)
    assert state.normalized_speed == pytest.approx(0.10)


def test_kinematics_acceleration():
    # Step 1: vel was (20.0, 0.0)
    # Step 2: moves by (30.0, 0.0) in 0.5s -> new vel = (60.0, 0.0)
    # dv = (40.0, 0.0), dt = 0.5 -> ax = 80.0, ay = 0.0
    b0 = Box(0.0, 0.0, 20.0, 20.0)          # center (10, 10)
    b1 = Box(30.0, 0.0, 50.0, 20.0)         # center (40, 10)

    state = compute_kinematics(
        current_box=b1,
        current_timestamp=1.0,
        previous_box=b0,
        previous_timestamp=0.5,
        previous_velocity=[20.0, 0.0],
    )

    assert state.velocity[0] == pytest.approx(60.0)
    assert state.acceleration[0] == pytest.approx(80.0)
    assert state.acceleration[1] == pytest.approx(0.0)
    assert state.acceleration_magnitude == pytest.approx(80.0)


def test_kinematics_aspect_ratio_fall_dynamics():
    # Standing person: width 40, height 160 (AR = 0.25)
    # Falls horizontal: width 160, height 40 (AR = 4.0)
    # dt = 0.5s -> d(AR)/dt = (4.0 - 0.25) / 0.5 = 7.5 s^-1
    b_standing = Box(50.0, 10.0, 90.0, 170.0)
    b_fallen = Box(10.0, 100.0, 170.0, 140.0)

    state = compute_kinematics(
        current_box=b_fallen,
        current_timestamp=0.5,
        previous_box=b_standing,
        previous_timestamp=0.0,
    )

    assert state.aspect_ratio == pytest.approx(4.0)
    assert state.aspect_ratio_velocity == pytest.approx(7.5)


def test_kinematics_handles_zero_delta_t_gracefully():
    b0 = Box(10.0, 10.0, 30.0, 50.0)
    b1 = Box(20.0, 20.0, 40.0, 60.0)

    # Identical timestamps (zero delta_t) must not raise ZeroDivisionError
    state = compute_kinematics(
        current_box=b1,
        current_timestamp=1.0,
        previous_box=b0,
        previous_timestamp=1.0,
        previous_velocity=[10.0, 10.0],
    )
    assert state.velocity == [10.0, 10.0]
    assert state.acceleration == [0.0, 0.0]


def test_kinematics_state_serialization():
    box = Box(10.0, 10.0, 50.0, 90.0)
    state = compute_kinematics(box, current_timestamp=0.0)
    d = state.to_dict()
    assert isinstance(d["centroid"], list)
    assert isinstance(d["speed"], float)
    assert d["aspect_ratio"] == pytest.approx(40.0 / 80.0)
