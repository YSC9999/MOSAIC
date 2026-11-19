"""Unit tests for reproducible seeding."""

import random

import numpy as np

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
