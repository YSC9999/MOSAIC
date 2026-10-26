"""Deterministic seeding.

Computers produce "random" numbers from a *seed*. The same seed always gives the
same sequence, which is what makes an experiment repeatable. We seed every
random-number source we use, and we derive independent sub-seeds from a single
base seed so that different components (the data sampler, the synthetic video
generator, ...) do not accidentally share or disturb each other's streams.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np


import sys


def seed_everything(seed: int, seed_torch: Optional[bool] = None) -> None:
    """Seed Python's ``random``, NumPy and (optionally/automatically) PyTorch.

    PyTorch is seeded whenever it has already been imported or when ``seed_torch`` is True.
    If PyTorch is not imported and ``seed_torch`` is None/False, no PyTorch import occurs,
    so Phase 0 and clean-venv tests run without requiring PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    # Only affects child processes (hash randomisation is fixed at start-up).
    os.environ["PYTHONHASHSEED"] = str(seed)
    if seed_torch is True or (seed_torch is None and "torch" in sys.modules):
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass


def derive_seed(base_seed: int, *names: str) -> int:
    """Derive a stable 32-bit sub-seed from a base seed and a list of names."""
    key = f"{base_seed}:" + "/".join(names)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def make_rng(base_seed: int, *names: str) -> np.random.Generator:
    """Create an independent NumPy random generator for one named component."""
    return np.random.default_rng(derive_seed(base_seed, *names))
