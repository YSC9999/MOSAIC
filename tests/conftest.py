"""Shared test setup.

* Tests are marked automatically from the folder they live in
  (tests/unit -> ``unit``, tests/integration -> ``integration``, ...).
* The unit and integration tiers run with the network switched OFF, which
  proves they need no downloads.
* Synthetic videos are generated once per test session (they are deterministic).
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Dict

import pytest

from mosaic.synthetic import SCENARIOS, SyntheticVideo, generate_synthetic

ROOT = Path(__file__).resolve().parents[1]
TIER_FOLDERS = ("unit", "integration", "model_smoke", "slow")


def pytest_collection_modifyitems(items):
    for item in items:
        for tier in TIER_FOLDERS:
            if tier in item.path.parts:
                item.add_marker(getattr(pytest.mark, tier))


@pytest.fixture(autouse=True)
def _network_is_off_for_fast_tiers(request, monkeypatch):
    """Unit and integration tests must work without any network access."""
    if request.node.get_closest_marker("model_smoke") or request.node.get_closest_marker("slow"):
        return

    def _blocked(*args, **kwargs):
        raise RuntimeError("network access is blocked in the unit and integration test tiers")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def synthetic_videos() -> Dict[str, SyntheticVideo]:
    return {name: generate_synthetic(name, seed=1337) for name in SCENARIOS}


@pytest.fixture(scope="session")
def encoded_videos(tmp_path_factory, synthetic_videos) -> Dict[str, Path]:
    """The synthetic videos encoded as real .avi files (a stand-in for a tiny real clip)."""
    folder = tmp_path_factory.mktemp("encoded_videos")
    return {name: video.write(folder / f"{name}.avi") for name, video in synthetic_videos.items()}
