"""Where results are stored.

Everything the project produces (manifests, checkpoints, metrics, reports, ...)
goes under one ``artifacts/`` directory with a fixed set of sub-folders, so you
never have to guess where something was saved.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SUBDIRS = (
    "manifests",  # lists of video clips + their properties
    "checkpoints",  # saved trained models
    "metrics",  # numeric results
    "reports",  # human-readable reports
    "runs",  # one folder per execution (config snapshot, environment, diagnostics, log)
    "token_cache",  # cached Track Tokens (Phase 6)
    "fixtures",  # generated test videos
    "logs",
)

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ArtifactLayout:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def ensure(self) -> "ArtifactLayout":
        """Create the root and every standard sub-folder (idempotent)."""
        for name in SUBDIRS:
            (self.root / name).mkdir(parents=True, exist_ok=True)
        return self

    def path(self, name: str) -> Path:
        if name not in SUBDIRS:
            raise KeyError(f"unknown artifact folder {name!r}; known folders: {list(SUBDIRS)}")
        return self.root / name

    def run_dir(self, run_id: str, create: bool = True) -> Path:
        """Folder for one run. ``run_id`` may not contain path separators or '..'."""
        if not _RUN_ID_RE.match(run_id) or ".." in run_id:
            raise ValueError(f"unsafe run id {run_id!r}")
        folder = self.path("runs") / run_id
        if create:
            folder.mkdir(parents=True, exist_ok=True)
        return folder


def make_run_id(experiment_id: str, now: Optional[datetime] = None) -> str:
    """``<experiment_id>_<UTC timestamp>`` - unique per execution."""
    moment = now if now is not None else datetime.now(timezone.utc)
    return f"{experiment_id}_{moment.strftime('%Y%m%dT%H%M%S%fZ')}"
