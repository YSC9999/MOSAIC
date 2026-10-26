"""Deterministic synthetic videos with known ground truth.

Real datasets (like RWF-2000) have no answer key for things such as "where is
the person in frame 37?". To test our gating, tracking and kinematics code we
need videos where we *do* know the right answer. So we draw them ourselves:
a coloured rectangle ("person") on a fixed random-texture background, moving
along a scripted path. The generator reports the exact box of the rectangle in
every frame.

Scenarios
---------
``static``     Rectangle never moves. Every frame is identical.
``moving``     Rectangle moves in a straight line at constant speed.
``burst``      Still for the first third, a sudden fast move in the middle
               third, still again for the last third.
``occlusion``  Rectangle moves horizontally behind a static wall (the
               "occluder") and re-appears on the other side.

Same seed -> byte-identical frames. A different seed changes the background
texture.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .contracts import Box
from .ingestion import ArrayFrameSource
from .seeding import make_rng

SCENARIOS = ("static", "moving", "burst", "occlusion")
OBJECT_COLOR = (40, 40, 230)  # BGR (reddish)
OCCLUDER_COLOR = (230, 120, 30)  # BGR (bluish)
_MARGIN = 10


@dataclass(frozen=True)
class GroundTruthObject:
    """Exact truth about the rectangle in one frame."""

    object_id: int
    frame_index: int
    timestamp: float
    box: Box
    visible_fraction: float  # 1.0 = fully visible, 0.0 = completely hidden

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "box": self.box.to_list(),
            "visible_fraction": self.visible_fraction,
        }


@dataclass
class SyntheticVideo:
    scenario: str
    seed: int
    fps: float
    width: int
    height: int
    frames: np.ndarray  # (num_frames, height, width, 3) uint8
    ground_truth: List[GroundTruthObject]  # one entry per frame
    occluder: Optional[Box] = None
    events: Dict[str, int] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def timestamps(self) -> List[float]:
        return [i / self.fps for i in range(self.num_frames)]

    def frame_source(self, source_id: Optional[str] = None) -> ArrayFrameSource:
        name = source_id if source_id is not None else f"synthetic-{self.scenario}-seed{self.seed}"
        return ArrayFrameSource(self.frames, self.fps, source_id=name, kind="synthetic")

    def ground_truth_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "seed": self.seed,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "num_frames": self.num_frames,
            "occluder": None if self.occluder is None else self.occluder.to_list(),
            "events": dict(self.events),
            "objects": [item.to_dict() for item in self.ground_truth],
        }

    def write(self, path: Path | str) -> Path:
        """Encode the frames as an MJPG ``.avi`` file (and return its path)."""
        target = Path(path)
        if target.suffix.lower() != ".avi":
            raise ValueError(f"synthetic videos are written as .avi files, got {target.name!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(target), cv2.VideoWriter_fourcc(*"MJPG"), float(self.fps), (self.width, self.height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"OpenCV could not open a video writer for {target}")
        try:
            for frame in self.frames:
                writer.write(frame)
        finally:
            writer.release()
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError(f"video file was not written: {target}")
        return target

    def write_ground_truth(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.ground_truth_dict(), indent=2), encoding="utf-8")
        return target


def _positions(scenario: str, n: int, width: int, height: int, bw: int, bh: int) -> List[tuple]:
    y_center = (height - bh) // 2
    x_start = _MARGIN
    x_end = width - _MARGIN - bw
    result = []
    for i in range(n):
        if scenario == "static":
            x, y = (width - bw) // 2, y_center
        elif scenario == "moving":
            t = i / (n - 1)
            x = int(round(x_start + t * (x_end - x_start)))
            y = int(round(_MARGIN + t * (height - _MARGIN - bh - _MARGIN)))
        elif scenario == "burst":
            first, last = n // 3, (2 * n) // 3
            t = min(max((i - first) / (last - first), 0.0), 1.0)
            x = int(round(x_start + t * (x_end - x_start)))
            y = y_center
        else:  # occlusion
            t = i / (n - 1)
            x = int(round(x_start + t * (x_end - x_start)))
            y = y_center
        result.append((x, y))
    return result


def generate_synthetic(
    scenario: str,
    seed: int = 0,
    width: int = 160,
    height: int = 120,
    fps: float = 30.0,
    duration_seconds: float = 3.0,
) -> SyntheticVideo:
    """Create a synthetic video plus its exact ground truth."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose one of {list(SCENARIOS)}")
    if not isinstance(width, int) or not isinstance(height, int) or width < 96 or height < 72:
        raise ValueError("width/height must be integers of at least 96x72")
    if not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be a positive finite number, got {fps!r}")
    if not isinstance(duration_seconds, (int, float)) or not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError(f"duration_seconds must be positive and finite, got {duration_seconds!r}")
    n = int(round(duration_seconds * fps))
    if n < 12:
        raise ValueError(f"a synthetic video needs at least 12 frames, got {n}")

    rng = make_rng(seed, "synthetic", "background")
    background = rng.integers(50, 90, size=(height, width, 3), dtype=np.uint8)
    bw, bh = max(8, width // 8), max(8, height // 5)
    positions = _positions(scenario, n, width, height, bw, bh)

    occluder: Optional[Box] = None
    if scenario == "occlusion":
        y_center = (height - bh) // 2
        occluder = Box(
            int(0.4 * width), max(0, y_center - bh), int(0.6 * width), min(height, y_center + 2 * bh)
        )

    frames = np.empty((n, height, width, 3), dtype=np.uint8)
    truth: List[GroundTruthObject] = []
    for i, (x, y) in enumerate(positions):
        frame = background.copy()
        frame[y : y + bh, x : x + bw] = OBJECT_COLOR
        visible = 1.0
        if occluder is not None:
            ox1, oy1, ox2, oy2 = (int(v) for v in occluder.to_list())
            frame[oy1:oy2, ox1:ox2] = OCCLUDER_COLOR
            overlap_w = max(0, min(x + bw, ox2) - max(x, ox1))
            overlap_h = max(0, min(y + bh, oy2) - max(y, oy1))
            visible = 1.0 - (overlap_w * overlap_h) / float(bw * bh)
        frames[i] = frame
        truth.append(GroundTruthObject(0, i, i / fps, Box(x, y, x + bw, y + bh), visible))

    events: Dict[str, int] = {}
    if scenario == "burst":
        events = {"motion_start_frame": n // 3, "motion_end_frame": (2 * n) // 3}
    elif scenario == "occlusion":
        hidden = [t.frame_index for t in truth if t.visible_fraction == 0.0]
        partial = [t.frame_index for t in truth if t.visible_fraction < 1.0]
        if hidden:
            events["fully_occluded_start"] = hidden[0]
            events["fully_occluded_end"] = hidden[-1]
        if partial:
            events["partially_occluded_start"] = partial[0]
            events["partially_occluded_end"] = partial[-1]

    return SyntheticVideo(
        scenario=scenario,
        seed=seed,
        fps=float(fps),
        width=width,
        height=height,
        frames=frames,
        ground_truth=truth,
        occluder=occluder,
        events=events,
    )
