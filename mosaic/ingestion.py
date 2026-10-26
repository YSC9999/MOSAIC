"""Video ingestion: turning a video file (or an in-memory array) into frames.

Every frame comes with a *frame index* and a *timestamp*. The timestamp is
always ``frame_index / fps`` - computed by us, not trusted from the video codec -
so it is identical on every machine and every replay.

Rules from the plan that this module enforces:

* **Deterministic replay** - the same file always yields the same frames,
  indices and timestamps.
* **Explicit failure** - a file that is missing, unreadable, empty or has an
  invalid frame rate is reported, never guessed at.
* **No silent substitution** - if a frame cannot be decoded we stop and say so.
  We never repeat the previous frame or insert a blank one.
* **Same report for every source** - real files and synthetic arrays produce the
  same ``DecodeDiagnostics`` structure.
"""

from __future__ import annotations

import abc
import math
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

import cv2
import numpy as np

from .config import DecodeConfig
from .contracts import DecodeDiagnostics


class DecodeError(RuntimeError):
    """The video source could not be opened or its frame rate is unusable."""


@dataclass(frozen=True)
class FrameRecord:
    """One decoded frame: BGR uint8 pixels of shape (height, width, 3)."""

    frame_index: int
    timestamp: float
    frame: np.ndarray


def _frame_problem(frame: object) -> Optional[str]:
    if not isinstance(frame, np.ndarray):
        return f"frame is not a numpy array (got {type(frame).__name__})"
    if frame.ndim != 3 or frame.shape[2] != 3:
        return f"frame must have shape (height, width, 3), got {frame.shape}"
    if frame.dtype != np.uint8:
        return f"frame must be uint8, got {frame.dtype}"
    if frame.shape[0] < 1 or frame.shape[1] < 1:
        return f"frame has zero size: {frame.shape}"
    return None


class _State:
    """Mutable bookkeeping while a source is being read."""

    def __init__(self, source_id: str, kind: str, tolerance: int) -> None:
        self.source_id = source_id
        self.kind = kind
        self.tolerance = tolerance
        self.fps: Optional[float] = None
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.declared: Optional[int] = None
        self.decoded = 0
        self.first_ts: Optional[float] = None
        self.last_ts: Optional[float] = None
        self.monotonic = True
        self.truncated = False
        self.reached_end = False
        self.fatal: List[str] = []
        self.partial: List[str] = []

    def note_frame(self, timestamp: float) -> None:
        if self.first_ts is None:
            self.first_ts = timestamp
        if self.last_ts is not None and not timestamp > self.last_ts:
            self.monotonic = False
        self.last_ts = timestamp
        self.decoded += 1

    def finalize(self) -> DecodeDiagnostics:
        errors = list(self.fatal) + list(self.partial)
        warnings: List[str] = []
        if self.fatal:
            status = "failed"
        elif self.decoded == 0:
            status = "empty"
            errors.append("no frames could be decoded")
        elif self.partial:
            status = "partial"
        elif (
            self.reached_end
            and self.declared is not None
            and not self.truncated
            and abs(self.decoded - self.declared) > self.tolerance
        ):
            status = "partial"
            errors.append(
                f"container declared {self.declared} frames but only {self.decoded} were decoded"
            )
        else:
            status = "ok"
        if status == "ok" and not self.reached_end and not self.truncated:
            warnings.append("iteration stopped before the end of the stream; counts may be incomplete")
        duration = (self.decoded / self.fps) if (self.fps and self.decoded) else None
        return DecodeDiagnostics(
            source_id=self.source_id,
            source_kind=self.kind,
            status=status,
            decoded_frame_count=self.decoded,
            fps=self.fps,
            width=self.width,
            height=self.height,
            declared_frame_count=self.declared,
            first_timestamp=self.first_ts,
            last_timestamp=self.last_ts,
            duration_seconds=duration,
            timestamps_monotonic=self.monotonic,
            truncated=self.truncated,
            errors=errors,
            warnings=warnings,
        )


class FrameSource(abc.ABC):
    """Anything that yields ``FrameRecord`` objects and can report how it went."""

    def __init__(self, source_id: str) -> None:
        self.source_id = source_id
        self._state: Optional[_State] = None

    @abc.abstractmethod
    def iter_frames(self) -> Iterator[FrameRecord]:
        """Yield frames in order. Raises ``DecodeError`` if the source is unusable."""

    @property
    def diagnostics(self) -> DecodeDiagnostics:
        """Report on the most recent iteration (available once iteration has started)."""
        if self._state is None:
            raise RuntimeError("diagnostics are only available after iter_frames() has been used")
        return self._state.finalize()

    def read_all(self) -> List[FrameRecord]:
        return list(self.iter_frames())


class VideoReader(FrameSource):
    """Reads a video file with OpenCV."""

    def __init__(
        self,
        path: Path | str,
        decode: Optional[DecodeConfig] = None,
        source_id: Optional[str] = None,
    ) -> None:
        self.path = Path(path)
        super().__init__(source_id if source_id is not None else self.path.name)
        self._decode = decode if decode is not None else DecodeConfig()

    def iter_frames(self) -> Iterator[FrameRecord]:
        state = _State(self.source_id, "file", self._decode.frame_count_tolerance)
        self._state = state

        def fail(message: str) -> DecodeError:
            state.fatal.append(message)
            return DecodeError(message)

        if not self.path.is_file():
            raise fail(f"file not found: {self.path}")
        capture = cv2.VideoCapture(str(self.path))
        try:
            if not capture.isOpened():
                raise fail(f"could not open video (unreadable or unsupported format): {self.path}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(fps) or fps <= 0:
                raise fail(f"video reports an invalid frame rate ({fps}); refusing to guess")
            state.fps = fps
            declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            state.declared = declared if declared > 0 else None

            index = 0
            while True:
                if self._decode.max_frames is not None and index >= self._decode.max_frames:
                    state.truncated = True
                    break
                ok, frame = capture.read()
                if not ok:
                    state.reached_end = True
                    break
                problem = _frame_problem(frame)
                if problem is not None:
                    state.partial.append(f"frame {index}: {problem}")
                    break
                height, width = frame.shape[:2]
                if state.width is None:
                    state.width, state.height = int(width), int(height)
                elif (width, height) != (state.width, state.height):
                    state.partial.append(
                        f"frame {index} has size {width}x{height}, expected {state.width}x{state.height}"
                    )
                    break
                timestamp = index / fps
                state.note_frame(timestamp)
                yield FrameRecord(index, timestamp, frame)
                index += 1
        finally:
            capture.release()


class ArrayFrameSource(FrameSource):
    """Frames that already live in memory (used for synthetic videos)."""

    def __init__(
        self,
        frames: Sequence[np.ndarray] | np.ndarray,
        fps: float,
        source_id: str = "array",
        kind: str = "array",
    ) -> None:
        super().__init__(source_id)
        self._frames = frames
        self._fps = fps
        self._kind = kind

    def iter_frames(self) -> Iterator[FrameRecord]:
        state = _State(self.source_id, self._kind, tolerance=0)
        self._state = state
        fps = self._fps
        if isinstance(fps, bool) or not isinstance(fps, numbers.Real) or not math.isfinite(fps) or fps <= 0:
            message = f"invalid frame rate {fps!r}; refusing to guess"
            state.fatal.append(message)
            raise DecodeError(message)
        fps = float(fps)
        state.fps = fps
        state.declared = len(self._frames)
        for index, frame in enumerate(self._frames):
            problem = _frame_problem(frame)
            if problem is not None:
                state.partial.append(f"frame {index}: {problem}")
                return
            height, width = frame.shape[:2]
            if state.width is None:
                state.width, state.height = int(width), int(height)
            elif (width, height) != (state.width, state.height):
                state.partial.append(
                    f"frame {index} has size {width}x{height}, expected {state.width}x{state.height}"
                )
                return
            timestamp = index / fps
            state.note_frame(timestamp)
            yield FrameRecord(index, timestamp, frame)
        state.reached_end = True


def probe_video(
    path: Path | str,
    decode: Optional[DecodeConfig] = None,
    source_id: Optional[str] = None,
) -> DecodeDiagnostics:
    """Fully decode a video (discarding frames) and report what was found.

    Never raises for unreadable files - the failure is returned as diagnostics so
    corrupt clips can be listed in a manifest instead of crashing a whole scan.
    """
    reader = VideoReader(path, decode=decode, source_id=source_id)
    try:
        for _ in reader.iter_frames():
            pass
    except DecodeError:
        pass
    return reader.diagnostics
