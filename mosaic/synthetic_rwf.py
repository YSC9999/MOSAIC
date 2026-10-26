"""Deterministic mock RWF-2000 dataset fixture generator.

Creates a miniature, self-contained RWF-2000 dataset folder for offline unit
and integration testing:
* Produces valid video clips (.avi) with authentic decode properties.
* Strict folder structure:
    target_dir/
      train/
        Fight/
        NonFight/
      val/
        Fight/
        NonFight/
* Distinct visual dynamics:
  - Fight clips: Rapid, erratic bursts of motion and high pixel variance.
  - NonFight clips: Slow, gentle horizontal drift or static scenes.
* 100% deterministic and runs in milliseconds without network downloads.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from mosaic.seeding import derive_seed, make_rng


def generate_motion_frames(
    scenario: str,
    num_frames: int = 16,
    width: int = 64,
    height: int = 64,
    seed: int = 1337,
) -> np.ndarray:
    """Generate a sequence of synthetic frames with distinct motion profiles."""
    rng = make_rng(seed)
    frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)

    # Dark background with subtle noise
    base_bg = rng.integers(20, 50, size=(height, width, 3), dtype=np.uint8)

    if scenario == "fight":
        # Rapid, erratic bursts of motion
        x = width // 2
        y = height // 2
        for t in range(num_frames):
            frame = base_bg.copy()
            # Sudden high-speed jumps
            dx = int(rng.integers(-12, 13))
            dy = int(rng.integers(-12, 13))
            x = int(np.clip(x + dx, 10, width - 20))
            y = int(np.clip(y + dy, 10, height - 20))

            # Draw energetic bright bounding boxes
            color1 = (
                int(rng.integers(150, 255)),
                int(rng.integers(50, 150)),
                int(rng.integers(50, 150)),
            )
            cv2.rectangle(frame, (x, y), (x + 16, y + 16), color1, -1)

            # Secondary erratic actor
            x2 = int(np.clip(x + rng.integers(-10, 10), 5, width - 15))
            y2 = int(np.clip(y + rng.integers(-10, 10), 5, height - 15))
            color2 = (
                int(rng.integers(50, 150)),
                int(rng.integers(150, 255)),
                int(rng.integers(50, 150)),
            )
            cv2.rectangle(frame, (x2, y2), (x2 + 12, y2 + 12), color2, -1)
            frames[t] = frame

    elif scenario == "nonfight":
        # Calm, smooth, slow linear drift
        x = 5.0
        y = float(height // 2 - 8)
        vx = float((width - 25) / max(1, num_frames - 1))
        color = (180, 180, 60)

        for t in range(num_frames):
            frame = base_bg.copy()
            cur_x = int(round(x + t * vx))
            cur_y = int(round(y))
            cv2.rectangle(frame, (cur_x, cur_y), (cur_x + 14, cur_y + 14), color, -1)
            frames[t] = frame
    else:
        raise ValueError(f"unknown scenario {scenario}")

    return frames


def write_synthetic_clip(
    path: Path | str,
    frames: np.ndarray,
    fps: float = 15.0,
) -> None:
    """Write frame array [T, H, W, 3] to a video file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(p), fourcc, fps, (w, h), isColor=True)
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VideoWriter for {p}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def create_mock_rwf_dataset(
    target_dir: Path | str,
    clips_per_class: int = 2,
    num_frames: int = 16,
    width: int = 64,
    height: int = 64,
    fps: float = 15.0,
    seed: int = 1337,
) -> Path:
    """Create a fully structured mock RWF-2000 dataset in target_dir.

    Generates:
      target_dir/train/Fight/
      target_dir/train/NonFight/
      target_dir/val/Fight/
      target_dir/val/NonFight/
    """
    root = Path(target_dir)
    splits = ["train", "val"]
    labels = ["Fight", "NonFight"]

    clip_count = 0
    for split in splits:
        for label in labels:
            folder = root / split / label
            folder.mkdir(parents=True, exist_ok=True)
            scenario = "fight" if label == "Fight" else "nonfight"

            for i in range(clips_per_class):
                clip_count += 1
                clip_seed = derive_seed(seed, f"{split}_{label}_{i}")
                clip_name = f"{split}_{label.lower()}_{i+1:02d}.avi"
                clip_path = folder / clip_name

                frames = generate_motion_frames(
                    scenario=scenario,
                    num_frames=num_frames,
                    width=width,
                    height=height,
                    seed=clip_seed,
                )
                write_synthetic_clip(clip_path, frames, fps=fps)

    return root
