"""Fraction of sampled frames kept by the motion gate for several settings (uses cached 64x64 thumbnails).

Usage: python scripts/gate_sweep.py DET_DIR [--max-clips N]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mosaic.motion_gate import MotionGate  # noqa: E402

SETTINGS = {  # name: (threshold, burst_threshold, burst_hold, keep_alive)
    "default": (0.010, 0.020, 5, 15),
    "t.02": (0.020, 0.040, 5, 15),
    "t.03": (0.030, 0.060, 3, 15),
    "t.05": (0.050, 0.100, 3, 15),
    "t.08": (0.080, 0.150, 2, 15),
    "t.12": (0.120, 0.250, 2, 15),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("det_dir")
    ap.add_argument("--max-clips", type=int, default=400)
    a = ap.parse_args()
    files = sorted(Path(a.det_dir).rglob("*.pkl"))[: a.max_clips]
    clips = []
    for f in files:
        with f.open("rb") as fh:
            d = pickle.load(fh)
        clips.append(([fr["gray64"] for fr in d["frames"]], "Fight" in str(f) and "NonFight" not in str(f)))
    print(f"{len(clips)} clips ({sum(c[1] for c in clips)} fight)")
    for name, (t, b, h, k) in SETTINGS.items():
        g = MotionGate(True, t, b, h, k)
        fr_all, fr_fight, fr_non, min_win = [], [], [], []
        for grays, fight in clips:
            sel = g.filter_clip(grays).selected_indices
            frac = len(sel) / len(grays)
            fr_all.append(frac)
            (fr_fight if fight else fr_non).append(frac)
            starts = [0, 8, 16, 24, 32, 40, 48]
            min_win.append(min(sum(s <= i < s + 16 for i in sel) for s in starts))
        print(f"{name:8s} kept {np.mean(fr_all):.2f} (fight {np.mean(fr_fight):.2f} / non-fight {np.mean(fr_non):.2f}) | "
              f"clips with a window having <4 kept frames: {np.mean([m < 4 for m in min_win]):.2f}")


if __name__ == "__main__":
    main()
