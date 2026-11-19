"""Tracking stage over a gated detection cache (built by build_det.py). Gating is not optional. Run with .venv.

For each clip: take the frames the gate kept (stored in the cache), run ByteTrack over them (cleaned settings: high 0.5,
NMS 0.7), and store window membership on the 16-frame window layout of the 64 sampled frames.
Usage: .venv/Scripts/python.exe scripts/build_tracks.py --det-dir D --out-dir O [--setting gdef --min-frames 4]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mosaic.motion_gate import DEFAULT_GATE  # noqa: E402
from mosaic.track_join import drop_short_tracks, gated_windows, track_clip  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--setting", default=DEFAULT_GATE, help="gate whose kept frames to track (must be in the cache metadata)")
    ap.add_argument("--min-frames", type=int, default=4)
    ap.add_argument("--high", type=float, default=0.5)
    ap.add_argument("--nms-iou", type=float, default=0.7)
    a = ap.parse_args()
    det_dir, out_dir = Path(a.det_dir), Path(a.out_dir)
    st = {"clips": 0, "sampled": 0, "kept": 0, "windows": 0, "w_ge1": 0, "w_ge2": 0, "people": 0}
    t0 = time.time()
    for p in sorted(det_dir.rglob("*.pkl")):
        out = out_dir / p.relative_to(det_dir)
        if out.exists():
            continue
        with p.open("rb") as fh:
            d = pickle.load(fh)
        frames = d["frames"]
        if a.setting not in d["meta"].get("gate_kept", {}):
            raise SystemExit(f"{p}: cache has no gate {a.setting!r}; rebuild it with build_det.py --gates {a.setting}")
        kept = list(d["meta"]["gate_kept"][a.setting])
        by_pos = {f["pos"]: f for f in frames}
        sub = [by_pos[q] for q in kept]
        n_sampled = int(d["meta"]["n_sampled"])
        ids = drop_short_tracks(track_clip(sub, high=a.high, nms_iou=a.nms_iou))
        gw = gated_windows(kept, ids, [f["scores"] for f in sub], n_sampled, min_frames=a.min_frames)
        payload = {"meta": {"rel_path": d["meta"]["rel_path"], "n_sampled": n_sampled, "gate": vars(a)},
                   "track_ids": ids, "kept_pos": kept, **gw}
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        st["clips"] += 1; st["sampled"] += n_sampled; st["kept"] += len(kept)
        for w in gw["windows"]:
            k = len(w["members"]); st["windows"] += 1; st["w_ge1"] += k >= 1; st["w_ge2"] += k >= 2; st["people"] += k
    if st["clips"]:
        print(json.dumps({"clips": st["clips"], "share_frames_kept": st["kept"] / st["sampled"],
                          "mean_people_per_window": st["people"] / st["windows"],
                          "share_windows_ge1": st["w_ge1"] / st["windows"], "share_windows_ge2": st["w_ge2"] / st["windows"],
                          "gate": vars(a)}, indent=1))
    print(f"finished in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
