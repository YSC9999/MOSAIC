"""Track every cached detection clip and compute window membership. Run with .venv.

Skips clips without a detection pickle yet and clips already tracked.
Usage: .venv/Scripts/python.exe scripts/build_tracks.py
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

from mosaic.track_join import drop_short_tracks, track_clip, window_members  # noqa: E402
from mosaic.windows import window_index_matrix  # noqa: E402

CACHE = Path(os.environ.get("MOSAIC_CACHE", "C:/mosaic_cache"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-dir", default=str(CACHE / "det_rfdetr_seg_large"))
    ap.add_argument("--out-dir", default=str(CACHE / "tracks_bytetrack"))
    ap.add_argument("--high", type=float, default=0.25)
    ap.add_argument("--nms-iou", type=float, default=None)
    args = ap.parse_args()
    det_dir, out_dir = Path(args.det_dir), Path(args.out_dir)
    done = skipped = 0
    stats = {"clips": 0, "dets": 0, "tracked_dets": 0, "tracks": 0, "windows": 0,
             "windows_ge1": 0, "windows_ge2": 0, "people_per_window": 0}
    t0 = time.time()
    for p in sorted(det_dir.rglob("*.pkl")):
        rel = p.relative_to(det_dir)
        out = out_dir / rel
        if out.exists():
            skipped += 1
            continue
        with p.open("rb") as fh:
            d = pickle.load(fh)
        frames = d["frames"]
        ids = drop_short_tracks(track_clip(frames, high=args.high, nms_iou=args.nms_iou))
        rows = window_index_matrix(len(frames)).tolist()
        wins = window_members(ids, [f["scores"] for f in frames], rows)
        payload = {"meta": {"rel_path": d["meta"]["rel_path"], "n_sampled": len(frames)},
                   "track_ids": ids, "window_rows": rows, "windows": wins}
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        done += 1
        stats["clips"] += 1
        stats["dets"] += sum(len(a) for a in ids)
        stats["tracked_dets"] += sum(int((a >= 0).sum()) for a in ids)
        stats["tracks"] += len({t for a in ids for t in a.tolist() if t >= 0})
        for w in wins:
            k = len(w["members"])
            stats["windows"] += 1
            stats["windows_ge1"] += k >= 1
            stats["windows_ge2"] += k >= 2
            stats["people_per_window"] += k
    if stats["clips"]:
        s = dict(stats)
        s["tracks_per_clip"] = s["tracks"] / s["clips"]
        s["share_dets_tracked"] = s["tracked_dets"] / max(1, s["dets"])
        s["mean_people_per_window"] = s["people_per_window"] / max(1, s["windows"])
        s["share_windows_ge1"] = s["windows_ge1"] / max(1, s["windows"])
        s["share_windows_ge2"] = s["windows_ge2"] / max(1, s["windows"])
        print(json.dumps(s, indent=1))
    print(f"finished: done={done} skipped={skipped} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
