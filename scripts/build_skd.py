"""Build per-clip silhouette-kinematic tensors from the det + track caches. Run with .venv.

Output npz per clip (float16 nodes/edges): nodes (W,P,T,29), node_mask, edges (W,T,P,P,9),
count, members. Resumable; skips clips without tracks yet or already done.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mosaic.skd import build_clip_tensors  # noqa: E402

CACHE = Path(os.environ.get("MOSAIC_CACHE", "C:/mosaic_cache"))


def main() -> None:
    trk_name = os.environ.get("MOSAIC_TRACKS", "tracks_bytetrack")
    det_dir, trk_dir, out_dir = CACHE / os.environ.get("MOSAIC_DET", "det_rfdetr_seg_large"), CACHE / trk_name, CACHE / os.environ.get("MOSAIC_SKD", "skd")
    done = skipped = failed = 0
    t0 = time.time()
    for tp in sorted(trk_dir.rglob("*.pkl")):
        rel = tp.relative_to(trk_dir)
        out = out_dir / rel.with_suffix(".npz")
        if out.exists():
            skipped += 1
            continue
        try:
            d = pickle.load((det_dir / rel).open("rb"))
            t = pickle.load(tp.open("rb"))
            if "kept_pos" in t:   # motion-gated clip: detections exist only for the kept frames
                if "pos" in d["frames"][0]:     # gated det cache: rows addressed by their position
                    by_pos = {f["pos"]: f for f in d["frames"]}
                else:
                    by_pos = {q: f for q, f in enumerate(d["frames"])}
                d = {"meta": d["meta"], "frames": [dict(by_pos[q], pos=q) for q in t["kept_pos"]]}
            arrays = build_clip_tensors(d, t)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, **arrays)
            done += 1
        except Exception as exc:
            failed += 1
            print("FAILED", rel, repr(exc), flush=True)
        if (done + failed) % 50 == 0 and (done + failed):
            print(f"done={done} failed={failed} {(done + failed) / (time.time() - t0):.2f} clips/s", flush=True)
    print(f"finished: done={done} skipped={skipped} failed={failed} in {time.time() - t0:.0f} s")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
