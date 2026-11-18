"""Cache per-window motion scores -> C:/mosaic_cache/motion/<rel>.npy (resumable)."""
from __future__ import annotations

import json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mosaic.motion import gray_frames, window_motion  # noqa: E402

out_dir = Path(os.environ.get("MOSAIC_CACHE", "C:/mosaic_cache")) / "motion"
m = json.loads((ROOT / os.environ.get("MOSAIC_MANIFEST", "artifacts/manifests/rwf2000_manifest.json")).read_text(encoding="utf-8"))
a = json.loads((ROOT / os.environ.get("MOSAIC_SPLIT", "artifacts/manifests/rwf2000_v3_split.json")).read_text(encoding="utf-8"))["assignment"]
n = 0
for e in m["entries"]:
    if e["rel_path"] not in a:
        continue
    p = out_dir / (e["rel_path"].rsplit(".", 1)[0] + ".npy")
    if p.exists():
        continue
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, window_motion(gray_frames(e["abs_path"], n_frames=int(e["frame_count"]))))
    n += 1
print(f"finished: wrote {n}")
