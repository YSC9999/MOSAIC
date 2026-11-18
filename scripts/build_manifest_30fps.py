"""Manifest for the 30 fps RWF-2000 copy (data/RWF-2000_30fps), same schema as the 10 fps manifest.

official val -> split "test"; official train -> split "train" (the grouped internal split is built later).
Exact duplicates (sha256) are removed keeping the TEST copy (test set stays intact), as for the 10 fps copy.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "RWF-2000_30fps" / "RWF-2000"
OUT = ROOT / "artifacts" / "manifests" / "rwf2000_30fps_manifest.json"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    entries, corrupt = [], []
    paths = sorted(BASE.glob("*/*/*.avi"), key=lambda p: (p.parts[-3] == "train", str(p)))   # val/test first, so duplicates are dropped from train
    for i, p in enumerate(paths):
        split_dir, label = p.parts[-3], p.parts[-2]
        rel = f"{split_dir}/{label}/{p.name}"
        cap = cv2.VideoCapture(str(p))
        ok = cap.isOpened()
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if ok else 0
        fps = float(cap.get(cv2.CAP_PROP_FPS)) if ok else 0.0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        first_ok = ok and cap.read()[0]
        cap.release()
        e = {"clip_id": f"{label}/{p.name}", "abs_path": str(p), "rel_path": rel, "label": label,
             "target": 1 if label == "Fight" else 0, "split": "train" if split_dir == "train" else "test",
             "sha256": sha256(p), "frame_count": n, "fps": fps, "duration_seconds": round(n / fps, 2) if fps else 0.0,
             "width": w, "height": h, "decode_status": "ok" if (first_ok and n > 0 and fps > 0) else "failed", "error": None}
        (entries if e["decode_status"] == "ok" else corrupt).append(e)
        if (i + 1) % 250 == 0:
            print(f"{i + 1}/{len(paths)}", flush=True)
    seen, kept, removed = {}, [], []
    for e in entries:
        if e["sha256"] in seen:
            k = seen[e["sha256"]]
            removed.append({"kept_split": k["split"], "kept_clip": k["rel_path"], "removed_split": e["split"],
                            "removed_clip": e["rel_path"], "sha256": e["sha256"]})
        else:
            seen[e["sha256"]] = e
            kept.append(e)
    import collections
    meta = {"schema_version": "manifest-0.1", "dataset_name": "RWF-2000 (30 fps, Kaggle vulamnguyen/rwf2000)",
            "created_utc": datetime.now(timezone.utc).isoformat(), "corrupt_count": len(corrupt),
            "removed_exact_duplicate_count": len(removed), "removed_exact_duplicates": removed,
            "frame_count_histogram": dict(collections.Counter(e["frame_count"] for e in kept).most_common(5)),
            "fps_histogram": dict(collections.Counter(round(e["fps"]) for e in kept)),
            "name_map": "data/RWF-2000_30fps/name_map.json"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"meta": meta, "entries": kept, "corrupt_entries": corrupt, "near_duplicates": []},
                              indent=1, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "removed_exact_duplicates"}, indent=1))
    print("kept", len(kept), collections.Counter((e["split"], e["label"]) for e in kept))


if __name__ == "__main__":
    main()
