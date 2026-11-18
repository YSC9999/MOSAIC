"""Build the RF-DETR-Seg person/mask cache. Run with the .venv interpreter.

Resumable: existing clip pickles are skipped.
Usage: .venv/Scripts/python.exe scripts/build_det_cache.py [--limit N] [--model RFDETRSegLarge]
"""

from __future__ import annotations

import argparse
import os
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mosaic.det_cache import person_class_ids, process_frame  # noqa: E402
from mosaic.windows import select_frame_indices  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="artifacts/manifests/rwf2000_manifest.json")
    ap.add_argument("--split", default="artifacts/manifests/rwf2000_v3_split.json")
    ap.add_argument("--cache-dir", default=str(Path(os.environ.get("MOSAIC_CACHE", "C:/mosaic_cache")) / "det_rfdetr_seg_large"))
    ap.add_argument("--model", default="RFDETRSegLarge")
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import rfdetr
    import torch

    torch.manual_seed(42)
    model = getattr(rfdetr, args.model)()
    pids = person_class_ids()
    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    assignment = json.loads((ROOT / args.split).read_text(encoding="utf-8"))["assignment"]
    entries = [e for e in manifest["entries"] if e["rel_path"] in assignment]
    if args.limit:
        entries = entries[: args.limit]
    cache_dir = ROOT / args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    log = cache_dir / "build_log.jsonl"

    done = skipped = failed = 0
    t0 = time.time()
    for i, e in enumerate(entries):
        out = cache_dir / (e["rel_path"].rsplit(".", 1)[0] + ".pkl")
        if out.exists():
            skipped += 1
            continue
        try:
            # Stream frames: only the current frame and its detections are held in memory.
            cap = cv2.VideoCapture(e["abs_path"])
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            n_declared = int(e.get("frame_count") or cap.get(cv2.CAP_PROP_FRAME_COUNT))
            idx = select_frame_indices(n_declared)
            wanted = set(idx)
            frames = []
            n_people = 0
            fi = 0
            h = w = None
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                if h is None:
                    h, w = f.shape[:2]
                if fi in wanted:
                    rec = process_frame(model, cv2.cvtColor(f, cv2.COLOR_BGR2RGB), args.threshold, pids)
                    rec["frame_index"] = int(fi)
                    rec["timestamp"] = fi / fps
                    n_people += len(rec["boxes"])
                    frames.append(rec)
                fi += 1
            cap.release()
            if not frames:
                raise RuntimeError("no frames decoded")
            if len(frames) != len(idx):
                raise RuntimeError(f"decoded {fi} frames, expected {n_declared}; got {len(frames)}/{len(idx)} sampled")
            payload = {
                "meta": {
                    "rel_path": e["rel_path"], "n_frames": fi, "fps": fps, "height": h, "width": w,
                    "model": args.model, "rfdetr_version": rfdetr.__version__ if hasattr(rfdetr, "__version__") else None,
                    "threshold": args.threshold, "person_class_ids": pids, "sampled_indices": idx,
                },
                "frames": frames,
            }
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp")
            with tmp.open("wb") as fh:
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(out)
            done += 1
            rec_log = {"rel_path": e["rel_path"], "frames": len(frames), "dets": n_people,
                       "valid_masks": int(sum(f["mask_valid"].sum() for f in frames))}
        except Exception as exc:
            failed += 1
            rec_log = {"rel_path": e["rel_path"], "error": repr(exc)}
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec_log, ensure_ascii=False) + "\n")
        if (done + failed) % 25 == 0:
            rate = (done + failed) / (time.time() - t0)
            print(f"[{i + 1}/{len(entries)}] done={done} failed={failed} {rate * 60:.1f} clips/min "
                  f"eta={(len(entries) - i - 1) / max(rate, 1e-9) / 3600:.2f} h", flush=True)
    el = time.time() - t0
    print(f"finished: done={done} skipped={skipped} failed={failed} in {el / 3600:.2f} h "
          f"({el / max(done, 1):.1f} s/clip)", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
