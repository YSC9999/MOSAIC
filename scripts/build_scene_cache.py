"""Cache frozen Swin3D-T window features for every clip in the v3 split.

Resumable: clips whose cache file already exists are skipped.
Usage: python scripts/build_scene_cache.py [--limit N] [--cache-dir DIR]
"""

from __future__ import annotations

import argparse
import os
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mosaic.scene_cache import cache_path, extract_clip_features, load_backbone, read_clip_frames  # noqa: E402
from mosaic.seeding import seed_everything  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="artifacts/manifests/rwf2000_manifest.json")
    ap.add_argument("--split", default="artifacts/manifests/rwf2000_v3_split.json")
    ap.add_argument("--cache-dir", default=str(Path(os.environ.get("MOSAIC_CACHE", "C:/mosaic_cache")) / "swin3d_t_k400"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=3)
    args = ap.parse_args()

    seed_everything(42)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    assignment = json.loads((ROOT / args.split).read_text(encoding="utf-8"))["assignment"]
    entries = [e for e in manifest["entries"] if e["rel_path"] in assignment]
    if args.limit:
        entries = entries[: args.limit]
    cache_dir = ROOT / args.cache_dir

    model = load_backbone(device)
    done = skipped = failed = 0
    t0 = time.time()
    log = cache_dir / "build_log.jsonl"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for i, e in enumerate(entries):
        out = cache_path(cache_dir, e["rel_path"])
        if out.exists():
            skipped += 1
            continue
        try:
            frames, is_pad = read_clip_frames(e["abs_path"], n_frames=int(e["frame_count"]))
            feats = extract_clip_features(model, frames, device, batch=args.batch)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, feats)
            done += 1
            rec = {"rel_path": e["rel_path"], "shape": list(feats.shape), "n_pad": int(sum(is_pad))}
        except Exception as exc:  # record and continue; failures are reported at the end
            failed += 1
            rec = {"rel_path": e["rel_path"], "error": repr(exc)}
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if (done + failed) % 50 == 0:
            rate = (done + failed) / (time.time() - t0)
            left = (len(entries) - i - 1) / max(rate, 1e-9)
            print(f"[{i + 1}/{len(entries)}] done={done} failed={failed} {rate:.2f} clips/s eta={left / 60:.1f} min", flush=True)
    elapsed = time.time() - t0
    print(f"finished: done={done} skipped={skipped} failed={failed} in {elapsed / 60:.1f} min "
          f"({elapsed / max(done, 1):.2f} s/clip)")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
