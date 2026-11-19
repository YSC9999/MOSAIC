"""Detection stage: motion gate first, then detect ONLY the kept frames (RF-DETR-Seg). Gating is not optional. Run with .venv.

For each clip: decode the 64 sampled frames, run the motion gate (--gates, default gdef) on 64x64 thumbnails, detect only the frames
kept by it (the union if several gates are named), and store those rows with their position ``pos`` (0..63). ``meta["gate_kept"][name]`` lists the
positions each setting keeps, so tracks for each setting can be built from the same cache.
Clips already detected on all 64 frames (--reuse-dir) are converted without running the detector (identical rows).
Usage: .venv/Scripts/python.exe scripts/build_det_gated.py --manifest M --split S --model RFDETRSegNano --cache-dir D [--reuse-dir R]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mosaic.det_cache import person_class_ids, process_frame  # noqa: E402
from mosaic.motion_gate import DEFAULT_GATE, GATE_SETTINGS, make_gate  # noqa: E402
from mosaic.windows import sample_indices  # noqa: E402

def gate_kept(grays, ts, gates):
    """Positions kept by each named gate, e.g. {"gdef": [0, 1, 4, ...]}."""
    return {name: [int(i) for i in make_gate(name).filter_clip(grays, ts).selected_indices] for name in gates}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--reuse-dir", default=None)
    ap.add_argument("--model", default="RFDETRSegNano")
    ap.add_argument("--gates", default=DEFAULT_GATE, help=f"comma-separated gate names from {sorted(GATE_SETTINGS)}")
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    gates = [g for g in a.gates.split(",") if g]
    unknown = [g for g in gates if g not in GATE_SETTINGS]
    if not gates or unknown:
        ap.error(f"--gates must name gates from {sorted(GATE_SETTINGS)}, got {a.gates!r}")
    manifest = json.loads((ROOT / a.manifest).read_text(encoding="utf-8"))
    assign = json.loads((ROOT / a.split).read_text(encoding="utf-8"))["assignment"]
    entries = [e for e in manifest["entries"] if e["rel_path"] in assign]
    if a.limit:
        entries = entries[: a.limit]
    cache, reuse = Path(a.cache_dir), (Path(a.reuse_dir) if a.reuse_dir else None)
    cache.mkdir(parents=True, exist_ok=True)
    model = pids = None
    done = conv = failed = frames_total = frames_det = 0
    t0 = time.time()
    for i, e in enumerate(entries):
        out = cache / (e["rel_path"].rsplit(".", 1)[0] + ".pkl")
        if out.exists():
            continue
        try:
            old = reuse / out.relative_to(cache) if reuse else None
            if old is not None and old.exists():                              # convert an all-frames cache
                with old.open("rb") as fh:
                    d = pickle.load(fh)
                fr = d["frames"]
                kept = gate_kept([f["gray64"] for f in fr], [f["timestamp"] for f in fr], gates)
                union = sorted(set().union(*kept.values()))
                rows = [dict(fr[p], pos=p) for p in union]
                n_samp = len(fr)
                meta = dict(d["meta"])
                conv += 1
            else:
                if model is None:
                    import rfdetr
                    import torch
                    torch.manual_seed(42)
                    model, pids = getattr(rfdetr, a.model)(), person_class_ids()
                cap = cv2.VideoCapture(e["abs_path"])
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                n_decl = int(e["frame_count"])
                idx, is_pad = sample_indices(n_decl)                              # always 64 positions; short clips are padded
                real = [p for p in range(len(idx)) if not is_pad[p]]
                wanted = {idx[p]: p for p in real}
                held, fi = {}, 0
                while True:                                                    # hold only the 64 sampled frames
                    ok, f = cap.read()
                    if not ok:
                        break
                    if fi in wanted:
                        held[wanted[fi]] = f
                    fi += 1
                    if fi > idx[-1]:
                        break
                cap.release()
                if len(held) != len(real):
                    raise RuntimeError(f"decoded {len(held)}/{len(real)} sampled frames")
                for p in range(len(idx)):                                          # padded positions copy the last real frame
                    if is_pad[p]:
                        held[p] = held[real[-1]]
                grays = [cv2.resize(cv2.cvtColor(held[p], cv2.COLOR_BGR2GRAY), (64, 64), interpolation=cv2.INTER_AREA)
                         for p in range(len(idx))]
                ts = [idx[p] / fps for p in range(len(idx))]
                kept = gate_kept(grays, ts, gates)
                kept = {g: [q for q in v if not is_pad[q]] for g, v in kept.items()}   # never detect padded positions
                union = sorted(set().union(*kept.values()))
                rows = []
                for p in union:
                    rec = process_frame(model, cv2.cvtColor(held[p], cv2.COLOR_BGR2RGB), a.threshold, pids)
                    rec.update(frame_index=int(idx[p]), timestamp=ts[p], pos=p, gray64=grays[p])
                    rows.append(rec)
                h, w = held[0].shape[:2]
                n_samp = len(idx)
                meta = {"rel_path": e["rel_path"], "n_frames": n_decl, "fps": fps, "height": h, "width": w,
                        "model": a.model, "threshold": a.threshold, "person_class_ids": pids, "sampled_indices": idx,
                        "is_pad": is_pad}
                del held
                done += 1
            meta.update(n_sampled=n_samp, gate_kept=kept, gates={g: list(GATE_SETTINGS[g]) for g in gates})
            frames_total += n_samp
            frames_det += len(rows)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp")
            with tmp.open("wb") as fh:
                pickle.dump({"meta": meta, "frames": rows}, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(out)
        except Exception as exc:
            failed += 1
            print("FAILED", e["rel_path"], repr(exc), flush=True)
        n = done + conv + failed
        if n % 25 == 0 and n:
            print(f"[{i + 1}/{len(entries)}] detected={done} converted={conv} failed={failed} "
                  f"share_of_frames_detected={frames_det / max(frames_total, 1):.2f} {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"finished: detected={done} converted={conv} failed={failed} "
          f"share_of_frames_detected={frames_det / max(frames_total, 1):.3f} in {(time.time() - t0) / 3600:.2f} h", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
