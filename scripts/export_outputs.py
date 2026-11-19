"""Spatio-temporal outputs for test clips from a saved v3 checkpoint.

Per clip: clip probability and class, per-window violence probabilities with time spans, violent interval(s), and one
group box per frame of every violent window (attention-weighted union of the high-attention people's boxes).
Writes <out>/<clip>.json for every clip and an overlay mp4 for the first --render clips.

Usage: python scripts/export_outputs.py CKPT --out DIR [--n 50] [--render 10] [--split test] [--seed 7]
Environment (same as training): MOSAIC_CACHE, MOSAIC_MANIFEST, MOSAIC_SPLIT, MOSAIC_SKD, MOSAIC_DET, MOSAIC_TRACKS.
Attention is the model's focus, not verified involvement. Run with the interpreter that has torch + cv2.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mosaic.models.fusion_v3 import FusionV3, clip_logit  # noqa: E402
from mosaic.outputs import group_boxes, intervals_from_windows, window_time_spans  # noqa: E402
from mosaic.train_v3 import CACHE, load_all, make_batch  # noqa: E402
from mosaic.windows import top_k_for, window_index_matrix  # noqa: E402


def run_model(model, d, idx, stats, device):
    """Window probabilities and attention for clip ``idx``, averaged over the original and flipped views."""
    probs, attns, clip_p = [], [], []
    for flip in (False, True):
        b = make_batch(d, torch.tensor([idx]), torch.tensor([flip]), stats, device)
        with torch.no_grad():
            out = model.window_logits(b["scene"], b["nodes"], b["mask"], b["edges"], b["motion"], b["count"])
        z = out["logit"].view(1, b["W"])
        probs.append(torch.sigmoid(z)[0].cpu().numpy())
        clip_p.append(float(torch.sigmoid(clip_logit(z))[0]))
        attns.append(out["attn"].cpu().numpy() if out["attn"] is not None else None)
    attn = None if attns[0] is None else (attns[0] + attns[1]) / 2
    return np.mean(probs, axis=0), attn, float(np.mean(clip_p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--render", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--only-violent", action="store_true", help="choose clips from the Fight class only")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = FusionV3(use_scene=cfg["use_scene"], use_person=cfg["use_person"], use_gat=cfg["use_gat"],
                     dropout=cfg["dropout"], scene_in_dropout=cfg["scene_in_dropout"]).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    stats = ck["stats"]
    manifest = os.environ.get("MOSAIC_MANIFEST", "artifacts/manifests/rwf2000_manifest.json")
    split = os.environ.get("MOSAIC_SPLIT", "artifacts/manifests/rwf2000_v3_split.json")
    data = load_all(ROOT / manifest, ROOT / split)[a.split]
    entries = {e["rel_path"]: e for e in json.loads((ROOT / manifest).read_text(encoding="utf-8"))["entries"]}
    det_dir = CACHE / os.environ.get("MOSAIC_DET", "det_rfdetr_seg_large")
    trk_dir = CACHE / os.environ.get("MOSAIC_TRACKS", "tracks_v2")
    skd_dir = CACHE / os.environ.get("MOSAIC_SKD", "skd_v2")

    pool = [i for i in range(len(data["y"])) if (not a.only_violent or data["y"][i] == 1)]
    rng = np.random.default_rng(a.seed)
    chosen = sorted(rng.choice(pool, size=min(a.n, len(pool)), replace=False).tolist())
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for n_done, i in enumerate(chosen):
        rel = data["rel"][i]
        stem = rel.rsplit(".", 1)[0]
        with (det_dir / (stem + ".pkl")).open("rb") as fh:
            det = pickle.load(fh)
        with (trk_dir / (stem + ".pkl")).open("rb") as fh:
            trk = pickle.load(fh)
        members = np.load(skd_dir / (stem + ".npz"))["members"]                  # (W, P) track ids
        wp, attn, clip_p = run_model(model, data, i, stats, device)
        meta = det["meta"]
        n_sampled = int(meta.get("n_sampled", len(det["frames"])))
        ts = [fi / meta["fps"] for fi in meta["sampled_indices"]]                      # time of every sampled position
        mat = window_index_matrix(n_sampled)
        spans = window_time_spans(ts, [int(r[0]) for r in mat])
        iv = intervals_from_windows(wp, spans)
        label = int(clip_p >= 0.5)
        W0, H0 = meta["width"], meta["height"]
        # gated caches store rows only for kept frames (each row carries its position); tracks cover the gate's kept frames
        by_pos = {f.get("pos", r): f for r, f in enumerate(det["frames"])}
        kept = trk.get("kept_pos", list(range(len(det["frames"]))))
        boxes_by_pos = {}
        for pos, ids in zip(kept, trk["track_ids"]):
            boxes_by_pos[int(pos)] = {int(t): by_pos[pos]["boxes"][k] for k, t in enumerate(ids) if t >= 0}
        win_boxes = {}
        if label and attn is not None:
            for w in iv["windows"]:
                if members[w].max() < 0:
                    continue
                win_boxes[int(w)] = group_boxes(attn[w], members[w], boxes_by_pos, [int(p) for p in mat[w]], (W0, H0))
        rec = {"clip": rel, "true_label": int(data["y"][i]), "clip_prob": clip_p, "pred_label": label,
               "window_prob": [float(x) for x in wp], "window_smoothed": [float(x) for x in iv["smoothed"]],
               "window_span_s": spans.tolist(), "violent_windows": iv["windows"],
               "intervals_s": [[r[0], r[1]] for r in iv["intervals"]], "primary_interval": iv["primary"],
               "sampled_timestamps_s": ts, "group_boxes": {str(w): {str(p): b for p, b in bb.items()} for w, bb in win_boxes.items()},
               "note": "attention is the model's focus, not verified involvement"}
        (out_dir / (Path(rel).stem + ".json")).write_text(json.dumps(rec), encoding="utf-8")
        summary.append({"clip": rel, "true": rec["true_label"], "prob": clip_p, "pred": label, "intervals": rec["intervals_s"]})
        if n_done < a.render and label:
            render(entries[rel]["abs_path"], meta["sampled_indices"], rec, out_dir / (Path(rel).stem + ".mp4"), n_sampled)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    acc = np.mean([s["true"] == s["pred"] for s in summary])
    print(f"exported {len(summary)} clips to {out_dir}; clip accuracy on this sample {acc:.3f}")


def render(video: str, sampled_idx, rec, out_path: Path, n_pos: int) -> None:
    """Overlay: group box in violent windows, probability timeline, interval caption. 12.8 fps on the sampled frames."""
    cap = cv2.VideoCapture(video)
    want = {fi: p for p, fi in enumerate(sampled_idx)}
    held, fi = {}, 0
    while True:
        ok, f = cap.read()
        if not ok or fi > sampled_idx[-1]:
            break
        if fi in want:
            held[want[fi]] = f
        fi += 1
    cap.release()
    h0, w0 = held[0].shape[:2]
    sc = min(1.0, 960 / w0)
    W, H = int(w0 * sc), int(h0 * sc)
    ts = rec["sampled_timestamps_s"]
    fps = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 else 12.8
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H + 40))
    win_of = {}
    for w, bb in rec["group_boxes"].items():
        for p, b in bb.items():
            win_of.setdefault(int(p), b)
    spans = rec["window_span_s"]
    for p in range(n_pos):
        img = cv2.resize(held[p], (W, H))
        b = win_of.get(p)
        inside = any(s[0] <= ts[p] < s[1] for s in rec["intervals_s"])
        if b is not None and inside:
            cv2.rectangle(img, (int(b[0] * sc), int(b[1] * sc)), (int(b[2] * sc), int(b[3] * sc)), (0, 0, 255), 3)
        bar = np.full((40, W, 3), 30, np.uint8)
        t_end = ts[-1] + 1 / fps
        for (s0, s1), pr in zip(spans, rec["window_prob"]):
            x0, x1 = int(W * s0 / t_end), int(W * min(s1, t_end) / t_end)
            cv2.rectangle(bar, (x0, 8), (x1, 32), (int(255 * (1 - pr)), 60, int(255 * pr)), -1)
        cv2.line(bar, (int(W * ts[p] / t_end),) * 1 + (0,), (int(W * ts[p] / t_end), 39), (255, 255, 255), 2)
        iv = rec["intervals_s"][rec["primary_interval"]]
        cv2.putText(img, f"violent p={rec['clip_prob']:.2f}  interval {iv[0]:.1f}-{iv[1]:.1f}s  (box = model focus area)",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        vw.write(np.vstack([img, bar]))
    vw.release()


if __name__ == "__main__":
    main()
