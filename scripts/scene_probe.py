"""Linear probes on cached frozen Swin3D-T features (Day-1 scene-branch check).

(a) logistic regression on the window-mean clip feature
(b) per-window linear logit, clip logit = mean of top-k windows, plus BCE(z_k, 0)
    on every window of non-violent clips (lambda = 0.5)

Training uses the original and flipped views as two samples per clip. Evaluation
averages the probabilities of the two views (flip test-time augmentation).
Hyperparameters are selected on internal_val only; test is scored once at the end.
"""

from __future__ import annotations

import argparse
import os
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mosaic.scene_cache import cache_path  # noqa: E402
from mosaic.seeding import seed_everything  # noqa: E402
from mosaic.windows import top_k_for  # noqa: E402


def load(split_path: Path, manifest_path: Path, cache_dir: Path):
    assignment = json.loads(split_path.read_text(encoding="utf-8"))["assignment"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = {}
    for e in manifest["entries"]:
        part = assignment.get(e["rel_path"])
        if part is None:
            continue
        x = np.load(cache_path(cache_dir, e["rel_path"])).astype(np.float32)  # (2, W, D)
        data.setdefault(part, {"x": [], "y": []})
        data[part]["x"].append(x)
        data[part]["y"].append(int(e["target"]))
    return {k: (np.stack(v["x"]), np.asarray(v["y"])) for k, v in data.items()}


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "n": int(len(y)),
    }


def probe_logreg(d):
    xf, yf = d["fit"]
    mu = xf.mean(axis=(0, 1, 2))
    sd = xf.std(axis=(0, 1, 2)) + 1e-6
    clip = lambda x: ((x - mu) / sd).mean(axis=2)  # (N, 2, D)
    xtr = clip(xf).reshape(-1, xf.shape[-1])
    ytr = np.repeat(yf, 2)
    best = None
    for c in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0]:
        m = LogisticRegression(C=c, max_iter=5000).fit(xtr, ytr)
        xv, yv = d["internal_val"]
        pv = np.mean([m.predict_proba(clip(xv)[:, v])[:, 1] for v in range(2)], axis=0)
        r = metrics(yv, pv)
        key = (r["accuracy"], r["roc_auc"])
        if best is None or key > best[0]:
            best = (key, c, m, r)
    _, c, m, rv = best
    xt, yt = d["test"]
    pt = np.mean([m.predict_proba(clip(xt)[:, v])[:, 1] for v in range(2)], axis=0)
    return {"C": c, "internal_val": rv, "test": metrics(yt, pt)}


def probe_topk(d, device, lam=0.5, epochs=300):
    xf, yf = d["fit"]
    mu = torch.tensor(xf.mean(axis=(0, 1, 2)))
    sd = torch.tensor(xf.std(axis=(0, 1, 2)) + 1e-6)
    prep = lambda x: ((torch.tensor(x) - mu) / sd).to(device)
    k = top_k_for(xf.shape[2])

    def clip_logit(z):
        return z.topk(k, dim=-1).values.mean(-1)

    def predict(model, x):
        with torch.no_grad():
            z = model(prep(x)).squeeze(-1)  # (N, 2, W)
            return torch.sigmoid(clip_logit(z)).mean(1).cpu().numpy()

    xtr = prep(xf).reshape(-1, xf.shape[2], xf.shape[-1])  # (2N, W, D)
    ytr = torch.tensor(np.repeat(yf, 2), dtype=torch.float32, device=device)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    best = None
    for wd in [1e-3, 1e-2, 1e-1, 1.0]:
        seed_everything(42)
        model = torch.nn.Linear(xf.shape[-1], 1).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
        for ep in range(1, epochs + 1):
            model.train()
            perm = torch.randperm(len(ytr), device=device)
            for i in range(0, len(perm), 64):
                b = perm[i : i + 64]
                z = model(xtr[b]).squeeze(-1)
                loss = bce(clip_logit(z), ytr[b])
                neg = ytr[b] == 0
                if neg.any():
                    loss = loss + lam * bce(z[neg], torch.zeros_like(z[neg]))
                opt.zero_grad()
                loss.backward()
                opt.step()
            if ep % 10 == 0:
                model.eval()
                xv, yv = d["internal_val"]
                r = metrics(yv, predict(model, xv))
                key = (r["accuracy"], r["roc_auc"])
                if best is None or key > best[0]:
                    best = (key, wd, ep, {n: p.detach().clone() for n, p in model.state_dict().items()}, r)
    _, wd, ep, state, rv = best
    model = torch.nn.Linear(xf.shape[-1], 1).to(device)
    model.load_state_dict(state)
    model.eval()
    xt, yt = d["test"]
    return {"weight_decay": wd, "epoch": ep, "top_k": k, "lambda": lam,
            "internal_val": rv, "test": metrics(yt, predict(model, xt))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="artifacts/manifests/rwf2000_manifest.json")
    ap.add_argument("--split", default="artifacts/manifests/rwf2000_v3_split.json")
    ap.add_argument("--cache-dir", default=str(Path(os.environ.get("MOSAIC_CACHE", "C:/mosaic_cache")) / "swin3d_t_k400"))
    ap.add_argument("--out", default="artifacts/reports/v3/scene_probe_swin3d_t.json")
    args = ap.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = load(ROOT / args.split, ROOT / args.manifest, ROOT / args.cache_dir)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backbone": "swin3d_t KINETICS400_V1, frozen, head=identity",
        "data": "local RWF-2000 copy (50 frames @ 10 fps), v3 grouped split",
        "sizes": {k: int(len(v[1])) for k, v in d.items()},
        "threshold": 0.5,
        "tta": "mean of original + horizontal flip probabilities",
        "a_logreg_window_mean": probe_logreg(d),
        "b_linear_topk_mil": probe_topk(d, device),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
