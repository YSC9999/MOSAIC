"""v3 Stage-A training on cached features (spec sections 8-12).

Everything heavy is cached: Swin window features (2 views), person descriptors
(SKD), motion scores. Standardization statistics come from the ``fit`` split only.
Selection uses ``internal_val`` clip accuracy (ties: lower loss); ``test`` is scored
once with the selected checkpoint.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .models.fusion_v3 import FusionV3, clip_logit
from .scene_cache import cache_path
from .seeding import seed_everything
from .skd import EDGE_FEATURES, FLIP_MIRROR_NODE, FLIP_NEGATE_EDGE, FLIP_NEGATE_NODE, NODE_FEATURES

# Mask-derived features (A4 box-only ablation): fill ratio, 7 shape feats, reach/mask-IoU rates, overlap, mask flag.
MASK_NODE_FEATURES = ["mask_fill", "log_mask_aspect", "reach", "solidity", "upper_width", "eccentricity", "cos2t",
                      "sin2t", "dreach", "mask_iou_prev", "max_overlap", "mask_valid"]
MASK_EDGE_FEATURES = ["mask_overlap"]

CACHE = Path(os.environ.get("MOSAIC_CACHE", "C:/mosaic_cache"))


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 30
    warmup_epochs: int = 2
    batch_clips: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.05
    patience: int = 6
    window_neg_weight: float = 0.5
    use_scene: bool = True
    use_person: bool = True
    use_gat: bool = False           # default: no interaction graph (GAT). It never helped; enable with --gat
    box_only: bool = False          # A4: zero mask-derived node/edge features
    dropout: float = 0.3            # fusion head dropout
    scene_in_dropout: float = 0.0   # dropout on the standardized scene features
    overfit_clips: int = 0          # >0: gate run, train+eval on this many fit clips
    name: str = "a1_full"
    save_ckpt: bool = False         # write <out_dir>/<name>.pt (best weights + fit statistics) for export / inference
    out_dir: str = "artifacts/reports/v3/runs"


# ----------------------------------------------------------------------------- data
def load_all(manifest_path: Path, split_path: Path, cache: Path = CACHE) -> Dict[str, Dict[str, torch.Tensor]]:
    assignment = json.loads(split_path.read_text(encoding="utf-8"))["assignment"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: Dict[str, List] = {}
    for e in manifest["entries"]:
        part = assignment.get(e["rel_path"])
        if part is None:
            continue
        stem = e["rel_path"].rsplit(".", 1)[0]
        z = np.load(cache / os.environ.get("MOSAIC_SKD", "skd") / (stem + ".npz"))
        rows.setdefault(part, []).append({
            "scene": np.load(cache_path(cache / "swin3d_t_k400", e["rel_path"])),
            "motion": np.load(cache / "motion" / (stem + ".npy")),
            "nodes": z["nodes"], "mask": z["node_mask"], "edges": z["edges"], "count": z["count"],
            "y": int(e["target"]), "rel": e["rel_path"],
        })
    out = {}
    for part, rs in rows.items():
        out[part] = {
            "scene": torch.from_numpy(np.stack([r["scene"] for r in rs])),        # (N,2,W,768) f16
            "motion": torch.from_numpy(np.stack([r["motion"] for r in rs])),      # (N,W)
            "nodes": torch.from_numpy(np.stack([r["nodes"] for r in rs])),        # (N,W,P,T,29) f16
            "mask": torch.from_numpy(np.stack([r["mask"] for r in rs])),          # (N,W,P,T)
            "edges": torch.from_numpy(np.stack([r["edges"] for r in rs])),        # (N,W,T,P,P,9) f16
            "count": torch.from_numpy(np.stack([r["count"] for r in rs])),        # (N,W)
            "y": torch.tensor([r["y"] for r in rs], dtype=torch.float32),
            "rel": [r["rel"] for r in rs],
        }
    return out


def _running_mean_std(chunks) -> tuple:
    """Mean/std over rows from an iterator of (n, D) float tensors, using float64 sums (no big copies)."""
    n, s1, s2 = 0, None, None
    for c in chunks:
        c = c.double()
        n += c.shape[0]
        s1 = c.sum(0) if s1 is None else s1 + c.sum(0)
        s2 = (c * c).sum(0) if s2 is None else s2 + (c * c).sum(0)
    mu = s1 / n
    sd = (s2 / n - mu * mu).clamp_min(0).sqrt().clamp_min(1e-6)
    return mu.float(), sd.float()


def fit_stats(d: Dict[str, torch.Tensor], chunk: int = 64) -> Dict[str, torch.Tensor]:
    """Mean/std from fit clips only: valid node entries, valid person pairs, scene dims, motion.

    Processed ``chunk`` clips at a time so memory stays small (edges are the big tensor)."""
    N = len(d["y"])
    rng = range(0, N, chunk)

    def nodes_iter():
        for i in rng:
            yield d["nodes"][i : i + chunk].float()[d["mask"][i : i + chunk]]

    def edges_iter():
        for i in rng:
            m = d["mask"][i : i + chunk].permute(0, 1, 3, 2)                       # (n,W,T,P)
            pair = m.unsqueeze(-1) & m.unsqueeze(-2)                               # (n,W,T,P,P)
            yield d["edges"][i : i + chunk].float()[pair]

    def scene_iter():
        for i in rng:
            yield d["scene"][i : i + chunk].float().reshape(-1, d["scene"].shape[-1])

    stats = {}
    stats["node_mu"], stats["node_sd"] = _running_mean_std(nodes_iter())
    stats["edge_mu"], stats["edge_sd"] = _running_mean_std(edges_iter())
    stats["scene_mu"], stats["scene_sd"] = _running_mean_std(scene_iter())
    mo = d["motion"].float().reshape(-1)
    stats["motion_mu"], stats["motion_sd"] = mo.mean(), mo.std().clamp_min(1e-6)
    return stats


def make_batch(d: Dict[str, torch.Tensor], idx: torch.Tensor, flip: torch.Tensor, stats, device) -> Dict[str, torch.Tensor]:
    """Gather clips, apply per-clip flip to features, standardize (clip +-5), flatten clips*windows."""
    N, W = len(idx), d["motion"].shape[1]
    f = flip.to(device)
    nodes = d["nodes"][idx].to(device).float()
    edges = d["edges"][idx].to(device).float()
    mask = d["mask"][idx].to(device)
    fn = f.view(N, 1, 1, 1, 1)
    nodes_f = nodes.clone()
    nodes_f[..., FLIP_MIRROR_NODE] = 1.0 - nodes_f[..., FLIP_MIRROR_NODE]
    nodes_f[..., FLIP_NEGATE_NODE] = -nodes_f[..., FLIP_NEGATE_NODE]
    nodes = torch.where(fn, nodes_f, nodes)
    edges_f = edges.clone()
    edges_f[..., FLIP_NEGATE_EDGE] = -edges_f[..., FLIP_NEGATE_EDGE]
    edges = torch.where(f.view(N, 1, 1, 1, 1, 1), edges_f, edges)
    s = {k: v.to(device) for k, v in stats.items()}
    nodes = ((nodes - s["node_mu"]) / s["node_sd"]).clamp(-5, 5) * s["node_keep"]
    edges = ((edges - s["edge_mu"]) / s["edge_sd"]).clamp(-5, 5) * s["edge_keep"]
    scene = d["scene"][idx, f.long().cpu()].to(device).float()                      # (N,W,768)
    scene = ((scene - s["scene_mu"]) / s["scene_sd"])
    motion = (d["motion"][idx].to(device).float() - s["motion_mu"]) / s["motion_sd"]
    P, T = nodes.shape[2], nodes.shape[3]
    return {
        "scene": scene.reshape(N * W, -1), "motion": motion.reshape(-1),
        "nodes": nodes.reshape(N * W, P, T, -1), "mask": mask.reshape(N * W, P, T),
        "edges": edges.reshape(N * W, T, P, P, -1), "count": d["count"][idx].to(device).reshape(-1),
        "N": N, "W": W,
    }


def forward_clips(model: FusionV3, b: Dict[str, torch.Tensor]):
    out = model.window_logits(b["scene"], b["nodes"], b["mask"], b["edges"], b["motion"], b["count"])
    z = out["logit"].view(b["N"], b["W"])
    return clip_logit(z), z, out["attn"]


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def predict(model: FusionV3, d, stats, device, batch: int = 16) -> Dict[str, np.ndarray]:
    """Probabilities with flip TTA (also returns the non-TTA ones) and per-window probs."""
    model.eval()
    N = len(d["y"])
    p_orig, p_flip, win = [], [], []
    for i in range(0, N, batch):
        idx = torch.arange(i, min(N, i + batch))
        zs = []
        for fl in (False, True):
            b = make_batch(d, idx, torch.full((len(idx),), fl), stats, device)
            c, z, _ = forward_clips(model, b)
            (p_flip if fl else p_orig).append(torch.sigmoid(c).cpu())
            zs.append(torch.sigmoid(z).cpu())
        win.append((zs[0] + zs[1]) / 2)
    po, pf = torch.cat(p_orig).numpy(), torch.cat(p_flip).numpy()
    return {"p_single": po, "p_tta": (po + pf) / 2, "p_window": torch.cat(win).numpy()}


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf = np.where(p >= 0.5, p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(tot)


def metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    pred = (p >= 0.5).astype(int)
    return {"accuracy": float(accuracy_score(y, pred)), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)), "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, p)), "ece": ece(p, y), "n": int(len(y))}


# ----------------------------------------------------------------------------- train
def lr_at(step: int, total: int, warm: int, base: float) -> float:
    if step < warm:
        return base * (step + 1) / warm
    return base * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def subset(d: Dict, idx: torch.Tensor) -> Dict:
    return {k: (v[idx] if torch.is_tensor(v) else [v[i] for i in idx.tolist()]) for k, v in d.items()}


def run(cfg: TrainConfig, data: Dict[str, Dict], device: torch.device) -> Dict:
    seed_everything(cfg.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    fit, val, test = data["fit"], data["internal_val"], data["test"]
    if cfg.overfit_clips:
        g = torch.Generator().manual_seed(cfg.seed)
        fit = subset(fit, torch.randperm(len(fit["y"]), generator=g)[: cfg.overfit_clips])
        val = test = fit
    stats = fit_stats(fit)
    nk, ek = torch.ones(len(NODE_FEATURES)), torch.ones(len(EDGE_FEATURES))
    if cfg.box_only:
        for n in MASK_NODE_FEATURES:
            nk[NODE_FEATURES.index(n)] = 0
        for n in MASK_EDGE_FEATURES:
            ek[EDGE_FEATURES.index(n)] = 0
    stats["node_keep"], stats["edge_keep"] = nk, ek
    model = FusionV3(use_scene=cfg.use_scene, use_person=cfg.use_person, use_gat=cfg.use_gat, dropout=cfg.dropout,
                     scene_in_dropout=cfg.scene_in_dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    N = len(fit["y"])
    steps_per_epoch = math.ceil(N / cfg.batch_clips)
    total, warm = cfg.epochs * steps_per_epoch, cfg.warmup_epochs * steps_per_epoch
    gen = torch.Generator().manual_seed(cfg.seed)
    best = {"acc": -1.0, "loss": float("inf"), "state": None, "epoch": 0}
    history, step, bad = [], 0, 0
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(N, generator=gen)
        tot = 0.0
        for i in range(0, N, cfg.batch_clips):
            idx = perm[i : i + cfg.batch_clips]
            flip = torch.rand(len(idx), generator=gen) < 0.5
            b = make_batch(fit, idx, flip, stats, device)
            y = fit["y"][idx].to(device)
            c, z, _ = forward_clips(model, b)
            loss = F.binary_cross_entropy_with_logits(c, y)
            neg = (y == 0).unsqueeze(-1).expand_as(z)
            if neg.any():
                loss = loss + cfg.window_neg_weight * F.binary_cross_entropy_with_logits(z[neg], torch.zeros_like(z[neg]))
            for g_ in opt.param_groups:
                g_["lr"] = lr_at(step, total, warm, cfg.lr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            tot += float(loss.detach()) * len(idx)
        pv = predict(model, val, stats, device)["p_tta"]
        yv = val["y"].numpy()
        vloss = float(F.binary_cross_entropy(torch.tensor(pv).clamp(1e-6, 1 - 1e-6), torch.tensor(yv)))
        vacc = float(((pv >= 0.5) == yv).mean())
        history.append({"epoch": epoch, "train_loss": tot / N, "val_loss": vloss, "val_acc": vacc})
        print(f"[{cfg.name}] epoch {epoch:2d} train_loss {tot / N:.4f} val_loss {vloss:.4f} val_acc {vacc:.3f}", flush=True)
        if (vacc, -vloss) > (best["acc"], -best["loss"]):
            best.update(acc=vacc, loss=vloss, epoch=epoch, state={k: v.detach().clone() for k, v in model.state_dict().items()})
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best["state"])
    if cfg.save_ckpt:
        ck = Path(cfg.out_dir)
        ck.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": asdict(cfg), "state": best["state"], "stats": stats}, ck / f"{cfg.name}.pt")
    res = {"config": asdict(cfg), "best_epoch": best["epoch"], "train_seconds": time.time() - t0, "history": history,
           "n_params": int(sum(p.numel() for p in model.parameters())), "stats_source": "fit split only"}
    for name, part in (("internal_val", val), ("test", test)):
        pr = predict(model, part, stats, device)
        y = part["y"].numpy()
        res[name] = {"single": metrics(y, pr["p_single"]), "tta": metrics(y, pr["p_tta"])}
    return res
