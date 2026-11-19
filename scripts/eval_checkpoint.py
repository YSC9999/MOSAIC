"""Evaluate a saved model on any dataset whose caches have been built (run_pipeline.py --skip train).

Typical use: train on RWF-2000 (run_pipeline.py ... --save-ckpt), build the caches for another dataset, then

    python scripts/eval_checkpoint.py CKPT configs/datasets/other.yaml --cache C:/mosaic_cache/other

The model's own normalisation statistics (from its training data) are used, nothing is refit and nothing is trained.
Writes artifacts/eval/<ckpt>__on__<dataset>.json with accuracy / precision / recall / F1 / ROC-AUC / ECE at the 0.5 threshold
and a per-clip prediction list. --part all (default) scores every clip; test / internal_val / fit score one split.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mosaic.datasets import DatasetError, load_dataset_config  # noqa: E402
from mosaic.models.fusion_v3 import FusionV3  # noqa: E402


def merge_parts(data: dict, part: str) -> dict:
    if part != "all":
        if part not in data:
            raise SystemExit(f"split {part!r} not in the dataset's split file (have {sorted(data)})")
        return data[part]
    parts = [data[k] for k in sorted(data)]
    return {k: (torch.cat([p[k] for p in parts]) if torch.is_tensor(parts[0][k]) else sum((p[k] for p in parts), []))
            for k in parts[0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("dataset", help="dataset config (YAML)")
    ap.add_argument("--cache", default=None, help="cache folder of that dataset (default C:/mosaic_cache/<name>)")
    ap.add_argument("--gate", default=None, help="gate used to build the caches (default: the checkpoint's training gate or gdef)")
    ap.add_argument("--part", default="all", choices=["all", "fit", "internal_val", "test"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    try:
        cfg = load_dataset_config(a.dataset)
    except DatasetError as exc:
        sys.exit(f"dataset config problem: {exc}")
    gate = a.gate or "gdef"
    cache = a.cache or f"C:/mosaic_cache/{cfg.name}"
    os.environ.update(MOSAIC_CACHE=cache, MOSAIC_SKD=f"skd_{gate}")
    from mosaic.train_v3 import load_all, metrics, predict  # imported after the environment is set

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    c = ck["cfg"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionV3(use_scene=c["use_scene"], use_person=c["use_person"], use_gat=c["use_gat"], dropout=c["dropout"],
                     scene_in_dropout=c["scene_in_dropout"]).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    data = merge_parts(load_all(cfg.manifest_path, cfg.split_path, Path(cache)), a.part)
    pr = predict(model, data, ck["stats"], device)
    y = data["y"].numpy()
    res = {"checkpoint": Path(a.ckpt).name, "trained_as": c["name"], "dataset": cfg.name, "part": a.part, "gate": gate,
           "clips": int(len(y)), "positive_clips": int(y.sum()), "threshold": 0.5,
           "tta": metrics(y, pr["p_tta"]), "single": metrics(y, pr["p_single"]),
           "predictions": [{"clip": r, "target": int(t), "prob": float(p)} for r, t, p in zip(data["rel"], y, pr["p_tta"])]}
    out = Path(a.out) if a.out else ROOT / "artifacts" / "eval" / f"{Path(a.ckpt).stem}__on__{cfg.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    m = res["tta"]
    print(f"{cfg.name} ({a.part}, {res['clips']} clips): accuracy {m['accuracy']:.3f}  F1 {m['f1']:.3f}  AUC {m['roc_auc']:.3f}  ECE {m['ece']:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
