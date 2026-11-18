"""Train one v3 variant. Examples:
  python scripts/train_v3.py --name gate --overfit 32 --epochs 60
  python scripts/train_v3.py --name a1_full --seed 42
  python scripts/train_v3.py --name a2_scene --no-person --seed 42
  python scripts/train_v3.py --name a3_person --no-scene --seed 42
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mosaic.train_v3 import TrainConfig, load_all, run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="a1_full")
ap.add_argument("--out-dir", default="artifacts/reports/v3/runs")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--overfit", type=int, default=0)
ap.add_argument("--no-scene", action="store_true")
ap.add_argument("--no-person", action="store_true")
ap.add_argument("--no-gat", action="store_true")
ap.add_argument("--box-only", action="store_true")
ap.add_argument("--dropout", type=float, default=0.3)
ap.add_argument("--scene-in-dropout", type=float, default=0.0)
ap.add_argument("--wd", type=float, default=0.05)
a = ap.parse_args()
cfg = TrainConfig(seed=a.seed, epochs=a.epochs, overfit_clips=a.overfit, use_scene=not a.no_scene,
                  use_person=not a.no_person, out_dir=a.out_dir, use_gat=not a.no_gat, box_only=a.box_only, dropout=a.dropout,
                  scene_in_dropout=a.scene_in_dropout, weight_decay=a.wd, name=f"{a.name}_s{a.seed}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = load_all(ROOT / os.environ.get("MOSAIC_MANIFEST", "artifacts/manifests/rwf2000_manifest.json"),
                ROOT / os.environ.get("MOSAIC_SPLIT", "artifacts/manifests/rwf2000_v3_split.json"))
res = run(cfg, data, device)
out = ROOT / cfg.out_dir / f"{cfg.name}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("best_epoch", "internal_val", "test")}, indent=1))
