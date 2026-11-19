"""Build the manifest for any dataset described by a config in configs/datasets/.

Usage: python scripts/build_manifest.py configs/datasets/rwf2000_30fps.yaml [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mosaic.datasets import DatasetError, build_manifest, load_dataset_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="dataset config (YAML)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    try:
        cfg = load_dataset_config(a.dataset)
        man = build_manifest(cfg, progress=lambda i, n: print(f"{i}/{n}", flush=True))
    except DatasetError as exc:
        sys.exit(f"dataset config problem: {exc}")
    out = Path(a.out) if a.out else cfg.manifest_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(man, indent=1, ensure_ascii=True), encoding="utf-8")
    m = man["meta"]
    print(json.dumps({k: m[k] for k in ("dataset_name", "layout", "group", "corrupt_count", "removed_exact_duplicate_count", "counts", "fps_histogram", "frame_count_histogram")}, indent=1))
    print(f"wrote {out} ({len(man['entries'])} clips)")


if __name__ == "__main__":
    main()
