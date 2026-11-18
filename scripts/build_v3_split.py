"""Build the v3 source-video grouped split from the deduplicated manifest.

Usage: python scripts/build_v3_split.py [--manifest PATH] [--out PATH] [--seed 42] [--holdout 0.1]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mosaic.splits import build_grouped_split, split_summary  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="artifacts/manifests/rwf2000_manifest.json")
    ap.add_argument("--out", default="artifacts/manifests/rwf2000_v3_split.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout", type=float, default=0.1)
    args = ap.parse_args()

    manifest = json.loads((ROOT / args.manifest).read_text(encoding="utf-8"))
    entries = [e for e in manifest["entries"] if e.get("decode_status") == "ok"]
    assignment = build_grouped_split(entries, holdout_fraction=args.holdout, seed=args.seed)
    summary = split_summary(entries, assignment)

    out = {
        "meta": {
            "schema_version": "v3-split-0.1",
            "source_manifest": args.manifest,
            "source_manifest_created_utc": manifest["meta"].get("created_utc"),
            "seed": args.seed,
            "holdout_fraction": args.holdout,
            "grouping": "source video (file stem without trailing _<n>)",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
        },
        "assignment": assignment,
    }
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
