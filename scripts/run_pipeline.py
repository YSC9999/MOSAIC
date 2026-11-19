"""Run the whole MOSAIC v3 pipeline on any dataset described by a config in configs/datasets/ (every step is resumable).

    manifest -> split -> swin features -> motion scores -> GATED detection -> tracks -> person features -> training

Gating is not optional: detection only runs on the frames the chosen gate keeps (--gate, default gdef).
Steps that need rfdetr / pycocotools / trackers use the project's .venv; the others use the current interpreter.
Names, manifests, splits and caches are all derived from the dataset config, so a new dataset needs only a YAML file.

Examples:
    python scripts/run_pipeline.py configs/datasets/rwf2000_30fps.yaml --cache C:/mosaic_cache_30fps
    python scripts/run_pipeline.py configs/datasets/my_dataset.yaml --dry-run          # print the commands only
    python scripts/run_pipeline.py configs/datasets/my_dataset.yaml --skip train       # build the caches only
    python scripts/run_pipeline.py configs/datasets/my_dataset.yaml --from-step detect # resume from a step
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mosaic.datasets import DatasetError, load_dataset_config  # noqa: E402
from mosaic.motion_gate import DEFAULT_GATE, GATE_SETTINGS  # noqa: E402

STEPS = ["manifest", "split", "swin", "motion", "detect", "tracks", "skd", "train"]
VARIANTS = {   # name -> extra train_v3.py flags (no GAT by default; masks stay on)
    "main": ["--name", "main_nogat"],
    "r1": ["--name", "r1_full_reg", "--gat", "--dropout", "0.5", "--wd", "0.2", "--scene-in-dropout", "0.3"],
    "a4": ["--name", "a4_boxonly", "--box-only"],
    "a2": ["--name", "a2_scene", "--no-person"],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="dataset config (YAML), e.g. configs/datasets/rwf2000_30fps.yaml")
    ap.add_argument("--cache", default=None, help="cache folder (default C:/mosaic_cache/<dataset name>)")
    ap.add_argument("--gate", default=DEFAULT_GATE, choices=sorted(GATE_SETTINGS))
    ap.add_argument("--model", default="RFDETRSegNano", help="RF-DETR segmentation model for detection")
    ap.add_argument("--variants", nargs="+", default=["main"], choices=sorted(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2024])
    ap.add_argument("--from-step", choices=STEPS, default=STEPS[0])
    ap.add_argument("--skip", nargs="*", default=[], choices=STEPS)
    ap.add_argument("--save-ckpt", action="store_true", help="write the trained weights next to the run metrics")
    ap.add_argument("--venv-python", default=str(ROOT / ".venv" / "Scripts" / "python.exe"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    try:
        cfg = load_dataset_config(a.dataset)
    except DatasetError as exc:
        sys.exit(f"dataset config problem: {exc}")
    rel = lambda p: str(Path(p).relative_to(ROOT)).replace("\\", "/")
    manifest, split = rel(cfg.manifest_path), rel(cfg.split_path)
    cache = (a.cache or f"C:/mosaic_cache/{cfg.name}").replace("\\", "/")
    det, trk, skd = "det_gated", f"tracks_{a.gate}", f"skd_{a.gate}"
    out_runs = f"artifacts/runs/{cfg.name}/{a.gate}"
    env = dict(os.environ, MOSAIC_CACHE=cache, MOSAIC_MANIFEST=manifest, MOSAIC_SPLIT=split, MOSAIC_DET=det,
               MOSAIC_TRACKS=trk, MOSAIC_SKD=skd, CUBLAS_WORKSPACE_CONFIG=":4096:8")
    py, vpy = sys.executable, a.venv_python

    cmds = {
        "manifest": [[py, "scripts/build_manifest.py", a.dataset]],
        "split": [[py, "scripts/build_v3_split.py", "--manifest", manifest, "--out", split]],
        "swin": [[py, "scripts/build_scene_cache.py", "--manifest", manifest, "--split", split]],
        "motion": [[py, "scripts/build_motion.py"]],
        "detect": [[vpy, "scripts/build_det.py", "--manifest", manifest, "--split", split, "--model", a.model,
                    "--gates", a.gate, "--cache-dir", f"{cache}/{det}"]],
        "tracks": [[vpy, "scripts/build_tracks.py", "--det-dir", f"{cache}/{det}", "--out-dir", f"{cache}/{trk}",
                    "--setting", a.gate]],
        "skd": [[vpy, "scripts/build_skd.py"]],
        "train": [[py, "scripts/train_v3.py", "--out-dir", out_runs, "--seed", str(s), *VARIANTS[v],
                   *(["--save-ckpt"] if a.save_ckpt else [])] for s in a.seeds for v in a.variants],
    }
    start = STEPS.index(a.from_step)
    print(f"dataset {cfg.name}: manifest {manifest}, cache {cache}, gate {a.gate}", flush=True)
    for step in STEPS[start:]:
        if step in a.skip:
            continue
        for cmd in cmds[step]:
            print(f"[{step}] " + " ".join(cmd), flush=True)
            if not a.dry_run:
                r = subprocess.run(cmd, cwd=ROOT, env=env)
                if r.returncode != 0:
                    sys.exit(f"step {step!r} failed (exit {r.returncode}); fix and rerun with --from-step {step}")
    print("done" if not a.dry_run else "dry run only; nothing was executed")


if __name__ == "__main__":
    main()
