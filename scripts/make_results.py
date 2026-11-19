"""Regenerate results/<dataset>/ (small, git-friendly) from the run files in artifacts/.

    python scripts/make_results.py --dataset rwf2000

Reads  artifacts/reports/v3_30fps/runs, artifacts/reports/v3_30fps_nano/runs (+ tracks logs, scene probe, outputs).
Writes results/rwf2000/{README.md, tables/, runs/, figures/, outputs/, splits/} and updates results/README.md.
Only the 30 fps copy of RWF-2000 is reported. Nothing large is copied (no caches, no videos beyond a few demos).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "artifacts" / "reports"

SETTINGS = {  # key: (label, runs dir, file prefix, tracks log or None)
    "nano_gdef": ("RF-DETR-Seg Nano, default gate", REPORTS / "v3_30fps_nano" / "runs", "gdef_", REPORTS / "v3_30fps_nano" / "tracks_gdef.log"),
    "nano_g02": ("RF-DETR-Seg Nano, strong gate", REPORTS / "v3_30fps_nano" / "runs", "g02_", REPORTS / "v3_30fps_nano" / "tracks_g02.log"),
    "large_nogate": ("RF-DETR-Seg Large, no gate", REPORTS / "v3_30fps" / "runs", "", REPORTS / "v3_30fps" / "tracks.log"),
}
VARIANTS = {  # file stem -> label
    "a5_nogat": "No GAT, masks on (MAIN)",
    "r1_full_reg": "+ GAT and stronger regularization",
    "a4_boxonly": "Box-only (mask features off), with GAT",
    "a2_scene": "Scene only",
    "r2_scene_reg": "Scene only + regularization",
    "a3_person": "People only",
}
SEEDS = (42, 1337, 2024)


def load_runs():
    out = {}
    for key, (_, d, pre, _) in SETTINGS.items():
        for v in VARIANTS:
            rs = []
            for s in SEEDS:
                f = d / f"{pre}{v}_s{s}.json"
                if f.exists():
                    rs.append((s, f, json.loads(f.read_text(encoding="utf-8"))))
            if rs:
                out[(key, v)] = rs
    return out


def stat(rs, sp, m, k="tta"):
    v = [r[sp][k][m] for _, _, r in rs]
    return float(np.mean(v)), float(np.std(v))


def fmt(rs, sp="test", m="accuracy"):
    a, b = stat(rs, sp, m)
    return f"{a:.3f} ± {b:.3f}"


def tracks_stats(path: Path):
    if path is None or not path.exists():
        return {}
    t = path.read_text(encoding="utf-8", errors="ignore")
    try:
        return json.loads(t[t.find("{"): t.rfind("}") + 1])
    except Exception:
        return {}


def table(runs, setting, variants, title_cols=("Variant", "Val acc", "Test acc", "F1", "AUC", "ECE")):
    lines = ["| " + " | ".join(title_cols) + " |", "|" + "---|" * len(title_cols)]
    for v in variants:
        rs = runs.get((setting, v))
        if not rs:
            continue
        name = ("**" + VARIANTS[v] + "**") if v == "a5_nogat" else VARIANTS[v]
        lines.append(f"| {name} | {fmt(rs, 'internal_val')} | {fmt(rs)} | {stat(rs, 'test', 'f1')[0]:.3f} | "
                     f"{stat(rs, 'test', 'roc_auc')[0]:.3f} | {stat(rs, 'test', 'ece')[0]:.3f} |")
    return "\n".join(lines)


def output_stats(out_dir: Path):
    fs = [f for f in out_dir.glob("*.json") if f.name != "summary.json"]
    if not fs:
        return None
    rs = [json.loads(f.read_text(encoding="utf-8")) for f in fs]
    pred = [r for r in rs if r["pred_label"] == 1]
    cov, whole, multi, nwin, lo = [], 0, 0, [], []
    for r in pred:
        dur = r["window_span_s"][-1][1]
        c = sum(b - a for a, b in r["intervals_s"]) / dur
        cov.append(c)
        whole += c >= 0.99
        multi += len(r["intervals_s"]) > 1
        nwin.append(len(r["violent_windows"]))
        lo.append(min(r["window_prob"]))
    return {"clips": len(rs), "predicted_violent": len(pred), "interval_coverage_mean": float(np.mean(cov)),
            "interval_coverage_median": float(np.median(cov)), "intervals_covering_whole_clip": int(whole),
            "clips_with_2plus_intervals": int(multi), "violent_windows_per_clip_mean": float(np.mean(nwin)),
            "lowest_window_prob_mean": float(np.mean(lo)), "clips_with_group_boxes": int(sum(bool(r["group_boxes"]) for r in pred))}


def make_figure(runs, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = []
    for key in ("large_nogate", "nano_gdef", "nano_g02"):
        rs = runs.get((key, "a5_nogat"))
        if not rs:
            continue
        kept = 1.0 if key == "large_nogate" else tracks_stats(SETTINGS[key][3]).get("share_frames_kept", np.nan)
        a, b = stat(rs, "test", "accuracy")
        pts.append((100 * kept, a, b, {"large_nogate": "No gate (Large)", "nano_gdef": "Default gate (Nano)", "nano_g02": "Strong gate (Nano)"}[key]))
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for x, y, e, lab in pts:
        ax.errorbar(x, y, yerr=e, fmt="o", capsize=4, label=lab)
    ax.set_xlabel("Share of frames the detector runs on (%)")
    ax.set_ylabel("Test accuracy (mean ± std, 3 seeds)")
    ax.set_title("Motion gating trade-off (main model, 30 fps)")
    ax.set_ylim(0.85, 0.92)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def build_rwf2000():
    dst = ROOT / "results" / "rwf2000"
    for sub in ("tables", "runs", "figures", "outputs", "splits"):
        (dst / sub).mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if not runs:
        sys.exit("no run files found under artifacts/reports/v3_30fps*/runs")

    # runs/ and the flat table
    rows = []
    for (key, v), rs in runs.items():
        for s, f, r in rs:
            (dst / "runs" / key).mkdir(exist_ok=True)
            shutil.copyfile(f, dst / "runs" / key / f"{v}_s{s}.json")
            rows.append({"setting": key, "variant": v, "seed": s,
                         "val_acc": r["internal_val"]["tta"]["accuracy"], "test_acc": r["test"]["tta"]["accuracy"],
                         "test_acc_no_flip_avg": r["test"]["single"]["accuracy"], "test_f1": r["test"]["tta"]["f1"],
                         "test_auc": r["test"]["tta"]["roc_auc"], "test_ece": r["test"]["tta"]["ece"],
                         "best_epoch": r["best_epoch"]})
    with (dst / "tables" / "all_results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # splits (clip names and labels only)
    sp_file = ROOT / "artifacts" / "manifests" / "rwf2000_30fps_v3_split.json"
    mf_file = ROOT / "artifacts" / "manifests" / "rwf2000_30fps_manifest.json"
    if sp_file.exists() and mf_file.exists():
        assign = json.loads(sp_file.read_text(encoding="utf-8"))["assignment"]
        label = {e["rel_path"]: e["label"] for e in json.loads(mf_file.read_text(encoding="utf-8"))["entries"]}
        with (dst / "splits" / "rwf2000_30fps_split.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["clip", "label", "split"])
            for rel, part in sorted(assign.items()):
                w.writerow([rel, label.get(rel, ""), part])

    make_figure(runs, dst / "figures" / "gating_tradeoff.png")
    ms_img = REPORTS / "v3_30fps" / "outputs" / "mask_examples.jpg"
    if ms_img.exists():
        shutil.copyfile(ms_img, dst / "figures" / "mask_examples.jpg")
    probe = REPORTS / "v3_30fps" / "scene_probe_swin3d_t.json"
    if probe.exists():
        shutil.copyfile(probe, dst / "tables" / "scene_probe_swin3d_t.json")

    # outputs: statistics + a few demo videos (from the GAT model R1, see README)
    out_src = REPORTS / "v3_30fps" / "outputs"
    st = output_stats(out_src)
    if st:
        (dst / "outputs" / "interval_stats.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
    for name in ("0Ow4cotKOuw_0.mp4", "39BFeYnbu-I_6.mp4", "masks_fight_g6jdqkcdgqk_1.mp4", "masks_fight_YDOJvzChqSg_1.mp4"):
        if (out_src / name).exists():
            (dst / "outputs" / "demo").mkdir(exist_ok=True)
            shutil.copyfile(out_src / name, dst / "outputs" / "demo" / name)

    # README with the tables
    main = runs.get(("nano_gdef", "a5_nogat"))
    t6 = ["| Setting | Frames detected | Test acc | AUC | People per window | Windows with 1+ person |", "|---|---|---|---|---|---|"]
    for key, lab in (("large_nogate", "No gate (Large)"), ("nano_gdef", "Default gate (Nano)"), ("nano_g02", "Strong gate (Nano)")):
        rs = runs.get((key, "a5_nogat"))
        if not rs:
            continue
        t = tracks_stats(SETTINGS[key][3])
        kept = 1.0 if key == "large_nogate" else t.get("share_frames_kept", float("nan"))
        t6.append(f"| {lab} | {100 * kept:.0f}% | {fmt(rs)} | {stat(rs, 'test', 'roc_auc')[0]:.3f} | "
                  f"{t.get('mean_people_per_window', float('nan')):.2f} | {100 * t.get('share_windows_ge1', float('nan')):.0f}% |")
    t1_rows = ["| Setting | Val acc | Test acc | F1 | AUC | ECE |", "|---|---|---|---|---|---|"]
    for key in ("nano_gdef", "nano_g02", "large_nogate"):
        rs = runs.get((key, "a5_nogat"))
        if rs:
            lab = {"nano_gdef": "**Nano, default gate (46% of frames): current pipeline**", "nano_g02": "Nano, strong gate (28%)",
                   "large_nogate": "Large, no gate"}[key]
            t1_rows.append(f"| {lab} | {fmt(rs, 'internal_val')} | {fmt(rs)} | {stat(rs, 'test', 'f1')[0]:.3f} | "
                           f"{stat(rs, 'test', 'roc_auc')[0]:.3f} | {stat(rs, 'test', 'ece')[0]:.3f} |")
    probe_txt = ""
    if probe.exists():
        p = json.loads(probe.read_text(encoding="utf-8"))
        probe_txt = ("| Probe on frozen Swin-T features (scene only) | Test acc | AUC |\n|---|---|---|\n"
                     f"| Logistic regression on the window average | {p['a_logreg_window_mean']['test']['accuracy']:.3f} | {p['a_logreg_window_mean']['test']['roc_auc']:.3f} |\n"
                     f"| Linear head, top-3 window pooling | {p['b_linear_topk_mil']['test']['accuracy']:.3f} | {p['b_linear_topk_mil']['test']['roc_auc']:.3f} |")
    ot = ""
    if st:
        ot = (f"On {st['clips']} test fight clips the model called {st['predicted_violent']} violent. The violent intervals cover on "
              f"average {100 * st['interval_coverage_mean']:.0f}% of the clip (median {100 * st['interval_coverage_median']:.0f}%); "
              f"{st['intervals_covering_whole_clip']} intervals cover the whole clip and {st['clips_with_2plus_intervals']} clips have two or "
              f"more separate intervals. These describe the outputs; they are **not scored** against ground truth (RWF-2000 has no time or box labels).")
    md = f"""# RWF-2000 results

Official test set (400 clips). Data: the 30 fps copy (150 frames per clip, 64 sampled, 7 windows). Every number is the mean ± std over
3 seeds (42, 1337, 2024); F1, AUC and ECE are means; flip test-time averaging is on. The model is selected on an internal
validation split grouped by source video, and the test set is used only for the final score.

**Main model:** no interaction graph (GAT), mask features on, person branch = per-person GRU + attention pooling, fused with
frozen Video Swin-T scene features. Default pipeline: motion gate (always on) -> RF-DETR-Seg Nano -> ByteTrack.

## T1  Main model

{chr(10).join(t1_rows)}

## T2  Ablations, Nano + default gate

{table(runs, 'nano_gdef', ['a5_nogat', 'r1_full_reg', 'a4_boxonly'])}

## T3  Ablations, Large detector, no gate

{table(runs, 'large_nogate', ['a5_nogat', 'r1_full_reg', 'a4_boxonly', 'a2_scene', 'r2_scene_reg', 'a3_person'])}

## T6  Gating trade-off (main model)

{chr(10).join(t6)}

![gating trade-off](figures/gating_tradeoff.png)

## T7  Scene-only reference

{probe_txt}

## Reading the tables

- The ablations are **within seed noise of each other**: GAT, no GAT, box-only and the full model all land between about 0.88 and 0.89.
  The GAT never helped, so the main model omits it. Masks did not help either (box-only matches); they are kept for now.
- At 30 fps the **scene-only linear probe (0.885) matches the main model**, so the person branch is not clearly ahead of a good scene baseline.
- The **default gate** detects about half the frames at no measurable accuracy cost; the strong gate costs about 1 point.
- Published results are about 94-95% ([CUE-Net, CVPRW 2024](https://arxiv.org/pdf/2404.18952): 94.00%; a
  [Nov 2025 preprint](https://arxiv.org/pdf/2511.10866): 95.25%), so this work is not at the state of the art yet.

## Spatio-temporal outputs

{ot}

The demo videos in `outputs/demo/` and the statistics above come from the **GAT model (R1, seed 42) on Large detections with no gate**,
not from the main model: only that checkpoint was saved. Red box = the model's attention-weighted focus area, not a judgment of
individuals. `figures/mask_examples.jpg` shows RF-DETR-Seg **Large** masks on four clips.

## Files

`tables/all_results.csv` (every run), `runs/<setting>/<variant>_s<seed>.json` (full metrics and training history),
`splits/rwf2000_30fps_split.csv` (fit / internal_val / test per clip), `figures/`, `outputs/`.
Regenerate with `python scripts/make_results.py --dataset rwf2000`.
"""
    (dst / "README.md").write_text(md, encoding="utf-8")
    a, b = stat(main, "test", "accuracy") if main else (float("nan"), float("nan"))
    auc = stat(main, "test", "roc_auc")[0] if main else float("nan")
    return {"dataset": "RWF-2000 (30 fps)", "model": "No GAT, Nano + default gate (46% of frames)",
            "test_acc": f"{a:.3f} ± {b:.3f}", "auc": f"{auc:.3f}", "folder": "rwf2000/"}


def build_generic(name: str):
    """results/<name>/ from artifacts/eval/*__on__<name>*.json (models trained elsewhere, evaluated on this dataset)."""
    files = sorted((ROOT / "artifacts" / "eval").glob(f"*__on__{name}*.json"))
    if not files:
        sys.exit(f"nothing to report for {name!r}: run scripts/eval_checkpoint.py first (writes artifacts/eval/*__on__{name}.json)")
    dst = ROOT / "results" / name
    (dst / "runs").mkdir(parents=True, exist_ok=True)
    (dst / "tables").mkdir(exist_ok=True)
    rows, lines = [], ["| Model (trained on) | Part | Clips | Accuracy | Precision | Recall | F1 | AUC | ECE |", "|---|---|---|---|---|---|---|---|---|"]
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        m = r["tta"]
        slim = {k: v for k, v in r.items() if k != "predictions"}
        (dst / "runs" / f.name).write_text(json.dumps(slim, indent=1), encoding="utf-8")
        with (dst / "runs" / (f.stem + "_predictions.csv")).open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["clip", "target", "prob"])
            w.writerows([[x["clip"], x["target"], f"{x['prob']:.5f}"] for x in r["predictions"]])
        rows.append({"checkpoint": r["checkpoint"], "part": r["part"], "clips": r["clips"], **{k: m[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "ece")}})
        lines.append(f"| {r['trained_as']} ({r['checkpoint']}) | {r['part']} | {r['clips']} | {m['accuracy']:.3f} | {m['precision']:.3f} | "
                     f"{m['recall']:.3f} | {m['f1']:.3f} | {m['roc_auc']:.3f} | {m['ece']:.3f} |")
    with (dst / "tables" / "all_results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    NL = chr(10)
    head = [f"# {name} results", "",
            "Models evaluated on this dataset **without training or fine-tuning on it**: the model's own normalisation statistics are used,",
            "the threshold is 0.5 and flip test-time averaging is on.", ""]
    foot = ["", "`runs/*.json` hold the metrics and `runs/*_predictions.csv` the per-clip probabilities.",
            f"Regenerate with `python scripts/make_results.py --dataset {name}`."]
    md = NL.join(head + lines + foot) + NL
    (dst / "README.md").write_text(md, encoding="utf-8")
    best = max(rows, key=lambda x: x["accuracy"])
    return {"dataset": name, "model": f"evaluated, not trained on it ({best['checkpoint']})", "test_acc": f"{best['accuracy']:.3f}",
            "auc": f"{best['roc_auc']:.3f}", "folder": f"{name}/"}


def write_index(entries):
    lines = ["# Results", "",
             "One folder per dataset, small files only (metrics, tables, figures, a few demos). Caches, checkpoints and videos of the "
             "datasets stay out of git (`artifacts/`, `data/`). Regenerate a dataset's folder with `scripts/make_results.py`.", "",
             "| Dataset | Main model and setting | Test acc (3 seeds) | AUC | Details |", "|---|---|---|---|---|"]
    for e in entries:
        lines.append(f"| {e['dataset']} | {e['model']} | {e['test_acc']} | {e['auc']} | [{e['folder']}]({e['folder']}README.md) |")
    (ROOT / "results" / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rwf2000", help="rwf2000 (full tables) or the name of any dataset evaluated with eval_checkpoint.py")
    a = ap.parse_args()
    index = ROOT / "results" / "index.json"
    known = json.loads(index.read_text(encoding="utf-8")) if index.exists() else {}
    known[a.dataset] = build_rwf2000() if a.dataset == "rwf2000" else build_generic(a.dataset)
    index.write_text(json.dumps(known, indent=1), encoding="utf-8")
    write_index(list(known.values()))
    print("wrote results/" + a.dataset)


if __name__ == "__main__":
    main()
