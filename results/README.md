# Results

One folder per dataset, small files only (metrics, tables, figures, a few demos). Caches, checkpoints and videos of the datasets stay out of git (`artifacts/`, `data/`). Regenerate a dataset's folder with `scripts/make_results.py`.

| Dataset | Main model and setting | Test acc (3 seeds) | AUC | Details |
|---|---|---|---|---|
| RWF-2000 (30 fps) | No GAT, Nano + default gate (46% of frames) | 0.888 ± 0.004 | 0.958 | [rwf2000/](rwf2000/README.md) |
