import torch

from mosaic.train_v3 import TrainConfig, fit_stats, lr_at, make_batch, metrics, run


def _fake(n=24, W=6, P=4, T=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = (torch.arange(n) % 2).float()
    nodes = torch.randn(n, W, P, T, 29, generator=g) * 0.1
    nodes[..., 15] += y.view(n, 1, 1, 1) * 1.0           # speed channel carries the label
    mask = torch.zeros(n, W, P, T, dtype=torch.bool)
    mask[:, :, :2] = True
    return {"scene": (torch.randn(n, 2, W, 768, generator=g) + y.view(n, 1, 1, 1) * 0.3).half(),
            "motion": torch.rand(n, W, generator=g), "nodes": nodes.half(), "mask": mask,
            "edges": torch.randn(n, W, T, P, P, 9, generator=g).half(), "count": torch.full((n, W), 2.0),
            "y": y, "rel": [str(i) for i in range(n)]}


def test_make_batch_flip_changes_only_expected_channels():
    d = _fake()
    st = fit_stats(d)
    st["node_keep"], st["edge_keep"] = torch.ones(29), torch.ones(9)
    idx = torch.arange(4)
    a = make_batch(d, idx, torch.zeros(4, dtype=torch.bool), st, "cpu")
    b = make_batch(d, idx, torch.ones(4, dtype=torch.bool), st, "cpu")
    assert a["nodes"].shape == (24, 4, 16, 29) and a["edges"].shape == (24, 16, 4, 4, 9)
    assert a["scene"].shape == (24, 768) and not torch.allclose(a["scene"], b["scene"])   # different cached view
    assert torch.allclose(a["nodes"][..., 6], b["nodes"][..., 6])                         # log_mask_aspect unchanged
    assert not torch.allclose(a["nodes"][..., 0], b["nodes"][..., 0])                     # cx mirrored


def test_lr_schedule():
    assert lr_at(0, 100, 10, 1.0) < lr_at(9, 100, 10, 1.0) <= 1.0
    assert lr_at(99, 100, 10, 1.0) < 0.01


def test_tiny_training_learns_and_ablations_run(tmp_path):
    d = _fake()
    data = {"fit": d, "internal_val": d, "test": d}
    for kw in ({}, {"use_person": False}, {"use_scene": False}, {"use_gat": False}):
        cfg = TrainConfig(epochs=30, warmup_epochs=2, patience=99, batch_clips=8, name="t", **kw)
        res = run(cfg, data, torch.device("cpu"))
        assert res["test"]["tta"]["accuracy"] >= 0.9, (kw, res["test"])
        assert res["history"][-1]["train_loss"] < res["history"][0]["train_loss"]


def test_metrics_keys():
    import numpy as np
    m = metrics(np.array([0, 1, 1, 0]), np.array([0.2, 0.8, 0.6, 0.4]))
    assert m["accuracy"] == 1.0 and m["roc_auc"] == 1.0 and 0 <= m["ece"] <= 0.5


def test_chunked_stats_match_direct():
    from tests.unit.test_train_v3 import _fake  # noqa: F401
    d = _fake(n=20)
    a = fit_stats(d, chunk=3)
    b = fit_stats(d, chunk=64)
    for k in a:
        assert torch.allclose(a[k].float(), b[k].float(), atol=1e-4), k
    nv = d["nodes"].float()[d["mask"]]
    assert torch.allclose(a["node_mu"], nv.mean(0), atol=1e-4) and torch.allclose(a["node_sd"], nv.std(0, unbiased=False), atol=1e-3)


def test_box_only_zeroes_mask_features_and_still_runs():
    from mosaic.skd import NODE_FEATURES
    from mosaic.train_v3 import MASK_NODE_FEATURES
    d = _fake()
    cfg = TrainConfig(epochs=2, warmup_epochs=1, box_only=True, scene_in_dropout=0.3, dropout=0.5, name="t")
    res = run(cfg, {"fit": d, "internal_val": d, "test": d}, torch.device("cpu"))
    assert "test" in res
    st = fit_stats(d)
    st["node_keep"], st["edge_keep"] = torch.ones(29), torch.ones(9)
    for n in MASK_NODE_FEATURES:
        st["node_keep"][NODE_FEATURES.index(n)] = 0
    b = make_batch(d, torch.arange(2), torch.zeros(2, dtype=torch.bool), st, "cpu")
    assert (b["nodes"][..., [NODE_FEATURES.index(n) for n in MASK_NODE_FEATURES]] == 0).all()
