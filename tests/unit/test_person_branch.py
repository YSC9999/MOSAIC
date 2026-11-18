import torch

from mosaic.models.fusion_v3 import FusionV3, clip_logit
from mosaic.models.person_branch import DenseGATv2Layer, PersonBranch


def _batch(B=3, P=5, T=6, present=None):
    torch.manual_seed(0)
    nodes = torch.randn(B, P, T, 29)
    mask = torch.zeros(B, P, T, dtype=torch.bool)
    mask[:, :3] = True if present is None else present
    edges = torch.randn(B, T, P, P, 9)
    return nodes, mask, edges


def test_shapes_and_attention_sums_to_one():
    pb = PersonBranch().eval()
    n, m, e = _batch()
    pooled, attn, has = pb(n, m, e)
    assert pooled.shape == (3, 128) and attn.shape == (3, 5) and has.all()
    assert torch.allclose(attn.sum(-1), torch.ones(3), atol=1e-5)
    assert (attn[:, 3:] == 0).all()                       # absent people get exactly zero weight


def test_padded_people_do_not_affect_output():
    pb = PersonBranch().eval()
    n, m, e = _batch()
    out1 = pb(n, m, e)[0]
    n2, e2 = n.clone(), e.clone()
    n2[:, 3:] = 99.0                                      # garbage in padded slots
    e2[:, :, 3:, :] = 99.0
    e2[:, :, :, 3:] = 99.0
    assert torch.allclose(out1, pb(n2, m, e2)[0], atol=1e-5)


def test_permutation_invariance_over_people():
    pb = PersonBranch().eval()
    n, m, e = _batch()
    perm = torch.tensor([2, 0, 1, 3, 4])
    out = pb(n, m, e)[0]
    out_p = pb(n[:, perm], m[:, perm], e[:, :, perm][:, :, :, perm])[0]
    assert torch.allclose(out, out_p, atol=1e-4)


def test_empty_window_uses_learned_vector_and_has_flag():
    pb = PersonBranch().eval()
    n, m, e = _batch()
    m[1] = False
    pooled, attn, has = pb(n, m, e)
    assert has.tolist() == [True, False, True]
    assert torch.allclose(pooled[1], pb.empty) and (attn[1] == 0).all()
    assert torch.isfinite(pooled).all()


def test_gru_state_carried_over_missing_frames():
    pb = PersonBranch(use_gat=False).eval()
    n, m, e = _batch(B=1, P=1, T=6)
    m[:] = True
    m[0, 0, 2:4] = False                                  # person missing in the middle
    a = pb(n, m, e)[0]
    n2 = n.clone(); n2[0, 0, 2:4] = 50.0                  # values at missing frames must not matter
    assert torch.allclose(a, pb(n2, m, e)[0], atol=1e-5)


def test_gat_gradients_finite_and_single_person_ok():
    pb = PersonBranch()
    n, m, e = _batch(B=2, P=4, T=4)
    m[:] = False
    m[:, 0] = True                                        # only one person: attention must be self-loop only
    out = pb(n, m, e)[0]
    out.sum().backward()
    assert torch.isfinite(out).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in pb.parameters())


def test_gat_layer_masks_invalid_sources():
    layer = DenseGATv2Layer().eval()
    x = torch.randn(1, 3, 64); e = torch.randn(1, 3, 3, 64)
    valid = torch.tensor([[True, True, False]])
    y1 = layer(x, e, valid)
    x2 = x.clone(); x2[0, 2] = 123.0
    assert torch.allclose(y1[0, :2], layer(x2, e, valid)[0, :2], atol=1e-5)


def test_fusion_ablation_variants_and_topk():
    n, m, e = _batch(B=4)
    scene, motion, count = torch.randn(4, 768), torch.rand(4), torch.full((4,), 3.0)
    for kw in ({}, {"use_person": False}, {"use_scene": False}, {"use_gat": False}):
        f = FusionV3(**kw).eval()
        out = f.window_logits(scene, n, m, e, motion, count)
        assert out["logit"].shape == (4,)
    z = torch.tensor([[1., 5., 3., 2., 4., 0.]])
    assert clip_logit(z).item() == (5 + 4 + 3) / 3        # W=6 -> k=3
