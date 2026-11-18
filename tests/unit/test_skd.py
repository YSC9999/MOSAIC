import numpy as np

from mosaic.skd import (EDGE_DIM, NODE_DIM, NODE_FEATURES, flip_edges, flip_nodes, shape_features, edge_features,
                        crop_iou)


def _ellipse(h=60, w=20, size=100):
    yy, xx = np.mgrid[:size, :size]
    return (((xx - 50) / (w / 2)) ** 2 + ((yy - 50) / (h / 2)) ** 2) <= 1


def test_dims():
    assert NODE_DIM == 29 and EDGE_DIM == 9 and len(set(NODE_FEATURES)) == 29


def test_shape_vertical_ellipse():
    m = _ellipse()
    f = shape_features(m, (0, 0, 100, 100), np.array([40, 20, 60, 80], np.float32))
    assert f[0] < 0                       # taller than wide -> log aspect < 0
    assert f[2] > 0.95                    # convex -> solidity ~1
    assert f[4] > 0.9                     # elongated -> eccentric
    assert abs(f[5] + 1) < 0.05 and abs(f[6]) < 0.05   # vertical: cos2t=-1, sin2t=0


def test_shape_empty_is_zero():
    assert not shape_features(np.zeros((10, 10), bool), (0, 0, 10, 10), np.array([0, 0, 5, 5], np.float32)).any()


def test_flip_involution_and_signs():
    x = np.random.default_rng(0).normal(size=(3, 4, NODE_DIM)).astype(np.float32)
    x[..., 0] = np.abs(x[..., 0]) % 1
    y = flip_nodes(x)
    assert np.allclose(flip_nodes(y), x, atol=1e-6)
    assert np.allclose(y[..., 0], 1 - x[..., 0])
    i_vx, i_sin = NODE_FEATURES.index("vx"), NODE_FEATURES.index("sin2t")
    assert np.allclose(y[..., i_vx], -x[..., i_vx]) and np.allclose(y[..., i_sin], -x[..., i_sin])
    keep = [i for i in range(NODE_DIM) if i not in (0, i_vx, i_sin)]
    assert np.allclose(y[..., keep], x[..., keep])
    e = np.random.default_rng(1).normal(size=(5, EDGE_DIM)).astype(np.float32)
    assert np.allclose(flip_edges(flip_edges(e)), e) and np.allclose(flip_edges(e)[..., 1], -e[..., 1])


def test_edge_closing_speed_sign():
    a = {"cx": 0, "cy": 0, "h": 100, "vx": 0, "vy": 0, "kin": True, "box": np.array([-20, -50, 20, 50.])}
    b = {"cx": 100, "cy": 0, "h": 100, "vx": 0, "vy": 0, "kin": True, "box": np.array([80, -50, 120, 50.])}
    e = edge_features(a, b, 0.0, prev_dist=(2.0, 0.1))     # distance went 2.0 -> 1.0 in 0.1 s
    assert abs(e[5] - 10.0) < 1e-4 and e[8] == 1.0 and e[0] == 1.0


def test_crop_iou():
    a = np.ones((10, 10), bool)
    assert crop_iou(a, (0, 0, 10, 10), a, (5, 0, 15, 10)) == 5 * 10 / (200 - 50)
