"""Silhouette-kinematic descriptors (spec section 5).

Raw (unstandardized) features are cached; standardization with train-fit
statistics, clipping to +-5 and horizontal flip are applied at training time
(``FLIP_NEGATE_NODE`` / ``FLIP_MIRROR_NODE`` / ``FLIP_NEGATE_EDGE``).

Acceleration and jerk are stored as log1p of the magnitude (capped at ACCEL_CAP / JERK_CAP):
measured on real clips at 10 fps, a +-20 clip saturated jerk (mean 18) and carried no signal.

Units: lengths are divided by the person's box height h; rates by the real time
between the two sampled frames involved, so features do not depend on
resolution or frame rate.

Mask decoding needs pycocotools (``.venv``); it is imported lazily.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

NODE_FEATURES = [
    # geometry (6)
    "cx", "cy", "w", "h", "log_aspect", "mask_fill",
    # shape (7), zero when the mask is invalid
    "log_mask_aspect", "reach", "solidity", "upper_width", "eccentricity", "cos2t", "sin2t",
    # kinematics (9)
    "vx", "vy", "speed", "accel", "jerk", "dlog_area", "daspect", "dreach", "mask_iou_prev",
    # interaction (3)
    "max_overlap", "n_near", "nearest_dist",
    # quality / flags (4)
    "det_conf", "mask_valid", "gap", "has_prev",
]
EDGE_FEATURES = ["dist", "dx", "dy", "mask_overlap", "box_iou", "closing_speed", "rel_speed",
                 "log_h_ratio", "both_kin"]
NODE_DIM = len(NODE_FEATURES)  # 29
EDGE_DIM = len(EDGE_FEATURES)  # 9
assert NODE_DIM == 29 and EDGE_DIM == 9

FLIP_MIRROR_NODE = [NODE_FEATURES.index("cx")]  # x -> 1 - x
FLIP_NEGATE_NODE = [NODE_FEATURES.index("vx"), NODE_FEATURES.index("sin2t")]
FLIP_NEGATE_EDGE = [EDGE_FEATURES.index("dx")]

VEL_CLIP = 10.0
RATE_CLIP = 20.0
ACCEL_CAP = 200.0   # heights/s^2 before log compression
JERK_CAP = 2000.0   # heights/s^3 before log compression
DIST_CLIP = 5.0
NEAR_RADIUS = 1.5
GAP_NORM = 16.0
SHAPE_SLICE = slice(6, 13)


# ----------------------------------------------------------------------------- masks
def decode_crop(rle: Optional[dict]) -> Optional[np.ndarray]:
    if rle is None:
        return None
    from pycocotools import mask as mask_utils

    r = dict(rle)
    if isinstance(r["counts"], str):
        r["counts"] = r["counts"].encode("ascii")
    return mask_utils.decode(r).astype(bool)


def shape_features(mask: np.ndarray, crop: Sequence[int], box: np.ndarray) -> np.ndarray:
    """7 shape features from a crop mask whose top-left is at (crop[0], crop[1])."""
    out = np.zeros(7, np.float32)
    h = max(float(box[3] - box[1]), 1.0)
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return out
    mw, mh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    out[0] = np.log(mw / mh)
    cx, cy = xs.mean(), ys.mean()
    m8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = np.concatenate([c.reshape(-1, 2) for c in contours]) if contours else np.stack([xs, ys], 1)
    out[1] = float(np.sqrt(((pts - [cx, cy]) ** 2).sum(1)).max()) / h
    hull = cv2.convexHull(pts.astype(np.int32))
    hull_area = float(cv2.contourArea(hull))
    out[2] = min(1.0, len(xs) / hull_area) if hull_area > 0 else 1.0
    y_mid = ys.min() + mh / 2.0
    up = xs[ys < y_mid]
    out[3] = (up.max() - up.min() + 1) / h if len(up) else 0.0
    mu20, mu02 = ((xs - cx) ** 2).mean(), ((ys - cy) ** 2).mean()
    mu11 = ((xs - cx) * (ys - cy)).mean()
    common = np.sqrt(((mu20 - mu02) / 2) ** 2 + mu11 ** 2)
    l1, l2 = (mu20 + mu02) / 2 + common, (mu20 + mu02) / 2 - common
    out[4] = float(np.sqrt(max(0.0, 1 - l2 / l1))) if l1 > 0 else 0.0
    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    out[5], out[6] = np.cos(2 * theta), np.sin(2 * theta)
    return out


def crop_iou(ma: np.ndarray, ca: Sequence[int], mb: np.ndarray, cb: Sequence[int]) -> float:
    x1, y1 = max(ca[0], cb[0]), max(ca[1], cb[1])
    x2, y2 = min(ca[2], cb[2]), min(ca[3], cb[3])
    inter = 0
    if x2 > x1 and y2 > y1:
        a = ma[y1 - ca[1]:y2 - ca[1], x1 - ca[0]:x2 - ca[0]]
        b = mb[y1 - cb[1]:y2 - cb[1], x1 - cb[0]:x2 - cb[0]]
        inter = int(np.logical_and(a, b).sum())
    union = int(ma.sum()) + int(mb.sum()) - inter
    return inter / union if union > 0 else 0.0


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / u) if u > 0 else 0.0


# ----------------------------------------------------------------------------- per-observation
def clip_observations(det: Dict[str, Any], track_ids: List[np.ndarray]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Node features for every tracked observation, keyed by (frame_row, track_id).

    Each value holds ``x`` (29,), plus centroid / height / velocity / kin flag used for edges.
    """
    meta, frames = det["meta"], det["frames"]
    W0, H0 = float(meta["width"]), float(meta["height"])
    ts = np.array([f["timestamp"] for f in frames], np.float64)
    pos = [f.get("pos", r) for r, f in enumerate(frames)]   # position in the 64 sampled frames (gated clips skip some)
    obs: Dict[Tuple[int, int], Dict[str, Any]] = {}
    history: Dict[int, List[Dict[str, Any]]] = {}

    for r, f in enumerate(frames):
        ids = track_ids[r]
        boxes = np.asarray(f["boxes"], np.float32)
        rows = [k for k in range(len(ids)) if ids[k] >= 0]
        # frame-level interaction context among tracked people
        cents = {k: ((boxes[k, 0] + boxes[k, 2]) / 2, (boxes[k, 1] + boxes[k, 3]) / 2) for k in rows}
        for k in rows:
            t = int(ids[k])
            b = boxes[k]
            w, h = max(float(b[2] - b[0]), 1.0), max(float(b[3] - b[1]), 1.0)
            cx, cy = cents[k]
            x = np.zeros(NODE_DIM, np.float32)
            valid = bool(f["mask_valid"][k])
            x[0:6] = [cx / W0, cy / H0, w / W0, h / H0, np.log(w / h),
                      (f["mask_area"][k] / (w * h)) if valid else 0.0]
            mask = crop = None
            if valid:
                mask = decode_crop(f["mask_rle"][k])
                crop = tuple(int(v) for v in f["mask_crop"][k])
                x[SHAPE_SLICE] = shape_features(mask, crop, b)
            # interaction
            ov = np.asarray(f["overlap"], np.float32)
            others = [j for j in rows if j != k]
            x[22] = float(ov[k, [j for j in range(len(ids)) if j != k]].max()) if len(ids) > 1 else 0.0
            dists = [np.hypot(cents[j][0] - cx, cents[j][1] - cy) / h for j in others]
            x[23] = float(sum(d <= NEAR_RADIUS for d in dists))
            x[24] = float(min(DIST_CLIP, min(dists))) if dists else DIST_CLIP
            x[25] = float(f["scores"][k])
            x[26] = float(valid)
            hist = history.setdefault(t, [])
            vx = vy = 0.0
            kin = False
            if hist:
                p = hist[-1]
                dt = ts[r] - ts[p["r"]]
                if dt <= 0:
                    raise ValueError(f"non-positive time step between rows {p['r']} and {r}")
                hm = (h + p["h"]) / 2
                vx = float(np.clip((cx - p["cx"]) / hm / dt, -VEL_CLIP, VEL_CLIP))
                vy = float(np.clip((cy - p["cy"]) / hm / dt, -VEL_CLIP, VEL_CLIP))
                kin = True
                x[13], x[14], x[15] = vx, vy, np.hypot(vx, vy)
                if len(hist) >= 2 and hist[-1]["kin"]:
                    ax, ay = (vx - p["vx"]) / dt, (vy - p["vy"]) / dt
                    x[16] = float(np.log1p(min(ACCEL_CAP, np.hypot(ax, ay))))
                    if len(hist) >= 3 and p.get("acc") is not None:
                        jx, jy = (ax - p["acc"][0]) / dt, (ay - p["acc"][1]) / dt
                        x[17] = float(np.log1p(min(JERK_CAP, np.hypot(jx, jy))))
                    acc = (ax, ay)
                else:
                    acc = None
                x[18] = np.clip((np.log(w * h) - np.log(p["w"] * p["h"])) / dt, -RATE_CLIP, RATE_CLIP)
                x[19] = np.clip((np.log(w / h) - np.log(p["w"] / p["h"])) / dt, -RATE_CLIP, RATE_CLIP)
                if valid and p["valid"]:
                    x[20] = np.clip((x[7] - p["reach"]) / dt, -RATE_CLIP, RATE_CLIP)
                    x[21] = crop_iou(mask, crop, p["mask"], p["crop"])
                x[27] = min(1.0, (pos[r] - pos[p["r"]]) / GAP_NORM)
                x[28] = 1.0
            else:
                acc = None
            hist.append({"r": r, "cx": cx, "cy": cy, "w": w, "h": h, "vx": vx, "vy": vy, "kin": kin,
                         "acc": acc, "valid": valid, "reach": float(x[7]), "mask": mask, "crop": crop})
            if len(hist) > 3:
                hist.pop(0)
            obs[(r, t)] = {"x": x, "cx": cx, "cy": cy, "h": h, "vx": vx, "vy": vy, "kin": kin,
                           "box": b, "k": k}
    return obs


def edge_features(a: Dict[str, Any], b: Dict[str, Any], overlap: float,
                  prev_dist: Optional[Tuple[float, float]]) -> np.ndarray:
    """Edge i->j. ``prev_dist`` = (distance in heights, dt) at the pair's previous common frame."""
    e = np.zeros(EDGE_DIM, np.float32)
    hm = (a["h"] + b["h"]) / 2
    dist = np.hypot(b["cx"] - a["cx"], b["cy"] - a["cy"]) / hm
    e[0] = min(DIST_CLIP, dist)
    e[1] = np.clip((b["cx"] - a["cx"]) / hm, -DIST_CLIP, DIST_CLIP)
    e[2] = np.clip((b["cy"] - a["cy"]) / hm, -DIST_CLIP, DIST_CLIP)
    e[3] = overlap
    e[4] = box_iou(a["box"], b["box"])
    if prev_dist is not None:
        e[5] = np.clip(-(dist - prev_dist[0]) / prev_dist[1], -RATE_CLIP, RATE_CLIP)
    both = a["kin"] and b["kin"]
    if both:
        e[6] = min(RATE_CLIP, float(np.hypot(a["vx"] - b["vx"], a["vy"] - b["vy"])))
    e[7] = np.log(b["h"] / a["h"])
    e[8] = float(both)
    return e


# ----------------------------------------------------------------------------- windows
def build_clip_tensors(det: Dict[str, Any], tracks: Dict[str, Any], max_people: int = 16) -> Dict[str, np.ndarray]:
    """Arrays for one clip.

    nodes (W, P, T, 29), node_mask (W, P, T), edges (W, T, P, P, 9), count (W,), members (W, P)
    with P = max_people, T = window length.
    """
    frames = det["frames"]
    ts = np.array([f["timestamp"] for f in frames], np.float64)
    obs = clip_observations(det, tracks["track_ids"])
    rows = tracks["window_slots"] if "window_slots" in tracks else tracks["window_rows"]   # per window: row index or -1 for a skipped frame
    W, T, P = len(rows), len(rows[0]), max_people
    nodes = np.zeros((W, P, T, NODE_DIM), np.float32)
    mask = np.zeros((W, P, T), bool)
    edges = np.zeros((W, T, P, P, EDGE_DIM), np.float32)
    members = np.full((W, P), -1, np.int64)
    count = np.zeros(W, np.float32)
    last_pair: Dict[Tuple[int, int], Tuple[float, int]] = {}
    pair_dist_cache: Dict[Tuple[int, int, int], float] = {}

    def pair_dist(r: int, ti: int, tj: int) -> Optional[float]:
        key = (r, min(ti, tj), max(ti, tj))
        if key not in pair_dist_cache:
            a, b = obs.get((r, ti)), obs.get((r, tj))
            if a is None or b is None:
                return None
            pair_dist_cache[key] = float(np.hypot(b["cx"] - a["cx"], b["cy"] - a["cy"]) / ((a["h"] + b["h"]) / 2))
        return pair_dist_cache[key]

    for wk, win in enumerate(tracks["windows"]):
        mem = win["members"][:P]
        members[wk, : len(mem)] = mem
        count[wk] = win["count_uncapped"]
        for ti, r in enumerate(rows[wk]):
            if r < 0:
                continue
            present = [(pi, t) for pi, t in enumerate(mem) if (r, t) in obs]
            for pi, t in present:
                nodes[wk, pi, ti] = obs[(r, t)]["x"]
                mask[wk, pi, ti] = True
            ov = np.asarray(frames[r]["overlap"], np.float32)
            for pi, t in present:
                for pj, u in present:
                    if pi == pj:
                        continue
                    a, b = obs[(r, t)], obs[(r, u)]
                    prev = None
                    for rp in range(r - 1, max(-1, r - 17), -1):  # previous common frame, within patience
                        d = pair_dist(rp, t, u)
                        if d is not None:
                            prev = (d, ts[r] - ts[rp])
                            break
                    edges[wk, ti, pi, pj] = edge_features(a, b, float(ov[a["k"], b["k"]]), prev)
    if not (np.isfinite(nodes).all() and np.isfinite(edges).all()):
        raise ValueError("non-finite descriptor values")
    return {"nodes": nodes.astype(np.float16), "node_mask": mask, "edges": edges.astype(np.float16),
            "count": count, "members": members}


def flip_nodes(x: np.ndarray) -> np.ndarray:
    """Horizontal flip on raw node features (..., 29)."""
    y = x.copy()
    y[..., FLIP_MIRROR_NODE] = 1.0 - y[..., FLIP_MIRROR_NODE]
    y[..., FLIP_NEGATE_NODE] = -y[..., FLIP_NEGATE_NODE]
    return y


def flip_edges(e: np.ndarray) -> np.ndarray:
    y = e.copy()
    y[..., FLIP_NEGATE_EDGE] = -y[..., FLIP_NEGATE_EDGE]
    return y
