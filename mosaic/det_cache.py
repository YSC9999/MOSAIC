"""RF-DETR-Seg person detection + mask cache (spec section 3).

Runs in the isolated ``.venv`` (rfdetr, pycocotools). Heavy imports are lazy so
the pure helpers can be unit-tested in the main environment.

Per clip pickle::

    {"meta": {...},
     "frames": [{"frame_index", "timestamp", "boxes" (n,4) float32 xyxy,
                 "scores" (n,), "det_idx" (n,) int, "mask_valid" (n,) bool,
                 "mask_rle" [n COCO RLE dicts | None], "mask_area" (n,),
                 "mask_centroid" (n,2), "mask_bbox" (n,4), "overlap" (n,n) float16}, ...]}

``overlap[i, j]`` = |M_i ∩ M_j| / min(|M_i|, |M_j|) for valid masks, else 0.

Cost control (measured: full-frame mask handling of ~70 detections/frame took
0.30 s/frame vs 0.13 s for the detector itself):

- masks are processed only for the ``MAX_MASKED`` highest-scoring persons per
  frame; the rest keep their box/score with ``mask_valid = False``;
- each mask is stored as COCO RLE of the crop ``mask_crop`` (box expanded by
  ``CROP_MARGIN`` of its width/height, clamped) with the crop's (x1, y1, x2, y2)
  offset. Pixels outside that crop are dropped.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PERSON_CLASS_NAME = "person"
MIN_BOX_W = 4.0
MIN_BOX_H = 8.0
MIN_MASK_PIXELS = 16
MIN_MASK_BOX_IOU = 0.5
MAX_MASKED = 24
CROP_MARGIN = 0.25


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def clamp_and_filter(boxes: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """Clamp xyxy boxes to the frame; return (clamped boxes, keep mask)."""
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, width)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, height)
    keep = ((b[:, 2] - b[:, 0]) >= MIN_BOX_W) & ((b[:, 3] - b[:, 1]) >= MIN_BOX_H)
    return b, keep


def crop_region(box: np.ndarray, width: int, height: int, margin: float = CROP_MARGIN) -> Tuple[int, int, int, int]:
    bw, bh = box[2] - box[0], box[3] - box[1]
    x1 = int(max(0, np.floor(box[0] - margin * bw)))
    y1 = int(max(0, np.floor(box[1] - margin * bh)))
    x2 = int(min(width, np.ceil(box[2] + margin * bw)))
    y2 = int(min(height, np.ceil(box[3] + margin * bh)))
    return x1, y1, x2, y2


def mask_stats(mask: np.ndarray, box: np.ndarray, offset: Tuple[int, int] = (0, 0)) -> Dict[str, Any]:
    """Area, centroid, mask bbox and validity for one boolean mask (crop at ``offset``)."""
    ys, xs = np.nonzero(mask)
    xs = xs + offset[0]
    ys = ys + offset[1]
    area = int(len(xs))
    if area == 0:
        return {"area": 0, "centroid": (0.0, 0.0), "bbox": (0.0, 0.0, 0.0, 0.0), "valid": False}
    mb = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)
    valid = area >= MIN_MASK_PIXELS and box_iou(mb, box) >= MIN_MASK_BOX_IOU
    return {"area": area, "centroid": (float(xs.mean()), float(ys.mean())), "bbox": tuple(mb.tolist()), "valid": bool(valid)}


def pairwise_overlap(masks: List[Optional[np.ndarray]], boxes: np.ndarray, areas: np.ndarray, valid: np.ndarray,
                     crops: Optional[List[Tuple[int, int, int, int]]] = None) -> np.ndarray:
    """Intersection over the smaller area. ``masks[i]`` covers ``crops[i]`` (full frame if None)."""
    n = len(masks)
    out = np.zeros((n, n), dtype=np.float16)
    for i in range(n):
        if not valid[i]:
            continue
        for j in range(i + 1, n):
            if not valid[j]:
                continue
            ci = crops[i] if crops else (0, 0, masks[i].shape[1], masks[i].shape[0])
            cj = crops[j] if crops else (0, 0, masks[j].shape[1], masks[j].shape[0])
            x1, y1 = max(ci[0], cj[0]), max(ci[1], cj[1])
            x2, y2 = min(ci[2], cj[2]), min(ci[3], cj[3])
            if x2 <= x1 or y2 <= y1:
                continue
            a = masks[i][y1 - ci[1]:y2 - ci[1], x1 - ci[0]:x2 - ci[0]]
            b = masks[j][y1 - cj[1]:y2 - cj[1], x1 - cj[0]:x2 - cj[0]]
            inter = int(np.logical_and(a, b).sum())
            v = inter / max(1, min(areas[i], areas[j]))
            out[i, j] = out[j, i] = v
    return out


def person_class_ids() -> List[int]:
    from rfdetr.utilities.class_names import COCO_CLASSES

    ids = [int(k) for k, v in COCO_CLASSES.items() if v == PERSON_CLASS_NAME]
    if not ids:
        raise RuntimeError("person class not found in rfdetr COCO_CLASSES")
    return ids


def process_frame(model, rgb: np.ndarray, threshold: float, person_ids: List[int]) -> Dict[str, Any]:
    from pycocotools import mask as mask_utils

    h, w = rgb.shape[:2]
    det = model.predict(rgb, threshold=threshold)
    if len(det):
        sel = np.isin(det.class_id, person_ids)
        boxes, keep = clamp_and_filter(det.xyxy[sel], w, h)
        scores = det.confidence[sel].astype(np.float32)
        masks_all = det.mask[sel] if det.mask is not None else None
        idx = np.nonzero(keep)[0]
        boxes, scores = boxes[idx], scores[idx]
        masks = [masks_all[k] for k in idx] if masks_all is not None else [None] * len(idx)
    else:
        boxes, scores, masks = np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), []
    n = len(boxes)
    areas = np.zeros(n, np.int64)
    cents = np.zeros((n, 2), np.float32)
    mbbox = np.zeros((n, 4), np.float32)
    valid = np.zeros(n, bool)
    crops: List[Tuple[int, int, int, int]] = [(0, 0, 0, 0)] * n
    crop_masks: List[Optional[np.ndarray]] = [None] * n
    rles: List[Optional[dict]] = [None] * n
    order = np.argsort(-scores, kind="stable")[:MAX_MASKED]
    for k in order.tolist():
        if masks[k] is None:
            continue
        c = crop_region(boxes[k], w, h)
        cm = np.ascontiguousarray(masks[k][c[1]:c[3], c[0]:c[2]])
        st = mask_stats(cm, boxes[k], offset=(c[0], c[1]))
        areas[k], cents[k], mbbox[k], valid[k] = st["area"], st["centroid"], st["bbox"], st["valid"]
        rle = mask_utils.encode(np.asfortranarray(cm.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("ascii")
        rles[k], crops[k], crop_masks[k] = rle, c, cm
    overlap = pairwise_overlap(crop_masks, boxes, areas, valid, crops)
    return {
        "boxes": boxes, "scores": scores, "det_idx": np.arange(n, dtype=np.int32),
        "mask_valid": valid, "mask_rle": rles, "mask_crop": np.asarray(crops, np.int32).reshape(-1, 4),
        "mask_area": areas, "mask_centroid": cents, "mask_bbox": mbbox, "overlap": overlap,
    }
