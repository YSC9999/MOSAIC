"""Auxiliary threat and object evidence detection (Block 4).

Extracts secondary object evidence (e.g. handheld items, sticks, bags) and associates
them with proximate human tracks:
1. Auxiliary Role: Object evidence is strictly supportive. It never sets the scene
   violence label by itself.
2. Proximity Association: Associates detected objects with the nearest person bounding box
   when spatial overlap or proximity is detected.
3. Modality Independence: When object evidence is absent or disabled, the pipeline operates
   cleanly using person tracking and kinematics alone.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .contracts import Box, ObjectEvidence


class ObjectError(ValueError):
    """Raised when object evidence inputs are invalid."""


def compute_box_center_distance(b1: Box, b2: Box) -> float:
    """Compute Euclidean distance between the centers of two bounding boxes."""
    c1_x, c1_y = (b1.x1 + b1.x2) / 2.0, (b1.y1 + b1.y2) / 2.0
    c2_x, c2_y = (b2.x1 + b2.x2) / 2.0, (b2.y1 + b2.y2) / 2.0
    dx = c1_x - c2_x
    dy = c1_y - c2_y
    return float(math.sqrt(dx * dx + dy * dy))


def compute_box_iou(b1: Box, b2: Box) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes."""
    inter_x1 = max(b1.x1, b2.x1)
    inter_y1 = max(b1.y1, b2.y1)
    inter_x2 = min(b1.x2, b2.x2)
    inter_y2 = min(b1.y2, b2.y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0

    union_area = b1.area + b2.area - inter_area
    if union_area <= 0.0:
        return 0.0

    return float(inter_area / union_area)


def associate_objects_to_tracks(
    objects: Sequence[ObjectEvidence],
    person_boxes: Sequence[Tuple[int, Box]],
    max_distance_pixels: float = 80.0,
) -> List[ObjectEvidence]:
    """Associate detected objects with proximate person track IDs.

    person_boxes is a list of (track_id, Box) pairs.
    """
    if not objects or not person_boxes:
        return list(objects)

    associated: List[ObjectEvidence] = []
    for obj in objects:
        best_track_id: Optional[int] = None
        min_dist = max_distance_pixels

        for track_id, p_box in person_boxes:
            # Check overlap first
            iou = compute_box_iou(obj.box, p_box)
            if iou > 0.05:
                best_track_id = track_id
                break

            # Fall back to center distance
            dist = compute_box_center_distance(obj.box, p_box)
            if dist < min_dist:
                min_dist = dist
                best_track_id = track_id

        # Update associated track id if found
        updated = ObjectEvidence(
            object_id=obj.object_id,
            category=obj.category,
            box=obj.box,
            confidence=obj.confidence,
            frame_index=obj.frame_index,
            associated_track_id=best_track_id if best_track_id is not None else obj.associated_track_id,
        )
        associated.append(updated)

    return associated
