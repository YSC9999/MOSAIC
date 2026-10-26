"""Kinematic feature calculation from bounding box observations (Block 4 / Support Path).

Calculates physical motion derivatives across time:
1. Centroid: (cx, cy) center of bounding box.
2. Velocity: (vx, vy) displacement per second, strictly accounting for non-uniform
   delta_t intervals resulting from motion gating.
3. Acceleration: (ax, ay) velocity change per second.
4. Aspect Ratio & Dynamics: width / height and d(AR)/dt (essential for detecting
   sudden falls, tackles, and posture changes during physical altercations).
5. Frame-Normalized Coordinates: Scales coordinates by frame dimensions (W, H)
   so physical motion quantities are directly comparable across resolutions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .contracts import Box


@dataclass(frozen=True)
class KinematicState:
    """Calculated kinematic properties of a single observation."""

    centroid: List[float]               # [cx, cy] in pixel coordinates
    velocity: List[float]               # [vx, vy] in pixels / second
    acceleration: List[float]           # [ax, ay] in pixels / second^2
    speed: float                        # ||v|| in pixels / second
    acceleration_magnitude: float       # ||a|| in pixels / second^2
    aspect_ratio: float                 # width / height
    aspect_ratio_velocity: float        # d(AR) / dt
    normalized_centroid: List[float]    # [cx/W, cy/H] in [0, 1]
    normalized_velocity: List[float]    # [vx/W, vy/H] in 1 / second
    normalized_speed: float             # ||norm_v|| in 1 / second

    def to_dict(self) -> Dict[str, Any]:
        return {
            "centroid": [float(v) for v in self.centroid],
            "velocity": [float(v) for v in self.velocity],
            "acceleration": [float(v) for v in self.acceleration],
            "speed": float(self.speed),
            "acceleration_magnitude": float(self.acceleration_magnitude),
            "aspect_ratio": float(self.aspect_ratio),
            "aspect_ratio_velocity": float(self.aspect_ratio_velocity),
            "normalized_centroid": [float(v) for v in self.normalized_centroid],
            "normalized_velocity": [float(v) for v in self.normalized_velocity],
            "normalized_speed": float(self.normalized_speed),
        }


def compute_centroid(box: Box) -> Tuple[float, float]:
    """Compute (cx, cy) midpoint of an axis-aligned bounding box."""
    return (float((box.x1 + box.x2) / 2.0), float((box.y1 + box.y2) / 2.0))


def compute_kinematics(
    current_box: Box,
    current_timestamp: float,
    previous_box: Optional[Box] = None,
    previous_timestamp: Optional[float] = None,
    previous_velocity: Optional[List[float]] = None,
    frame_width: int = 1280,
    frame_height: int = 720,
) -> KinematicState:
    """Compute physical kinematics for an observation given previous temporal context.

    Safely handles motion-gating timestamp gaps (delta_t > 0) and guards against
    division-by-zero when delta_t is zero or negative.
    """
    w_frame = max(1.0, float(frame_width))
    h_frame = max(1.0, float(frame_height))

    cx, cy = compute_centroid(current_box)
    ar = float(current_box.width / max(1e-4, current_box.height))

    norm_cx = float(cx / w_frame)
    norm_cy = float(cy / h_frame)

    # Initial observation without prior context
    if previous_box is None or previous_timestamp is None:
        return KinematicState(
            centroid=[cx, cy],
            velocity=[0.0, 0.0],
            acceleration=[0.0, 0.0],
            speed=0.0,
            acceleration_magnitude=0.0,
            aspect_ratio=ar,
            aspect_ratio_velocity=0.0,
            normalized_centroid=[norm_cx, norm_cy],
            normalized_velocity=[0.0, 0.0],
            normalized_speed=0.0,
        )

    delta_t = float(current_timestamp - previous_timestamp)
    if delta_t <= 1e-5:
        # Zero or negative delta_t: cannot compute derivative, maintain prior velocity
        vx = previous_velocity[0] if previous_velocity else 0.0
        vy = previous_velocity[1] if previous_velocity else 0.0
        speed = math.sqrt(vx * vx + vy * vy)
        norm_vx = vx / w_frame
        norm_vy = vy / h_frame
        return KinematicState(
            centroid=[cx, cy],
            velocity=[vx, vy],
            acceleration=[0.0, 0.0],
            speed=speed,
            acceleration_magnitude=0.0,
            aspect_ratio=ar,
            aspect_ratio_velocity=0.0,
            normalized_centroid=[norm_cx, norm_cy],
            normalized_velocity=[norm_vx, norm_vy],
            normalized_speed=math.sqrt(norm_vx * norm_vx + norm_vy * norm_vy),
        )

    # 1. Velocity (displacement / delta_t)
    prev_cx, prev_cy = compute_centroid(previous_box)
    vx = float((cx - prev_cx) / delta_t)
    vy = float((cy - prev_cy) / delta_t)
    speed = float(math.sqrt(vx * vx + vy * vy))

    # Normalized velocity
    norm_vx = float(vx / w_frame)
    norm_vy = float(vy / h_frame)
    norm_speed = float(math.sqrt(norm_vx * norm_vx + norm_vy * norm_vy))

    # 2. Acceleration (delta_v / delta_t)
    if previous_velocity is not None:
        ax = float((vx - previous_velocity[0]) / delta_t)
        ay = float((vy - previous_velocity[1]) / delta_t)
        acc_mag = float(math.sqrt(ax * ax + ay * ay))
    else:
        ax, ay, acc_mag = 0.0, 0.0, 0.0

    # 3. Aspect ratio velocity (d(AR) / dt)
    prev_ar = float(previous_box.width / max(1e-4, previous_box.height))
    ar_vel = float((ar - prev_ar) / delta_t)

    return KinematicState(
        centroid=[cx, cy],
        velocity=[vx, vy],
        acceleration=[ax, ay],
        speed=speed,
        acceleration_magnitude=acc_mag,
        aspect_ratio=ar,
        aspect_ratio_velocity=ar_vel,
        normalized_centroid=[norm_cx, norm_cy],
        normalized_velocity=[norm_vx, norm_vy],
        normalized_speed=norm_speed,
    )
