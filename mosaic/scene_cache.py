"""Frozen Video Swin-T (Kinetics-400) window features (spec section 7, Stage A).

For each clip: decode with OpenCV, select frames (``windows.select_frame_indices``),
resize each frame straight to 224x224 (bilinear, aspect not preserved), apply
ImageNet mean/std, cut 16-frame windows and run the frozen backbone with its
classification head removed. Two views are cached: original and horizontal flip.

Output per clip: float16 array of shape (2, W, 768) - view 0 original, view 1 flipped.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import nn

from .ingestion import VideoReader
from .windows import pad_indices, select_frame_indices, window_index_matrix

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SCENE_DIM = 768
WINDOW_LEN = 16
STRIDE = 8


def load_backbone(device: torch.device) -> nn.Module:
    """Swin3D-T with Kinetics-400 weights and the head replaced by identity, in eval mode."""
    from torchvision.models.video import Swin3D_T_Weights, swin3d_t

    weights = Swin3D_T_Weights.KINETICS400_V1
    tf = weights.transforms()
    if tuple(round(v, 3) for v in tf.mean) != IMAGENET_MEAN or tuple(round(v, 3) for v in tf.std) != IMAGENET_STD:
        raise RuntimeError(f"unexpected Swin3D normalization: mean={tf.mean} std={tf.std}")
    model = swin3d_t(weights=weights)
    model.head = nn.Identity()
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def read_clip_frames(path: Path | str, max_frames: int = 64, size: int = 224,
                     n_frames: Optional[int] = None) -> Tuple[np.ndarray, List[bool]]:
    """Return (T, size, size, 3) RGB uint8 frames and the per-frame padding flags.

    With ``n_frames`` (the clip's frame count) frames are streamed and only the selected ones are kept,
    so memory stays at a few frames instead of the whole clip.
    """
    if n_frames is None:
        records = VideoReader(path).read_all()
        if not records:
            raise RuntimeError(f"no frames decoded from {path}")
        idx = select_frame_indices(len(records), max_frames)
        idx, is_pad = pad_indices(idx, WINDOW_LEN)
        frames = np.stack([_prep(records[i].frame, size) for i in idx])
        return frames, is_pad
    idx = select_frame_indices(n_frames, max_frames)
    idx, is_pad = pad_indices(idx, WINDOW_LEN)
    wanted, last, got = set(idx), max(idx), {}
    for rec in VideoReader(path).iter_frames():
        if rec.frame_index in wanted:
            got[rec.frame_index] = _prep(rec.frame, size)
        if rec.frame_index >= last:
            break
    missing = sorted(wanted - set(got))
    if missing:
        raise RuntimeError(f"{path}: frames {missing[:5]} not decodable (declared {n_frames})")
    return np.stack([got[i] for i in idx]), is_pad


def _prep(bgr: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (size, size), interpolation=cv2.INTER_LINEAR)


def frames_to_windows(frames: np.ndarray) -> torch.Tensor:
    """(T, H, W, 3) uint8 -> (W, 3, 16, H, W) normalized float tensor."""
    mat = window_index_matrix(len(frames), WINDOW_LEN, STRIDE)
    x = torch.from_numpy(frames).float().div_(255.0)  # T,H,W,C
    mean = torch.tensor(IMAGENET_MEAN).view(1, 1, 1, 3)
    std = torch.tensor(IMAGENET_STD).view(1, 1, 1, 3)
    x = (x - mean) / std
    x = x.permute(3, 0, 1, 2)  # C,T,H,W
    return torch.stack([x[:, row] for row in torch.from_numpy(mat)])  # W,C,16,H,W


@torch.no_grad()
def extract_clip_features(model: nn.Module, frames: np.ndarray, device: torch.device, batch: int = 3) -> np.ndarray:
    """Return (2, W, 768) float16 features for original and flipped views."""
    views = []
    for flip in (False, True):
        f = frames[:, :, ::-1].copy() if flip else frames
        wins = frames_to_windows(f)
        outs = []
        for i in range(0, len(wins), batch):
            xb = wins[i : i + batch].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                outs.append(model(xb).float().cpu())
        views.append(torch.cat(outs))
    feats = torch.stack(views).numpy()
    if feats.shape[-1] != SCENE_DIM or not np.isfinite(feats).all():
        raise RuntimeError(f"bad feature tensor shape={feats.shape} finite={np.isfinite(feats).all()}")
    return feats.astype(np.float16)


def cache_path(cache_dir: Path, rel_path: str) -> Path:
    return cache_dir / (rel_path.replace("\\", "/").rsplit(".", 1)[0] + ".npy")
