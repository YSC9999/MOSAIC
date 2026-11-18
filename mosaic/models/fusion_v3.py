"""v3 fusion (spec sections 8-9): scene + person + motion -> window logit -> clip logit (top-k mean)."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import nn

from .person_branch import PersonBranch


class FusionV3(nn.Module):
    """use_scene / use_person switch the ablations (A2 scene only, A3 person only)."""

    def __init__(self, scene_dim: int = 768, use_scene: bool = True, use_person: bool = True,
                 use_gat: bool = True, dropout: float = 0.3, modality_dropout: float = 0.2,
                 scene_in_dropout: float = 0.0) -> None:
        super().__init__()
        if not (use_scene or use_person):
            raise ValueError("need at least one branch")
        self.use_scene, self.use_person = use_scene, use_person
        self.modality_dropout = modality_dropout
        self.scene_in_drop = nn.Dropout(scene_in_dropout)
        self.scene_proj = nn.Linear(scene_dim, 256)
        self.person = PersonBranch(use_gat=use_gat) if use_person else None
        self.head = nn.Sequential(nn.Linear(256 + 128 + 3, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def window_logits(self, scene: torch.Tensor, nodes: Optional[torch.Tensor], node_mask: Optional[torch.Tensor],
                      edges: Optional[torch.Tensor], motion: torch.Tensor, count: torch.Tensor) -> Dict[str, torch.Tensor]:
        """All inputs flattened over windows (B = clips * W). Returns logits (B,) and person attention."""
        B = motion.shape[0]
        s = self.scene_proj(self.scene_in_drop(scene)) if self.use_scene else motion.new_zeros(B, 256)
        if self.use_person:
            p, attn, has = self.person(nodes, node_mask, edges)
        else:
            p, attn, has = motion.new_zeros(B, 128), None, torch.zeros(B, dtype=torch.bool, device=motion.device)
        if self.training and self.use_scene and self.use_person and self.modality_dropout > 0:
            r = torch.rand(B, device=motion.device)
            drop_s = r < self.modality_dropout
            drop_p = (~drop_s) & (torch.rand(B, device=motion.device) < self.modality_dropout)
            s = s * (~drop_s).unsqueeze(-1)
            p = p * (~drop_p).unsqueeze(-1)
        x = torch.cat([s, p, motion.unsqueeze(-1), has.float().unsqueeze(-1), (count / 16.0).unsqueeze(-1)], -1)
        return {"logit": self.head(x).squeeze(-1), "attn": attn}


def clip_logit(window_logits: torch.Tensor, k: Optional[int] = None) -> torch.Tensor:
    """(N, W) -> (N,) mean of the top-k window logits, k = ceil(0.4 W) by default."""
    W = window_logits.shape[-1]
    k = k or max(1, math.ceil(0.4 * W))
    return window_logits.topk(k, dim=-1).values.mean(-1)
