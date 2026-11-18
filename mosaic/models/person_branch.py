"""Person branch (spec section 6): node embed -> per-frame dense GATv2 -> per-person GRU -> attention pooling.

Shapes (B = windows in the batch, P people, T frames):
    nodes (B, P, T, 29)   node_mask (B, P, T) bool   edges (B, T, P, P, 9)
Returns pooled person vector (B, 128), people attention (B, P), has_people (B,).
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class DenseGATv2Layer(nn.Module):
    """Fully connected, masked GATv2 with edge features, self-loops, residual and LayerNorm."""

    def __init__(self, dim: int = 64, heads: int = 4, edge_dim: int = 64) -> None:
        super().__init__()
        assert dim % heads == 0
        self.h, self.dk = heads, dim // heads
        self.w_l = nn.Linear(dim, dim, bias=False)
        self.w_r = nn.Linear(dim, dim, bias=False)
        self.w_e = nn.Linear(edge_dim, dim, bias=False)
        self.attn = nn.Parameter(torch.empty(heads, self.dk))
        nn.init.xavier_uniform_(self.attn)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, e: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """x (N, P, D), e (N, P, P, De) embedded edges, valid (N, P) bool."""
        N, P, D = x.shape
        l = self.w_l(x).view(N, P, 1, self.h, self.dk)           # target i
        r = self.w_r(x).view(N, 1, P, self.h, self.dk)           # source j
        ee = self.w_e(e).view(N, P, P, self.h, self.dk)
        score = (F.leaky_relu(l + r + ee, 0.2) * self.attn).sum(-1)   # (N, P, P, H)
        eye = torch.eye(P, dtype=torch.bool, device=x.device)
        allowed = (valid.unsqueeze(2) & valid.unsqueeze(1)) | eye   # self-loops always allowed
        score = score.masked_fill(~allowed.unsqueeze(-1), float("-inf"))
        a = torch.softmax(score, dim=2)
        msg = torch.einsum("nijh,njhd->nihd", a, self.w_r(x).view(N, P, self.h, self.dk)).reshape(N, P, D)
        y = self.norm(x + self.out(msg))
        return y * valid.unsqueeze(-1)                              # padded people stay zero


class PersonBranch(nn.Module):
    def __init__(self, node_dim: int = 29, edge_dim: int = 9, emb: int = 64, hidden: int = 128,
                 heads: int = 4, layers: int = 2, use_gat: bool = True) -> None:
        super().__init__()
        self.node_embed = nn.Sequential(nn.Linear(node_dim, emb), nn.LayerNorm(emb), nn.GELU())
        self.edge_embed = nn.Sequential(nn.Linear(edge_dim, emb), nn.GELU())
        self.use_gat = use_gat
        self.gats = nn.ModuleList([DenseGATv2Layer(emb, heads, emb) for _ in range(layers)]) if use_gat else nn.ModuleList()
        self.gru = nn.GRUCell(emb, hidden)
        self.pool_w = nn.Linear(hidden, hidden)
        self.pool_v = nn.Linear(hidden, 1, bias=False)
        self.empty = nn.Parameter(torch.zeros(hidden))
        self.hidden = hidden

    def forward(self, nodes: torch.Tensor, node_mask: torch.Tensor, edges: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, P, T, _ = nodes.shape
        x = self.node_embed(nodes) * node_mask.unsqueeze(-1)             # (B,P,T,D)
        if self.use_gat:
            e = self.edge_embed(edges)                                  # (B,T,P,P,De)
            xt = x.permute(0, 2, 1, 3).reshape(B * T, P, -1)
            vt = node_mask.permute(0, 2, 1).reshape(B * T, P)
            et = e.reshape(B * T, P, P, -1)
            for gat in self.gats:
                xt = gat(xt, et, vt)
            x = xt.view(B, T, P, -1).permute(0, 2, 1, 3)
        h = x.new_zeros(B * P, self.hidden)
        flat = x.reshape(B * P, T, -1)
        m = node_mask.reshape(B * P, T)
        for t in range(T):
            new = self.gru(flat[:, t], h)
            h = torch.where(m[:, t : t + 1], new, h)                    # carry state over missing frames
        h = h.view(B, P, -1)
        person_valid = node_mask.any(-1)                                # (B,P)
        scores = self.pool_v(torch.tanh(self.pool_w(h))).squeeze(-1)
        scores = scores.masked_fill(~person_valid, float("-inf"))
        has = person_valid.any(-1)
        attn = torch.softmax(torch.where(has.unsqueeze(-1), scores, torch.zeros_like(scores)), dim=-1)
        attn = attn * person_valid                                      # exact zeros for absent people
        pooled = (attn.unsqueeze(-1) * h).sum(1)
        pooled = torch.where(has.unsqueeze(-1), pooled, self.empty.expand(B, -1))
        return pooled, attn, has
