"""Baseline temporal classifier for RWF-2000 violence detection.

Combines a 2D spatial feature extractor with a recurrent temporal model (GRU):
1. Spatial Backbone:
   - ResNet-18 (512-dim features) or LightCNN (128-dim features with 4x4 spatial grid).
   - Can freeze backbone weights (keeping normalization layers in eval mode when frozen)
     to evaluate scene feature representations.
2. Temporal Sequence Model:
   - Gated Recurrent Unit (GRU) or LSTM over sampled time steps, conditioned on
     per-step time deltas (Plan Section 2.2).
   - Respects ``validity_mask`` so padded frames never corrupt the final hidden state.
3. Binary Classification Head:
   - Linear projection to a single scalar logit.
   - Sigmoid activation yields the violence probability in [0.0, 1.0].
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ModelError(ValueError):
    """Raised when model configuration or inputs are invalid."""


class LightCNNBackbone(nn.Module):
    """Compact convolutional feature extractor preserving a 4x4 spatial grid.

    Designed for lightweight edge execution and deterministic offline unit testing
    without external weight downloads. Uses GroupNorm and a 4x4 spatial grid so
    actor positions and inter-frame movement are preserved even on small batch sizes.
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 128, freeze: bool = False) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.frozen = freeze
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(32 * 4 * 4, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
        )
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True) -> "LightCNNBackbone":
        super().train(False if self.frozen else mode)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, feature_dim]
        out = self.features(x)
        return self.proj(out)


class ResNet18Backbone(nn.Module):
    """ResNet-18 backbone stripped of its final fully-connected classification layer."""

    def __init__(self, pretrained: bool = False, freeze: bool = True) -> None:
        super().__init__()
        self.feature_dim = 512
        self.frozen = freeze
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        base = tv_models.resnet18(weights=weights)

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        # Exclude the final fc classification layer
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        if freeze:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()

    def train(self, mode: bool = True) -> "ResNet18Backbone":
        # Keep frozen BatchNorm running statistics fixed in eval mode
        super().train(False if self.frozen else mode)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.imagenet_mean) / self.imagenet_std
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)


class BaselineTemporalClassifier(nn.Module):
    """Baseline video sequence classifier: 2D Backbone + GRU/LSTM + Masked Temporal Pooling Head."""

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = False,
        freeze_backbone: bool = True,
        hidden_dim: int = 128,
        rnn_type: str = "gru",
        rnn_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.rnn_type_name = rnn_type

        # 1. Instantiate spatial backbone
        if backbone == "resnet18":
            self.backbone = ResNet18Backbone(pretrained=pretrained, freeze=freeze_backbone)
            feat_dim = self.backbone.feature_dim
        elif backbone == "cnn_light":
            # Only freeze cnn_light when explicitly requested and pretrained; an untrained
            # light CNN has no external pretrained weights so its projection is trainable
            # unless explicitly frozen after loading a checkpoint.
            self.backbone = LightCNNBackbone(feature_dim=128, freeze=False)
            feat_dim = 128
            if freeze_backbone and pretrained:
                for p in self.backbone.parameters():
                    p.requires_grad = False
                self.backbone.frozen = True
        else:
            raise ModelError(f"unknown backbone {backbone!r}; supported: ['resnet18', 'cnn_light']")

        self.feat_dim = feat_dim
        self.dropout_p = float(dropout)
        self.app_norm = nn.LayerNorm(feat_dim)
        self.mot_norm = nn.LayerNorm(feat_dim)

        # 2. Instantiate temporal sequence model
        # Input combines normalized appearance f_t, frame-to-frame motion delta_f_t, and time_delta (+1)
        rnn_input_dim = feat_dim * 2 + 1
        rnn_dropout = dropout if rnn_layers > 1 else 0.0
        if rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=rnn_input_dim,
                hidden_size=hidden_dim,
                num_layers=rnn_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=rnn_input_dim,
                hidden_size=hidden_dim,
                num_layers=rnn_layers,
                batch_first=True,
                dropout=rnn_dropout,
            )
        else:
            raise ModelError(f"unknown rnn_type {rnn_type!r}; supported: ['gru', 'lstm']")

        # 3. Masked temporal pooling (last + mean + max) -> binary classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, 1),
        )

    def extract_spatial_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract per-frame 2D spatial backbone features of shape (B, T, feat_dim)."""
        if frames.ndim != 5:
            raise ModelError(
                f"expected 5D tensor (B, T, C, H, W), got tensor with {frames.ndim} dimensions: {frames.shape}"
            )
        b, t, c, h, w = frames.shape
        frames_flat = frames.reshape(b * t, c, h, w)
        if getattr(self.backbone, "frozen", False):
            with torch.no_grad():
                features = self.backbone(frames_flat)
        else:
            features = self.backbone(frames_flat)
        return features.view(b, t, -1)

    def forward_from_features(
        self,
        feat_seq: torch.Tensor,
        validity_mask: Optional[torch.Tensor] = None,
        time_deltas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run temporal sequence encoder and classification head from spatial features (B, T, feat_dim)."""
        b, t, _ = feat_seq.shape

        if validity_mask is not None:
            if validity_mask.shape != (b, t):
                raise ModelError(
                    f"validity_mask must have shape ({b}, {t}), got {tuple(validity_mask.shape)}"
                )
            valid_bool = validity_mask.to(device=feat_seq.device, dtype=torch.bool)
            valid_f = valid_bool.to(dtype=feat_seq.dtype).unsqueeze(-1)  # [B, T, 1]
            feat_seq = feat_seq * valid_f
        else:
            valid_bool = None
            valid_f = None

        # Compute per-step feature motion delta_f_t = f_t - f_{t-1}
        feat_diff = torch.zeros_like(feat_seq)
        if t > 1:
            feat_diff[:, 1:, :] = feat_seq[:, 1:, :] - feat_seq[:, :-1, :]
        if valid_f is not None:
            feat_diff = feat_diff * valid_f

        norm_app = self.app_norm(feat_seq)
        norm_mot = self.mot_norm(feat_diff)
        if self.training and self.dropout_p > 0.0:
            keep_p = 1.0 - self.dropout_p
            ch_mask_a = (
                torch.empty((b, 1, feat_seq.size(-1)), device=feat_seq.device, dtype=feat_seq.dtype)
                .bernoulli_(keep_p)
                .div_(keep_p)
            )
            ch_mask_m = (
                torch.empty((b, 1, feat_seq.size(-1)), device=feat_seq.device, dtype=feat_seq.dtype)
                .bernoulli_(keep_p)
                .div_(keep_p)
            )
            norm_app = norm_app * ch_mask_a
            norm_mot = norm_mot * ch_mask_m
        if valid_f is not None:
            norm_app = norm_app * valid_f
            norm_mot = norm_mot * valid_f

        # Append per-step time delta channel [B, T, 1]
        if time_deltas is None:
            dt_channel = torch.zeros((b, t, 1), dtype=feat_seq.dtype, device=feat_seq.device)
        else:
            if time_deltas.shape != (b, t):
                raise ModelError(
                    f"time_deltas must have shape ({b}, {t}), got {tuple(time_deltas.shape)}"
                )
            dt_channel = time_deltas.to(device=feat_seq.device, dtype=feat_seq.dtype).unsqueeze(-1)
            if valid_f is not None:
                dt_channel = dt_channel * valid_f

        rnn_in = torch.cat([norm_app, norm_mot, dt_channel], dim=-1)  # [B, T, 2*feat_dim + 1]
        rnn_out, _ = self.rnn(rnn_in)  # [B, T, hidden_dim]

        if valid_bool is not None and valid_f is not None:
            valid_counts = valid_bool.long().sum(dim=1).clamp(min=1)  # [B]
            last_idx = (valid_counts - 1).view(b, 1, 1).expand(b, 1, rnn_out.size(2))
            last_valid = rnn_out.gather(1, last_idx).squeeze(1)
            mean_valid = (rnn_out * valid_f).sum(dim=1) / valid_counts.to(rnn_out.dtype).unsqueeze(-1)
            masked_for_max = rnn_out.masked_fill(~valid_bool.unsqueeze(-1), -1e9)
            max_valid = masked_for_max.max(dim=1).values
        else:
            last_valid = rnn_out[:, -1, :]
            mean_valid = rnn_out.mean(dim=1)
            max_valid = rnn_out.max(dim=1).values

        pooled = torch.cat([last_valid, mean_valid, max_valid], dim=-1)  # [B, 3 * hidden_dim]
        return self.classifier(pooled)

    def forward(
        self,
        frames: torch.Tensor,
        validity_mask: Optional[torch.Tensor] = None,
        time_deltas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass from raw frame tensor (B, T, C, H, W)."""
        feat_seq = self.extract_spatial_features(frames)
        return self.forward_from_features(
            feat_seq, validity_mask=validity_mask, time_deltas=time_deltas
        )

    @torch.no_grad()
    def predict_score(
        self,
        frames: torch.Tensor,
        validity_mask: Optional[torch.Tensor] = None,
        time_deltas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute violence probabilities in [0.0, 1.0] for a batch of clips."""
        self.eval()
        logits = self.forward(frames, validity_mask=validity_mask, time_deltas=time_deltas)
        return torch.sigmoid(logits)
