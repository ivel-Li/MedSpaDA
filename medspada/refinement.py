"""Shared class-agnostic spatial logit residual for generalist SAM3."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class GeneralistSpatialResidualC0(nn.Module):
    """Refine one selected query logit map without changing its acceptance.

    Every concept uses the same weights.  The final layer is initialized to
    zero, so installing C0 initially reproduces the frozen base prediction
    exactly.
    """

    def __init__(self, residual_limit: float = 8.0) -> None:
        super().__init__()
        self.residual_limit = float(residual_limit)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.GroupNorm(6, 24),
            nn.GELU(),
            nn.Conv2d(24, 24, 3, padding=1),
            nn.GroupNorm(6, 24),
            nn.GELU(),
        )
        self.down = nn.Sequential(
            nn.Conv2d(24, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(56, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 24, 3, padding=1),
            nn.GroupNorm(6, 24),
            nn.GELU(),
        )
        self.residual = nn.Conv2d(24, 1, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, logits: Tensor, scores: Tensor) -> tuple[Tensor, Tensor]:
        if logits.ndim != 3 or scores.ndim != 1 or logits.shape[0] != scores.shape[0]:
            raise ValueError("expected logits [N,H,W] and scores [N]")
        height, width = logits.shape[-2:]
        score_map = scores[:, None, None].expand(-1, height, width)
        features = self.stem(torch.stack(
            [torch.tanh(logits / 4.0), logits.sigmoid(), score_map], dim=1
        ))
        deep = self.down(features)
        deep = F.interpolate(deep, (height, width), mode="bilinear", align_corners=False)
        features = self.fuse(torch.cat([features, deep], dim=1))
        residual = self.residual_limit * torch.tanh(
            self.residual(features).squeeze(1) / self.residual_limit
        )
        return logits + residual, residual
