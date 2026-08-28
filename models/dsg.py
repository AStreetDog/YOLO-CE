"""DSG: Denoised Selective Guidance Module.

Implements selective P2-to-P3 detail guidance with three key components:
  1. Channel projection + denoising: reduce P2 to P3 channel width with DWConv denoising
  2. Gated selection: learned gate selects which spatial positions receive P2 detail
  3. Residual injection: controlled residual scale (alpha) prevents feature corruption

Reference: Section 3.3 of the paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv, DWConv


class P2GuidedP3DN(nn.Module):
    """Denoised Selective P2-to-P3 Guidance (DSG).

    Takes [P2, P3] feature maps as input. Downsamples P2 to P3 resolution,
    applies denoising, gates the detail signal, and injects it into P3 with
    a learned residual scale alpha.

    Args:
        channels: [P2_channels, P3_channels] after width scaling.
        alpha: Initial residual injection strength (default 0.075).
        gate_ratio: Fraction of P3 channels used for gate hidden dim.
        alpha_limit: Upper clamp for the learnable alpha parameter.
        denoise_repeats: Number of DWConv denoising layers (default 1).
        gate_min: Minimum gate hidden channels.
        act: Activation function flag.
    """

    def __init__(
        self,
        channels: list[int],
        alpha: float = 0.075,
        gate_ratio: float = 0.125,
        alpha_limit: float = 1.0,
        denoise_repeats: int = 1,
        gate_min: int = 8,
        act=True,
    ):
        super().__init__()
        if len(channels) != 2:
            raise ValueError("P2GuidedP3DN expects two input channels: [P2, P3].")

        p2_channels, p3_channels = channels
        gate_channels = max(int(gate_min), int(p3_channels * gate_ratio))
        denoise_repeats = max(0, int(denoise_repeats))

        # Step 1: Channel projection (1x1) + spatial downsample (DWConv stride=2)
        self.p2_down = nn.Sequential(
            Conv(p2_channels, p3_channels, 1, 1, act=act),
            DWConv(p3_channels, p3_channels, 3, 2, act=act),
        )

        # Step 2: Denoising via depthwise convolution
        self.detail_denoise = (
            nn.Sequential(*[DWConv(p3_channels, p3_channels, 3, 1, act=act) for _ in range(denoise_repeats)])
            if denoise_repeats
            else nn.Identity()
        )

        # Step 3: Gated selection
        self.gate = nn.Sequential(
            Conv(p3_channels * 2, gate_channels, 1, 1, act=act),
            nn.Conv2d(gate_channels, p3_channels, 1, bias=True),
            nn.Sigmoid(),
        )

        # Learnable residual scale
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.alpha_limit = float(alpha_limit)

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            xs: [P2_features, P3_features] tensor list.

        Returns:
            Enhanced P3 features with selective P2 detail injection.
        """
        if len(xs) != 2:
            raise ValueError("P2GuidedP3DN expects [P2, P3] feature tensors.")

        p2, p3 = xs

        # Downsample P2 and denoise
        p2_detail = self.detail_denoise(self.p2_down(p2))
        if p2_detail.shape[-2:] != p3.shape[-2:]:
            p2_detail = F.interpolate(p2_detail, size=p3.shape[-2:], mode="bilinear", align_corners=False)

        # Gated injection
        gate = self.gate(torch.cat((p2_detail, p3), dim=1))
        scale = self.alpha.clamp(0.0, self.alpha_limit)
        return p3 + scale * gate * p2_detail
