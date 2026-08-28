"""SAF-IoU: Scale-Adaptive Focusing IoU Loss.

Corrects the outlier-degree threshold in dynamic focusing (Wise-IoU style) losses
so that small targets are not systematically suppressed by the global statistics.

Key formula:
  φ(A_i) = 1 + κ * max(0, A_median / A_i - 1)     # scale correction factor
  β_i = L_i / (L_mean · φ(A_i))                     # corrected outlier degree
  f(β) = β · exp(1 - β)                             # non-monotonic focusing

Reference: Section 3.4 of the paper.
"""

import torch
import torch.nn as nn
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.metrics import bbox_iou


class SAFState:
    """Maintains EMA state for Scale-Adaptive Focusing.

    Tracks:
      - loss_mean: EMA of mean IoU loss across all targets
      - area_median: EMA of median target box area (for scale correction)
    """

    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum
        self.loss_mean: float = 0.5
        self.area_median: float = 1.0

    def update(self, loss_values: torch.Tensor, target_areas: torch.Tensor) -> None:
        with torch.no_grad():
            cur_loss_mean = loss_values.mean().item()
            self.loss_mean = self.momentum * self.loss_mean + (1 - self.momentum) * cur_loss_mean
            if target_areas.numel() > 0:
                cur_area_median = target_areas.median().item()
                self.area_median = self.momentum * self.area_median + (1 - self.momentum) * cur_area_median


# Module-level state (reset per training session via reset_saf_state)
_saf_state = SAFState()


def reset_saf_state(momentum: float = 0.9) -> None:
    """Reset EMA state at the start of each training session."""
    global _saf_state
    _saf_state = SAFState(momentum)


class SAFIoUBboxLoss(BboxLoss):
    """BboxLoss with Scale-Adaptive Focusing mechanism.

    Replaces standard CIoU loss with SAF-CIoU:
      1. Compute CIoU loss normally
      2. Track global loss mean and area median via EMA
      3. Compute scale correction factor φ(A) for each target
      4. Apply non-monotonic focusing f(β) = β·exp(1-β)

    Args:
        reg_max: Max regression range for DFL.
        kappa: Scale correction strength (default 1.5). Higher = stronger
               protection for small targets.
    """

    def __init__(self, reg_max=16, kappa: float = 1.5):
        super().__init__(reg_max)
        self.kappa = kappa

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask):
        """Compute SAF-CIoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pred_fg = pred_bboxes[fg_mask]
        target_fg = target_bboxes[fg_mask]

        # Standard CIoU
        iou = bbox_iou(pred_fg, target_fg, xywh=False, CIoU=True)
        raw_loss = 1.0 - iou

        # Scale-Adaptive Focusing
        with torch.no_grad():
            # Compute target box areas
            t_w = (target_fg[:, 2] - target_fg[:, 0]).clamp(min=1e-6)
            t_h = (target_fg[:, 3] - target_fg[:, 1]).clamp(min=1e-6)
            target_areas = (t_w * t_h).unsqueeze(-1)

            # Update EMA
            _saf_state.update(raw_loss.detach(), target_areas.squeeze(-1))

            # Scale correction: φ(A) = 1 + κ * max(0, A_med/A - 1)
            area_ratio = _saf_state.area_median / (target_areas + 1e-8)
            scale_correction = 1.0 + self.kappa * (area_ratio - 1.0).clamp(min=0.0)

            # Scale-corrected outlier degree
            corrected_threshold = _saf_state.loss_mean * scale_correction
            outlier_degree = raw_loss.detach() / (corrected_threshold + 1e-8)

            # Non-monotonic focusing: f(β) = β * exp(1-β)
            focusing = outlier_degree * torch.exp(1.0 - outlier_degree)
            focusing = focusing.clamp(min=0.1, max=1.5)

        loss_iou = raw_loss * focusing
        loss_iou = ((loss_iou * weight).sum() / target_scores_sum)

        # DFL loss (unchanged from base class)
        if self.dfl_loss is not None:
            target_ltrb = self.bbox2dist(anchor_points, target_bboxes, self.reg_max - 1)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max),
                                     target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl
