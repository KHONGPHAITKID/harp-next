import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_soft_pixel_targets(
    pt_labels,
    coors,
    batch_size,
    height,
    width,
    num_classes,
    ignore_index=255,
    entropy_eps=1e-8,
    min_weight=0.0,
):
    """Build per-pixel soft targets from point labels and projection coordinates.

    Args:
        pt_labels: [N] point labels in [0, num_classes-1] or ignore_index.
        coors: [N, 3] integer [batch_id, y, x] coordinates.
        batch_size: number of batch items B.
        height: range image height H.
        width: range image width W.
        num_classes: number of semantic classes C.
        ignore_index: ignored label value.
        entropy_eps: epsilon for numerical stability in entropy/log.
        min_weight: lower bound for confidence weight in [0, 1].

    Returns:
        soft_targets: [B, C, H, W] normalized class histograms.
        pixel_weight: [B, 1, H, W] entropy-based confidence weights.
        hard_targets: [B, H, W] argmax labels (ignore_index where empty).
    """
    if coors.ndim != 2 or coors.shape[1] != 3:
        raise ValueError(f"coors must have shape [N, 3], got {tuple(coors.shape)}")

    device = pt_labels.device
    num_pixels = batch_size * height * width
    hist = torch.zeros(num_pixels, num_classes, device=device, dtype=torch.float32)

    b = coors[:, 0].long()
    y = coors[:, 1].long()
    x = coors[:, 2].long()
    cls = pt_labels.long()

    valid = (
        (b >= 0)
        & (b < batch_size)
        & (y >= 0)
        & (y < height)
        & (x >= 0)
        & (x < width)
        & (cls != ignore_index)
        & (cls >= 0)
        & (cls < num_classes)
    )

    if valid.any():
        flat_idx = b[valid] * (height * width) + y[valid] * width + x[valid]
        one_hot = F.one_hot(cls[valid], num_classes=num_classes).to(torch.float32)
        hist.index_add_(0, flat_idx, one_hot)

    counts = hist.sum(dim=1, keepdim=True)
    soft_flat = hist / counts.clamp_min(1.0)
    empty = counts.squeeze(1) == 0

    hard_flat = soft_flat.argmax(dim=1)
    hard_flat[empty] = ignore_index

    entropy = -(soft_flat.clamp_min(entropy_eps).log() * soft_flat).sum(dim=1)
    denom = math.log(num_classes) if num_classes > 1 else 1.0
    weight_flat = 1.0 - entropy / denom
    weight_flat = weight_flat.clamp(min=min_weight, max=1.0)
    weight_flat[empty] = 0.0

    soft_targets = soft_flat.view(batch_size, height, width, num_classes).permute(0, 3, 1, 2).contiguous()
    pixel_weight = weight_flat.view(batch_size, 1, height, width)
    hard_targets = hard_flat.view(batch_size, height, width)
    return soft_targets, pixel_weight, hard_targets


class SoftPixelCELoss(nn.Module):
    """Cross-entropy for soft per-pixel targets with per-pixel weighting."""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pixel_logits, soft_targets, pixel_weight):
        logp = F.log_softmax(pixel_logits, dim=1)
        soft_targets = soft_targets.clamp_min(self.eps)
        soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True).clamp_min(self.eps)

        ce = -(soft_targets * logp).sum(dim=1, keepdim=True)
        weighted = ce * pixel_weight
        denom = pixel_weight.sum().clamp_min(1.0)
        return weighted.sum() / denom
