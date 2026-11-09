from __future__ import annotations

import numpy as np

def dice(prediction, target):
    denominator = float(prediction.sum() + target.sum())
    return 1.0 if denominator == 0 else float(
        2 * np.logical_and(prediction, target).sum() / denominator
    )


def selected_q0(output, num_o2o, torch):
    logits = output["pred_logits"].squeeze(-1)[:, :num_o2o]
    presence = output["presence_logit_dec"]
    if presence.ndim == 1:
        presence = presence[:, None]
    scores = logits.sigmoid() * presence.sigmoid()
    best = scores.argmax(dim=1)
    rows = torch.arange(scores.shape[0], device=scores.device)
    return output["pred_masks"][rows, best], scores[rows, best], best


def c0_window_loss(refined, residual, scores, target, threshold, torch):
    # [C,D,H,W] and [C,D]
    accepted = scores >= threshold
    positive = target.flatten(2).sum(-1) > 0
    pixel_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        refined, target, reduction="none"
    ).mean(dim=(-2, -1))

    def mean_or_zero(values, mask):
        return values[mask].mean() if mask.any() else values.sum() * 0

    positive_bce = mean_or_zero(pixel_bce, accepted & positive)
    negative_bce = mean_or_zero(pixel_bce, accepted & ~positive)
    if (accepted & positive).any() and (accepted & ~positive).any():
        balanced_bce = 0.5 * (positive_bce + negative_bce)
    else:
        balanced_bce = positive_bce + negative_bce

    probability = refined.sigmoid()
    intersection = (probability * target).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    row_dice = 1 - (2 * intersection + 1) / (denominator + 1)
    positive_row_dice = mean_or_zero(row_dice, accepted & positive)

    effective = probability * accepted[..., None, None]
    volume_losses = []
    for category in range(target.shape[0]):
        inter = (effective[category] * target[category]).sum()
        denom = effective[category].sum() + target[category].sum()
        volume_losses.append(1 - (2 * inter + 1) / (denom + 1))
    volume_dice = torch.stack(volume_losses).mean()
    residual_l1 = mean_or_zero(residual.abs().mean(dim=(-2, -1)), accepted)
    loss = 0.5 * balanced_bce + 0.5 * positive_row_dice + volume_dice + 1e-3 * residual_l1
    return loss, {
        "balanced_bce": float(balanced_bce.detach()),
        "positive_row_dice": float(positive_row_dice.detach()),
        "volume_dice": float(volume_dice.detach()),
        "residual_l1": float(residual_l1.detach()),
    }
