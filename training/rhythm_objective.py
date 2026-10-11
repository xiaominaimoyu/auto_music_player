"""Training-only elapsed-IOI supervision in score-relative physical seconds.

The target log ratio has already removed the fitted performer-wide tempo
scale.  This objective does not restore that oracle scale and never supplies
performance timing to the model.  Deployment at the score's fixed tempo is
a separate evaluation condition, not the meaning of this training loss.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from training.train_expression import TrainingRunError


DEFAULT_CUMULATIVE_SPANS = (1, 8, 32)


def validate_spans(spans: Sequence[int]) -> tuple[int, ...]:
    values = tuple(spans)
    if not values or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                         for value in values) or len(set(values)) != len(values):
        raise TrainingRunError("cumulative spans must be distinct positive interval counts")
    return values


def cumulative_ioi_loss(
    prediction_log_ratio: Tensor,
    target_log_ratio: Tensor,
    score_ioi_seconds: Tensor,
    trusted_mask: Tensor,
    *,
    spans: Sequence[int] = DEFAULT_CUMULATIVE_SPANS,
    log_ratio_clip: float = 4.0,
    minimum_elapsed_s: float = 1e-6,
) -> tuple[Tensor, dict[str, Any]]:
    """Equal-span mean SmoothL1 of relative cumulative elapsed-time errors.

    All tensors are shaped ``[batch, chord]``.  Each interval row represents
    the preceding IOI.  The caller must restrict ``trusted_mask`` to interval
    endpoints wholly inside a disjoint core and apply the native IOI mask.
    A span contributes only when every interval in it is trusted; unfolding
    the chord axis cannot cross samples, invalid gaps, halo, or padding.

    Only predictions are clamped before exp.  Ground truth is not clipped;
    malformed/overflowing trusted labels fail instead of silently changing
    supervision.  Float64 arithmetic keeps long-window summation stable.
    No valid span returns a differentiable zero and explicit zero coverage.
    """
    selected_spans = validate_spans(spans)
    if not math.isfinite(log_ratio_clip) or log_ratio_clip <= 0:
        raise TrainingRunError("cumulative prediction log-ratio clip must be finite and positive")
    if not math.isfinite(minimum_elapsed_s) or minimum_elapsed_s <= 0:
        raise TrainingRunError("cumulative minimum elapsed seconds must be finite and positive")
    tensors = (prediction_log_ratio, target_log_ratio, score_ioi_seconds, trusted_mask)
    if prediction_log_ratio.ndim != 2 or any(value.shape != prediction_log_ratio.shape for value in tensors):
        raise TrainingRunError("cumulative IOI arrays must have matching [batch, chord] shapes")
    if trusted_mask.dtype != torch.bool:
        raise TrainingRunError("cumulative IOI mask must be boolean")
    if any(value.device != prediction_log_ratio.device for value in tensors):
        raise TrainingRunError("cumulative IOI arrays must share a device")
    for name, value in zip(("prediction", "target", "score seconds"), tensors[:3]):
        if not torch.isfinite(value[trusted_mask]).all():
            raise TrainingRunError(f"non-finite trusted cumulative IOI {name}")
    if torch.any(score_ioi_seconds[trusted_mask] <= 0):
        raise TrainingRunError("trusted cumulative score IOIs must be positive seconds")

    # Mask first so even arbitrary padding data cannot poison exp/summation.
    zero = torch.zeros((), dtype=torch.float64, device=prediction_log_ratio.device)
    prediction = torch.where(trusted_mask, prediction_log_ratio.double(), zero)
    target = torch.where(trusted_mask, target_log_ratio.double(), zero)
    seconds = torch.where(trusted_mask, score_ioi_seconds.double(), zero)
    predicted_ioi = seconds * torch.exp(prediction.clamp(-log_ratio_clip, log_ratio_clip))
    target_ioi = seconds * torch.exp(target)
    if not torch.isfinite(predicted_ioi).all() or not torch.isfinite(target_ioi).all():
        raise TrainingRunError("cumulative IOI exponentiation overflow")
    # Multiplication by zero preserves a gradient path without inspecting
    # invalid prediction rows that could contain NaN.
    loss = prediction.sum() * 0.0
    active_span_losses = []
    span_counts: dict[str, int] = {}
    span_relative_mae: dict[str, float | None] = {}
    for span in selected_spans:
        key = str(span)
        if prediction.shape[1] < span:
            span_counts[key], span_relative_mae[key] = 0, None
            continue
        valid = trusted_mask.unfold(1, span, 1).all(dim=-1)
        count = int(valid.sum().item())
        span_counts[key] = count
        if not count:
            span_relative_mae[key] = None
            continue
        predicted_elapsed = predicted_ioi.unfold(1, span, 1).sum(dim=-1)[valid]
        target_elapsed = target_ioi.unfold(1, span, 1).sum(dim=-1)[valid]
        relative = (predicted_elapsed - target_elapsed) / target_elapsed.clamp_min(minimum_elapsed_s)
        if not torch.isfinite(relative).all():
            raise TrainingRunError("non-finite cumulative elapsed-time error")
        active_span_losses.append(F.smooth_l1_loss(relative, torch.zeros_like(relative)))
        span_relative_mae[key] = float(relative.detach().abs().mean().item())
    if active_span_losses:
        loss = torch.stack(active_span_losses).mean()
    return loss, {
        "span_counts": span_counts,
        "span_relative_mae": span_relative_mae,
        "valid_window_count": sum(span_counts.values()),
        "prediction_clipped_count": int((prediction_log_ratio.detach()[trusted_mask].abs() > log_ratio_clip).sum().item()),
    }
