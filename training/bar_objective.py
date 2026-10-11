"""Bar-level elapsed, tempo, and local auxiliary supervision in score-relative seconds.

For a complete span with score IOIs ``s_i`` and predicted log ratios ``r_i``, span tempo
is ``b = log(sum(s_i*exp(r_i))/sum(s_i))`` and local is ``r_i - b``, so a constant log
shift over a span moves only ``b`` and leaves local at zero.  Horizon means are taken
first and then averaged with equal weight over active horizons; otherwise one-bar spans
would outnumber 16-bar spans by an order of magnitude.

Window layout, core ownership, and span eligibility come from the caller
(``training.bar_windows`` plus the locked native masks and start trust).  This module
never plans windows, never reads labels into a model input, clamps predictions before
``exp``, and never clamps trusted labels.  ``bar_ioi_mask`` filters local interval rows
only; tempo and elapsed still see the whole complete span.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from training.bar_supervision import DEFAULT_BAR_HORIZONS, validate_horizons
from training.train_expression import TrainingRunError


def _expand_pairs(starts: Tensor, ends: Tensor, device: torch.device) -> tuple[Tensor, Tensor]:
    """Return (span index, chord row) for every interval row ``start+1 ... end``."""
    lengths = ends - starts
    if torch.any(lengths <= 0):
        raise TrainingRunError("bar span endpoints must be strictly ordered chord indices")
    offsets = torch.cumsum(lengths, dim=0) - lengths
    pair_span = torch.repeat_interleave(torch.arange(lengths.numel(), device=device, dtype=torch.int64), lengths)
    position = torch.arange(int(lengths.sum().item()), device=device, dtype=torch.int64)
    return pair_span, torch.repeat_interleave(starts + 1, lengths) + position - torch.repeat_interleave(offsets, lengths)


def _span_sum(values: Tensor, batch_index: Tensor, starts: Tensor, ends: Tensor) -> Tensor:
    """Sum ``values`` over interval rows ``start+1 ... end`` of each span."""
    cumulative = torch.cumsum(values, dim=1)
    padded = torch.cat((torch.zeros(values.shape[0], 1, dtype=values.dtype, device=values.device), cumulative), dim=1)
    return padded[batch_index, ends + 1] - padded[batch_index, starts + 1]


def true_bar_objective(prediction_log_ratio: Tensor, target_log_ratio: Tensor,
                       score_ioi_seconds: Tensor, span_rows: Tensor, span_mask: Tensor,
                       span_horizons: Tensor, *, horizons: Sequence[int] = DEFAULT_BAR_HORIZONS,
                       log_ratio_clip: float = 4.0, minimum_elapsed_s: float = 1e-6,
                       bar_ioi_mask: Tensor | None = None) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Return weighted bar losses and auditable counts for one padded window batch."""
    selected_horizons = validate_horizons(horizons)
    if not math.isfinite(log_ratio_clip) or log_ratio_clip <= 0:
        raise TrainingRunError("bar log ratio clip must be finite and positive")
    if not math.isfinite(minimum_elapsed_s) or minimum_elapsed_s <= 0:
        raise TrainingRunError("bar minimum elapsed seconds must be finite and positive")
    if prediction_log_ratio.ndim != 2:
        raise TrainingRunError("bar IOI tensors must be shaped [batch, chord]")
    batch, chords = prediction_log_ratio.shape
    for name, value in (("target", target_log_ratio), ("score seconds", score_ioi_seconds)):
        if value.shape != prediction_log_ratio.shape:
            raise TrainingRunError(f"bar {name} arrays must match [batch, chord]")
    if span_rows.ndim != 3 or span_rows.shape[0] != batch or span_rows.shape[2] != 2:
        raise TrainingRunError("bar span rows must be shaped [batch, span, 2]")
    if span_mask.shape != span_horizons.shape or span_mask.shape[0] != batch:
        raise TrainingRunError("bar span mask must match span horizons and batch")
    if span_rows.dtype != torch.int64 or span_horizons.dtype != torch.int64 or span_mask.dtype != torch.bool:
        raise TrainingRunError("bar span tensors need int64 rows and horizons with a boolean mask")
    arrays = [prediction_log_ratio, target_log_ratio, score_ioi_seconds, span_rows, span_mask, span_horizons]
    if bar_ioi_mask is not None:
        if bar_ioi_mask.shape != prediction_log_ratio.shape or bar_ioi_mask.dtype != torch.bool:
            raise TrainingRunError("bar IOI mask must be a [batch, chord] boolean")
        arrays.append(bar_ioi_mask)
    if any(value.device != prediction_log_ratio.device for value in arrays):
        raise TrainingRunError("bar objective arrays must share a device")
    device = prediction_log_ratio.device
    requested = torch.isin(span_horizons, torch.tensor(selected_horizons, device=device, dtype=torch.int64))
    selected = span_mask & requested
    excluded_unrequested = int((span_mask & ~requested).sum().item())

    counts = selected.sum(dim=1)
    batch_index = torch.arange(batch, device=device, dtype=torch.int64).repeat_interleave(counts)
    rows = span_rows[selected]
    starts, ends = rows[:, 0], rows[:, 1]
    span_horizon = span_horizons[selected]
    if bool(span_mask.any()) and (int(span_rows[span_mask].min()) < 0 or int(span_rows[span_mask].max()) >= chords):
        raise TrainingRunError("bar span rows escape the padded window")
    pair_span, pair_row = _expand_pairs(starts, ends, device)
    pair_batch = batch_index[pair_span]

    row_mask = torch.zeros(batch * chords, dtype=torch.bool, device=device)
    if pair_row.numel():
        row_mask.index_fill_(0, pair_batch * chords + pair_row, True)
    row_mask = row_mask.view(batch, chords)
    zero = torch.zeros((), dtype=torch.float64, device=device)
    prediction = torch.where(row_mask, prediction_log_ratio.double(), zero)
    target = torch.where(row_mask, target_log_ratio.double(), zero)
    seconds = torch.where(row_mask, score_ioi_seconds.double(), zero)

    metrics: dict[str, Any] = {
        "units": "score-relative seconds; performer-wide oracle tempo scale remains removed",
        "tempo_definition": "b = log(sum(score_IOI*exp(log_ratio))/sum(score_IOI)) for each complete span",
        "local_definition": "log_ratio - b, so a constant shift over a span leaves local at zero",
        "horizon_weighting": "mean of active-horizon means; span counts never weight horizons",
        "prediction_clamp": [-log_ratio_clip, log_ratio_clip], "target_clamp": None,
        "minimum_elapsed_seconds": minimum_elapsed_s,
        "bar_ioi_mask_scope": "local interval rows only; tempo and elapsed keep the complete span",
        "geometry_owner": "training.bar_windows plans windows; this objective consumes caller span rows",
        "model_input": False, "deployment_evidence": False, "horizons": {},
    }
    if not span_horizon.numel():
        metrics.update(valid_span_count=0, supervised_interval_count=0, local_interval_count=0,
                       active_horizon_count=0, excluded_unrequested_horizon_count=excluded_unrequested,
                       local_rows_excluded_by_mask_count=0, spans_without_local_row_count=0,
                       target_clipped_count=0)
        empty = prediction.sum() * 0.0
        return {"bar_elapsed": empty, "bar_tempo": empty, "bar_local": empty}, metrics

    gathered = seconds[pair_batch, pair_row]
    if not torch.isfinite(gathered).all() or torch.any(gathered <= 0):
        raise TrainingRunError("bar score IOI seconds must be finite and positive on supervised intervals")
    for name, value in (("prediction", prediction_log_ratio), ("target", target_log_ratio)):
        if not torch.isfinite(value.double()[pair_batch, pair_row]).all():
            raise TrainingRunError(f"non-finite trusted bar {name} log ratio")

    predicted_ioi = seconds * torch.exp(prediction.clamp(-log_ratio_clip, log_ratio_clip))
    target_ioi = seconds * torch.exp(target)
    if not torch.isfinite(predicted_ioi).all() or not torch.isfinite(target_ioi).all():
        raise TrainingRunError("bar IOI exponentiation overflow")
    predicted_elapsed = _span_sum(predicted_ioi, batch_index, starts, ends)
    target_elapsed = _span_sum(target_ioi, batch_index, starts, ends)
    score_sum = _span_sum(seconds, batch_index, starts, ends)
    if torch.any(score_sum <= 0):
        raise TrainingRunError("bar score IOI sums must be positive")
    if torch.any(target_elapsed <= 0):
        raise TrainingRunError("bar target elapsed sums must be positive")

    tempo_prediction = torch.log(predicted_elapsed.clamp_min(minimum_elapsed_s)) - torch.log(score_sum)
    tempo_target = torch.log(target_elapsed.clamp_min(minimum_elapsed_s)) - torch.log(score_sum)
    tempo_residual = tempo_prediction - tempo_target
    elapsed_relative = (predicted_elapsed - target_elapsed) / target_elapsed.clamp_min(minimum_elapsed_s)
    local_span, local_batch, local_row = pair_span, pair_batch, pair_row
    excluded_local_rows = spans_without_local_row = 0
    if bar_ioi_mask is not None and pair_span.numel():
        keep = bar_ioi_mask[pair_batch, pair_row]
        excluded_local_rows = int((~keep).sum().item())
        spans_without_local_row = int((torch.bincount(pair_span[keep], minlength=int(span_horizon.numel())) == 0).sum())
        local_span, local_batch, local_row = pair_span[keep], pair_batch[keep], pair_row[keep]
    local_residual = (prediction_log_ratio.double()[local_batch, local_row]
                      - target_log_ratio.double()[local_batch, local_row] - tempo_residual[local_span])
    for name, values in (("tempo", tempo_residual), ("elapsed", elapsed_relative), ("local", local_residual)):
        if not torch.isfinite(values).all():
            raise TrainingRunError(f"non-finite bar {name} residual")

    local_horizon = span_horizon[local_span]
    span_row_horizon = span_horizon[pair_span]
    clipped_rows = prediction_log_ratio.detach().double()[pair_batch, pair_row].abs() > log_ratio_clip
    losses: dict[str, list[Tensor]] = {"bar_elapsed": [], "bar_tempo": [], "bar_local": []}
    for horizon in selected_horizons:
        key = str(horizon)
        spans_of_horizon = span_horizon == horizon
        pairs_of_horizon = local_horizon == horizon
        span_count, pair_count = int(spans_of_horizon.sum().item()), int(pairs_of_horizon.sum().item())
        entry = {"span_count": span_count, "local_interval_count": pair_count,
                 "prediction_clipped_row_count": int((clipped_rows & (span_row_horizon == horizon)).sum().item())}
        if span_count:
            for name, values in (("bar_elapsed", elapsed_relative), ("bar_tempo", tempo_residual)):
                selected_values = values[spans_of_horizon]
                losses[name].append(F.smooth_l1_loss(selected_values, torch.zeros_like(selected_values)))
                entry[f"{name.removeprefix('bar_')}_mean_absolute"] = float(selected_values.detach().abs().mean())
        else:
            entry.update(elapsed_mean_absolute=None, tempo_mean_absolute=None)
        if pair_count:
            local_values = local_residual[pairs_of_horizon]
            losses["bar_local"].append(F.smooth_l1_loss(local_values, torch.zeros_like(local_values)))
            entry["local_mean_absolute"] = float(local_values.detach().abs().mean())
        else:
            entry["local_mean_absolute"] = None
        metrics["horizons"][key] = entry

    def _combine(parts: list[Tensor]) -> Tensor:
        if not parts:
            return prediction.sum() * 0.0
        return torch.stack(parts).mean()

    metrics.update(valid_span_count=int(span_horizon.numel()), supervised_interval_count=int(pair_row.numel()),
                   local_interval_count=int(local_row.numel()),
                   active_horizon_count=sum(1 for horizon in selected_horizons if (span_horizon == horizon).any()),
                   excluded_unrequested_horizon_count=excluded_unrequested,
                   local_rows_excluded_by_mask_count=excluded_local_rows,
                   spans_without_local_row_count=spans_without_local_row,
                   target_clipped_count=int((target_elapsed < minimum_elapsed_s).sum().item()))
    return ({"bar_elapsed": _combine(losses["bar_elapsed"]), "bar_tempo": _combine(losses["bar_tempo"]),
             "bar_local": _combine(losses["bar_local"])}, metrics)
