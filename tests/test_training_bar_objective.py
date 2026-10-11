from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.bar_objective import true_bar_objective
from training.bar_supervision import BarSupervisionError, build_bar_spans, decompose_bar_targets
from training.train_expression import TrainingRunError


def _batch(seconds, predictions, targets, spans, horizons):
    """Build one [1, chord] batch plus its [1, span, 2] span tensors."""
    def vector(values, dtype):
        return torch.as_tensor(np.asarray(values, dtype=dtype)[None, ...])
    rows = torch.as_tensor(np.asarray([[start, end] for start, end in spans], dtype=np.int64)[None, ...])
    return (vector(predictions, np.float32), vector(targets, np.float32), vector(seconds, np.float64),
            rows, torch.ones(1, len(spans), dtype=torch.bool), vector(horizons, np.int64))


def _shape_batch(batch, position, shape):
    tensors = list(batch[:6])
    tensors[position] = torch.zeros(shape, dtype=tensors[position].dtype)
    return true_bar_objective(*tensors)


def _closed_form_case():
    """Two one-bar spans and one four-bar span over eight unit-seconds intervals."""
    seconds = [0.0] + [1.0] * 8
    prediction = [0.0] + [0.2] * 4 + [-0.2] * 4
    target = [0.0] * 9
    return _batch(seconds, prediction, target, [(0, 4), (4, 8), (0, 8)], [1, 1, 4])


def test_identity_targets_produce_zero_losses():
    rng = np.random.default_rng(20261006)
    seconds = [0.0] + list(np.arange(1, 30) * 0.05 + 0.5)
    target = list(rng.normal(0.0, 0.08, 30).astype(np.float32))
    batch = _batch(seconds, target, target, [(0, 6), (6, 13), (0, 13), (13, 29)], [1, 1, 4, 16])
    losses, metrics = true_bar_objective(*batch)
    assert all(float(value) == pytest.approx(0.0, abs=1e-12) for value in losses.values())
    assert metrics["valid_span_count"] == 4 and metrics["supervised_interval_count"] == 42
    assert metrics["target_clipped_count"] == 0 and metrics["active_horizon_count"] == 3


def test_constant_shift_lands_in_tempo_and_not_local():
    seconds = [0.0] + [1.0] * 8
    base = [0.0] + [0.15, -0.05, 0.3, -0.2, 0.05, 0.2, -0.1, 0.1]
    shifted = [value + 0.4 for value in base]
    spans, kinds = [(0, 4), (4, 8), (0, 8)], [1, 1, 4]
    losses, metrics = true_bar_objective(*_batch(seconds, shifted, base, spans, kinds))
    assert float(losses["bar_local"]) == pytest.approx(0.0, abs=1e-15)
    for horizon in ("1", "4"):
        assert metrics["horizons"][horizon]["tempo_mean_absolute"] == pytest.approx(0.4, abs=1e-6)
        assert metrics["horizons"][horizon]["local_mean_absolute"] == pytest.approx(0.0, abs=1e-6)


def test_equal_horizon_weighting_differs_from_span_count_weighting():
    losses, metrics = true_bar_objective(*_closed_form_case())
    shift = math.log(math.cosh(0.2))
    smooth = lambda value: 0.5 * value * value
    tempo_one, tempo_four = smooth(0.2), smooth(shift)
    elapsed_one = 0.5 * (smooth(math.exp(0.2) - 1) + smooth(1 - math.exp(-0.2)))
    elapsed_four = smooth(math.cosh(0.2) - 1)
    local_one, local_four = 0.0, 0.5 * (0.04 + shift * shift)
    assert float(losses["bar_tempo"]) == pytest.approx((tempo_one + tempo_four) / 2, abs=1e-9)
    assert float(losses["bar_elapsed"]) == pytest.approx((elapsed_one + elapsed_four) / 2, abs=1e-9)
    assert float(losses["bar_local"]) == pytest.approx((local_one + local_four) / 2, abs=1e-9)
    pooled = (2 * tempo_one + tempo_four) / 3
    assert float(losses["bar_tempo"]) != pytest.approx(pooled, abs=1e-4)
    entry = metrics["horizons"]["1"]
    assert (entry["span_count"], entry["local_interval_count"]) == (2, 8)
    assert entry["tempo_mean_absolute"] == pytest.approx(0.2, abs=1e-6)
    assert entry["local_mean_absolute"] == pytest.approx(0.0, abs=1e-9)
    assert (metrics["horizons"]["4"]["span_count"], metrics["horizons"]["4"]["local_interval_count"]) == (1, 8)
    assert metrics["horizons"]["16"]["span_count"] == 0
    assert metrics["horizons"]["16"]["tempo_mean_absolute"] is None
    assert metrics["active_horizon_count"] == 2


def _reference(bar_sizes):
    seconds, beats, bars, phases = [], [], [], []
    for index, size in enumerate(bar_sizes):
        for position in range(size):
            bars.append(index)
            phases.append(position / size)
            beats.append(4 * index + 4 * position / size)
            seconds.append((4 * index + 4 * position / size) * 0.5)
    return {"chord_score_seconds": np.asarray(seconds), "chord_score_beats": np.asarray(beats),
            "chord_bar_index": np.asarray(bars, dtype=np.int64), "chord_bar_phase": np.asarray(phases),
            "partial_bar_indices": np.empty(0, dtype=np.int64)}


def test_tempo_and_local_match_bar_supervision_decomposition():
    reference = _reference([6, 8, 10, 7, 9, 5])
    spans = build_bar_spans(reference, (1,))
    count = spans.chord_count
    # The objective consumes interval seconds, never the absolute score clock.
    intervals = spans.score_ioi_seconds
    assert intervals[0] == 0.0 and bool(np.all(intervals[1:] > 0))
    target = np.random.default_rng(11).normal(0.0, 0.06, count).astype(np.float32)
    trusted = np.ones(count, dtype=bool)
    decomposition = decompose_bar_targets(spans, trusted, trusted, target)
    one_bar = spans.horizons == 1
    # A zero prediction makes b_prediction vanish, so the residuals become exactly
    # -b_target and -local_target from the already audited diagnostic module.
    batch = _batch(list(intervals), np.zeros(count, dtype=np.float32), target,
                   list(zip(spans.starts[one_bar].tolist(), spans.ends[one_bar].tolist())),
                   [1] * int(one_bar.sum()))
    losses, metrics = true_bar_objective(*batch)
    entry = metrics["horizons"]["1"]
    assert entry["span_count"] == decomposition["bar_count"] == int(one_bar.sum())
    assert entry["tempo_mean_absolute"] == pytest.approx(
        float(np.abs(decomposition["bar_tempo_log_ratio"]).mean()), abs=1e-6)
    covered = np.isfinite(decomposition["local_log_ioi_ratio"])
    assert entry["local_mean_absolute"] == pytest.approx(
        float(np.abs(decomposition["local_log_ioi_ratio"][covered]).mean()), abs=1e-6)
    assert entry["local_interval_count"] == int(covered.sum())
    assert float(losses["bar_tempo"]) > 0.0 and float(losses["bar_local"]) > 0.0


def test_gradient_reaches_only_supervised_interval_rows():
    batch = _closed_form_case()
    prediction = batch[0].clone().requires_grad_(True)
    losses, _ = true_bar_objective(prediction, *batch[1:])
    sum(losses.values()).backward()
    gradient = prediction.grad[0].abs()
    assert bool(torch.isfinite(gradient).all())
    assert [int(index) for index in torch.nonzero(gradient > 0).flatten()] == list(range(1, 9))


def test_unselected_rows_and_other_batch_padding_may_be_nan():
    base = _closed_form_case()
    prediction, target, seconds, rows, mask, kinds = base
    chords = seconds.shape[1]
    padded_prediction = torch.full((2, chords), float("nan"), dtype=torch.float32)
    padded_prediction[0] = prediction[0]
    padded_target = torch.zeros(2, chords, dtype=torch.float32)
    padded_target[0] = target[0]
    padded_seconds = torch.zeros(2, chords, dtype=torch.float64)
    padded_seconds[0] = seconds[0]
    padded_rows = torch.zeros(2, rows.shape[1], 2, dtype=torch.int64)
    padded_rows[0] = rows[0]
    padded_mask = torch.zeros(2, rows.shape[1], dtype=torch.bool)
    padded_mask[0] = mask[0]
    padded_kinds = torch.zeros(2, rows.shape[1], dtype=torch.int64)
    padded_kinds[0] = kinds[0]
    reference, metrics = true_bar_objective(*base)
    losses, padded_metrics = true_bar_objective(padded_prediction, padded_target, padded_seconds,
                                                padded_rows, padded_mask, padded_kinds)
    for name in reference:
        assert float(losses[name]) == pytest.approx(float(reference[name]), abs=1e-12)
    assert metrics["valid_span_count"] == padded_metrics["valid_span_count"] == 3


def test_unrequested_horizons_are_excluded_and_counted():
    seconds = [0.0] + [1.0] * 8
    zero = [0.0] * 9
    shifted = [0.0] + [0.5] * 8
    batch = _batch(seconds, shifted, zero, [(0, 4), (4, 8), (0, 8)], [1, 1, 2])
    losses, metrics = true_bar_objective(*batch)
    assert metrics["excluded_unrequested_horizon_count"] == 1
    assert metrics["horizons"]["1"]["span_count"] == 2
    assert float(losses["bar_tempo"]) == pytest.approx(0.5 * 0.25, abs=1e-9)


def test_bar_ioi_mask_filters_local_rows_only():
    seconds = [0.0] + [1.0] * 8
    zero = [0.0] * 9
    prediction = [0.0] + [0.5, 0.5, -0.5, -0.5, 0.2, 0.2, 0.2, 0.2]
    args = _batch(seconds, prediction, zero, [(0, 4), (4, 8)], [1, 1])
    unmasked, full_metrics = true_bar_objective(*args)
    keep = torch.tensor([[False, False, False, False, False, True, True, True, True]], dtype=torch.bool)
    masked, mask_metrics = true_bar_objective(*args, bar_ioi_mask=keep)
    assert float(masked["bar_tempo"]) == pytest.approx(float(unmasked["bar_tempo"]), abs=1e-12)
    assert float(masked["bar_elapsed"]) == pytest.approx(float(unmasked["bar_elapsed"]), abs=1e-12)
    assert float(unmasked["bar_local"]) > 0.0
    assert float(masked["bar_local"]) == pytest.approx(0.0, abs=1e-15)
    assert mask_metrics["local_rows_excluded_by_mask_count"] == 4
    assert mask_metrics["spans_without_local_row_count"] == 1
    assert mask_metrics["supervised_interval_count"] == full_metrics["supervised_interval_count"] == 8
    assert mask_metrics["local_interval_count"] == 4 and full_metrics["local_interval_count"] == 8


def test_prediction_clamp_bounds_overflow():
    seconds = [0.0] + [1.0] * 8
    zero = [0.0] * 9
    batch = _batch(seconds, [0.0] + [10.0] * 8, zero, [(0, 4), (4, 8)], [1, 1])
    losses, metrics = true_bar_objective(*batch)
    # SmoothL1 is linear beyond |x| = 1, so the clamped b of 4.0 costs 4.0 - 0.5.
    assert float(losses["bar_tempo"]) == pytest.approx(3.5, abs=1e-9)
    assert float(losses["bar_local"]) == pytest.approx(5.5, abs=1e-6)
    assert float(losses["bar_elapsed"]) == pytest.approx(math.exp(4.0) - 1.0 - 0.5, abs=1e-6)
    assert metrics["horizons"]["1"]["prediction_clipped_row_count"] == 8


def test_empty_span_set_returns_differentiable_zero():
    batch = _closed_form_case()
    prediction = batch[0].clone().requires_grad_(True)
    empty_rows = torch.zeros(1, 0, 2, dtype=torch.int64)
    empty_mask = torch.zeros(1, 0, dtype=torch.bool)
    empty_kinds = torch.zeros(1, 0, dtype=torch.int64)
    losses, metrics = true_bar_objective(prediction, batch[1], batch[2], empty_rows, empty_mask, empty_kinds)
    assert all(float(value.detach()) == 0.0 for value in losses.values())
    assert metrics["valid_span_count"] == 0 and metrics["supervised_interval_count"] == 0
    assert metrics["active_horizon_count"] == 0 and metrics["horizons"] == {}
    losses["bar_tempo"].backward()
    assert float(prediction.grad.abs().sum()) == 0.0


@pytest.mark.parametrize("position, shape, message", (
    (0, (1, 8), r"must match \[batch, chord\]"),
    (3, (1, 3), r"shaped \[batch, span, 2\]"),
    (4, (1, 4), r"must match span horizons"),
))
def test_mismatched_shapes_are_rejected(position, shape, message):
    with pytest.raises(TrainingRunError, match=message):
        _shape_batch(_closed_form_case(), position, shape)


def test_dtype_and_device_and_range_guards():
    batch = list(_closed_form_case())
    batch[3] = batch[3].float()
    with pytest.raises(TrainingRunError, match="need int64 rows"):
        true_bar_objective(*batch)
    batch = list(_closed_form_case())
    batch[3][0, 0, 1] = 9
    with pytest.raises(TrainingRunError, match="escape the padded window"):
        true_bar_objective(*batch)
    batch = list(_closed_form_case())
    batch[3][0, 0, 1] = 0
    with pytest.raises(TrainingRunError, match="strictly ordered"):
        true_bar_objective(*batch)
    with pytest.raises(BarSupervisionError, match="nonempty and distinct"):
        true_bar_objective(*_closed_form_case(), horizons=(1, 1))


def test_non_finite_or_non_positive_supervised_values_are_rejected():
    for index, value in ((0, float("nan")), (1, float("inf"))):
        batch = list(_closed_form_case())
        batch[index][0, 2] = value
        with pytest.raises(TrainingRunError, match="non-finite trusted bar"):
            true_bar_objective(*batch)
    batch = list(_closed_form_case())
    batch[2][0, 3] = 0.0
    with pytest.raises(TrainingRunError, match="finite and positive on supervised intervals"):
        true_bar_objective(*batch)
    batch = list(_closed_form_case())
    batch[2][0, 3] = float("nan")
    with pytest.raises(TrainingRunError, match="finite and positive on supervised intervals"):
        true_bar_objective(*batch)
