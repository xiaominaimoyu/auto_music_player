"""NumPy-only true-bar geometry and label diagnostics, never model inputs.

The MIDI meter grid assumes tick zero starts a bar; pickup/printed measure
numbering is unknown. Exact chord endpoints are required: missing boundaries
are reported, not replaced by interpolated human timing. This module does not
change training, prediction, native masks, or playback adoption criteria.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_BAR_HORIZONS = (1, 4, 16)
SCORE_GEOMETRY_FIELDS = (
    "chord_score_seconds", "chord_score_beats", "chord_bar_index",
    "chord_bar_phase", "partial_bar_indices",
)
NORMALIZED_TARGET_DOMAIN = "score_tempo_normalized_human_reference"


class BarSupervisionError(ValueError):
    """Malformed geometry or supervision must not produce usable diagnostics."""


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise BarSupervisionError(f"{name} must be a positive integer")
    return int(value)


def validate_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    try:
        values = tuple(_positive_integer(value, "bar horizon") for value in horizons)
    except TypeError as exc:
        raise BarSupervisionError("bar horizons must be a sequence") from exc
    if not values or len(set(values)) != len(values):
        raise BarSupervisionError("bar horizons must be nonempty and distinct")
    return values


def _array(value: Any, name: str, *, dtype: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BarSupervisionError(f"invalid {name} array") from exc
    if shape is not None and result.shape != shape:
        raise BarSupervisionError(f"wrong {name} shape")
    if not np.isfinite(result).all():
        raise BarSupervisionError(f"nonfinite {name}")
    return result


def _integers(value: Any, name: str, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    original = np.asarray(value)
    if original.dtype.kind not in "iu" or original.dtype.kind == "b":
        raise BarSupervisionError(f"{name} must contain integers")
    result = _array(value, name, dtype=np.int64, shape=shape)
    if np.any(result < 0):
        raise BarSupervisionError(f"{name} must be nonnegative")
    return result


def _boolean(value: Any, name: str, count: int) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype.kind != "b" or result.shape != (count,):
        raise BarSupervisionError(f"{name} must be a chord-shaped boolean mask")
    return result


def _readonly(value: np.ndarray) -> np.ndarray:
    result = value.copy()
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class BarSpans:
    """Intervals for each window are exactly ``start + 1 ... end`` inclusive."""

    chord_count: int
    starts: np.ndarray
    ends: np.ndarray
    horizons: np.ndarray
    score_ioi_seconds: np.ndarray
    requested_horizons: tuple[int, ...]
    boundary_count: int
    partial_bar_count: int
    missing_end_counts: Mapping[str, int]
    partial_excluded_counts: Mapping[str, int]


def build_bar_spans(score_reference: Mapping[str, Any],
                    horizons: Sequence[int] = DEFAULT_BAR_HORIZONS) -> BarSpans:
    """Build score-only exact-boundary spans; ignore every nonallowlisted field.

    No performance seconds, fitted global scale, labels, masks, or trust flags
    are accessed. Partial meter-change bars invalidate windows containing
    them. A missing endpoint remains unavailable, including silence at a
    boundary and the terminal bar when no following chord is recorded.
    """
    selected = validate_horizons(horizons)
    missing = [name for name in SCORE_GEOMETRY_FIELDS if name not in score_reference]
    if missing:
        raise BarSupervisionError(f"missing score geometry fields: {', '.join(missing)}")
    seconds = _array(score_reference["chord_score_seconds"], "score seconds", dtype=np.float64)
    if seconds.ndim != 1 or not len(seconds):
        raise BarSupervisionError("score seconds must be a nonempty chord vector")
    count = len(seconds)
    beats = _array(score_reference["chord_score_beats"], "score beats", dtype=np.float64, shape=(count,))
    bars = _integers(score_reference["chord_bar_index"], "bar indices", shape=(count,))
    phases = _array(score_reference["chord_bar_phase"], "bar phases", dtype=np.float64, shape=(count,))
    partial = _integers(score_reference["partial_bar_indices"], "partial bar indices")
    if partial.ndim != 1 or len(np.unique(partial)) != len(partial):
        raise BarSupervisionError("partial bar indices must be a unique vector")
    if np.any(seconds < 0) or np.any(beats < 0) or np.any(np.diff(seconds) <= 0) or np.any(np.diff(beats) <= 0):
        raise BarSupervisionError("score seconds/beats must be nonnegative and strictly increasing")
    if np.any(phases < 0) or np.any(phases >= 1) or np.any(np.diff(bars) < 0):
        raise BarSupervisionError("invalid ordered bar coordinates")
    same_bar = np.diff(bars) == 0
    if np.any(np.diff(phases)[same_bar] <= 0):
        raise BarSupervisionError("bar phases must increase inside each bar")
    boundary_indices = np.flatnonzero(np.isclose(phases, 0.0, rtol=0, atol=1e-9))
    boundary = {int(bars[index]): int(index) for index in boundary_indices}
    if len(boundary) != len(boundary_indices):
        raise BarSupervisionError("duplicate chord endpoint for one bar boundary")
    partial_set = set(map(int, partial))
    starts, ends, sizes = [], [], []
    missing_counts, partial_counts = {}, {}
    for horizon in selected:
        key = str(horizon)
        missing_counts[key] = partial_counts[key] = 0
        for bar, start in sorted(boundary.items()):
            end = boundary.get(bar + horizon)
            if end is None:
                missing_counts[key] += 1
                continue
            if any(value in partial_set for value in range(bar, bar + horizon)):
                partial_counts[key] += 1
                continue
            if end <= start:
                raise BarSupervisionError("bar endpoint order is inconsistent")
            starts.append(start)
            ends.append(end)
            sizes.append(horizon)
    return BarSpans(
        count, _readonly(np.asarray(starts, dtype=np.int64)),
        _readonly(np.asarray(ends, dtype=np.int64)), _readonly(np.asarray(sizes, dtype=np.int64)),
        _readonly(np.diff(seconds, prepend=seconds[0])), selected, len(boundary), len(partial),
        missing_counts, partial_counts,
    )


def supervised_span_eligibility(spans: BarSpans, native_ioi_mask: Any,
                                start_trusted: Any) -> dict[str, np.ndarray]:
    """Native eligibility is supervision-only and separate from score geometry.

    The two rejection masks are disjoint: an untrusted start takes precedence
    over invalid intervals. No native mask is rewritten or inferred.
    """
    native = _boolean(native_ioi_mask, "native IOI mask", spans.chord_count)
    trusted = _boolean(start_trusted, "start trusted mask", spans.chord_count)
    untrusted = ~trusted[spans.starts]
    invalid_prefix = np.r_[0, np.cumsum(~native, dtype=np.int64)]
    invalid = (invalid_prefix[spans.ends + 1] - invalid_prefix[spans.starts + 1]) > 0
    return {"eligible": ~(untrusted | invalid), "untrusted_start": untrusted,
            "invalid_ioi_after_trusted_start": invalid & ~untrusted}


def span_coverage(spans: BarSpans, native_ioi_mask: Any, start_trusted: Any, *,
                  core_size: int = 128, max_chords: int | None = None) -> dict[str, Any]:
    """Report exact endpoint ownership, optional capacity exclusion and gaps.

    ``max_chords`` counts both endpoints (``end - start + 1``), not IOIs.
    A core includes chord indices ``[k*size, (k+1)*size)``; merely fitting the
    interval length is insufficient when the window crosses a core boundary.
    """
    core = _positive_integer(core_size, "core size")
    maximum = None if max_chords is None else _positive_integer(max_chords, "max chords")
    eligibility = supervised_span_eligibility(spans, native_ioi_mask, start_trusted)
    eligible = eligibility["eligible"]
    capacity = np.ones(len(eligible), dtype=bool) if maximum is None else spans.ends - spans.starts + 1 <= maximum
    same_core = spans.starts // core == spans.ends // core
    result = {}
    for horizon in spans.requested_horizons:
        key = str(horizon)
        selected = spans.horizons == horizon
        active = selected & eligible
        retained = active & capacity
        widths = spans.ends[active] - spans.starts[active]
        result[key] = {
            "exact_geometry_window_count": int(selected.sum()),
            "missing_end_boundary_count": spans.missing_end_counts[key],
            "partial_bar_excluded_window_count": spans.partial_excluded_counts[key],
            "untrusted_start_count": int((selected & eligibility["untrusted_start"]).sum()),
            "invalid_ioi_after_trusted_start_count": int((selected & eligibility["invalid_ioi_after_trusted_start"]).sum()),
            "eligible_window_count": int(active.sum()),
            "max_chords_excluded_window_count": int((active & ~capacity).sum()),
            "capacity_retained_window_count": int(retained.sum()),
            "same_core_window_count": int((retained & same_core).sum()),
            "cross_core_window_count": int((retained & ~same_core).sum()),
            "eligible_interval_count_quantiles": dict(zip(("min", "p50", "p95", "max"),
                map(float, np.quantile(widths, (0, .5, .95, 1))))) if len(widths) else None,
        }
    return {"chord_count": spans.chord_count, "boundary_count": spans.boundary_count,
            "partial_bar_count": spans.partial_bar_count, "core_size": core, "max_chords": maximum,
            "max_chords_policy": "both chord endpoints counted; exclusions never imputed",
            "pickup_status": "unknown; MIDI tick zero assumed meter-grid bar start",
            "horizons": result}


def _reconstruct_normalized_ioi(spans: BarSpans, log_ratio: Any, mask: np.ndarray) -> np.ndarray:
    ratio = _array(log_ratio, "log IOI ratio", dtype=np.float64, shape=(spans.chord_count,))
    result = np.zeros(spans.chord_count, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result[mask] = spans.score_ioi_seconds[mask] * np.exp(ratio[mask])
    if not np.isfinite(result[mask]).all() or np.any(result[mask] <= 0):
        raise BarSupervisionError("normalized IOIs must be finite and positive on trusted intervals")
    return result


def decompose_bar_targets(spans: BarSpans, native_ioi_mask: Any, start_trusted: Any,
                          target_log_ioi: Any) -> dict[str, Any]:
    """Diagnostic tempo/local decomposition on complete eligible one-bar spans.

    For each bar, b=log(sum(s*exp(r))/sum(s)), local=r-b. Consequently
    sum(s*exp(local))=sum(s). This removes the two-head constant-offset
    ambiguity but is not a fitted slow curve, a model, or a training target
    installation. NaN local values explicitly denote unavailable intervals.
    """
    if 1 not in spans.requested_horizons:
        raise BarSupervisionError("one-bar geometry is required for target decomposition")
    native = _boolean(native_ioi_mask, "native IOI mask", spans.chord_count)
    target = _array(target_log_ioi, "target log IOI", dtype=np.float64, shape=(spans.chord_count,))
    eligible = supervised_span_eligibility(spans, native, start_trusted)["eligible"] & (spans.horizons == 1)
    intervals = np.zeros(spans.chord_count, dtype=bool)
    local = np.full(spans.chord_count, np.nan, dtype=np.float64)
    starts, ends = spans.starts[eligible], spans.ends[eligible]
    tempo, normalized_error, reconstruction_error = [], [], []
    reconstructed = _reconstruct_normalized_ioi(spans, target, native & (np.arange(spans.chord_count) > 0))
    for start, end in zip(starts, ends):
        index = slice(int(start) + 1, int(end) + 1)
        if intervals[index].any():
            raise BarSupervisionError("complete one-bar spans must not overlap in IOI ownership")
        score = spans.score_ioi_seconds[index]
        expected = reconstructed[index]
        bar_tempo = float(np.log(expected.sum() / score.sum()))
        residual = target[index] - bar_tempo
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            normalized = score * np.exp(residual)
            recovered = score * np.exp(bar_tempo + residual)
        if not np.isfinite(normalized).all() or not np.isfinite(recovered).all():
            raise BarSupervisionError("bar decomposition overflow")
        local[index] = residual
        intervals[index] = True
        tempo.append(bar_tempo)
        normalized_error.append(abs(float(normalized.sum() / score.sum() - 1)))
        reconstruction_error.append(float(np.max(np.abs(recovered - expected))))
    return {"bar_starts": starts.copy(), "bar_ends": ends.copy(),
            "bar_tempo_log_ratio": np.asarray(tempo), "local_log_ioi_ratio": local,
            "local_available": intervals, "bar_count": len(starts),
            "covered_interval_count": int(intervals.sum()),
            "normalization_max_relative_error": max(normalized_error) if normalized_error else None,
            "reconstruction_max_error_seconds": max(reconstruction_error) if reconstruction_error else None,
            "diagnostic_only": True, "model_input": False, "slow_curve_fitted": False}


def evaluate_normalized_timing(spans: BarSpans, prediction_log_ioi: Any, target_log_ioi: Any,
                               native_ioi_mask: Any, start_trusted: Any, *,
                               human_ioi_seconds: Any | None = None,
                               global_tempo_scale: float | None = None) -> dict[str, Any]:
    """Evaluate s*exp(pred) against normalized native labels, diagnostic only.

    Optional human IOIs plus global scale cross-check target reconstruction
    against human/g. They do not calibrate predictions. This new, explicit
    domain cannot retroactively replace old fixed-score adoption results.
    """
    native = _boolean(native_ioi_mask, "native IOI mask", spans.chord_count).copy()
    native[0] = False
    trusted = _boolean(start_trusted, "start trusted mask", spans.chord_count)
    predicted = _reconstruct_normalized_ioi(spans, prediction_log_ioi, native)
    target = _reconstruct_normalized_ioi(spans, target_log_ioi, native)
    if (human_ioi_seconds is None) != (global_tempo_scale is None):
        raise BarSupervisionError("human IOIs and global tempo scale must be provided together")
    roundtrip = None
    if human_ioi_seconds is not None:
        human = _array(human_ioi_seconds, "human IOI seconds", dtype=np.float64, shape=(spans.chord_count,))
        scale_array = _array(global_tempo_scale, "global tempo scale", dtype=np.float64)
        if scale_array.shape != () or float(scale_array) <= 0 or np.any(human[native] <= 0):
            raise BarSupervisionError("invalid positive human IOIs/global tempo scale")
        difference = np.abs(target[native] - human[native] / float(scale_array))
        roundtrip = float(difference.max()) if len(difference) else None
    error = predicted - target
    eligible = supervised_span_eligibility(spans, native, trusted)["eligible"]
    cumulative = np.r_[0.0, np.cumsum(error)]
    drift = {}
    for horizon in spans.requested_horizons:
        selected = eligible & (spans.horizons == horizon)
        values = np.abs(cumulative[spans.ends[selected] + 1] - cumulative[spans.starts[selected] + 1]) * 1000
        drift[str(horizon)] = {"window_count": len(values),
            "mae_ms": float(values.mean()) if len(values) else None,
            "p95_ms": float(np.quantile(values, .95)) if len(values) else None}
    absolute = np.abs(error[native]) * 1000
    return {"domain": NORMALIZED_TARGET_DOMAIN,
            "prediction_policy": "score_IOI * exp(pred); no performer calibration of predictions",
            "reference_policy": "score_IOI * exp(native_target) equals human_IOI/global_scale",
            "uses_performance_derived_reference": True, "uses_prediction_performance_calibration": False,
            "diagnostic_only": True, "deployment_evidence": False, "adoption_gate_changed": False,
            "valid_ioi_count": int(native.sum()),
            "ioi_mae_ms": float(absolute.mean()) if len(absolute) else None,
            "native_target_roundtrip_max_error_seconds": roundtrip, "bar_drift": drift}
