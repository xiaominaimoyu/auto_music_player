"""Score-only bar-aligned window planning; labels, masks, and trust never enter.

Fixed 128-chord cores cannot supervise a 16-bar span: the balanced validation set has
2221 eligible 16-bar spans and only 394 have both endpoints inside one fixed core.
Cores here are cut at recorded bar boundaries so a complete span lives in a single
core.  ``max_window_chords`` bounds the model input with halos included, and the
widest eligible span in that set is 310 chords, so no span is lost to capacity.

Endpoints are exact.  A span that cannot fit a core is reported as excluded, never
truncated.  A bar longer than the core capacity forces a mid-bar cut, which is
reported because it invalidates every span crossing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from training.bar_supervision import BarSpans, BarSupervisionError

DEFAULT_MAX_WINDOW_CHORDS = 512


@dataclass(frozen=True)
class BarWindow:
    """Core interval ``[core_start, core_end)`` plus its score-only halo."""

    start: int
    end: int
    core_start: int
    core_end: int


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise BarSupervisionError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise BarSupervisionError(f"{name} must be a nonnegative integer")
    return int(value)


def bar_boundary_indices(score_reference: Mapping[str, Any]) -> np.ndarray:
    """Return the chord indices that start a bar, reading ``chord_bar_phase`` only."""
    if "chord_bar_phase" not in score_reference:
        raise BarSupervisionError("missing score geometry field: chord_bar_phase")
    phases = np.asarray(score_reference["chord_bar_phase"], dtype=np.float64)
    if phases.ndim != 1 or not len(phases):
        raise BarSupervisionError("bar phases must be a nonempty chord vector")
    if not np.isfinite(phases).all() or np.any(phases < 0) or np.any(phases >= 1):
        raise BarSupervisionError("invalid bar phases")
    result = np.flatnonzero(np.isclose(phases, 0.0, rtol=0.0, atol=1e-9)).astype(np.int64)
    result.flags.writeable = False
    return result


def _validated_boundaries(boundary_indices: Sequence[int], chord_count: int) -> np.ndarray:
    try:
        values = np.asarray(boundary_indices, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BarSupervisionError("invalid bar boundary indices") from exc
    if values.ndim != 1:
        raise BarSupervisionError("bar boundary indices must be a vector")
    if np.any(values < 0) or np.any(values >= chord_count):
        raise BarSupervisionError("bar boundary indices escape the sample")
    if np.any(np.diff(values) <= 0):
        raise BarSupervisionError("bar boundary indices must be strictly increasing")
    return values


def _core_starts(chord_count: int, boundaries: np.ndarray, core_capacity: int) -> tuple[list[int], int]:
    """Cut as late as capacity allows so the fewest spans straddle a core boundary."""
    cuts: list[int] = [0]
    forced = 0
    position = 0
    while position + core_capacity < chord_count:
        limit = position + core_capacity
        candidates = boundaries[(boundaries > position) & (boundaries <= limit)]
        if len(candidates):
            position = int(candidates[-1])
        else:
            position = limit
            forced += 1
        cuts.append(position)
    return cuts, forced


def _validate_span_geometry(spans: BarSpans, boundaries: np.ndarray, chord_count: int) -> None:
    """Reject span sets that did not come from the supplied bar geometry."""
    if spans.chord_count != chord_count:
        raise BarSupervisionError("bar span chord count differs from window planning")
    if np.any(spans.ends <= spans.starts):
        raise BarSupervisionError("bar span endpoints must be strictly ordered")
    allowed = {int(value) for value in boundaries}
    for name, values in (("start", spans.starts), ("end", spans.ends)):
        escaped = sorted({int(value) for value in values if int(value) not in allowed})
        if escaped:
            raise BarSupervisionError(f"{name} index {escaped[:5]} is not a bar boundary")


def plan_bar_windows(chord_count: int, boundary_indices: Sequence[int], spans: BarSpans,
                     context_size: int,
                     max_window_chords: int = DEFAULT_MAX_WINDOW_CHORDS) -> list[BarWindow]:
    """Plan windows whose cores tile the sample once, cut at bar boundaries.

    Every chord belongs to exactly one core, so no chord can receive a second core or
    auxiliary loss.  A halo may overlap a neighbouring core but is never a core.
    """
    count = _positive_integer(chord_count, "chord count")
    context = _nonnegative_integer(context_size, "context size")
    maximum = _positive_integer(max_window_chords, "max window chords")
    capacity = maximum - 2 * context
    if capacity <= 0:
        raise BarSupervisionError("max window chords must exceed twice the context size")
    if not isinstance(spans, BarSpans):
        raise BarSupervisionError("bar window planning requires BarSpans from score geometry")
    boundaries = _validated_boundaries(boundary_indices, count)
    _validate_span_geometry(spans, boundaries, count)
    cuts, _ = _core_starts(count, boundaries, capacity)
    windows: list[BarWindow] = []
    for index, core_start in enumerate(cuts):
        core_end = count if index + 1 == len(cuts) else cuts[index + 1]
        windows.append(BarWindow(max(0, core_start - context), min(count, core_end + context),
                                 core_start, core_end))
    return windows


def _window_cores(windows: Sequence[BarWindow], chord_count: int) -> tuple[np.ndarray, np.ndarray]:
    if not windows:
        raise BarSupervisionError("window plan is empty")
    starts = np.asarray([window.core_start for window in windows], dtype=np.int64)
    ends = np.asarray([window.core_end for window in windows], dtype=np.int64)
    if starts[0] != 0 or ends[-1] != chord_count:
        raise BarSupervisionError("window cores do not cover the sample")
    if np.any(np.diff(starts) <= 0) or not np.array_equal(starts[1:], ends[:-1]):
        raise BarSupervisionError("window cores overlap or leave a chord uncovered")
    for window in windows:
        if not 0 <= window.start <= window.core_start < window.core_end <= window.end <= chord_count:
            raise BarSupervisionError("window halo escapes its own core bounds")
    return starts, ends


def bar_window_coverage(windows: Sequence[BarWindow], spans: BarSpans,
                        boundary_indices: Sequence[int]) -> dict[str, Any]:
    """Report which complete spans survive the plan; never modify spans or masks."""
    boundaries = _validated_boundaries(boundary_indices, spans.chord_count)
    allowed = {int(value) for value in boundaries}
    starts, ends = _window_cores(windows, spans.chord_count)
    start_core = np.searchsorted(starts, spans.starts, side="right") - 1
    end_core = np.searchsorted(starts, spans.ends, side="right") - 1
    contained = start_core == end_core
    outside_bar = np.asarray([index > 0 and int(start) not in allowed
                              for index, start in enumerate(starts)], dtype=np.int64)
    prefix = np.r_[0, np.cumsum(outside_bar, dtype=np.int64)]
    forced_crossing = (prefix[end_core + 1] - prefix[start_core + 1]) > 0
    result: dict[str, Any] = {}
    for horizon in spans.requested_horizons:
        key = str(horizon)
        selected = spans.horizons == horizon
        active = selected & contained
        crossed = selected & ~contained
        intervals = spans.ends[active] - spans.starts[active]
        result[key] = {
            "exact_geometry_window_count": int(selected.sum()),
            "retained_window_count": int(active.sum()),
            "crosses_bar_aligned_cut_window_count": int((crossed & ~forced_crossing).sum()),
            "capacity_forced_split_window_count": int((crossed & forced_crossing).sum()),
            "retained_interval_count": int(intervals.sum()),
            "retained_interval_count_quantiles": dict(zip(("min", "p50", "p95", "max"),
                map(float, np.quantile(intervals, (0, .5, .95, 1))))) if len(intervals) else None,
        }
    core_lengths = ends - starts
    widths = np.asarray([window.end - window.start for window in windows], dtype=np.int64)
    return {
        "chord_count": spans.chord_count, "core_count": len(windows),
        "core_length": {"min": int(core_lengths.min()), "p50": float(np.quantile(core_lengths, .5)),
                        "max": int(core_lengths.max())},
        "window_length": {"min": int(widths.min()), "max": int(widths.max())},
        "capacity_forced_cut_count": int(outside_bar.sum()),
        "horizons": result,
        "core_ownership": "cores tile every chord exactly once; halos never form a second core",
        "exclusion_policy": "spans crossing a core cut are excluded and counted by cut kind, never truncated",
        "model_input": False, "training_changed": False,
    }
