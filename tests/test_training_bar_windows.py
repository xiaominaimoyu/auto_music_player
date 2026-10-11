from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

from training.bar_supervision import BarSupervisionError, build_bar_spans, span_coverage
from training.bar_windows import (
    DEFAULT_MAX_WINDOW_CHORDS, BarWindow, bar_boundary_indices, bar_window_coverage, plan_bar_windows,
)


def _reference(bar_sizes, partial=()):
    seconds, beats, bars, phases = [], [], [], []
    for index, size in enumerate(bar_sizes):
        for position in range(size):
            bars.append(index)
            phases.append(position / size)
            beats.append(4 * index + 4 * position / size)
            seconds.append((4 * index + 4 * position / size) * 0.5)
    return {"chord_score_seconds": np.asarray(seconds), "chord_score_beats": np.asarray(beats),
            "chord_bar_index": np.asarray(bars, dtype=np.int64), "chord_bar_phase": np.asarray(phases),
            "partial_bar_indices": np.asarray(list(partial), dtype=np.int64)}


def _plan(bar_sizes, *, partial=(), context=16, maximum=DEFAULT_MAX_WINDOW_CHORDS, horizons=(1, 4, 16)):
    reference = _reference(bar_sizes, partial)
    spans = build_bar_spans(reference, horizons)
    boundaries = bar_boundary_indices(reference)
    count = int(reference["chord_score_seconds"].size)
    windows = plan_bar_windows(count, boundaries, spans, context, maximum)
    return reference, spans, boundaries, windows


def _owns_each_chord_once(windows, chord_count):
    owners = np.zeros(chord_count, dtype=np.int64)
    for window in windows:
        owners[window.core_start:window.core_end] += 1
    assert np.all(owners == 1)


def test_cores_tile_every_chord_exactly_once_and_fit_capacity():
    reference, spans, boundaries, windows = _plan([8] * 400)
    count = int(reference["chord_score_seconds"].size)
    _owns_each_chord_once(windows, count)
    capacity = DEFAULT_MAX_WINDOW_CHORDS - 2 * 16
    assert [window.core_end - window.core_start for window in windows] == [capacity] * 6 + [count - capacity * 6]
    assert all(window.end - window.start <= DEFAULT_MAX_WINDOW_CHORDS for window in windows)
    assert sorted(windows, key=lambda window: window.core_start) == windows


def test_interior_windows_reach_the_declared_input_limit_exactly():
    _, _, _, windows = _plan([8] * 400)
    interior = windows[1:-1]
    assert {window.end - window.start for window in interior} == {DEFAULT_MAX_WINDOW_CHORDS}
    assert windows[0].start == 0 and windows[0].end == DEFAULT_MAX_WINDOW_CHORDS - 16


def test_bar_aligned_cores_supervise_16_bar_spans_that_fixed_cores_cut():
    reference, spans, boundaries, windows = _plan([8] * 400)
    count = int(reference["chord_score_seconds"].size)
    trusted = np.ones(count, dtype=bool)
    native = np.ones(count, dtype=bool)
    fixed = span_coverage(spans, native, trusted, core_size=128)["horizons"]["16"]
    planned = bar_window_coverage(windows, spans, boundaries)["horizons"]["16"]
    assert fixed["same_core_window_count"] == 0
    assert planned["retained_window_count"] == 288
    assert planned["retained_window_count"] + planned["crosses_bar_aligned_cut_window_count"] \
        == fixed["exact_geometry_window_count"]


def test_single_bar_sample_is_one_core():
    reference, spans, boundaries, windows = _plan([7], horizons=(1,))
    assert windows == [BarWindow(0, 7, 0, 7)]
    coverage = bar_window_coverage(windows, spans, boundaries)["horizons"]["1"]
    assert coverage["exact_geometry_window_count"] == 0 and coverage["retained_window_count"] == 0


def test_oversized_bar_forces_capacity_cut_and_only_crossing_spans_are_lost():
    reference, spans, boundaries, windows = _plan([600, 8, 8, 8], horizons=(1,))
    report = bar_window_coverage(windows, spans, boundaries)
    assert [window.core_end - window.core_start for window in windows] == [480, 144]
    assert report["capacity_forced_cut_count"] == 1
    horizon = report["horizons"]["1"]
    assert horizon["exact_geometry_window_count"] == 3
    assert horizon["capacity_forced_split_window_count"] == 1
    assert horizon["retained_window_count"] == 2
    assert horizon["crosses_bar_aligned_cut_window_count"] == 0


def test_span_touching_the_next_core_is_excluded_but_starting_a_core_is_retained():
    reference, spans, boundaries, windows = _plan([8] * 200, horizons=(1,))
    pairs = list(zip(spans.starts.tolist(), spans.ends.tolist()))
    assert (472, 480) in pairs and (480, 488) in pairs

    def naive_contained(start: int, end: int) -> bool:
        return any(window.core_start <= start and end < window.core_end for window in windows)

    expected = {str(horizon): sum(naive_contained(int(start), int(end)) for start, end, size in
                                  zip(spans.starts, spans.ends, spans.horizons) if int(size) == horizon)
                for horizon in spans.requested_horizons}
    coverage = bar_window_coverage(windows, spans, boundaries)["horizons"]["1"]
    assert len(pairs) == 199
    assert coverage["retained_window_count"] == expected["1"] == 196
    assert coverage["crosses_bar_aligned_cut_window_count"] == 3
    assert coverage["capacity_forced_split_window_count"] == 0
    assert not naive_contained(472, 480) and naive_contained(480, 488)


def test_partial_bars_never_enter_the_supervised_set():
    _, spans, boundaries, windows = _plan([8] * 10, partial=(1,), horizons=(1, 4))
    report = bar_window_coverage(windows, spans, boundaries)
    assert boundaries.tolist() == list(range(0, 80, 8))
    assert spans.boundary_count == 10 and spans.partial_bar_count == 1
    assert report["horizons"]["1"]["exact_geometry_window_count"] == 8
    for horizon in ("1", "4"):
        assert report["horizons"][horizon]["retained_window_count"] \
            == report["horizons"][horizon]["exact_geometry_window_count"]
    assert 8 not in spans.starts.tolist() and 72 not in spans.starts.tolist()


def test_missing_end_boundary_is_reported_and_never_imputed():
    _, spans, boundaries, windows = _plan([8] * 10, horizons=(1,))
    report = bar_window_coverage(windows, spans, boundaries)["horizons"]["1"]
    assert spans.missing_end_counts["1"] == 1
    assert report["exact_geometry_window_count"] == 9 and report["retained_window_count"] == 9
    assert spans.ends.max() == 72


def test_halo_may_overlap_a_neighbour_core_but_is_never_a_core():
    reference, spans, boundaries, windows = _plan([8] * 200, context=16)
    count = int(reference["chord_score_seconds"].size)
    for index, window in enumerate(windows):
        expected_start = max(0, window.core_start - 16)
        expected_end = min(count, window.core_end + 16)
        assert (window.start, window.end) == (expected_start, expected_end)
        if index:
            assert window.start < windows[index - 1].core_end <= window.core_start
    _owns_each_chord_once(windows, count)


def test_planning_is_insensitive_to_label_bearing_seconds():
    reference, spans, boundaries, windows = _plan([9] * 60, horizons=(1, 4, 16))
    scrambled = replace(spans, score_ioi_seconds=np.flip(spans.score_ioi_seconds).copy())
    assert plan_bar_windows(spans.chord_count, boundaries, scrambled, 16) == windows
    names = set(inspect.signature(plan_bar_windows).parameters)
    assert names == {"chord_count", "boundary_indices", "spans", "context_size", "max_window_chords"}
    assert not names & {"chord_masks", "chord_trusted", "targets", "native_ioi_mask", "start_trusted"}


def test_boundary_indices_reuse_the_geometry_that_produced_the_spans():
    reference, spans, boundaries, _ = _plan([5, 9, 4, 12], horizons=(1, 4))
    assert boundaries.tolist() == [0, 5, 14, 18]
    assert not boundaries.flags.writeable
    assert set(spans.starts.tolist()) | set(spans.ends.tolist()) <= set(boundaries.tolist())
    with pytest.raises(BarSupervisionError, match="not a bar boundary"):
        plan_bar_windows(int(reference["chord_score_seconds"].size), np.asarray([0, 5, 9, 14]), spans, 16)


@pytest.mark.parametrize(("count", "boundaries", "context", "maximum", "message"), (
    (0, [0], 16, 512, "chord count must be a positive integer"),
    (64, [0], -1, 512, "context size must be a nonnegative integer"),
    (64, [0], 32, 64, "must exceed twice the context size"),
    (64, [8, 4], 16, 512, "strictly increasing"),
    (64, [0, 64], 16, 512, "escape the sample"),
))
def test_invalid_planning_inputs_are_rejected(count, boundaries, context, maximum, message):
    reference = _reference([8] * 8)
    spans = build_bar_spans(reference, (1,))
    with pytest.raises(BarSupervisionError, match=message):
        plan_bar_windows(count, np.asarray(boundaries, dtype=np.int64), spans, context, maximum)


def test_coverage_rejects_windows_that_do_not_tile_the_sample():
    reference, spans, boundaries, windows = _plan([8] * 100, horizons=(1,))
    assert len(windows) == 2
    broken = [windows[0], replace(windows[1], core_start=windows[1].core_start + 1)]
    with pytest.raises(BarSupervisionError, match="overlap or leave a chord uncovered"):
        bar_window_coverage(broken, spans, boundaries)
    with pytest.raises(BarSupervisionError, match="do not cover the sample"):
        bar_window_coverage([windows[0]], spans, boundaries)
    with pytest.raises(BarSupervisionError, match="halo escapes"):
        bar_window_coverage([replace(windows[0], end=801), windows[1]], spans, boundaries)
