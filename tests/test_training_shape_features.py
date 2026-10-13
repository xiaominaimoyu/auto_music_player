from __future__ import annotations

import math

import numpy as np
import pytest

from training.shape_features import SHAPE_FEATURE_NAMES, derive_shape_features


def _note_features(rows: list[tuple[float, float, float]]) -> np.ndarray:
    """rows are (pitch, onset_beats, duration_beats); width matches the shipped 15 columns."""
    features = np.zeros((len(rows), 15), dtype=np.float32)
    for index, (pitch, onset, duration) in enumerate(rows):
        features[index, 0] = (pitch - 60.0) / 24.0
        features[index, 3] = onset
        features[index, 4] = math.log1p(duration)
    return features


# three notes, one chord each, positions 0..3 so the span is 3 beats:
#   n0 key 60 @0 length 2 -> ends 2      n1 key 62 @1 length 3 -> ends 4      n2 key 60 @2 length 1 -> ends 3
ROWS = [(60.0, 0.0, 2.0), (62.0, 1.0, 3.0), (60.0, 2.0, 1.0)]
POSITIONS = np.array([0.0, 1.0, 2.0, 3.0])
MAPPING = np.array([0, 1, 2], dtype=np.int64)


def _block(rows=ROWS, positions=POSITIONS, mapping=MAPPING, names=SHAPE_FEATURE_NAMES) -> np.ndarray:
    return derive_shape_features(_note_features(rows), positions, mapping, names)


def test_the_shipped_block_is_six_named_columns_in_order():
    assert SHAPE_FEATURE_NAMES == ("duration_versus_local_mean_log", "voices_sounding_at_offset_log1p",
                                  "pitch_repeats_previous_chord", "concurrent_notes_at_onset_log1p",
                                  "same_pitch_previous_span_fraction", "same_pitch_next_span_fraction")
    block = _block()
    assert block.shape == (3, len(SHAPE_FEATURE_NAMES))
    assert block.dtype == np.float32
    assert np.isfinite(block).all()


def test_duration_versus_local_mean_compares_a_note_to_its_neighbourhood():
    # nine-note window over durations [2,2,2,2,2,3,1,1,1] etc -> local means 16/9, 15/9, 14/9
    expected = [math.log(2 / (16 / 9)), math.log(3 / (15 / 9)), math.log(1 / (14 / 9))]
    assert _block()[:, 0].tolist() == pytest.approx(expected, abs=1e-5)


def test_voices_sounding_at_each_offset_never_goes_negative():
    # at n0's end n1 sounds (1); at n1's end nothing does (0); at n2's end n1 still does (1)
    assert _block()[:, 1].tolist() == pytest.approx([math.log1p(1), 0.0, math.log1p(1)], abs=1e-5)


def test_a_note_that_ends_exactly_when_another_begins_is_not_called_overlapping():
    # n0 runs 0..2 and n1 starts at 2.0: sequential, not concurrent, whatever the float32 round
    # trip does to the recovered duration
    rows = [(60.0, 0.0, 2.0), (62.0, 2.0, 2.0)]
    block = _block(rows, np.array([0.0, 4.0]), np.array([0, 1], dtype=np.int64))
    concurrent = block[:, SHAPE_FEATURE_NAMES.index("concurrent_notes_at_onset_log1p")]
    assert concurrent.tolist() == pytest.approx([0.0, 0.0], abs=1e-6)


def test_concurrent_voices_count_only_notes_already_started_and_not_yet_ended():
    # a bass held from 0 to 8, an upper voice from 1 to 3 and another from 2 to 4: when the
    # third note begins, two others are still sounding, and none of them is the note itself
    rows = [(48.0, 0.0, 8.0), (60.0, 1.0, 2.0), (64.0, 2.0, 2.0)]
    mapping = np.array([0, 1, 2], dtype=np.int64)
    block = _block(rows, np.array([0.0, 8.0]), mapping)
    concurrent = block[:, SHAPE_FEATURE_NAMES.index("concurrent_notes_at_onset_log1p")]
    assert concurrent.tolist() == pytest.approx([0.0, math.log1p(1), math.log1p(2)], abs=1e-6)
    sounding = block[:, SHAPE_FEATURE_NAMES.index("voices_sounding_at_offset_log1p")]
    assert sounding.tolist() == pytest.approx([0.0, math.log1p(2), math.log1p(1)], abs=1e-6)


def test_pitch_repeats_previous_chord_sees_only_the_chord_immediately_before():
    # key 60 returns at chord 2 but chord 1 struck key 62 alone, so it is not a repeat of the
    # preceding chord -- the column is about voice writing, not about any earlier occurrence
    rows = [(60.0, 0.0, 1.0), (62.0, 1.0, 1.0), (60.0, 2.0, 1.0)]
    assert _block(rows, np.array([0.0, 3.0]), MAPPING)[:, 2].tolist() == pytest.approx([0.0, 0.0, 0.0])
    # the same writing where the returning note reuses the previous chord's key does flag
    rows = [(60.0, 0.0, 1.0), (62.0, 1.0, 1.0), (62.0, 2.0, 1.0)]
    assert _block(rows, np.array([0.0, 3.0]), MAPPING)[:, 2].tolist() == pytest.approx([0.0, 0.0, 1.0])


def test_notes_starting_together_share_a_chord_and_are_not_marked_as_returns():
    rows = [(60.0, 0.0, 1.0), (64.0, 0.0, 1.0), (60.0, 1.0, 1.0)]
    block = _block(rows, np.array([0.0, 1.0]), np.array([0, 0, 1], dtype=np.int64))
    assert block[:, 2].tolist() == pytest.approx([0.0, 0.0, 1.0])


def test_same_pitch_gaps_are_span_normalised_and_absence_reads_as_one_full_span():
    block = _block()
    previous = block[:, SHAPE_FEATURE_NAMES.index("same_pitch_previous_span_fraction")]
    following = block[:, SHAPE_FEATURE_NAMES.index("same_pitch_next_span_fraction")]
    assert previous.tolist() == pytest.approx([math.log1p(1.0), math.log1p(1.0), math.log1p(2 / 3)], abs=1e-5)
    assert following.tolist() == pytest.approx([math.log1p(2 / 3), math.log1p(1.0), math.log1p(1.0)], abs=1e-5)


def test_a_chord_repeated_by_different_hands_counts_once_per_note():
    # both notes of chord 1 return keys from chord 0, two beats apart, over a four-beat span
    rows = [(60.0, 0.0, 2.0), (64.0, 0.0, 2.0), (60.0, 2.0, 1.0), (64.0, 2.0, 1.0)]
    mapping = np.array([0, 0, 1, 1], dtype=np.int64)
    block = _block(rows, np.array([0.0, 4.0]), mapping)
    assert block[:, 2].tolist() == pytest.approx([0.0, 0.0, 1.0, 1.0])
    gaps = block[:, SHAPE_FEATURE_NAMES.index("same_pitch_previous_span_fraction")]
    assert np.allclose(gaps[2:], math.log1p(2 / 4), atol=1e-5)


def test_selecting_a_subset_keeps_the_requested_order():
    names = ("concurrent_notes_at_onset_log1p", "duration_versus_local_mean_log")
    block = _block(names=names)
    full = _block()
    assert np.array_equal(block[:, 0], full[:, SHAPE_FEATURE_NAMES.index("concurrent_notes_at_onset_log1p")])
    assert np.array_equal(block[:, 1], full[:, 0])


def test_an_unknown_column_name_is_refused_rather_than_silently_dropped():
    with pytest.raises(ValueError, match="unknown shape features"):
        _block(names=("rest_after_log1p",))


def test_geometry_the_only_source_of_the_block():
    # identical geometry must give identical numbers no matter what the labels say
    rows = ROWS
    first = _block(rows)
    second = _block(rows)
    assert np.array_equal(first, second)


def test_non_finite_durations_and_empty_inputs_are_refused():
    features = _note_features([(60.0, 0.0, 1.0)])
    features[0, 4] = np.inf
    with pytest.raises(ValueError, match="not finite"):
        derive_shape_features(features, np.array([0.0, 1.0]), np.array([0], dtype=np.int64))
    with pytest.raises(ValueError, match="non-empty"):
        derive_shape_features(np.zeros((0, 15), dtype=np.float32), np.array([0.0]),
                              np.array([], dtype=np.int64))
    with pytest.raises(ValueError, match="at least 5 columns"):
        derive_shape_features(np.zeros((2, 4), dtype=np.float32), np.array([0.0, 1.0]),
                              np.array([0, 1], dtype=np.int64))
