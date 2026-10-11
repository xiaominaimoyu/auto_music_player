from __future__ import annotations

import math

import numpy as np
import pytest

from training.shape_features import SHAPE_FEATURE_NAMES, derive_shape_features, _recover_pitches


def _note_features(rows: list[tuple[float, float, float]]) -> np.ndarray:
    """rows are (pitch, onset_beats, duration_beats); width matches the shipped 15 columns."""
    features = np.zeros((len(rows), 15), dtype=np.float32)
    for index, (pitch, onset, duration) in enumerate(rows):
        features[index, 0] = (pitch - 60.0) / 24.0
        features[index, 3] = onset
        features[index, 4] = math.log1p(duration)
    return features


def test_the_shape_columns_are_the_three_announced_names():
    features = derive_shape_features(_note_features([(60.0, 0.0, 1.0)]), np.array([0.0, 1.0]))
    assert features.shape == (1, len(SHAPE_FEATURE_NAMES))
    assert SHAPE_FEATURE_NAMES == ("same_pitch_previous_span_fraction", "same_pitch_next_span_fraction",
                                   "voices_sounding_at_offset_log1p")


def test_pitches_come_back_exactly_from_the_stored_column():
    features = _note_features([(21.0, 0.0, 1.0), (60.0, 0.0, 1.0), (88.0, 0.0, 1.0), (108.0, 0.0, 1.0)])
    assert _recover_pitches(features).tolist() == [21, 60, 88, 108]


def test_same_pitch_gaps_are_measured_in_fractions_of_the_piece_span():
    # span 4: n0 and n2 share key 60 two beats apart, every other key appears once
    features = _note_features([(60.0, 0.0, 5.0), (62.0, 1.0, 1.0), (60.0, 2.0, 1.0), (64.0, 3.0, 4.0)])
    derived = derive_shape_features(features, np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    previous, following = np.log1p([1.0, 1.0, 0.5, 1.0]), np.log1p([0.5, 1.0, 1.0, 1.0])
    assert derived[:, 0].tolist() == pytest.approx(previous.tolist(), abs=1e-6)
    assert derived[:, 1].tolist() == pytest.approx(following.tolist(), abs=1e-6)


def test_a_key_never_struck_again_reads_one_full_span_rather_than_a_magic_number():
    features = _note_features([(60.0, 0.0, 1.0), (62.0, 1.0, 1.0)])
    derived = derive_shape_features(features, np.array([0.0, 4.0]))
    assert np.allclose(derived[:, :2], np.log1p(1.0), atol=1e-6)


def test_voices_sounding_at_each_note_s_own_offset_exclude_the_note_itself():
    # n0 (0..5) still sounds when n1 ends at 2, when n2 ends at 3, and nothing sounds at n3's end
    features = _note_features([(60.0, 0.0, 5.0), (62.0, 1.0, 1.0), (59.0, 2.0, 1.0), (64.0, 3.0, 4.0)])
    derived = derive_shape_features(features, np.array([0.0, 4.0]))
    assert derived[:, 2].tolist() == pytest.approx([np.log1p(1.0), np.log1p(1.0), np.log1p(1.0), 0.0], abs=1e-6)


def test_adjacent_notes_do_not_look_overlapped_because_of_the_storage_round_trip():
    # a note that ends exactly when the next begins: expm1(float32(log1p(2))) lands a few
    # nanobeats late, so without the tie tolerance this pair would be called overlapping
    features = _note_features([(60.0, 0.0, 2.0), (62.0, 2.0, 2.0)])
    derived = derive_shape_features(features, np.array([0.0, 4.0]))
    assert derived[:, 2].tolist() == pytest.approx([0.0, 0.0], abs=1e-6)
    # a genuine overlap of half a beat is still seen, and only where it is true: n0 ends while
    # n1 is sounding, but at n1's own end nothing else is left
    overlapping = _note_features([(60.0, 0.0, 2.0), (62.0, 1.5, 2.0)])
    assert derive_shape_features(overlapping, np.array([0.0, 4.0]))[:, 2].tolist() == pytest.approx(
        [np.log1p(1.0), 0.0], abs=1e-6)


def test_features_are_deterministic_and_finite_even_for_a_degenerate_score():
    features = _note_features([(60.0, 0.0, 1.0), (60.0, 0.0, 1.0)])
    positions = np.array([0.0, 0.0])
    first = derive_shape_features(features, positions)
    second = derive_shape_features(features, positions)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_non_finite_durations_are_refused_instead_of_propagating_nan():
    features = _note_features([(60.0, 0.0, 1.0)])
    features[0, 4] = np.inf
    with pytest.raises(ValueError, match="not finite"):
        derive_shape_features(features, np.array([0.0, 1.0]))


def test_empty_inputs_are_refused():
    with pytest.raises(ValueError, match="non-empty"):
        derive_shape_features(np.zeros((0, 15), dtype=np.float32), np.array([0.0]))
