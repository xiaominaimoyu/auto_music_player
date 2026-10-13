from __future__ import annotations

import math

import numpy as np
import pytest

from core.humanize import HumanizeParams
from training.rules_baseline import (RuleTimings, UNPARSEABLE_NOTE_ID, duration_log_ratio_predictions,
                                     element_gaps, ioi_log_ratio_predictions, leap_rule_coverage,
                                     onset_predictions, rule_timings)

ZERO_JITTER = HumanizeParams(jitter_ms=0.0, jitter_correlation=0.0)


def test_an_element_is_the_interval_until_the_next_one_arrives():
    # onsets at 0, 1, 3 beats: the first two elements occupy 1 and 2 beats, and the last keeps
    # its longest written voice because nothing follows it
    onsets = np.array([0.0, 1.0, 3.0])
    notes = np.array([0.5, 1.0, 2.0])
    mapping = np.array([0, 1, 2], dtype=np.int64)
    assert element_gaps(onsets, notes, mapping).tolist() == [1.0, 2.0, 2.0]


def test_element_gaps_refuse_an_escaping_mapping_or_a_repeated_onset():
    with pytest.raises(ValueError, match="escapes"):
        element_gaps(np.array([0.0, 1.0]), np.array([1.0]), np.array([4], dtype=np.int64))
    with pytest.raises(ValueError, match="strictly increasing"):
        element_gaps(np.array([0.0, 0.5, 0.5]), np.array([1.0, 1.0, 1.0]), np.array([0, 1, 2], dtype=np.int64))
    with pytest.raises(ValueError, match="non-empty"):
        element_gaps(np.array([]), np.array([1.0]), np.array([0], dtype=np.int64))


def test_hold_tiers_follow_the_written_length_of_the_element():
    # 0.25 beats is short, 1.0 is mid, 2.0 reaches the long tier's threshold exactly
    beats = np.array([0.25, 1.0, 2.0])
    seconds = np.array([0.0, 0.5, 1.0])
    timings = rule_timings(seconds, beats, seconds_per_beat=0.5, params=HumanizeParams(), sessions=6, seed=7)
    for session in range(6):
        hold = timings.hold[session]
        assert 0.58 <= hold[0] <= 0.70
        assert 0.70 <= hold[1] <= 0.85
        assert 0.86 <= hold[2] <= 0.98


def test_a_different_session_seed_gives_a_different_realization_and_a_constant_one_does_not():
    beats = np.array([1.0, 1.0, 1.0])
    seconds = np.array([0.0, 0.5, 1.0])
    first = rule_timings(seconds, beats, seconds_per_beat=0.5, sessions=3, seed=11)
    second = rule_timings(seconds, beats, seconds_per_beat=0.5, sessions=3, seed=11)
    assert np.array_equal(first.offset_ms, second.offset_ms)
    assert np.array_equal(first.hold, second.hold)
    assert not np.allclose(first.hold[0], first.hold[1])
    still = rule_timings(seconds, beats, seconds_per_beat=0.5, params=ZERO_JITTER, sessions=3, seed=5)
    assert np.allclose(still.offset_ms, 0.0)


def test_the_breath_rule_survives_zero_jitter_and_fires_after_a_long_element():
    # element 0 is written two beats long, so element 1 starts a new phrase and gets 25 ms
    # times U(0.6, 1.0); with the jitter switched off that pause is the whole prediction
    beats = np.array([2.0, 1.0, 1.0])
    seconds = np.array([0.0, 4.0, 5.0])
    timings = rule_timings(seconds, beats, seconds_per_beat=1.0, params=ZERO_JITTER, sessions=24, seed=3)
    assert np.allclose(timings.offset_ms[:, 0], 0.0)
    breath = timings.offset_ms[:, 1]
    assert np.all(breath >= 25.0 * 0.6 - 1e-9) and np.all(breath <= 25.0 * 1.0 + 1e-9)
    assert float(breath.mean()) == pytest.approx(25.0 * 0.8, abs=1.0)
    assert np.allclose(timings.offset_ms[:, 2], 0.0)


def test_the_chained_anti_overlap_bound_delays_an_onset_a_held_element_would_run_over():
    # element 0 is written two beats long at one second per beat, and the long tier holds at
    # least 0.86 of that, so it cannot release before 1720 ms -- while the next score onset is
    # only 500 ms away.  The player could never emit the raw offset, so the bound moves it.
    beats = np.array([2.0, 1.0])
    seconds = np.array([0.0, 0.5])
    timings = rule_timings(seconds, beats, seconds_per_beat=1.0, params=ZERO_JITTER, sessions=4, seed=9)
    assert np.all(timings.offset_ms[:, 1] >= 1720.0 + 1.5 - 500.0 - 1e-6)


def test_rule_timings_refuse_an_impossible_timeline():
    with pytest.raises(ValueError, match="non-decreasing"):
        rule_timings(np.array([1.0, 0.5]), np.array([1.0, 1.0]), seconds_per_beat=0.5)
    with pytest.raises(ValueError, match="positive"):
        rule_timings(np.array([0.0, 1.0]), np.array([1.0, 0.0]), seconds_per_beat=0.5)
    with pytest.raises(ValueError, match="finite and positive"):
        rule_timings(np.array([0.0]), np.array([1.0]), seconds_per_beat=0.0)
    with pytest.raises(ValueError, match="at least 1"):
        rule_timings(np.array([0.0]), np.array([1.0]), seconds_per_beat=0.5, sessions=0)


def test_a_constant_shift_of_the_whole_timeline_leaves_every_interval_untouched():
    timings = RuleTimings(np.full((2, 4), 37.5), np.full((2, 4), 0.8))
    predictions = ioi_log_ratio_predictions(timings, np.array([1.0, 2.0, 3.5, 4.0]))
    assert np.isnan(predictions[0, 0])
    assert np.allclose(predictions[:, 1:], 0.0)


def test_an_interval_the_player_could_not_emit_is_missing_coverage_not_zero_error():
    # the second onset lands 1500 ms earlier relative to the first than the score allows, so
    # the interval would be negative: reporting zero error there would flatter the baseline
    timings = RuleTimings(np.array([[500.0, -1000.0]]), np.array([[0.8, 0.8]]))
    predictions = ioi_log_ratio_predictions(timings, np.array([0.0, 0.2]))
    assert np.isnan(predictions[0, 1])


def test_ioi_prediction_scales_a_stretched_interval_by_the_same_ratio():
    timings = RuleTimings(np.array([[0.0, 100.0]]), np.array([[0.8, 0.8]]))
    predictions = ioi_log_ratio_predictions(timings, np.array([0.0, 1.0]))
    assert predictions[0, 1] == pytest.approx(math.log(1.1))


def test_duration_prediction_divides_the_held_element_length_by_the_written_note_length():
    # element 0 occupies 2 s and element 1 occupies 1 s; holds are 0.5 and 0.25, so element 0
    # sounds 1.0 s and element 1 sounds 0.25 s -- and each note is compared against its own
    # notated length, which is what the label's denominator is
    timings = RuleTimings(np.zeros((1, 2)), np.array([[0.5, 0.25]]))
    per_note = duration_log_ratio_predictions(timings, np.array([1, 1, 0], dtype=np.int64),
                                              np.array([2.0, 1.0]), np.array([1.0, 4.0, 2.0]))
    assert per_note.shape == (1, 3)
    assert per_note[0].tolist() == pytest.approx([math.log(0.25), math.log(0.0625), math.log(0.5)])


def test_duration_prediction_refuses_a_zero_written_length_and_a_mismatched_element_count():
    timings = RuleTimings(np.zeros((1, 2)), np.array([[0.5, 0.5]]))
    with pytest.raises(ValueError, match="positive and finite"):
        duration_log_ratio_predictions(timings, np.array([0, 1], dtype=np.int64),
                                       np.array([1.0, 1.0]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="match the rule timings"):
        duration_log_ratio_predictions(timings, np.array([0, 1], dtype=np.int64),
                                       np.array([1.0]), np.array([1.0, 1.0]))


def test_onset_prediction_is_the_clamped_offset_in_milliseconds():
    timings = RuleTimings(np.array([[0.0, 12.5]]), np.array([[0.8, 0.8]]))
    assert onset_predictions(timings).tolist() == [[0.0, 12.5]]


def test_the_projection_carries_no_pitch_so_the_leap_delay_is_excluded_by_design():
    # the same seed, the same durations, the same everything -- the only difference is whether
    # the element has a degree name the leap rule can read.  PianoCoRe pitches never get one,
    # so the shipping leap delay is disclosed as missing rather than re-invented here.
    from core.humanize import build_plan

    placeholder = build_plan([{"notes": [UNPARSEABLE_NOTE_ID], "dur": 1.0} for _ in range(2)],
                             ZERO_JITTER, seed=21)
    named = build_plan([{"notes": ["mid_1"], "dur": 1.0}, {"notes": ["mid_7"], "dur": 1.0}],
                       ZERO_JITTER, seed=21)
    assert placeholder.timings[1][0] == pytest.approx(0.0)
    # mid_1 -> mid_7 is six degrees, past LEAP_THRESHOLD, so the real app would add 5 ms x U(.5,1)
    assert 2.5 <= named.timings[1][0] <= 5.0


def test_leap_rule_coverage_counts_only_consecutive_pairs():
    coverage = leap_rule_coverage([[60, 62, 70], [60]], threshold=4)
    assert coverage["consecutive_pairs"] == 2 and coverage["would_fire"] == 1
    assert coverage["coverage"] == pytest.approx(0.5)
    assert leap_rule_coverage([[60]], threshold=4)["coverage"] == 0.0
