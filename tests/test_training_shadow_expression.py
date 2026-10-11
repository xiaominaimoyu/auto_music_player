"""Compiler-only tests: no input driver, GUI, GPU, or performance reference."""
from dataclasses import replace
import io

import mido
import numpy as np
import pytest

from core.compiler import ChordPolicy, CompileParams, ModifierPolicy, compile_score
from core.humanize import HumanizeParams, plan_timings
from core.ir import IRRest, to_storage
from core.profile import load_profiles
from core.scenario import get_scenario
from training.midi_features import parse_midi_bytes
from training.render_expression_candidates import original_study_midi, score_inference_arrays
from training.shadow_expression import shadow_projection


def _score(notes, tempos=((0, 500000),)):
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    events = [(beat * 480, -1, mido.MetaMessage("set_tempo", tempo=tempo))
              for beat, tempo in tempos]
    for beat, duration, pitch in notes:
        events.extend([(round(beat * 480), 1, mido.Message("note_on", note=pitch, velocity=80)),
                       (round((beat + duration) * 480), 0, mido.Message("note_off", note=pitch))])
    previous = 0
    for tick, _, message in sorted(events, key=lambda item: (item[0], item[1])):
        tick = int(tick)
        track.append(message.copy(time=tick - previous))
        previous = tick
    data = io.BytesIO()
    midi.save(file=data)
    parsed = parse_midi_bytes(data.getvalue())
    ticks = sorted({note.start_tick for note in parsed.notes})
    mapping = np.array([ticks.index(note.start_tick) for note in parsed.notes], dtype=np.int64)
    return parsed, mapping


def _prediction(score, mapping):
    return np.zeros((len(score.notes), 6)), np.zeros((int(np.max(mapping)) + 1, 2))


def _direct_params(**kwargs):
    return CompileParams(bpm=120, modifier_buttons={}, **kwargs)


def test_default_does_not_activate_supplied_model_and_identity_is_actual_compiler():
    score, mapping = _score([(0, .5, 60), (1, .5, 62)])
    ny, cy = _prediction(score, mapping)
    ny[:, 1], cy[:, 1] = 10, 10
    actual = shadow_projection(score, mapping, ny, cy, params=_direct_params())
    expected = compile_score(actual["elements"], actual["compile_params"])
    assert actual["strategy"] == "identity"
    assert actual["events"] == expected.events
    assert not actual["model_enabled_by_default"]
    assert not actual["actual_game_input_sent"]
    assert not actual["uses_oracle_performance"]


def test_rules_are_exact_current_production_plan_and_reproducible():
    score, mapping = _score([(0, .5, 60), (1, 3, 67), (5, .25, 72)])
    params, humanize = CompileParams(bpm=120), HumanizeParams()
    result = shadow_projection(score, mapping, strategy="rules", seed=77, params=params, humanize=humanize)
    expected_params = replace(get_scenario("free_play").apply_intervals(params), min_gap_ms=humanize.min_gap_ms)
    expected = compile_score(result["elements"], expected_params,
                             timings=plan_timings(to_storage(result["elements"]), humanize, seed=77))
    assert result["events"] == expected.events
    again = shadow_projection(score, mapping, strategy="rules", seed=77, params=params, humanize=humanize)
    assert again["events"] == result["events"]
    assert result["all_events_balanced"]


def test_model_integrates_fixed_score_seconds_and_does_not_use_aux_micro_velocity_pedal():
    score, mapping = _score([(0, .2, 60), (1, .2, 62), (2, .2, 64), (3, .2, 65)],
                            tempos=((0, 500000), (2, 1000000)))
    ny, cy = _prediction(score, mapping)
    cy[:, 1] = np.log(1.1)
    result = shadow_projection(score, mapping, ny, cy, strategy="model", strength=1, params=_direct_params(), cumulative_onset_limit_s=2)
    # bpm controls IR units, not a rescaling of the integrated score seconds.
    np.testing.assert_allclose(result["score_intervals_s"], [0, .5, .5, 1])
    np.testing.assert_allclose(result["desired_onsets_s"], [0, .55, 1.1, 2.2])
    np.testing.assert_allclose(result["actual_onsets_s"], result["desired_onsets_s"], atol=1e-10)
    ny[:, 0], ny[:, 2:], cy[:, 0] = 1e9, 1e9, -1e9
    changed = shadow_projection(score, mapping, ny, cy, strategy="model", strength=1, params=_direct_params(), cumulative_onset_limit_s=2)
    assert changed["events"] == result["events"]
    another_bpm = shadow_projection(score, mapping, ny, cy, strategy="model", strength=1,
                                    params=replace(_direct_params(), bpm=60), cumulative_onset_limit_s=2)
    np.testing.assert_allclose(another_bpm["desired_onsets_s"], result["desired_onsets_s"])
    np.testing.assert_allclose(another_bpm["actual_onsets_s"], result["actual_onsets_s"])


def test_real_rests_leading_and_long_silence_remain_in_same_nominal_ir():
    score, mapping = _score([(2, .5, 60), (50, .5, 62)])
    ny, cy = _prediction(score, mapping)
    baseline = shadow_projection(score, mapping, params=_direct_params())
    learned = shadow_projection(score, mapping, ny, cy, strategy="model", params=_direct_params())
    assert baseline["elements"] == learned["elements"]
    rests = [element for element in baseline["elements"] if isinstance(element, IRRest)]
    assert len(rests) >= 3
    assert all(0 < element.dur <= 16 for element in rests)
    total_s = sum(element.dur for element in baseline["elements"]) * .5
    assert total_s == pytest.approx(25.25)
    np.testing.assert_allclose(learned["actual_onsets_s"], [1, 25])
    np.testing.assert_array_equal(learned["element_to_chord"][learned["chord_to_element"]], [0, 1])


def test_model_clipping_reports_limits_and_all_events_have_balanced_releases():
    score, mapping = _score([(0, .01, 60), (1, 12, 62), (2, .5, 64)])
    ny, cy = _prediction(score, mapping)
    ny[:, 1], cy[:, 1] = 1000, -1000
    result = shadow_projection(score, mapping, ny, cy, strategy="model", params=_direct_params(max_hold_ms=100), strength=1)
    assert result["clipping"]["ioi"] == 3
    assert result["clipping"]["duration_ratio"] == 3
    assert result["clipping"]["raw_numeric_exponent_cap"] == 6
    assert result["clipping"]["hold_floor"] >= 1
    assert result["clipping"]["hold_slot"] >= 1
    assert result["clipping"]["hold_max"] >= 1
    assert result["all_events_balanced"]
    assert np.isfinite(result["actual_onsets_s"]).all()
    assert np.all(result["actual_holds_s"] > 0)
    assert np.all(result["actual_holds_s"] <= .1 + 1e-10)


def test_absolute_cursor_offsets_do_not_accumulate_jitter_with_modifier_settle():
    score, mapping = _score([(i, .4, 48 if i % 2 else 74) for i in range(12)])
    ny, cy = _prediction(score, mapping)
    cy[:, 1] = .2
    result = shadow_projection(score, mapping, ny, cy, strategy="model", strength=1)
    # First onset needs physical modifier setup. Later roomy slots hit target.
    assert result["actual_onsets_s"][0] >= .03
    np.testing.assert_allclose(result["actual_onsets_s"][1:], result["desired_onsets_s"][1:], atol=1e-9)
    assert result["all_events_balanced"]
    assert result["baseline_offset_mode"] == "absolute_cursor_translation"


def test_minimum_gap_projection_is_reported_not_silently_accepted():
    score, mapping = _score([(i * .05, 1, 60) for i in range(6)])
    ny, cy = _prediction(score, mapping)
    cy[:, 1] = -.25
    result = shadow_projection(score, mapping, ny, cy, strategy="model", strength=1,
                               params=_direct_params(min_gap_ms=25))
    assert result["compiler_delayed_chords"] > 0
    assert result["max_compiler_onset_change_ms"] > 0
    down = [event for event in result["events"] if event.device == "kb" and event.action == "down"]
    up = [event for event in result["events"] if event.device == "kb" and event.action == "up"]
    assert all(after.t_ms >= before.t_ms + 25 - 1e-8 for before, after in zip(up[:-1], down[1:]))


def test_simultaneous_chord_highest_selection_fold_and_compiler_loss_are_explicit():
    score, mapping = _score([(0, .5, 36), (0, .8, 73), (0, .2, 76), (1, .5, 62)])
    ny, cy = _prediction(score, mapping)
    result = shadow_projection(score, mapping, ny, cy, strategy="model",
                               params=CompileParams(bpm=120, chord_policy=ChordPolicy.CHORD_HIGHEST))
    assert result["selected_note_indices"][0] == next(i for i, n in enumerate(score.notes) if n.pitch == 76)
    assert result["unrepresented_source_voices"] == 2
    assert len(result["folded_note_indices"]) == 1
    assert any(item["reason"] == "highest" for item in result["degradations"])
    assert result["projected_voice_count"] == 2
    assert len([event for event in result["events"] if event.device == "kb" and event.action == "down"]) == 2


def test_rejected_chords_are_nan_with_coverage_not_fabricated_zero():
    score, mapping = _score([(0, .5, 60), (0, .5, 64), (1, .5, 62)])
    ny, cy = _prediction(score, mapping)
    result = shadow_projection(score, mapping, ny, cy, strategy="model",
                               params=CompileParams(bpm=120, chord_policy=ChordPolicy.CHORD_REJECT))
    assert np.isnan(result["actual_onsets_s"][0])
    assert not result["coverage_mask"][0]
    assert result["covered_chord_count"] == 1
    assert result["all_events_balanced"]


def test_no_oracle_arguments_and_mapping_and_finiteness_are_fail_closed():
    score, mapping = _score([(0, .5, 60), (1, .5, 62)])
    ny, cy = _prediction(score, mapping)
    with pytest.raises(TypeError):
        shadow_projection(score, mapping, performance_reference={"global_scale": .2})
    with pytest.raises(ValueError):
        shadow_projection(score, mapping[::-1], ny, cy, strategy="model")
    with pytest.raises(ValueError):
        shadow_projection(score, mapping, ny[:-1], cy, strategy="model")
    with pytest.raises(ValueError):
        shadow_projection(score, mapping, ny.astype(complex), cy, strategy="model")
    ny[0, 4] = np.nan
    with pytest.raises(ValueError):
        shadow_projection(score, mapping, ny, cy, strategy="model")


def test_npc_disallows_model_and_rules_but_identity_is_available():
    score, mapping = _score([(0, .5, 60), (1, .5, 62)])
    ny, cy = _prediction(score, mapping)
    for strategy in ("rules", "model"):
        with pytest.raises(ValueError, match="NPC"):
            shadow_projection(score, mapping, ny, cy, strategy=strategy, scenario_id="npc_quest")
    result = shadow_projection(score, mapping, scenario_id="npc_quest")
    assert result["strategy"] == "identity"
    # No NPC post-roll is emitted by a compiler-only audit.
    assert all(event.key != "Q" for event in result["events"])


def test_actual_delta_profile_and_original_scores_have_balanced_compile_output():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    profile = next(profile for profile in load_profiles(str(root / "profiles"))
                   if profile.id == "delta_force_harmonica")
    params = profile.build_compile_params(bpm=120, settle_ms=30, release_settle_ms=20,
                                          hold_ratio=1, max_hold_ms=3000, gap_ms=0)
    for study in range(3):
        score = parse_midi_bytes(original_study_midi(study))
        arrays = score_inference_arrays(score)
        ny, cy = _prediction(score, arrays["note_to_chord"])
        for strategy in ("identity", "rules", "model"):
            result = shadow_projection(score, arrays["note_to_chord"], ny, cy,
                                       strategy=strategy, params=params)
            assert result["all_events_balanced"]
            assert not result["actual_game_input_sent"]
            assert result["source_chord_count"] == len(cy)
            assert result["unrepresented_source_voices"] > 0


def test_arpeggiation_is_not_misreported_as_single_voice_accuracy():
    score, mapping = _score([(0, .5, 60), (0, .5, 64)])
    with pytest.raises(ValueError, match="many-events"):
        shadow_projection(score, mapping, params=CompileParams(chord_policy=ChordPolicy.CHORD_ARPEGGIATE))


def test_up_events_sort_before_down_at_equal_time_and_end_without_stuck_key():
    score, mapping = _score([(0, 1, 60), (1, 1, 60)])
    result = shadow_projection(score, mapping, params=_direct_params(gap_ms=0, min_gap_ms=0))
    assert result["all_events_balanced"]
    for before, after in zip(result["events"], result["events"][1:]):
        if before.t_ms == after.t_ms:
            assert not (before.action == "down" and after.action == "up")


def test_long_biased_sequence_keeps_score_only_budget_and_positive_intervals():
    score, mapping = _score([(i * .2, .1, 60) for i in range(500)])
    ny, cy = _prediction(score, mapping)
    for bias in (-.25, .25):
        cy[:, 1] = bias
        result = shadow_projection(score, mapping, ny, cy, strategy="model", strength=1,
                                   params=_direct_params(), cumulative_onset_limit_s=.15)
        assert result["clipping"]["cumulative_onset_budget"] > 0
        assert result["max_cumulative_offset_before_budget_s"] > 1
        assert result["max_cumulative_offset_after_budget_s"] <= .15 + 1e-10
        assert np.all(np.diff(result["desired_onsets_s"]) > 0)
        assert result["desired_onsets_s"][0] == result["score_onsets_s"][0]
        assert result["all_events_balanced"]


def test_modifier_release_guard_prevents_late_up_canceling_new_down():
    score, mapping = _score([(i * .5, .42, 74) for i in range(16)])
    ny, cy = _prediction(score, mapping)
    result = shadow_projection(score, mapping, ny, cy, strategy="model")
    assert result["all_events_balanced"]
    assert result["compiler_delayed_chords"] > 0
    assert result["actual_cumulative_budget_exceeded_chords"] > 0
    assert result["max_actual_score_offset_s"] > .15
    # The constraint is explicitly a possible quality failure, not game proof.
    np.testing.assert_allclose(result["desired_onsets_s"], result["score_onsets_s"])
    modifier_events = [event for event in result["events"] if event.device == "mouse"]
    assert all(event.action == ("down" if index % 2 == 0 else "up")
               for index, event in enumerate(modifier_events))


def test_modifier_conflict_skip_and_long_note_projection_losses_are_disclosed():
    score, mapping = _score([(0, 40, 73), (41, .5, 60)])
    ny, cy = _prediction(score, mapping)
    result = shadow_projection(score, mapping, ny, cy, strategy="model",
                               params=CompileParams(bpm=120, modifier_policy=ModifierPolicy.REJECT_NOTE))
    assert result["long_note_truncations"] == 1
    assert result["rest_element_count"] >= 2
    assert not result["coverage_mask"][0]
    assert np.isnan(result["actual_holds_s"][0])
    assert any(item["reason"] == "reject_note" for item in result["degradations"])


def test_prior_overlapping_bass_prevents_false_rest_and_phrase_breath_marker():
    score, mapping = _score([(0, 3.5, 48), (.5, .2, 64), (1, .2, 65), (4, .5, 67)])
    result = shadow_projection(score, mapping, params=CompileParams(bpm=120, chord_policy=ChordPolicy.CHORD_HIGHEST))
    first, second, third, fourth = result["chord_to_element"]
    assert second == first + 1
    assert third == second + 1
    assert fourth > third + 1  # All score keys silent from beat 3.5 to 4.
    assert "prior overlapping voices" in result["rest_semantics"]


@pytest.mark.parametrize("pitches", [[60] * 8, [74] * 8, [74, 50] * 4])
def test_absolute_adapter_never_cancels_free_play_full_release_gap(pitches):
    score, mapping = _score([(index, 1, pitch) for index, pitch in enumerate(pitches)])
    ny, cy = _prediction(score, mapping)
    result = shadow_projection(score, mapping, ny, cy, strategy="model",
                               params=CompileParams(bpm=120, gap_ms=0))
    assert result["compile_params"].gap_ms == 20
    assert result["required_model_full_release_gap_ms"] == 20
    assert result["minimum_actual_preparation_gap_ms"] >= 20 - 1e-8
    assert result["actual_full_release_gap_violation_count"] == 0
    assert result["actual_inter_event_gap_policy"]["absolute_model_enforced"]
    # Independently measure the actual events, including earlier mouse setup
    # and later mouse release on modifier changes, not only keyboard gaps.
    last_release = None
    for source in result["chord_to_element"]:
        events = [event for event in result["events"] if event.source_index == source]
        start = min(event.t_ms for event in events if event.action == "down")
        end = max(event.t_ms for event in events if event.action == "up")
        if last_release is not None:
            assert start - last_release >= 20 - 1e-8
        last_release = end


def test_larger_user_gap_and_min_gap_both_respected_by_absolute_adapter():
    score, mapping = _score([(index, 1, 60) for index in range(4)])
    ny, cy = _prediction(score, mapping)
    for gap, minimum in ((40, 1), (20, 35)):
        result = shadow_projection(score, mapping, ny, cy, strategy="model",
                                   params=CompileParams(bpm=120, gap_ms=gap, min_gap_ms=minimum))
        assert result["required_model_full_release_gap_ms"] == max(gap, minimum)
        assert result["minimum_actual_preparation_gap_ms"] >= max(gap, minimum) - 1e-8
        assert result["actual_full_release_gap_violation_count"] == 0
