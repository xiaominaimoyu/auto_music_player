from __future__ import annotations

import io
from dataclasses import replace

import mido
import numpy as np
import pytest

from core.midi_reader import read_midi_source
from core.profile import legacy_profile
from core.source_score import AdaptOptions
from tests.test_training_game_expressibility import DELTA
from training import source_game_mapping as mapping
from training.audit_game_expressibility import Voice
from training.tempo_reference import score_timing_reference


def _midi(tracks):
    """Fixture notes are (start_s, end_s, pitch[, channel]); one tick is 1 ms."""
    midi = mido.MidiFile(ticks_per_beat=500)
    for notes in tracks:
        events = []
        for position, note in enumerate(notes):
            start, end, pitch, *channel = note
            channel = channel[0] if channel else 0
            events.append((round(start * 1000), position * 2,
                           mido.Message("note_on", note=pitch, velocity=80, channel=channel)))
            events.append((round(end * 1000), position * 2 + 1,
                           mido.Message("note_off", note=pitch, channel=channel)))
        track = mido.MidiTrack()
        previous = 0
        for tick, _, message in sorted(events, key=lambda event: event[:2]):
            track.append(message.copy(time=tick - previous))
            previous = tick
        midi.tracks.append(track)
    buffer = io.BytesIO()
    midi.save(file=buffer)
    return buffer.getvalue()


def _reference(data):
    value = score_timing_reference(data)
    value["note_native_masks"] = np.ones((len(value["note_to_chord"]), 6), dtype=np.bool_)
    value["chord_native_masks"] = np.ones((len(value["chord_score_seconds"]), 2), dtype=np.bool_)
    value["chord_native_masks"][0, 1] = False
    value["chord_trusted"] = np.ones(len(value["chord_score_seconds"]), dtype=np.bool_)
    return value


def _build(tmp_path, tracks, *, options=None, profiles=None, min_playable_ms=60, mutate=None):
    data = _midi(tracks)
    path = tmp_path / "score.mid"
    path.write_bytes(data)
    reference = _reference(data)
    if mutate is not None:
        mutate(reference)
    return mapping.build_sample_game_mapping(
        score_bytes=data, production_read_result=read_midi_source(path), timing_reference=reference,
        profiles=profiles or [legacy_profile(), DELTA],
        options=options or AdaptOptions(track=None, style="preserve"), min_playable_ms=min_playable_ms), reference


def test_unique_join_uses_both_degrees_not_greedy_identity():
    voice = Voice(0, .5, 60, 0)
    result = mapping.join_unique_voices([voice, voice], [voice])
    assert result["target_to_source"].tolist() == [-1]
    assert result["source_to_target"].tolist() == [-1, -1]
    result = mapping.join_unique_voices([voice], [voice, voice])
    assert result["target_to_source"].tolist() == [-1, -1]
    assert result["source_candidate_count"].tolist() == [2]
    assert set(result["target_reason"]) == {"ambiguous_endpoint_identity"}


def test_unique_join_respects_provenance_and_continuous_epsilon():
    eps = mapping.EPSILON_S
    first = Voice(.49 * eps, 1 + .49 * eps, 60, 0)
    across_bucket = Voice(1.01 * eps, 1 + 1.01 * eps, 60, 0)
    assert mapping.join_unique_voices([first], [across_bucket])["target_to_source"].tolist() == [0]
    outside = Voice(1.5 * eps, 1 + 1.5 * eps, 60, 0)
    assert mapping.join_unique_voices([first], [outside])["target_to_source"].tolist() == [-1]
    result = mapping.join_unique_voices([first, first], [across_bucket], allowed_source_members=[(1,)])
    assert result["target_to_source"].tolist() == [1]
    assert result["candidate_indices"].tolist() == [1]


@pytest.mark.parametrize("kwargs", [{"epsilon_s": -1}, {"epsilon_s": float("nan")},
                                   {"allowed_source_members": []}, {"allowed_source_members": [[-1]]}])
def test_unique_join_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        mapping.join_unique_voices([Voice(0, 1, 60, 0)], [Voice(0, 1, 60, 0)], **kwargs)


def test_simple_identity_masks_are_diagnostic_and_do_not_alias_reference(tmp_path):
    result, reference = _build(tmp_path, [[(0, .5, 60), (.5, 1, 62), (1, 1.5, 64)]])
    arrays = result["arrays"]
    for stage in ("source", "import", "prepared", "profile_default", "profile_delta_force_harmonica"):
        assert arrays[f"{stage}_native_note_index"].tolist() == [0, 1, 2]
        assert arrays[f"{stage}_strict"].tolist() == [True] * 3
    for profile in ("default", "delta_force_harmonica"):
        assert arrays[f"profile_{profile}_ioi_candidate_mask"].tolist() == [False, True, True]
        assert arrays[f"profile_{profile}_singleton_duration_candidate_mask"].all()
    assert result["summary"]["diagnostic_only"] is True
    assert result["summary"]["training_applied"] is False
    assert result["summary"]["performance_targets_emitted"] is False
    for target, source in (("native_note_masks", "note_native_masks"),
                           ("native_chord_masks", "chord_native_masks"),
                           ("native_note_to_chord", "note_to_chord")):
        np.testing.assert_array_equal(arrays[target], reference[source])
        assert not np.shares_memory(arrays[target], reference[source])


def test_cross_track_fifo_pairing_difference_is_unknown_not_nearest_match(tmp_path):
    result, _ = _build(tmp_path, [[(0, 1, 60)], [(.25, .5, 60)]])
    arrays = result["arrays"]
    assert arrays["source_native_note_index"].tolist() == [-1, -1]
    assert arrays["native_to_source_note_index"].tolist() == [-1, -1]
    assert not arrays["profile_default_strict"].any()


def test_equal_endpoints_on_different_channels_remain_ambiguous(tmp_path):
    result, _ = _build(tmp_path, [[(0, .5, 60, 0), (0, .5, 60, 1)]])
    assert result["arrays"]["source_native_note_index"].tolist() == [-1, -1]
    assert not result["arrays"]["import_strict"].any()


def test_fold_cannot_alias_unselected_native_note(tmp_path):
    result, _ = _build(tmp_path, [[(0, .5, 48)], [(0, .5, 36)]],
                       options=AdaptOptions(track=1, style="preserve"))
    arrays = result["arrays"]
    assert arrays["import_pitch"].tolist() == [48]
    assert arrays["import_native_note_index"].tolist() == [-1]
    origins = arrays["import_candidate_source_indices"]
    assert arrays["source_pitch"][origins].tolist() == [36]
    assert "octave_folded" in arrays["import_reason"][0]


@pytest.mark.parametrize("kind", ["bridge", "adapter_split", "transport_absorb", "polyphony", "mixed_chord"])
def test_transformed_slots_are_never_reused_as_native_labels(tmp_path, kind):
    notes, options, stage, changed = {
        "bridge": ([(0, .45, 60), (.5, 1, 62)], AdaptOptions(track=None, style="piano"), "import", [0]),
        "adapter_split": ([(0, 9, 60)], AdaptOptions(track=None, style="preserve"), "import", [0, 1]),
        "transport_absorb": ([(0, 8, 60), (8, 8.02, 62)], AdaptOptions(track=None, style="preserve"), "prepared", [0, 1]),
        "polyphony": ([(0, .5, 60), (0, .5, 67)], AdaptOptions(track=None, style="piano"), "import", [0]),
        "mixed_chord": ([(0, .5, 60), (0, .5, 61)], AdaptOptions(track=None, style="preserve"), "import", [0]),
    }[kind]
    result, _ = _build(tmp_path, [notes], options=options)
    arrays = result["arrays"]
    assert not arrays[f"{stage}_strict"][changed].any()
    if kind != "bridge":
        assert not arrays["profile_default_duration_candidate_mask"].any()
    if kind == "bridge":
        assert arrays["import_strict"].tolist() == [False, True]
    if kind == "transport_absorb":
        assert arrays["prepared_keyup_s"][-1] == pytest.approx(8.02)
        assert arrays["prepared_candidate_source_indices"].tolist() == [0, 0, 1]


def test_full_chord_identity_is_required_for_ioi_and_singleton_hold(tmp_path):
    result, _ = _build(tmp_path, [[(0, .5, 60), (0, .5, 64), (.5, 1, 62), (.5, 1, 67)]])
    arrays = result["arrays"]
    assert arrays["profile_default_duration_candidate_mask"].tolist() == [True] * 4
    assert not arrays["profile_default_singleton_duration_candidate_mask"].any()
    assert arrays["profile_default_native_chord_index"].tolist() == [0, 1]
    assert arrays["profile_default_ioi_candidate_mask"].tolist() == [False, True]
    assert not arrays["profile_delta_force_harmonica_strict"].any()
    assert not arrays["profile_delta_force_harmonica_ioi_candidate_mask"].any()


def test_partial_native_chord_after_track_selection_is_not_a_whole_chord(tmp_path):
    result, _ = _build(tmp_path, [[(0, .5, 60), (.5, 1, 62)], [(0, .5, 48), (.5, 1, 50)]],
                       options=AdaptOptions(track=0, style="preserve"))
    arrays = result["arrays"]
    assert arrays["profile_default_strict"].all()
    assert arrays["profile_default_native_chord_index"].tolist() == [-1, -1]
    assert not arrays["profile_default_ioi_candidate_mask"].any()


def test_dropped_middle_chord_does_not_reuse_original_single_ioi(tmp_path):
    profile = replace(legacy_profile(), legacy_keymap={"low": list("ZXCVBNM"),
                      "mid": ["", *list("SDFGHJ")], "high": list("QWERTYU")})
    result, _ = _build(tmp_path, [[(0, .5, 62), (.5, 1, 60), (1, 1.5, 64)]], profiles=[profile])
    arrays = result["arrays"]
    assert arrays["profile_default_native_chord_index"].tolist() == [0, 2]
    assert arrays["profile_default_ioi_candidate_mask"].tolist() == [False, False]
    assert arrays["profile_default_ioi_native_index"].tolist() == [-1, -1]


@pytest.mark.parametrize("untrusted", [0, 1])
def test_ioi_candidate_requires_explicit_trust_at_both_endpoints(tmp_path, untrusted):
    def mutate(reference):
        reference["chord_trusted"][untrusted] = False
    result, _ = _build(tmp_path, [[(0, .5, 60), (.5, 1, 62)]], mutate=mutate)
    assert not result["arrays"]["profile_default_ioi_candidate_mask"].any()


def test_duration_native_mask_gate_is_not_replaced_by_identity(tmp_path):
    def mutate(reference):
        reference["note_native_masks"][0, 1] = False
    result, _ = _build(tmp_path, [[(0, .5, 60), (.5, 1, 62)]], mutate=mutate)
    assert result["arrays"]["profile_default_duration_candidate_mask"].tolist() == [False, True]
    assert result["arrays"]["profile_default_strict"].all()


def test_rests_do_not_create_fake_chords_or_break_consecutive_native_ioi(tmp_path):
    result, _ = _build(tmp_path, [[(.5, 1, 60), (1.5, 2, 62)]])
    arrays = result["arrays"]
    assert arrays["profile_default_group_element_index"].tolist() == [1, 3]
    assert arrays["profile_default_native_chord_index"].tolist() == [0, 1]
    assert arrays["profile_default_ioi_candidate_mask"].tolist() == [False, True]


def test_tempo_changes_retain_seconds_identity_not_uniform_tick_assumption(tmp_path):
    midi = mido.MidiFile(ticks_per_beat=500)
    midi.tracks.append(mido.MidiTrack([
        mido.MetaMessage("set_tempo", tempo=500000),
        mido.Message("note_on", note=60, velocity=80),
        mido.Message("note_off", note=60, time=500),
        mido.MetaMessage("set_tempo", tempo=1000000),
        mido.Message("note_on", note=62, velocity=80),
        mido.Message("note_off", note=62, time=500)]))
    buffer = io.BytesIO()
    midi.save(file=buffer)
    data = buffer.getvalue()
    path = tmp_path / "tempo.mid"
    path.write_bytes(data)
    result = mapping.build_sample_game_mapping(score_bytes=data, production_read_result=read_midi_source(path),
        timing_reference=_reference(data), profiles=[legacy_profile()],
        options=AdaptOptions(track=None, style="preserve"))
    assert result["arrays"]["profile_default_strict"].tolist() == [True, True]
    assert result["arrays"]["profile_default_keyup_s"].tolist() == pytest.approx([.5, 1.5])


def test_profile_pitch_degradation_is_distinct_from_source_identity(tmp_path):
    result, _ = _build(tmp_path, [[(0, .5, 61)]])
    arrays = result["arrays"]
    assert arrays["prepared_strict"].tolist() == [True]
    assert arrays["profile_default_strict"].tolist() == [False]
    assert arrays["profile_delta_force_harmonica_strict"].tolist() == [True]


def test_zero_duration_native_note_is_preserved_but_cannot_match(tmp_path):
    result, reference = _build(tmp_path, [[(0, 0, 60), (.5, 1, 62)]])
    arrays = result["arrays"]
    assert len(reference["note_native_masks"]) == 2
    assert arrays["native_to_source_note_index"].tolist() == [-1, 0]
    assert arrays["source_native_note_index"].tolist() == [1]


@pytest.mark.parametrize("invalid", ["order", "seconds", "masks", "trusted"])
def test_mismatched_native_reference_fails_closed(tmp_path, invalid):
    def mutate(reference):
        if invalid == "order":
            reference["note_to_chord"] = reference["note_to_chord"][::-1].copy()
        elif invalid == "seconds":
            reference["note_score_duration_seconds"][0] += .01
        elif invalid == "masks":
            reference["note_native_masks"] = reference["note_native_masks"].astype(np.int32)
        else:
            reference["chord_trusted"] = np.array([True])
    with pytest.raises(ValueError, match="native"):
        _build(tmp_path, [[(0, .5, 60), (.5, 1, 62)]], mutate=mutate)


def test_import_join_is_batched_over_all_source_voices(tmp_path, monkeypatch):
    original = mapping.join_unique_voices
    calls = []
    def observed(source, target, **kwargs):
        calls.append((len(source), len(target), kwargs.get("allowed_source_members")))
        return original(source, target, **kwargs)
    monkeypatch.setattr(mapping, "join_unique_voices", observed)
    _build(tmp_path, [[(i * .5, (i + 1) * .5, 60 + i % 3 * 2) for i in range(20)]])
    provenance_calls = [row for row in calls if row[2] is not None]
    assert len(provenance_calls) == 1
    assert provenance_calls[0][:2] == (20, 20)
