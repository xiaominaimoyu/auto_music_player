from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import mido
import pytest

from core.compiler import ChordPolicy
from core.profile import GameProfile, legacy_profile
from core.source_score import AdaptOptions, SourceNote, SourceSong
from training import audit_game_expressibility as audit
from training.tempo_reference import build_timing_reference
from tests.test_training_tempo_reference import _dataset_fixture


DELTA = GameProfile(id="delta_force_harmonica", name="Delta", chord_policy=ChordPolicy.CHORD_HIGHEST,
                    modifier_buttons={"lower": "left", "semitone": "middle", "higher": "right"},
                    pitch_direct_overrides={"high_1": ","})


def song(notes, *, duration=None, tempo_changes=1):
    values = [SourceNote(*note) for note in notes]
    return SourceSong("test", values, {track: "melody" for track in {n.track for n in values}},
                      max(n.end_s for n in values) if duration is None else duration, 120, tempo_changes)


def report(notes, *, style="preserve", settings=None, profiles=None, duration=None, min_playable_ms=audit.MIN_GAME_NOTE_MS):
    return audit.audit_song(song(notes, duration=duration), profiles=profiles or [legacy_profile(), DELTA],
        options=AdaptOptions(style=style, track=None), settings=settings or audit.PlaybackSettings(), min_playable_ms=min_playable_ms)


def projection(value, profile="default", scenario="free_play", baseline="mechanical"):
    return next(row for row in value["projection"] if (row["profile_id"], row["scenario_id"], row["baseline"]) ==
                (profile, scenario, baseline))


def test_natural_monophonic_full_source_fidelity_and_no_game_evidence():
    value = report([(0, .5, 60), (.5, 1, 62), (1, 1.5, 64)])
    assert value["source_to_import"]["exact_endpoint_voice_count"] == 3
    for row in value["projection"]:
        assert row["source_to_profile"]["exact_endpoint_voice_count"] == 3
        assert row["source_to_profile"]["pitch_time_coverage_fraction"] == pytest.approx(1)
        assert row["capabilities"]["measured_execution"] is False
    legacy = projection(value)
    assert legacy["release_budget"]["last_onset_delay_ms"] == pytest.approx(40)
    assert legacy["release_budget"]["nominal_transition_feasible_count"] == 2
    assert value["deployment_evidence"] is value["mask_applied"] is False


def test_legacy_chords_are_not_passed_to_monophonic_compiler(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("legacy must not compile")
    monkeypatch.setattr(audit, "compile_score", forbidden)
    value = report([(0, .5, 60), (0, .5, 64), (0, .5, 67)], profiles=[legacy_profile()])
    row = projection(value)
    assert row["logical_voice_count"] == row["nominal_actual_voice_count"] == 3
    assert row["source_to_profile"]["exact_endpoint_voice_count"] == 3
    assert row["release_budget"]["transition_count"] == 0
    assert row["capabilities"]["concurrent_chord_keys"] is True
    assert {r["scenario_id"] for r in value["projection"]} == {"free_play"}


def test_delta_chords_use_real_highest_policy_and_are_not_full_source_labels():
    value = report([(0, .5, 60), (0, .5, 64), (0, .5, 67)])
    row = projection(value, "delta_force_harmonica")
    assert row["logical_voice_count"] == 1
    assert row["source_to_profile"]["exact_endpoint_voice_count"] == 1
    assert row["suggested_source_full_fidelity_windows"][0]["feasible_count"] == 0
    assert row["capabilities"]["compiler_degradations"][0]["reason"] == "highest"
    assert row["capabilities"]["residual_held_keys"] == 0
    assert row["capabilities"]["event_pairing_violations"] == []


def test_chromatic_default_ignores_semitone_but_middle_delta_preserves_it():
    value = report([(0, .5, 61)])
    assert value["source_to_import"]["exact_endpoint_voice_count"] == 1
    assert projection(value)["source_to_profile"]["exact_endpoint_voice_count"] == 0
    delta = projection(value, "delta_force_harmonica")
    assert delta["source_to_profile"]["exact_endpoint_voice_count"] == 1
    assert delta["release_budget"]["last_onset_delay_ms"] == pytest.approx(30)


def test_delta_octave_and_semitone_conflict_counted_not_claimed_as_preserved():
    value = report([(0, .5, 49), (.5, 1, 73)])
    row = projection(value, "delta_force_harmonica")
    assert row["source_to_profile"]["exact_endpoint_voice_count"] == 0
    assert [d["reason"] for d in row["capabilities"]["compiler_degradations"]] == ["octave_first", "octave_first"]


def test_mixed_semitone_chord_reduction_is_import_not_model_error():
    value = report([(0, .5, 60), (0, .5, 61)])
    assert value["source_to_import"]["projected_voice_count"] == 1
    assert "mixed_semitone_chord" in {row["code"] for row in value["import_degradations"]}
    assert value["import_chord_faithful_count"] == 0


def test_octave_folding_is_counted_as_full_source_pitch_loss():
    value = report([(0, .5, 24), (.5, 1, 96)])
    assert value["source_out_of_C3_B5_count"] == 2
    assert value["source_to_import"]["exact_endpoint_voice_count"] == 0
    assert {row["code"]: row["count"] for row in value["import_degradations"]}["octave_folded"] == 2


def test_short_ornaments_have_budget_flags_not_guessed_hardware_minimum():
    value = report([(0, .01, 60), (.01, .02, 62), (.02, .5, 64)], min_playable_ms=0)
    assert value["source_short_key_duration_below_40ms"] == 2
    for row in value["projection"]:
        assert row["release_budget"]["nominal_transition_feasible_count"] == 0
        assert row["release_budget"]["transition_count"] == 2
        assert "not measured hardware limit" in row["release_budget"]["scope"]


def test_default_transport_absorbs_short_ornaments_before_real_game_routes():
    value = report([(0, .01, 60), (.01, .02, 62), (.02, .5, 64)])
    assert value["source_to_import"]["exact_endpoint_voice_count"] == 3
    assert value["source_to_transport"]["exact_endpoint_voice_count"] == 0
    assert value["transport"]["min_playable_ms"] == 60
    assert len(value["transport"]["degradations"]) == 2
    for row in value["projection"]:
        assert row["logical_voice_count"] == 1
        assert row["release_budget"]["transition_count"] == 0


def test_transport_splits_absorbed_sixteen_beat_note_before_real_game_route():
    # The 20 ms tail is absorbed and the 16.04-beat total gets legal pieces.
    # This introduces a repeated note, so endpoint fidelity remains diagnostic.
    value = report([(0, 8, 60), (8, 8.02, 62)], profiles=[legacy_profile()])
    assert len(value["transport"]["degradations"]) == 2
    assert projection(value)["logical_voice_count"] == 2
    assert value["source_to_transport"]["exact_endpoint_voice_count"] == 0


def test_modifier_same_latch_and_change_full_release_budget():
    # Low-natural same modifier, then high-natural requires releasing left
    # and settling right; high_2 has no high_1 direct override.
    value = report([(0, .5, 48), (.5, 1, 50), (1, 1.5, 74)])
    row = projection(value, "delta_force_harmonica")
    assert row["release_budget"]["modifier_cost_ms"]["mean"] == pytest.approx(25)
    assert row["release_budget"]["full_release_cost_ms"]["mean"] == pytest.approx(420)
    assert row["release_budget"]["last_onset_delay_ms"] == pytest.approx(100)


def test_rests_are_kept_gap_applies_to_rest_and_modifier_reacquired():
    value = report([(0, .5, 48), (1, 1.5, 50)])
    assert value["import_rest_count"] == 1
    legacy = projection(value)
    delta = projection(value, "delta_force_harmonica")
    assert legacy["release_budget"]["last_onset_delay_ms"] == pytest.approx(40)
    assert delta["release_budget"]["modifier_cost_ms"]["mean"] == pytest.approx(50)
    assert delta["release_budget"]["last_onset_delay_ms"] == pytest.approx(100)


def test_npc_uses_stricter_intervals_and_rules_remain_disabled():
    value = report([(0, .5, 60), (.5, 1, 62)])
    mechanical = projection(value, "delta_force_harmonica", "npc_quest")
    rules = projection(value, "delta_force_harmonica", "npc_quest", "rules")
    assert mechanical["capabilities"]["settle_ms"] == 60
    assert mechanical["capabilities"]["gap_ms"] == 120
    assert rules["capabilities"]["humanize_enabled"] is False
    assert rules["capabilities"]["humanize_seed"] is None
    assert rules["release_budget"] == mechanical["release_budget"]
    assert rules["capabilities"]["npc_submit_excluded"] is True


def test_rule_seed_repeatability_and_no_per_voice_hold_claim():
    notes = [(0, .25, 60), (0, .25, 64), (.25, .5, 67)]
    first, second = report(notes), report(notes)
    assert first == second
    assert projection(first, baseline="rules")["capabilities"]["independent_chord_voice_hold"] is False
    custom = report(notes, settings=replace(audit.PlaybackSettings(), seed=13))
    assert projection(first, baseline="rules")["release_budget"] != projection(custom, baseline="rules")["release_budget"]


def test_legacy_negative_targets_cannot_dispatch_before_blocking_hold_ends(monkeypatch):
    monkeypatch.setattr(audit, "plan_timings", lambda *args, **kwargs: [(-30, 1.0), (-100, 1.0)])
    source = report([(0, .1, 60), (.1, .2, 62)], profiles=[legacy_profile()])
    row = projection(source, baseline="rules")
    assert row["release_budget"]["onset_delay_ms"] == pytest.approx([0, 0])
    assert "not observed scheduler" in row["capabilities"]["evidence"]


def test_piano_short_gap_bridge_does_not_reuse_exact_original_duration_labels():
    value = report([(0, .4, 60), (.45, .9, 62)], style="piano")
    assert {row["code"] for row in value["import_degradations"]} == {"short_gap_bridged"}
    assert value["source_to_import"]["exact_endpoint_flags"] == [False, True]
    assert value["source_to_import_axis_end_error_ms"] == pytest.approx(0, abs=1e-8)


def test_auto_track_selection_reports_full_source_loss():
    source = SourceSong("multi", [SourceNote(0, .5, 60, 0), SourceNote(0, .5, 72, 1)],
                        {0: "bass", 1: "melody"}, .5, 120)
    value = audit.audit_song(source, profiles=[legacy_profile()])
    assert value["selected_track"] == 1
    assert value["source_to_import"]["source_voice_count"] == 2
    assert value["source_to_import"]["exact_endpoint_voice_count"] == 1


def test_missing_legacy_pitch_key_is_retention_loss():
    profile = replace(legacy_profile(), legacy_keymap={"low": list("ZXCVBNM"), "mid": ["", *list("SDFGHJ")], "high": list("QWERTYU")})
    value = report([(0, .5, 60)], profiles=[profile])
    assert projection(value)["logical_voice_count"] == 0
    assert projection(value)["import_to_profile"]["exact_endpoint_voice_count"] == 0


def test_tempo_changes_round_trip_seconds_without_quantization(tmp_path):
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack([
        mido.MetaMessage("set_tempo", tempo=500000, time=0),
        mido.Message("note_on", note=60, velocity=80, time=0),
        mido.Message("note_off", note=60, time=480),
        mido.MetaMessage("set_tempo", tempo=1000000, time=0),
        mido.Message("note_on", note=62, velocity=80, time=0),
        mido.Message("note_off", note=62, time=480)])
    midi.tracks.append(track)
    original = tmp_path / "tempo.mid"
    midi.save(original)
    parsed = audit.read_midi_source(original)
    value = audit.audit_song(parsed.song, profiles=[legacy_profile()], options=AdaptOptions(style="preserve", track=None))
    assert value["source_duration_s"] == pytest.approx(1.5)
    assert value["source_to_import"]["exact_endpoint_voice_count"] == 2
    assert value["source_to_import_axis_end_error_ms"] == pytest.approx(0)
    assert "tempo_flattened" in {row["code"] for row in value["import_degradations"]}


def test_windows_do_not_cross_short_sample_or_drop_failure():
    assert audit._windows([True, False, True], 2) == {"width": 2, "count": 2, "feasible_count": 0, "flags": [False, False]}
    assert audit._windows([True], 8)["count"] == 0


@pytest.mark.parametrize("kwargs", [{"gap_ms": -1}, {"hold_ratio": 0}, {"settle_ms": float("nan")}, {"max_hold_ms": float("inf")}])
def test_invalid_safety_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        audit.PlaybackSettings(**kwargs)


def test_source_imports_do_not_include_device_or_gui_modules():
    tree = ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
    names = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not set(names) & {"core.player", "core.event_player", "core.keyboard_driver", "gui.player_tab", "PyQt6.QtCore"}


def test_config_binding_includes_separate_player_source_and_profile_files():
    root = Path(audit.__file__).resolve().parents[1]
    bindings = audit._implementation_bindings(root / "profiles", root / "config.yaml")
    assert str(root / "core" / "player.py") in bindings
    assert str(root / "core" / "compiler.py") in bindings
    assert str(root / "core" / "transport.py") in bindings
    assert str(root / "training" / "tempo_reference.py") in bindings
    assert str(root / "profiles" / "default.yaml") in bindings
    assert str(root / "profiles" / "delta_force_harmonica.yaml") in bindings
    assert all(len(value) == 64 for value in bindings.values())


def _bound_reference(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    dataset, archive, selection = _dataset_fixture(inputs)
    reference = tmp_path / "reference"
    build_timing_reference(dataset_root=dataset, archive_path=archive,
                           sample_id_files=[selection], output_dir=reference)
    return reference, dataset


def test_reference_audit_seals_actual_import_and_both_separate_paths(tmp_path):
    reference, _ = _bound_reference(tmp_path)
    output = tmp_path / "output"
    result = audit.audit_reference_root(timing_reference=reference, output_dir=output, styles=["preserve"])
    assert result["sample_count"] == result["sample_style_count"] == 1
    assert result["status_counts"] == {"complete": 1}
    assert (output / "_SUCCESS").read_text().strip() == audit._sha256(output / "summary.json")
    assert not (output / "_INCOMPLETE").exists()
    sample = result["samples"][0]
    assert audit._sha256(output / sample["path"]) == sample["sha256"]
    row = json.loads((output / sample["path"]).read_text(encoding="utf-8"))
    assert {entry["route"] for entry in row["audit"]["projection"]} == {"legacy_static", "delta_pure_compiler"}
    for group in result["aggregate"]["groups"]:
        assert group["source_piece_group_count"] == 1
        assert group["transition_count"] == 5
        assert group["event_pairing_violation_count"] == group["residual_held_key_count"] == 0
        assert group["physical_events_checked_sample_count"] == (group["profile_id"] == "delta_force_harmonica")
    assert result["acceptance"]["game_input_sent"] is False


def test_transport_rejection_is_reported_with_denominator_and_without_projection(tmp_path, monkeypatch):
    reference, _ = _bound_reference(tmp_path)
    output = tmp_path / "output"

    def rejected(*args, **kwargs):
        raise audit.ScoreValidationError(["16-beat storage cap exceeded"])

    monkeypatch.setattr(audit, "prepare_score", rejected)
    result = audit.audit_reference_root(timing_reference=reference, output_dir=output, styles=["preserve"])
    assert result["sample_count"] == 1
    assert result["status_counts"] == {"transport_rejected": 1}
    assert result["aggregate"]["groups"] == []
    assert result["aggregate"]["status_groups"] == [{"split": "train", "performance_dataset": "demo",
        "style": "preserve", "status": "transport_rejected", "sample_count": 1}]
    sample = result["samples"][0]
    row = json.loads((output / sample["path"]).read_text(encoding="utf-8"))
    assert row["transport_errors"] == ["16-beat storage cap exceeded"]
    assert row["source_note_count"] == 6
    assert "audit" not in row


@pytest.mark.parametrize("target", ["reference_child", "dataset_child", "archive_sibling", "parent"])
def test_audit_output_cannot_overlap_bound_inputs(tmp_path, target):
    reference, dataset = _bound_reference(tmp_path)
    output = {"reference_child": reference / "output", "dataset_child": dataset / "output",
              "archive_sibling": dataset.parent / "output", "parent": tmp_path}[target]
    with pytest.raises(ValueError, match="output must be isolated"):
        audit.audit_reference_root(timing_reference=reference, output_dir=output)
    assert not (output / "_INCOMPLETE").exists()


@pytest.mark.parametrize("changed", ["summary", "shard", "implementation"])
def test_changes_during_audit_cannot_publish_success(tmp_path, monkeypatch, changed):
    reference, dataset = _bound_reference(tmp_path)
    output = tmp_path / "output"
    original_audit = audit.audit_song
    original_bindings = audit._implementation_bindings
    modified = False

    def mutate_once(*args, **kwargs):
        nonlocal modified
        result = original_audit(*args, **kwargs)
        if not modified:
            if changed == "summary":
                path = reference / "summary.json"
                path.write_bytes(path.read_bytes() + b" ")
            elif changed == "shard":
                path = dataset / "part.npz"
                path.write_bytes(path.read_bytes() + b"modified")
            modified = True
        return result

    def bindings(*args, **kwargs):
        result = original_bindings(*args, **kwargs)
        if modified and changed == "implementation":
            result["changed_after_start"] = "0" * 64
        return result

    monkeypatch.setattr(audit, "audit_song", mutate_once)
    monkeypatch.setattr(audit, "_implementation_bindings", bindings)
    with pytest.raises(ValueError, match="changed|shard hash mismatch"):
        audit.audit_reference_root(timing_reference=reference, output_dir=output, styles=["preserve"])
    assert (output / "_INCOMPLETE").exists()
    assert not (output / "_SUCCESS").exists()
