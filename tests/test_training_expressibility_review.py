"""Independent edge cases for source retention and configured game budgets."""
from dataclasses import replace

import pytest

from core.compiler import ChordPolicy
from core.humanize import HumanizeParams
from core.profile import GameProfile, legacy_profile
from core.source_score import AdaptOptions, SourceNote, SourceSong
from training import audit_game_expressibility as audit


DELTA = GameProfile(id="delta_force_harmonica", name="Delta",
    chord_policy=ChordPolicy.CHORD_HIGHEST,
    modifier_buttons={"lower": "left", "semitone": "middle", "higher": "right"})


def test_endpoint_retention_does_not_reuse_one_projected_voice_for_duplicates():
    source = [audit.Voice(0, .5, 60, 0), audit.Voice(0, .5, 60, 1)]
    projected = [audit.Voice(0, .5, 60, 0)]
    fidelity = audit._fidelity(source, projected)
    assert fidelity["exact_endpoint_voice_count"] == 1
    assert fidelity["exact_endpoint_voice_fraction"] == .5
    assert fidelity["exact_endpoint_flags"] == [True, False]
    # Union coverage intentionally answers a different pitch-time question,
    # and must not be interpreted as 100 percent independent voice fidelity.
    assert fidelity["pitch_time_coverage_fraction"] == 1


def test_custom_configuration_changes_full_release_budget_without_constants():
    source = SourceSong("switch", [SourceNote(0, .5, 48), SourceNote(.5, 1, 74)],
        {0: "melody"}, 1, 120)
    # All custom values exceed free-play scenario lower bounds.
    settings = audit.PlaybackSettings(hold_ratio=1, gap_ms=27, settle_ms=43, release_settle_ms=31)
    result = audit.audit_song(source, profiles=[DELTA], options=AdaptOptions(style="preserve", track=None), settings=settings)
    projection = next(row for row in result["projection"] if row["scenario_id"] == "free_play" and row["baseline"] == "mechanical")
    assert projection["release_budget"]["full_release_cost_ms"]["mean"] == pytest.approx(601)
    assert projection["release_budget"]["modifier_cost_ms"]["mean"] == pytest.approx(74)
    assert projection["release_budget"]["transition_feasible_flags"] == [False]
    assert projection["capabilities"]["gap_ms"] == 27
    assert projection["capabilities"]["settle_ms"] == 43
    assert projection["capabilities"]["release_settle_ms"] == 31


def test_rule_budget_uses_custom_min_gap_and_does_not_claim_hardware_limit(monkeypatch):
    items = [{"notes": ["low_1"], "dur": 1}, {"notes": ["low_2"], "dur": 1}]
    settings = replace(audit.PlaybackSettings(gap_ms=17), humanize=HumanizeParams(min_gap_ms=80))
    monkeypatch.setattr(audit, "plan_timings", lambda *args, **kwargs: [(0, .5), (0, .5)])
    logical, actual, metadata = audit._delta_projection(items, 120, DELTA, settings, "free_play", rules=True)
    budget = audit._release_budget(logical, actual, metadata)
    assert metadata["min_gap_ms"] == 80
    assert budget["full_release_cost_ms"]["mean"] == pytest.approx(330)
    assert "not measured hardware limit" in budget["scope"]


def test_legacy_chord_keeps_parallel_keys_and_one_transition_per_chord():
    items = [{"notes": ["mid_1", "mid_3", "mid_5"], "dur": 1},
             {"notes": ["mid_2", "mid_4"], "dur": 1}]
    logical, actual, metadata = audit._legacy_projection(items, 120, legacy_profile(), audit.PlaybackSettings(), rules=False)
    assert len(logical) == len(actual) == 5
    assert [voice.start_s for voice in actual[:3]] == [0, 0, 0]
    assert [voice.end_s for voice in actual[:3]] == [.375, .375, .375]
    assert audit._release_budget(logical, actual, metadata)["transition_count"] == 1
    assert metadata["concurrent_chord_keys"] is True
