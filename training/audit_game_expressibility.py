"""Score -> real importer -> profile expressibility audit, with no game I/O.

Legacy 21-key playback is *not* forced through the monophonic compiler. Its
nominal trajectory mirrors Player's score cursor and chord-wide key holds,
without importing Player/Qt/input drivers or claiming measured scheduling.
Delta uses the actual pure IR/compiler and scenario lower bounds. Suggested
window flags are observations, never replacement masks or training labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from core.compiler import Modifier, _flatten, compile_score, resolve_chord, resolve_modifier
from core.humanize import HumanizeParams, plan_timings
from core.ir import IRChord, IRNote, IRRest, from_storage
from core.midi_reader import MidiSourceError, read_midi_source
from core.profile import GameProfile, load_profiles
from core.scenario import get_scenario
from core.score_model import ScoreValidationError
from core.source_score import AdaptOptions, SourceSong, adapt_source_song
from core.transport import MIN_GAME_NOTE_MS, prepare_score
from training.pianocore_dataset import RefinedArchive
from training.tempo_reference import _canonical, _sha256, load_timing_reference


_NATURAL = (0, 2, 4, 5, 7, 9, 11)
_ROOT = Path(__file__).resolve().parents[1]
_EPSILON_S = 1e-7


@dataclass(frozen=True)
class PlaybackSettings:
    hold_ratio: float = .75
    gap_ms: float = 20.0
    settle_ms: float = 30.0
    release_settle_ms: float = 20.0
    # GUI build_event_plan uses None, not CompileParams' 3000 ms default.
    max_hold_ms: float | None = None
    humanize: HumanizeParams = HumanizeParams()
    seed: int = 20261003

    def __post_init__(self):
        if not math.isfinite(self.hold_ratio) or not 0 < self.hold_ratio <= 1:
            raise ValueError("hold_ratio must be finite and in (0, 1]")
        for name in ("gap_ms", "settle_ms", "release_settle_ms"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.max_hold_ms is not None and (not math.isfinite(self.max_hold_ms) or self.max_hold_ms <= 0):
            raise ValueError("max_hold_ms must be positive or None")


@dataclass(frozen=True)
class Voice:
    start_s: float
    end_s: float
    pitch: int
    source_index: int
    key: str = ""
    button: str | None = None


def load_playback_settings(path: str | Path = _ROOT / "config.yaml") -> PlaybackSettings:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["player"]
    human = config.get("humanize", {})
    return PlaybackSettings(hold_ratio=float(config["hold_ratio"]), gap_ms=float(config["gap_ms"]),
        settle_ms=float(config["modifier_settle_ms"]), release_settle_ms=float(config["modifier_release_ms"]),
        humanize=HumanizeParams(**{key: human[key] for key in
            ("jitter_ms", "breath_ms", "jitter_correlation") if key in human}))


def _pitch(note: IRNote, *, octave: int | None = None, semitone: int | None = None) -> int:
    return 60 + 12 * (note.octave if octave is None else octave) + _NATURAL[note.pitch - 1] + (
        note.semitone if semitone is None else semitone)


def imported_voices(items: Sequence[dict], bpm: int) -> list[Voice]:
    result, cursor = [], 0.0
    for index, element in enumerate(from_storage(items)):
        duration = float(element.dur) * 60.0 / bpm
        if not isinstance(element, IRRest):
            notes = element.notes if isinstance(element, IRChord) else (element,)
            result.extend(Voice(cursor, cursor + duration, _pitch(note), index) for note in notes)
        cursor += duration
    return result


def _statistics(values: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    return {"count": len(values), "min": float(values.min()) if len(values) else None,
            "mean": float(values.mean()) if len(values) else None,
            "p95": float(np.quantile(values, .95)) if len(values) else None,
            "max": float(values.max()) if len(values) else None}


def _voice_signature(voice: Voice) -> tuple[int, int, int]:
    # 0.1 us is numerical comparison tolerance, not rhythmic quantization.
    return (round(voice.start_s / _EPSILON_S), round(voice.end_s / _EPSILON_S), voice.pitch)


def _fidelity(source: Sequence[Voice], projected: Sequence[Voice]) -> dict[str, Any]:
    counter = Counter(_voice_signature(voice) for voice in projected)
    exact = []
    for voice in source:
        signature = _voice_signature(voice)
        exact.append(counter[signature] > 0)
        if counter[signature] > 0:
            counter[signature] -= 1
    # Duration coverage measures pitch/voice retention on the unshifted score
    # axis, not execution accuracy. Repeated/duplicate source voices retain
    # separate 1:1 endpoint flags above; coverage itself is pitch-time union.
    intervals = defaultdict(list)
    for voice in projected:
        intervals[voice.pitch].append((voice.start_s, voice.end_s))
    merged, starts = {}, {}
    for pitch, values in intervals.items():
        joined = []
        for left, right in sorted(values):
            if joined and left <= joined[-1][1] + _EPSILON_S:
                joined[-1] = (joined[-1][0], max(joined[-1][1], right))
            else:
                joined.append((left, right))
        merged[pitch] = joined
        starts[pitch] = [value[0] for value in joined]
    covered, total = 0.0, 0.0
    for voice in source:
        total += voice.end_s - voice.start_s
        values = merged.get(voice.pitch, [])
        at = max(0, bisect_left(starts.get(voice.pitch, []), voice.start_s) - 1)
        for left, right in values[at:]:
            if left >= voice.end_s:
                break
            covered += max(0.0, min(voice.end_s, right) - max(voice.start_s, left))
    return {"source_voice_count": len(source), "projected_voice_count": len(projected),
            "exact_endpoint_voice_count": sum(exact),
            "exact_endpoint_voice_fraction": sum(exact) / len(source) if source else None,
            "pitch_time_coverage_fraction": covered / total if total else None,
            "exact_endpoint_flags": exact}


def _legacy_projection(items: Sequence[dict], bpm: int, profile: GameProfile,
                       settings: PlaybackSettings, *, rules: bool) -> tuple[list[Voice], list[Voice], dict]:
    """Static zero-overhead Player timing, preserving concurrent chord voices."""
    if profile.legacy_keymap is None:
        raise ValueError("legacy projection needs legacy_keymap")
    timings = plan_timings(items, settings.humanize, seed=settings.seed) if rules else None
    logical, actual, cursor, prev_release = [], [], 0.0, None
    zero_overhead_clock = 0.0
    beat_s, gap_s = 60.0 / bpm, settings.gap_ms / 1000.0
    nominal_cursor = 0.0
    for index, element in enumerate(from_storage(items)):
        duration = element.dur * beat_s
        notes = element.notes if isinstance(element, IRChord) else (element,) if isinstance(element, IRNote) else ()
        voiced = []
        for note in notes:
            octave = ("low", "mid", "high")[note.octave + 1]
            keys = profile.legacy_keymap.get(octave, ())
            key = str(keys[note.pitch - 1]) if len(keys) > note.pitch - 1 and keys[note.pitch - 1] else ""
            if key:
                # Player.key_for uses note_id only. A stored semitone is ignored.
                voiced.append((_pitch(note, semitone=0), key))
        offset, ratio = timings[index] if timings is not None else (0.0, settings.hold_ratio)
        target = cursor + offset / 1000.0
        if voiced and prev_release is not None:
            target = max(target, prev_release + (settings.humanize.min_gap_ms / 1000.0 if rules else 0.0))
        # Player waits synchronously through holds. A negative/late target
        # cannot dispatch before the previous blocking hold has completed.
        # Its prev_release bookkeeping uses target, not measured dispatch.
        start = max(zero_overhead_clock, target)
        hold = duration * ratio if voiced else 0.0
        for pitch, key in voiced:
            logical.append(Voice(nominal_cursor, nominal_cursor + duration, pitch, index, key))
            actual.append(Voice(start, start + hold, pitch, index, key))
        if voiced:
            prev_release = target + hold
        zero_overhead_clock = start + hold
        nominal_cursor += duration
        cursor += duration + gap_s  # Includes rests, exactly as Player._run.
    return logical, actual, {"evidence": "static Player target and zero-overhead blocking-hold trace; zero latency compensation, not observed scheduler",
        "measured_execution": False, "concurrent_chord_keys": True, "semitone_supported": False,
        "independent_chord_voice_hold": False, "gap_ms": settings.gap_ms,
        "settle_ms": 0.0, "release_settle_ms": 0.0, "max_hold_ms": None,
        "min_gap_ms": settings.humanize.min_gap_ms if rules else 0.0,
        "humanize_enabled": rules, "humanize_seed": settings.seed if rules else None,
        "compiler_degradations": [], "physical_events_checked": False}


def _delta_projection(items: Sequence[dict], bpm: int, profile: GameProfile,
                      settings: PlaybackSettings, scenario_id: str, *, rules: bool) -> tuple[list[Voice], list[Voice], dict]:
    scenario = get_scenario(scenario_id)
    elements = from_storage(items, semitones=[item.get("semitone", 0) for item in items])
    params = scenario.apply_intervals(profile.build_compile_params(bpm=bpm, settle_ms=settings.settle_ms,
        release_settle_ms=settings.release_settle_ms, hold_ratio=settings.hold_ratio,
        max_hold_ms=settings.max_hold_ms, gap_ms=settings.gap_ms))
    enabled = rules and scenario.humanize
    timings = plan_timings(items, settings.humanize, seed=settings.seed) if enabled else None
    if enabled:
        params = replace(params, min_gap_ms=settings.humanize.min_gap_ms)
    plan, _, _ = _flatten(elements, params)
    result = compile_score(elements, params, timings=timings)
    # Resolve symbolic game notes using the same policies, but without
    # interpreting a key event as an empirically measured acoustic pitch.
    logical, cursor = [], 0.0
    for index, element in enumerate(elements):
        if isinstance(element, IRRest):
            cursor += element.dur * 60.0 / bpm
            continue
        resolved = resolve_chord(element, params.chord_policy)[0] if isinstance(element, IRChord) else [element]
        if not resolved:
            cursor += element.dur * 60.0 / bpm
        for note in resolved:
            duration = note.dur * 60.0 / bpm
            modifier, _ = resolve_modifier(note.octave, note.semitone, params.modifier_policy)
            override = params.pitch_direct_overrides.get(note.note_id) if not note.semitone else None
            if modifier is not None:
                button = None if override else params.modifier_buttons.get(modifier)
                key = str(override or params.pitch_keys[note.pitch - 1])
                octave = note.octave if override else -1 if button and modifier is Modifier.LOWER else 1 if button and modifier is Modifier.HIGHER else 0
                semi = 1 if button and modifier is Modifier.SEMITONE else 0
                logical.append(Voice(cursor, cursor + duration, _pitch(note, octave=octave, semitone=semi), index, key, button))
            cursor += duration
    planned = [item for item in plan if item["kind"] == "note"]
    down = [event for event in result.events if event.device == "kb" and event.action == "down"]
    up = [event for event in result.events if event.device == "kb" and event.action == "up"]
    if len(logical) != len(planned) or len(down) != len(logical) or len(up) != len(logical):
        raise ValueError("pure profile resolution and compiler keyboard counts differ")
    actual = []
    for voice, first, last in zip(logical, down, up):
        if (voice.source_index, voice.key) != (first.source_index, first.key) or (
                first.source_index, first.key) != (last.source_index, last.key):
            raise ValueError("profile note/event alignment differs")
        actual.append(replace(voice, start_s=first.t_ms / 1000, end_s=last.t_ms / 1000))
    held, violations = set(), []
    for event in result.events:
        key = event.device, event.key
        if event.action == "down":
            if key in held:
                violations.append("duplicate_down")
            held.add(key)
        else:
            if key not in held:
                violations.append("unpaired_up")
            held.discard(key)
    # scenario.plan can add NPC submit commands. They are not music and are
    # intentionally excluded from note/pitch/timing measurements.
    return logical, actual, {"evidence": "actual pure IR/compiler with scenario lower bounds; no EventPlayer",
        "measured_execution": False, "concurrent_chord_keys": False,
        "semitone_supported": "middle octave only under octave_first, subject to actual configured modifiers",
        "independent_chord_voice_hold": False, "gap_ms": params.gap_ms,
        "settle_ms": params.settle_ms, "release_settle_ms": params.release_settle_ms,
        "max_hold_ms": params.max_hold_ms, "humanize_enabled": enabled,
        "min_gap_ms": params.min_gap_ms if enabled else 0.0,
        "humanize_seed": settings.seed if enabled else None,
        "compiler_degradations": [asdict(value) for value in result.degradations],
        "physical_events_checked": True, "event_count": len(result.events),
        "event_pairing_violations": violations, "residual_held_keys": len(held),
        "npc_submit_excluded": scenario_id == "npc_quest"}


def _release_budget(logical: Sequence[Voice], actual: Sequence[Voice], metadata: dict) -> dict:
    """Nominal full-release costs, not a discovered game/device speed limit."""
    # Chords in the legacy route share one start/release; do not call their
    # concurrent keys zero-IOI sequential notes.
    groups = []
    for first, last in zip(logical, actual):
        if groups and first.source_index == groups[-1][0].source_index and abs(first.start_s - groups[-1][0].start_s) < _EPSILON_S:
            continue
        groups.append((first, last))
    slots, budgets, feasible, modifiers = [], [], [], []
    delays = [(last.start_s - first.start_s) * 1000 for first, last in groups]
    for (first, actual_first), (next_first, _) in zip(groups, groups[1:]):
        slot_ms = (next_first.start_s - first.start_s) * 1000
        hold_ms = (actual_first.end_s - actual_first.start_s) * 1000
        old, new = first.button, next_first.button
        release = metadata["release_settle_ms"] if old and old != new else 0.0
        settle = metadata["settle_ms"] if new and old != new else 0.0
        # Same-modifier latch needs neither release nor settle. Rests cause a
        # full release in compile_score even when the next voice has that same
        # modifier, which is detected using source_index discontinuity.
        if next_first.source_index > first.source_index + 1:
            release = metadata["release_settle_ms"] if old else 0.0
            settle = metadata["settle_ms"] if new else 0.0
        cost_ms = hold_ms + max(metadata["gap_ms"], metadata["min_gap_ms"]) + release + settle
        slots.append(slot_ms)
        budgets.append(cost_ms)
        feasible.append(slot_ms + _EPSILON_S * 1000 >= cost_ms)
        modifiers.append(release + settle)
    return {"scope": "configured nominal hold + gap + modifier-release/settle; not measured hardware limit and not proof of no-drift execution",
        "unshifted_score_ioi_ms": _statistics(slots), "full_release_cost_ms": _statistics(budgets),
        "modifier_cost_ms": _statistics(modifiers), "transition_feasible_flags": feasible,
        "transition_count": len(feasible), "nominal_transition_feasible_count": sum(feasible),
        "compiled_or_static_onset_delay_ms": _statistics(delays),
        "last_onset_delay_ms": delays[-1] if delays else None,
        "source_indices": [voice.source_index for voice, _ in groups],
        "onset_delay_ms": delays}


def _windows(flags: Sequence[bool], width: int) -> dict[str, Any]:
    mask = [all(flags[at:at + width]) for at in range(0, max(0, len(flags) - width + 1))]
    return {"width": width, "count": len(mask), "feasible_count": sum(mask), "flags": mask}


def audit_song(song: SourceSong, *, profiles: Sequence[GameProfile], options: AdaptOptions = AdaptOptions(),
               settings: PlaybackSettings = PlaybackSettings(), scenarios: Sequence[str] = ("free_play", "npc_quest"),
               min_playable_ms: float = MIN_GAME_NOTE_MS) -> dict:
    adapted = adapt_source_song(song, options)
    source = [Voice(note.start_s, note.end_s, note.pitch, index) for index, note in enumerate(song.notes)]
    imported = imported_voices(adapted.notes, adapted.bpm)
    # GUI _prepare_current_score stabilizes every imported MIDI before either
    # playback route. 60 ms is the current software policy, not our estimate
    # of a device/game minimum. No stored user transpose or segment is assumed.
    prepared = prepare_score(adapted.notes, transpose=0, fold_octaves=True, bpm=adapted.bpm,
                             min_playable_ms=min_playable_ms)
    transported = imported_voices(prepared.notes, adapted.bpm)
    source_fidelity = _fidelity(source, imported)
    source_chords = defaultdict(list)
    for index, voice in enumerate(source):
        source_chords[round(voice.start_s / _EPSILON_S)].append(index)
    full_chord_flags = [all(source_fidelity["exact_endpoint_flags"][i] for i in members)
                        for _, members in sorted(source_chords.items())]
    projected = []
    for profile in profiles:
        for scenario_id in scenarios:
            # Legacy Player has no NPC scenario adapter; do not manufacture it.
            if profile.legacy_keymap is not None and scenario_id != "free_play":
                continue
            for baseline in ("mechanical", "rules"):
                if profile.legacy_keymap is not None:
                    logical, actual, metadata = _legacy_projection(prepared.notes, adapted.bpm, profile, settings, rules=baseline == "rules")
                else:
                    logical, actual, metadata = _delta_projection(prepared.notes, adapted.bpm, profile, settings, scenario_id, rules=baseline == "rules")
                budget = _release_budget(logical, actual, metadata)
                final_fidelity = _fidelity(source, logical)
                final_flags = [all(final_fidelity["exact_endpoint_flags"][i] for i in members)
                               for _, members in sorted(source_chords.items())]
                projected.append({"profile_id": profile.id, "scenario_id": scenario_id, "baseline": baseline,
                    "route": "legacy_static" if profile.legacy_keymap is not None else "delta_pure_compiler",
                    "source_to_profile": final_fidelity, "import_to_profile": _fidelity(imported, logical),
                    "transport_to_profile": _fidelity(transported, logical),
                    "key_duration_abs_change_ms": _statistics([abs((last.end_s - last.start_s) - (first.end_s - first.start_s)) * 1000
                        for first, last in zip(logical, actual)]),
                    "capabilities": metadata, "release_budget": budget,
                    "suggested_source_full_fidelity_windows": [_windows(final_flags, width) for width in (1, 8, 32)],
                    "suggested_import_transition_windows": [_windows(budget["transition_feasible_flags"], width) for width in (1, 8, 32)],
                    "logical_voice_count": len(logical), "nominal_actual_voice_count": len(actual)})
    durations = [(note.end_s - note.start_s) * 1000 for note in song.notes]
    starts = sorted({note.start_s for note in song.notes})
    return {"source_note_count": len(source), "source_chord_count": len(starts),
        "source_duration_s": song.duration_s, "source_tracks": dict(song.tracks) if isinstance(song.tracks, dict) else song.tracks,
        "source_out_of_C3_B5_count": sum(not 48 <= note.pitch <= 83 for note in song.notes),
        "source_chromatic_count": sum(note.pitch % 12 not in _NATURAL for note in song.notes),
        "source_short_key_duration_below_40ms": sum(value < 40 for value in durations),
        "source_ioi_ms": _statistics(np.diff(starts) * 1000), "source_key_duration_ms": _statistics(durations),
        "options": asdict(options), "import_bpm": adapted.bpm, "selected_track": adapted.selected_track,
        "import_degradations": [asdict(value) for value in adapted.degradations],
        "import_element_count": len(adapted.notes), "import_rest_count": sum(not item["notes"] for item in adapted.notes),
        "source_to_import": source_fidelity, "import_chord_faithful_count": sum(full_chord_flags),
        "source_to_transport": _fidelity(source, transported), "import_to_transport": _fidelity(imported, transported),
        "transport": {"scope": "whole imported MIDI at import BPM, no playback transpose, no stored user preferences",
            "min_playable_ms": min_playable_ms, "minimum_origin": "current GUI software policy, not measured hardware speed",
            "degradations": [asdict(value) for value in prepared.degradations],
            "element_count": len(prepared.notes), "rest_count": sum(not item["notes"] for item in prepared.notes),
            "axis_end_error_ms": (sum(item["dur"] * 60 / adapted.bpm for item in prepared.notes) - song.duration_s) * 1000},
        "import_duration_s": sum(item["dur"] * 60 / adapted.bpm for item in adapted.notes),
        "source_to_import_axis_end_error_ms": (sum(item["dur"] * 60 / adapted.bpm for item in adapted.notes) - song.duration_s) * 1000,
        "tempo_policy": "source tempo integrated into seconds, then float beat slots at one BPM; tempo metadata flattened but seconds not automatically lost",
        "quantization_policy": "adapter preserves floating second durations; comparison epsilon 0.1 us is not quantization",
        "projection": projected, "deployment_evidence": False, "mask_applied": False}


def _implementation_bindings(profiles_dir: Path, config_path: Path) -> dict[str, str]:
    sources = [Path(__file__), *(_ROOT / "core" / f"{name}.py" for name in
        ("source_score", "midi_reader", "profile", "compiler", "humanize", "scenario", "ir", "player", "keymap", "transport", "score_io", "score_model")),
        *(_ROOT / "training" / f"{name}.py" for name in ("tempo_reference", "pianocore_dataset", "midi_features")),
        _ROOT / "gui/player_tab.py", config_path, *sorted(profiles_dir.glob("*.yaml")), *sorted(profiles_dir.glob("*.yml"))]
    return {str(path.resolve()): _sha256(path) for path in sources}


def _aggregate(rows: Sequence[dict]) -> dict:
    groups = defaultdict(list)
    status_counts = Counter((row["split"], row["performance_dataset"], row["style"], row["status"]) for row in rows)
    for row in rows:
        if row["status"] != "complete":
            continue
        for projection in row["audit"]["projection"]:
            key = (row["split"], row["performance_dataset"], row["style"], projection["profile_id"],
                   projection["scenario_id"], projection["baseline"])
            groups[key].append((row, projection))
    result = []
    for key, values in sorted(groups.items()):
        windows = {str(width): {"count": 0, "feasible_count": 0} for width in (1, 8, 32)}
        source = {str(width): {"count": 0, "feasible_count": 0} for width in (1, 8, 32)}
        for _, projection in values:
            for field, destination in (("suggested_import_transition_windows", windows), ("suggested_source_full_fidelity_windows", source)):
                for window in projection[field]:
                    for name in ("count", "feasible_count"):
                        destination[str(window["width"])][name] += window[name]
        result.append({"split": key[0], "performance_dataset": key[1], "style": key[2], "profile_id": key[3],
            "scenario_id": key[4], "baseline": key[5], "sample_count": len(values),
            "source_piece_group_count": len({row["piece_group"] for row, _ in values}),
            "unique_score_sha256_count": len({row["score_midi_sha256"] for row, _ in values}),
            "source_voice_count": sum(projection["source_to_profile"]["source_voice_count"] for _, projection in values),
            "projected_voice_count": sum(projection["source_to_profile"]["projected_voice_count"] for _, projection in values),
            "exact_endpoint_voice_count": sum(projection["source_to_profile"]["exact_endpoint_voice_count"] for _, projection in values),
            "pitch_time_coverage_fraction_sample_macro": _statistics([
                projection["source_to_profile"]["pitch_time_coverage_fraction"] for _, projection in values
                if projection["source_to_profile"]["pitch_time_coverage_fraction"] is not None]),
            "transition_count": sum(projection["release_budget"]["transition_count"] for _, projection in values),
            "nominal_transition_feasible_count": sum(projection["release_budget"]["nominal_transition_feasible_count"] for _, projection in values),
            "event_pairing_violation_count": sum(len(projection["capabilities"].get("event_pairing_violations", []))
                for _, projection in values),
            "residual_held_key_count": sum(projection["capabilities"].get("residual_held_keys", 0) for _, projection in values),
            "physical_events_checked_sample_count": sum(projection["capabilities"]["physical_events_checked"] for _, projection in values),
            "source_full_fidelity_windows": source, "import_transition_windows": windows,
            "last_onset_delay_ms_sample_macro": _statistics([projection["release_budget"]["last_onset_delay_ms"]
                for _, projection in values if projection["release_budget"]["last_onset_delay_ms"] is not None])})
    return {"scope": "descriptive counts among completed projections; rejected inputs remain in status_groups; not acceptance evidence",
        "status_groups": [{"split": key[0], "performance_dataset": key[1], "style": key[2], "status": key[3],
                           "sample_count": value} for key, value in sorted(status_counts.items())],
        "groups": result}


def audit_reference_root(*, timing_reference: str | Path, output_dir: str | Path,
                         profiles_dir: str | Path = _ROOT / "profiles", config_path: str | Path = _ROOT / "config.yaml",
                         styles: Sequence[str] = ("piano", "preserve")) -> dict:
    root, output = Path(timing_reference).resolve(), Path(output_dir).resolve()
    if root.is_file():
        root = root.parent
    reference_sha256 = _sha256(root / "summary.json")
    load_timing_reference(root)  # source manifest/shards/reference hashes; rejects test references
    if _sha256(root / "summary.json") != reference_sha256:
        raise ValueError("timing reference changed during initial verification")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    source_roots = [root, Path(summary["dataset_root"]).resolve(), Path(summary["archive_path"]).resolve().parent]
    if any(output.is_relative_to(source_root) or source_root.is_relative_to(output) for source_root in source_roots):
        raise ValueError("output must be isolated from timing reference, original dataset, and original archive directory")
    config_path, profiles_dir = Path(config_path).resolve(), Path(profiles_dir).resolve()
    bindings = _implementation_bindings(profiles_dir, config_path)
    profiles = [profile for profile in load_profiles(str(profiles_dir)) if profile.id in {"default", "delta_force_harmonica"}]
    if {profile.id for profile in profiles} != {"default", "delta_force_harmonica"}:
        raise ValueError("audit requires actual default and Delta profile files")
    if not styles or len(set(styles)) != len(styles) or not set(styles).issubset({"piano", "original", "preserve"}):
        raise ValueError("styles must be distinct importer options")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty")
    wanted = {row["id"]: row for row in summary["samples"]}
    selected = []
    with (Path(summary["dataset_root"]) / "manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] in wanted:
                selected.append(row)
    if len(selected) != len(wanted):
        raise ValueError("source selection binding differs")
    settings = load_playback_settings(config_path)
    output.mkdir(parents=True, exist_ok=True)
    (output / "_INCOMPLETE").write_text("game expressibility audit in progress\n", encoding="ascii")
    rows, score_cache = [], {}
    with RefinedArchive(summary["archive_path"]) as archive, tempfile.TemporaryDirectory(prefix="amp_score_audit_") as temporary:
        for position, row in enumerate(sorted(selected, key=lambda value: value["id"])):
            data = archive.read(row["score_midi_path"])
            digest = hashlib.sha256(data).hexdigest()
            if digest != row["score_midi_sha256"] or digest != wanted[row["id"]]["score_midi_sha256"]:
                raise ValueError("bound original score member SHA256 mismatch")
            for style in styles:
                cache_key = digest, style
                if cache_key not in score_cache:
                    original = Path(temporary) / "bound_score.mid"
                    original.write_bytes(data)
                    try:
                        parsed = read_midi_source(original)
                    except MidiSourceError as exc:
                        score_cache[cache_key] = {"status": "import_rejected", "error": str(exc)}
                    else:
                        try:
                            # Only parser/adapter validation errors are a
                            # descriptive rejection. A compiler/projection
                            # bug must fail closed and leave _INCOMPLETE.
                            adapted = adapt_source_song(parsed.song, AdaptOptions(style=style))
                        except ValueError as exc:
                            score_cache[cache_key] = {"status": "import_rejected", "error": str(exc)}
                        else:
                            try:
                                # Preparation is a distinct production-stage
                                # validation gate. In particular, short-fragment
                                # absorption can exceed the 16-beat storage cap.
                                # Record that refusal without fixing source data
                                # or swallowing compiler/projection exceptions.
                                prepare_score(adapted.notes, transpose=0, fold_octaves=True, bpm=adapted.bpm,
                                              min_playable_ms=MIN_GAME_NOTE_MS)
                            except ScoreValidationError as exc:
                                score_cache[cache_key] = {"status": "transport_rejected", "error": str(exc),
                                    "transport_errors": [str(value) for value in exc.errors],
                                    "transport_min_playable_ms": MIN_GAME_NOTE_MS,
                                    "source_note_count": len(parsed.song.notes), "source_duration_s": parsed.song.duration_s,
                                    "import_bpm": adapted.bpm, "selected_track": adapted.selected_track,
                                    "import_element_count": len(adapted.notes),
                                    "import_degradations": [asdict(value) for value in adapted.degradations]}
                    if cache_key not in score_cache:
                        audit = audit_song(parsed.song, profiles=profiles, options=AdaptOptions(style=style), settings=settings)
                        audit["reader_warnings"] = parsed.warnings
                        audit["source_ppq"] = parsed.source_metadata["ticks_per_beat"]
                        audit["source_explicit_tempo_event_count"] = len(parsed.source_metadata["tempo_events"])
                        score_cache[cache_key] = {"status": "complete", "audit": audit}
                item = {"id": row["id"], "split": row["split"], "performance_dataset": row["performance_dataset"],
                    "piece_group": row["piece_group"], "style": style, "score_midi_path": row["score_midi_path"],
                    "score_midi_sha256": digest, **score_cache[cache_key]}
                relative = f"samples/{hashlib.sha256((row['id'] + ':' + style).encode()).hexdigest()}.json"
                destination = output / relative
                destination.parent.mkdir(exist_ok=True)
                destination.write_bytes(_canonical(item) + b"\n")
                rows.append({**item, "path": relative, "sha256": _sha256(destination)})
            if (position + 1) % 25 == 0 or position + 1 == len(selected):
                print(f"game_expressibility={position + 1}/{len(selected)}", flush=True)
    if _sha256(root / "summary.json") != reference_sha256:
        raise ValueError("timing reference summary changed during audit; incomplete output must not be used")
    load_timing_reference(root)  # Freeze selected source manifest/shards/NPZs again before sealing.
    if _sha256(root / "summary.json") != reference_sha256:
        raise ValueError("timing reference changed during final verification")
    if bindings != _implementation_bindings(profiles_dir, config_path):
        raise ValueError("source/config changed during audit; incomplete output must not be used")
    report = {"schema_version": 1, "kind": "game_expressibility_audit", "status": "complete",
        "timing_reference_root": str(root), "timing_reference_summary_sha256": reference_sha256,
        "sample_count": len(selected), "sample_style_count": len(rows), "styles": list(styles),
        "implementation_and_config_sha256": bindings, "playback_settings": asdict(settings),
        "status_counts": dict(Counter(row["status"] for row in rows)), "aggregate": _aggregate(rows),
        "samples": [{key: row[key] for key in ("id", "split", "performance_dataset", "piece_group", "style", "score_midi_path",
            "score_midi_sha256", "status", "path", "sha256")} for row in rows],
        "policy": "suggested full-score fidelity and imported safety windows only; original native masks unchanged; no filtering or training",
        "logical_pitch_policy": "C-based storage interpretation, not measured in-game concert tuning; octave folds count as source loss",
        "scope": "same real SourceSong/adapt_source_song/GUI MIDI transport policy/profile code, pure Delta compiler or separate legacy static trace",
        "acceptance": {"game_input_sent": False, "actual_scheduler_measured": False, "training_mask_applied": False,
            "production_approved": False, "human_preferences_present": False}}
    (output / "summary.json").write_bytes(_canonical(report) + b"\n")
    (output / "_SUCCESS").write_text(_sha256(output / "summary.json") + "\n", encoding="ascii")
    (output / "_INCOMPLETE").unlink()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profiles-dir", default=str(_ROOT / "profiles"))
    parser.add_argument("--config", default=str(_ROOT / "config.yaml"))
    parser.add_argument("--styles", nargs="+", default=["piano", "preserve"])
    args = parser.parse_args()
    report = audit_reference_root(timing_reference=args.timing_reference, output_dir=args.output,
        profiles_dir=args.profiles_dir, config_path=args.config, styles=args.styles)
    print(json.dumps({"sample_count": report["sample_count"], "status_counts": report["status_counts"],
                      "summary_sha256": _sha256(Path(args.output) / "summary.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
