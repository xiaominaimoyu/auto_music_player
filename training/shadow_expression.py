"""Offline score -> safety -> production compiler audit; never sends input.

This is NOT a production MIDI importer or playback integration. Its C-major
pitch projection and one-voice-per-chord reduction are explicitly disclosed.
The returned events are only compiler output, not evidence of real execution.
No performance reference, labels, torch, Qt, or input driver is consulted.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from core.compiler import ChordPolicy, CompileParams, _flatten, compile_score
from core.humanize import HumanizeParams, plan_timings
from core.ir import IRChord, IRNote, IRRest, MAX_DUR_BEATS, to_storage
from core.scenario import get_scenario


_DIATONIC = {0: (1, 0), 1: (1, 1), 2: (2, 0), 3: (2, 1),
             4: (3, 0), 5: (4, 0), 6: (4, 1), 7: (5, 0),
             8: (5, 1), 9: (6, 0), 10: (6, 1), 11: (7, 0)}


def _score_arrays(score, mapping):
    mapping = np.asarray(mapping)
    if not score.notes or mapping.shape != (len(score.notes),) or mapping.dtype.kind not in "iu":
        raise ValueError("nonempty score and integer note-to-chord mapping required")
    ticks = sorted({note.start_tick for note in score.notes})
    expected = {tick: index for index, tick in enumerate(ticks)}
    if any(mapping[i] != expected[n.start_tick] for i, n in enumerate(score.notes)):
        raise ValueError("mapping does not match original score chord order")
    groups = [np.flatnonzero(mapping == index) for index in range(len(ticks))]
    onsets = np.array([score.notes[group[0]].start_s for group in groups], dtype=np.float64)
    durations = np.array([note.duration_s for note in score.notes], dtype=np.float64)
    if (not np.isfinite(onsets).all() or not np.isfinite(durations).all()
            or np.any(onsets < 0) or np.any(durations <= 0) or np.any(np.diff(onsets) <= 0)):
        raise ValueError("score contains invalid absolute timing or zero-duration notes")
    for group, start in zip(groups, onsets):
        if any(abs(score.notes[i].start_s - start) > 1e-8 for i in group):
            raise ValueError("simultaneous chord members disagree on start seconds")
    return groups, onsets, durations


def _pitch_ir(pitch, duration_beats):
    degree, semitone = _DIATONIC[int(pitch) % 12]
    original_octave = int(pitch) // 12 - 5  # C4=60 -> middle octave.
    octave = min(1, max(-1, original_octave))
    return IRNote(degree, octave, semitone, duration_beats), octave != original_octave


def _nominal_ir(score, groups, onsets, durations, params):
    beat_s = 60.0 / float(params.bpm)
    elements, element_to_chord, chord_to_element = [], [], []
    slot_s, selected, folded = [], [], []
    long_note_truncations = 0
    active_until = 0.0

    def rest(seconds):
        # Rest splitting preserves arbitrary long silence without violating IR limits.
        while seconds > 1e-10:
            amount = min(seconds, MAX_DUR_BEATS * beat_s)
            elements.append(IRRest(amount / beat_s))
            element_to_chord.append(-1)
            seconds -= amount

    rest(float(onsets[0]))
    for chord, group in enumerate(groups):
        # A long bass from an earlier onset can still sound under a later short
        # melody note. Only absence of ALL score voices defines a genuine rest;
        # do not hallucinate rest/breath markers from the latest chord alone.
        active_until = max(active_until, float(np.max(onsets[chord] + durations[group])))
        maximum = active_until - float(onsets[chord])
        interval = float(onsets[chord + 1] - onsets[chord]) if chord + 1 < len(groups) else maximum
        slot = min(maximum, interval)
        duration = min(slot, MAX_DUR_BEATS * beat_s)
        long_note_truncations += int(duration < slot - 1e-10)
        notes, fold_flags = [], []
        for index in group:
            note, is_folded = _pitch_ir(score.notes[index].pitch, duration / beat_s)
            notes.append(note)
            fold_flags.append(is_folded)
        if params.chord_policy is ChordPolicy.CHORD_HIGHEST:
            chosen = max(range(len(notes)), key=lambda i: (notes[i].octave, notes[i].pitch, notes[i].semitone))
        else:
            chosen = 0
        selected.append(int(group[chosen]))
        folded.extend(int(group[i]) for i, flag in enumerate(fold_flags) if flag)
        chord_to_element.append(len(elements))
        elements.append(notes[0] if len(notes) == 1 else IRChord(tuple(notes), duration / beat_s))
        element_to_chord.append(chord)
        slot_s.append(duration)
        # Preserve genuine score silence and any disclosed long-note truncation.
        rest(max(0.0, interval - duration))
    return (elements, np.asarray(element_to_chord, dtype=np.int64),
            np.asarray(chord_to_element, dtype=np.int64), np.asarray(slot_s),
            np.asarray(selected, dtype=np.int64), np.asarray(folded, dtype=np.int64),
            long_note_truncations)


def _absolute_timings(elements, params, desired_onsets, desired_holds, element_to_chord):
    """Translate absolute targets using compiler cursor bookkeeping, not jitter.

    _flatten is the real compiler's pitch/chord resolution. Only cursor and
    modifier costs are tracked here; event creation remains compile_score's
    responsibility. The resulting keyboard onsets are checked against this
    bookkeeping, making changes to compiler behavior fail loudly.
    """
    plan, _, _ = _flatten(elements, params)
    timings = [(0.0, None) for _ in elements]
    expected_onsets = np.full(len(desired_onsets), np.nan)
    cursor, held, previous_up, previous_release_end = 0.0, None, None, None
    beat_ms = 60000.0 / float(params.bpm)
    gap, settle, release = float(params.gap_ms), float(params.settle_ms), float(params.release_settle_ms)
    for index, item in enumerate(plan):
        if item["kind"] == "rest":
            if held:
                cursor += release
                held = None
            cursor += item["dur"] * beat_ms + gap
            continue
        chord = int(element_to_chord[item["index"]])
        button = item["button"]
        modifier_cost = (release if held else 0.0) + settle if button and button != held else 0.0
        preparation = max(0.0, float(desired_onsets[chord] * 1000 - modifier_cost))
        # The production compiler guards only keyboard release. Cancelling its
        # logical cursor with an absolute offset must not reorder a later mouse
        # release after a new modifier down. Constrain the whole physical chain.
        if previous_release_end is not None:
            preparation = max(preparation, previous_release_end + max(gap, float(params.min_gap_ms)))
        offset = preparation - cursor
        hold = float(desired_holds[chord] * 1000)
        timings[item["index"]] = (offset, hold / (item["dur"] * beat_ms))
        start = max(0.0, cursor + offset)
        if previous_up is not None:
            start = max(start, previous_up + float(params.min_gap_ms))
        nominal_start = cursor
        if button and button != held:
            start += modifier_cost
            nominal_start += modifier_cost
            held = button
        if params.max_hold_ms is not None:
            hold = min(hold, float(params.max_hold_ms))
        end = start + hold
        previous_up = end
        following = plan[index + 1] if index + 1 < len(plan) else None
        next_button = following["button"] if following and following["kind"] == "note" else None
        release_end, nominal_release_end = end, nominal_start + hold
        if held and held != next_button:
            release_end += release
            nominal_release_end += release
            held = None
        previous_release_end = release_end
        expected_onsets[chord] = start / 1000.0
        cursor = max(nominal_start + item["dur"] * beat_ms + gap,
                     nominal_release_end + gap, release_end + gap)
    return timings, expected_onsets


def _event_arrays(events, chord_to_element):
    """Validate a complete compiler event sequence without operating a device."""
    active = {}
    onset, hold = np.full(len(chord_to_element), np.nan), np.full(len(chord_to_element), np.nan)
    element_to_chord = {int(element): i for i, element in enumerate(chord_to_element)}
    previous_time = -1.0
    for event in events:
        if not np.isfinite(event.t_ms) or event.t_ms < previous_time or event.t_ms < 0:
            raise ValueError("compiler produced invalid event time")
        previous_time = event.t_ms
        key = (event.device, event.key)
        if event.action == "down":
            if key in active:
                raise ValueError("compiler produced a duplicate held-key down")
            active[key] = event
            if event.device == "kb" and event.source_index in element_to_chord:
                chord = element_to_chord[event.source_index]
                if np.isfinite(onset[chord]):
                    raise ValueError("audit only supports one keyboard voice per chord")
                onset[chord] = event.t_ms / 1000.0
        elif event.action == "up":
            down = active.pop(key, None)
            if down is None:
                raise ValueError("compiler produced an unmatched release")
            if event.device == "kb" and down.source_index in element_to_chord:
                chord = element_to_chord[down.source_index]
                hold[chord] = (event.t_ms - down.t_ms) / 1000.0
        else:
            raise ValueError("unknown compiler event action")
    if active:
        raise ValueError("compiler events leave held keys or modifiers")
    return onset, hold


def _actual_preparation_gaps(events, chord_to_element):
    """Measure full-release -> next preparation from ACTUAL compiler events.

    A held modifier shared between adjacent notes is not released/repressed;
    the previous keyboard release remains that transition's boundary. A new
    modifier press counts as preparation before the next keyboard onset.
    Native identity/rules are measured as-is, never rewritten to pass this test.
    """
    starts, releases = {}, {}
    for event in events:
        source = event.source_index
        if event.action == "down":
            starts[source] = min(starts.get(source, event.t_ms), event.t_ms)
        elif event.action == "up":
            releases[source] = max(releases.get(source, event.t_ms), event.t_ms)
    represented = [int(source) for source in chord_to_element if int(source) in starts]
    return np.asarray([starts[following] - releases[previous]
                       for previous, following in zip(represented, represented[1:])], dtype=float)


def shadow_projection(score, note_to_chord, note_prediction=None, chord_prediction=None,
                      *, strategy="identity", seed=20261003, params=None, strength=0.5,
                      scenario_id="free_play", humanize=None, cumulative_onset_limit_s=0.15):
    """Return an auditable, deterministic offline projection dictionary.

    ``identity`` and ``rules`` use the native production compiler behavior.
    ``model`` translates absolute score-tempo onsets into compiler offsets, so
    additionally test a zero-prediction model control to isolate adapter effects.
    Arrays are per ORIGINAL score chord; skipped events are NaN, never zero.
    A default call ignores predictions and does not activate any model.
    """
    if strategy not in {"identity", "rules", "model"}:
        raise ValueError("unknown shadow strategy")
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength must be finite and in [0,1]")
    if not np.isfinite(cumulative_onset_limit_s) or cumulative_onset_limit_s < 0:
        raise ValueError("cumulative onset limit must be finite and nonnegative")
    params = params or CompileParams()
    scenario = get_scenario(scenario_id)
    if strategy != "identity" and not scenario.humanize:
        raise ValueError("expressive projection forbidden for strict NPC/recognition scenarios")
    params = scenario.apply_intervals(params)
    if params.chord_policy is ChordPolicy.CHORD_ARPEGGIATE:
        raise ValueError("arpeggiation requires a separate many-events-per-chord audit")
    if not np.isfinite([params.settle_ms, params.release_settle_ms, params.gap_ms,
                        params.min_gap_ms, params.hold_ratio]).all():
        raise ValueError("non-finite compiler parameters")
    if params.max_hold_ms is not None and not np.isfinite(params.max_hold_ms):
        raise ValueError("non-finite max hold")
    groups, score_onsets, duration = _score_arrays(score, note_to_chord)
    elements, element_map, chord_map, slot_s, selected, folded, truncations = _nominal_ir(
        score, groups, score_onsets, duration, params)
    original_holds = duration[selected]
    raw_onsets, raw_holds = score_onsets.copy(), original_holds.copy()
    desired_onsets = score_onsets.copy()
    desired_holds = slot_s * float(params.hold_ratio)
    clipping = {"ioi": 0, "duration_ratio": 0, "hold_floor": 0, "hold_slot": 0,
                "hold_max": 0, "raw_numeric_exponent_cap": 0, "cumulative_onset_budget": 0}
    max_cumulative_before = max_cumulative_after = 0.0
    timings, expected = None, None
    if strategy == "rules":
        rule_params = humanize or HumanizeParams()
        params = replace(params, min_gap_ms=float(rule_params.min_gap_ms))
        timings = plan_timings(to_storage(elements), rule_params, seed=int(seed))
        offsets = np.array([timings[int(element)][0] for element in chord_map])
        ratios = np.array([timings[int(element)][1] for element in chord_map])
        desired_onsets = np.maximum(0, score_onsets + offsets / 1000)
        desired_holds = slot_s * ratios
        raw_onsets, raw_holds = desired_onsets.copy(), desired_holds.copy()
    elif strategy == "model":
        ny, cy = np.asarray(note_prediction), np.asarray(chord_prediction)
        if ny.shape != (len(score.notes), 6) or cy.shape != (len(groups), 2):
            raise ValueError("prediction heads must have original note/chord shapes")
        if ny.dtype.kind not in "fiu" or cy.dtype.kind not in "fiu":
            raise ValueError("prediction heads must be real numeric arrays")
        if not np.isfinite(ny).all() or not np.isfinite(cy).all():
            raise ValueError("prediction heads must be finite")
        log_ioi, log_duration = cy[:, 1], ny[selected, 1]
        intervals = np.diff(score_onsets, prepend=score_onsets[0])
        raw_onsets = score_onsets[0] + np.cumsum(intervals * np.exp(np.clip(log_ioi, -50, 50)))
        raw_holds = original_holds * np.exp(np.clip(log_duration, -50, 50))
        clipping["raw_numeric_exponent_cap"] = int(np.sum(np.abs(log_ioi) > 50) + np.sum(np.abs(log_duration) > 50))
        bounded_ioi = np.clip(log_ioi, -0.25, 0.25)
        bounded_duration = np.clip(log_duration, np.log(0.4), np.log(1.6))
        clipping["ioi"] = int(np.sum(bounded_ioi != log_ioi))
        clipping["duration_ratio"] = int(np.sum(bounded_duration != log_duration))
        desired_onsets = score_onsets[0] + np.cumsum(intervals * np.exp(bounded_ioi * strength))
        # Score-only absolute drift budget; it never resets clocks or anchors to
        # a performer. Clipping strictly increasing times to strictly increasing
        # score-centered bands preserves strictly increasing chord onsets.
        offsets = desired_onsets - score_onsets
        bounded_offsets = np.clip(offsets, -cumulative_onset_limit_s, cumulative_onset_limit_s)
        clipping["cumulative_onset_budget"] = int(np.sum(np.abs(offsets - bounded_offsets) > 1e-12))
        max_cumulative_before = float(np.max(np.abs(offsets)))
        desired_onsets = score_onsets + bounded_offsets
        if np.any(np.diff(desired_onsets) <= 0):
            raise ValueError("cumulative safety shaping produced nonmonotonic chord onsets")
        max_cumulative_after = float(np.max(np.abs(desired_onsets - score_onsets)))
        desired_holds = original_holds * np.exp(bounded_duration * strength)
        clipping["hold_floor"] = int(np.sum(desired_holds < 0.04))
        desired_holds = np.maximum(desired_holds, 0.04)
        clipping["hold_slot"] = int(np.sum(desired_holds > slot_s))
        desired_holds = np.minimum(desired_holds, slot_s)
        if params.max_hold_ms is not None:
            maximum = float(params.max_hold_ms) / 1000
            clipping["hold_max"] = int(np.sum(desired_holds > maximum))
            desired_holds = np.minimum(desired_holds, maximum)
        timings, expected = _absolute_timings(elements, params, desired_onsets, desired_holds, element_map)
    result = compile_score(elements, params, timings=timings)
    actual_onsets, actual_holds = _event_arrays(result.events, chord_map)
    actual_gaps = _actual_preparation_gaps(result.events, chord_map)
    required_gap = max(float(params.gap_ms), float(params.min_gap_ms))
    gap_violations = int(np.sum(actual_gaps < required_gap - 1e-8))
    if strategy == "model" and gap_violations:
        raise ValueError("absolute adapter violated full-release preparation gap")
    coverage = np.isfinite(actual_onsets) & np.isfinite(actual_holds)
    if expected is not None and not np.allclose(actual_onsets, expected, atol=1e-9, equal_nan=True):
        raise ValueError("compiler cursor behavior changed; absolute adapter requires review")
    degradation = [{"index": d.index, "requested": d.requested, "actual": d.actual,
                    "reason": d.reason} for d in result.degradations]
    return {
        "strategy": strategy, "scope": "offline_compiler_shadow_only",
        "uses_oracle_performance": False, "model_enabled_by_default": False,
        "scenario_id": scenario_id, "seed": int(seed), "strength": float(strength),
        "events": result.events, "elements": elements, "compile_params": params,
        "score_onsets_s": score_onsets, "score_intervals_s": np.diff(score_onsets, prepend=score_onsets[0]),
        "score_selected_key_holds_s": original_holds, "slot_holds_s": slot_s,
        "raw_onsets_s": raw_onsets, "raw_holds_s": raw_holds,
        "desired_onsets_s": desired_onsets, "desired_holds_s": desired_holds,
        "actual_onsets_s": actual_onsets, "actual_holds_s": actual_holds,
        "selected_note_indices": selected, "chord_to_element": chord_map,
        "element_to_chord": element_map, "coverage_mask": coverage,
        "source_chord_count": len(groups), "covered_chord_count": int(np.sum(coverage)),
        "source_note_count": len(score.notes), "projected_voice_count": int(np.sum(coverage)),
        "unrepresented_source_voices": int(len(score.notes) - np.sum(coverage)),
        "folded_note_indices": folded, "long_note_truncations": truncations,
        "rest_element_count": sum(isinstance(item, IRRest) for item in elements),
        "rest_semantics": "complete source-key silence; prior overlapping voices considered; sustain pedal unsupported",
        "clipping": clipping, "degradations": degradation,
        "compiler_delayed_chords": int(np.sum(coverage & (actual_onsets > desired_onsets + 1e-9))),
        "max_compiler_onset_change_ms": float(np.max(np.abs(actual_onsets[coverage] - desired_onsets[coverage])) * 1000) if np.any(coverage) else None,
        "max_actual_score_offset_s": float(np.max(np.abs(actual_onsets[coverage] - score_onsets[coverage]))) if np.any(coverage) else None,
        "actual_cumulative_budget_exceeded_chords": int(np.sum(coverage & (np.abs(actual_onsets - score_onsets) > cumulative_onset_limit_s + 1e-9))),
        "pitch_projection": "C-major degrees; out-of-range octaves folded; compiler policies disclosed",
        "hold_aggregation": "one common chord hold; model selects compiler first/highest voice",
        "baseline_offset_mode": "absolute_cursor_translation" if strategy == "model" else "native_compiler",
        "required_model_full_release_gap_ms": required_gap,
        "minimum_actual_preparation_gap_ms": float(np.min(actual_gaps)) if len(actual_gaps) else None,
        "actual_full_release_gap_violation_count": gap_violations,
        "actual_inter_event_gap_policy": {
            "scope": "previous full keyboard/modifier release to next keyboard/modifier preparation",
            "effective_compile_gap_ms": float(params.gap_ms),
            "effective_min_gap_ms": float(params.min_gap_ms),
            "absolute_model_enforced": strategy == "model",
            "native_identity_rules_unchanged": True,
        },
        "unsupported_expression": ["individual_chord_voices", "micro_offset", "velocity", "sustain_pedal"],
        "requires_zero_model_adapter_control": True,
        "cumulative_onset_limit_s": float(cumulative_onset_limit_s),
        "max_cumulative_offset_before_budget_s": max_cumulative_before,
        "max_cumulative_offset_after_budget_s": max_cumulative_after,
        "cumulative_budget_scope": "score-only desired timeline; compiler physical delays audited separately; engineering knob not musical proof",
        "all_events_balanced": True, "actual_game_input_sent": False,
    }
