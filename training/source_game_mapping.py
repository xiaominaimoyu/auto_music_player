"""Conservative, diagnostic-only native-note to game-fragment provenance.

Indices from the training parser and production MIDI reader are different
domains.  Only reciprocal unique endpoint identities may cross that boundary.
Adapter origins are traced before pitch folding, and every later whole-slot
transformation invalidates label eligibility.  No returned mask is installed
in a training dataset and no performance target is emitted.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Mapping, Sequence

import numpy as np

from core.midi_reader import MidiReadResult
from core.profile import GameProfile
from core.score_io import note_id_to_midi
from core.source_score import (
    AdaptOptions, SourceNote, _active_intervals, _append_storage_item,
    _bridge_short_gaps_with_count, _map_pitch, _output_bpm, _preserve_segments,
    _selected_highest_segments, _selected_source_notes, adapt_source_song,
)
from core.transport import MIN_GAME_NOTE_MS, prepare_score
from training.audit_game_expressibility import (
    PlaybackSettings, Voice, _delta_projection, _legacy_projection, imported_voices,
)
from training.midi_features import parse_midi_bytes

EPSILON_S = 1e-7


def _equal(left: float, right: float, epsilon: float = EPSILON_S) -> bool:
    return abs(left - right) <= epsilon


def join_unique_voices(source_voices: Sequence, target_voices: Sequence, *,
                       allowed_source_members: Sequence[Sequence[int]] | None = None,
                       epsilon_s: float = EPSILON_S) -> dict[str, np.ndarray]:
    """Match pitch/onset/key-up only when both candidate degrees equal one.

    The allowed sets are target-indexed original provenance, not nearest-note
    hints.  Continuous endpoint comparisons include adjacent numeric buckets;
    duplicate signatures invalidate every participant instead of picking one.
    """
    if not np.isfinite(epsilon_s) or epsilon_s < 0:
        raise ValueError("epsilon_s must be finite and nonnegative")
    if allowed_source_members is not None and len(allowed_source_members) != len(target_voices):
        raise ValueError("allowed_source_members must be target-indexed")
    for voice in (*source_voices, *target_voices):
        if not np.isfinite([voice.start_s, voice.end_s]).all() or voice.end_s <= voice.start_s:
            raise ValueError("voices must have finite positive intervals")
    by_pitch = defaultdict(list)
    for index, voice in enumerate(source_voices):
        by_pitch[int(voice.pitch)].append((float(voice.start_s), index))
    for values in by_pitch.values():
        values.sort()
    starts = {pitch: [entry[0] for entry in values] for pitch, values in by_pitch.items()}
    source_count = np.zeros(len(source_voices), dtype=np.int64)
    target_count = np.zeros(len(target_voices), dtype=np.int64)
    candidates = []
    for index, voice in enumerate(target_voices):
        allowed = None if allowed_source_members is None else set(allowed_source_members[index])
        if allowed is not None and any(isinstance(value, bool) or not isinstance(value, (int, np.integer))
                                       or not 0 <= value < len(source_voices) for value in allowed):
            raise ValueError("allowed source index out of range")
        values = by_pitch.get(int(voice.pitch), ())
        times = starts.get(int(voice.pitch), ())
        first = bisect_left(times, float(voice.start_s) - epsilon_s)
        last = bisect_right(times, float(voice.start_s) + epsilon_s)
        matches = [position for _, position in values[first:last]
                   if (allowed is None or position in allowed)
                   and _equal(float(source_voices[position].start_s), float(voice.start_s), epsilon_s)
                   and _equal(float(source_voices[position].end_s), float(voice.end_s), epsilon_s)]
        candidates.append(matches)
        target_count[index] = len(matches)
        for position in matches:
            source_count[position] += 1
    forward = np.full(len(source_voices), -1, dtype=np.int64)
    reverse = np.full(len(target_voices), -1, dtype=np.int64)
    for index, matches in enumerate(candidates):
        if len(matches) == 1 and source_count[matches[0]] == 1:
            forward[matches[0]] = index
            reverse[index] = matches[0]
    reason = np.asarray(["exact" if value >= 0 else "no_endpoint_identity" if target_count[i] == 0
                         else "ambiguous_endpoint_identity" for i, value in enumerate(reverse)], dtype="U40")
    offsets = np.cumsum([0, *(len(values) for values in candidates)], dtype=np.int64)
    return {"source_to_target": forward, "target_to_source": reverse,
            "source_candidate_count": source_count, "target_candidate_count": target_count,
            "target_reason": reason, "candidate_offsets": offsets,
            "candidate_indices": np.asarray([value for values in candidates for value in values], dtype=np.int64)}


def _slots(items: Sequence[dict], bpm: float) -> list[dict]:
    voices = imported_voices(items, bpm)
    grouped = defaultdict(list)
    for voice in voices:
        grouped[voice.source_index].append(voice)
    cursor, result = 0.0, []
    for index, item in enumerate(items):
        end = cursor + float(item["dur"]) * 60.0 / bpm
        result.append({"start": cursor, "end": end, "voices": grouped[index],
                       "pitches": tuple(sorted(voice.pitch for voice in grouped[index]))})
        cursor = end
    return result


def _trace_import(song, options: AdaptOptions, adapted) -> tuple[list[dict], list[dict]]:
    """Reuse adapter segment members, reproduce storage, and verify the result."""
    selected, track = _selected_source_notes(song, options)
    original_indices = [index for index, note in enumerate(song.notes) if track is None or note.track == track]
    if selected != [song.notes[index] for index in original_indices]:
        raise ValueError("selected source ordering differs")
    segments, _ = (_preserve_segments(selected) if options.style == "preserve"
                   else _selected_highest_segments(selected))
    # Mark every region where the adapter loses an active voice, even when
    # the winning voice happens to retain its complete original endpoints.
    lost_regions = []
    for left, right, active, groups, _ in _active_intervals(selected):
        if (options.style != "preserve" and len(active) > 1) or (options.style == "preserve" and len(groups) > 1):
            lost_regions.append((left, right))
    region_ends = [right for _, right in lost_regions]
    bridged = None
    if options.style == "piano":
        bridged, _ = _bridge_short_gaps_with_count([
            SourceNote(segment.start_s, segment.end_s, selected[segment.members[0]].pitch,
                       selected[segment.members[0]].track) for segment in segments])
    rendered = []
    for position, segment in enumerate(segments):
        left, right = segment.start_s, segment.end_s
        reasons = set()
        if any(not _equal(selected[index].start_s, left) or not _equal(selected[index].end_s, right)
               for index in segment.members):
            reasons.add("source_segment_clipped")
        at = bisect_right(region_ends, left + EPSILON_S)
        if at < len(lost_regions) and lost_regions[at][0] < right - EPSILON_S:
            reasons.add("polyphony_reduced")
        if bridged is not None:
            right = bridged[position].end_s
            if not _equal(right, segment.end_s):
                reasons.add("short_gap_bridged")
        mapped = []
        for index in segment.members:
            value = _map_pitch(selected[index].pitch, transpose=options.transpose, fold_octaves=options.fold_octaves)
            if value is None:
                reasons.add("unsupported_pitch")
                continue
            note_id, semi, folded = value
            if folded:
                reasons.add("octave_folded")
            if options.transpose:
                reasons.add("transposed")
            mapped.append((selected[index].pitch + options.transpose, note_id, semi, original_indices[index]))
        all_origins = tuple(original_indices[index] for index in segment.members)
        if not mapped:
            rendered.append((left, right, [], 0, {}, all_origins, reasons))
            continue
        if len(mapped) > 1 and len({value[2] for value in mapped}) > 1:
            mapped = [max(mapped, key=lambda value: value[0])]
            reasons.add("mixed_semitone_chord_reduced")
        mapped.sort(key=lambda value: value[0])
        ids, origins = [], defaultdict(list)
        for _, note_id, _, original in mapped:
            if note_id not in ids:
                ids.append(note_id)
            else:
                reasons.add("duplicate_pitch_reduced")
            origins[note_id].append(original)
        rendered.append((left, right, ids, mapped[0][2], dict(origins), all_origins, reasons))
    expected, provenance = [], []
    bpm, cursor = _output_bpm(song, options), 0.0
    def append(notes, beats, semi=0, origins=None, all_origins=(), reasons=()):
        old = len(expected)
        _append_storage_item(expected, notes, beats, semi)
        split = len(expected) - old > 1
        for _ in expected[old:]:
            provenance.append({"origins": origins or {}, "all_origins": tuple(all_origins),
                               "reasons": set(reasons) | ({"duration_split"} if split else set())})
    for left, right, ids, semi, origins, all_origins, reasons in rendered:
        if left > cursor + 1e-9:
            append([], (left - cursor) * (bpm / 60.0))
        event_start = max(left, cursor)
        if right > event_start + 1e-9:
            # Preserve the production multiplication order for exact replay.
            append(ids, (right - event_start) * (bpm / 60.0), semi, origins, all_origins, reasons)
            cursor = right
    if song.duration_s > cursor + 1e-9:
        append([], (song.duration_s - cursor) * (bpm / 60.0))
    # Rest conversion in production also uses beats_per_second multiplication.
    if len(expected) != len(adapted.notes) or any(
            first.get("notes") != second.get("notes")
            or first.get("semitone", 0) != second.get("semitone", 0)
            or abs(float(first["dur"]) - float(second["dur"])) > 1e-10
            for first, second in zip(expected, adapted.notes)):
        raise ValueError("provenance replay differs from actual adapter storage")
    slots = _slots(adapted.notes, adapted.bpm)
    for slot, trace in zip(slots, provenance):
        slot.update(trace)
    return slots, provenance


def _row(voice, origins, native_index=-1, reason="unknown") -> dict:
    origins = tuple(sorted(set(origins)))
    return {"voice": voice, "origins": origins,
            "source_index": origins[0] if len(origins) == 1 else -1,
            "native_index": int(native_index), "reason": reason}


def _add_stage(arrays: dict, summary: dict, prefix: str, rows: Sequence[dict]) -> None:
    arrays.update({
        f"{prefix}_native_note_index": np.asarray([row["native_index"] for row in rows], dtype=np.int64),
        f"{prefix}_source_note_index": np.asarray([row["source_index"] for row in rows], dtype=np.int64),
        f"{prefix}_element_index": np.asarray([row["voice"].source_index for row in rows], dtype=np.int64),
        f"{prefix}_pitch": np.asarray([row["voice"].pitch for row in rows], dtype=np.int16),
        f"{prefix}_onset_s": np.asarray([row["voice"].start_s for row in rows], dtype=np.float64),
        f"{prefix}_keyup_s": np.asarray([row["voice"].end_s for row in rows], dtype=np.float64),
        f"{prefix}_strict": np.asarray([row["native_index"] >= 0 for row in rows], dtype=np.bool_),
        f"{prefix}_reason": np.asarray([row["reason"] for row in rows], dtype="U256"),
        f"{prefix}_candidate_source_offsets": np.cumsum([0, *(len(row["origins"]) for row in rows)], dtype=np.int64),
        f"{prefix}_candidate_source_indices": np.asarray([index for row in rows for index in row["origins"]], dtype=np.int64),
    })
    strict = sum(row["native_index"] >= 0 for row in rows)
    summary["stages"][prefix] = {"voice_count": len(rows), "strict_voice_count": strict,
                                "unknown_voice_count": len(rows) - strict,
                                "reason_counts": dict(Counter(row["reason"] for row in rows))}


def _same_slot(first: dict, second: dict) -> bool:
    return (first["pitches"] == second["pitches"] and _equal(first["start"], second["start"])
            and _equal(first["end"], second["end"]))


def _propagate_slots(input_slots: Sequence[dict], input_rows: Sequence[dict],
                     output_slots: Sequence[dict], *, reason: str) -> list[dict]:
    grouped = defaultdict(list)
    for row in input_rows:
        grouped[row["voice"].source_index].append(row)
    ends = [slot["end"] for slot in input_slots]
    output = []
    for slot in output_slots:
        at = bisect_right(ends, slot["start"] + EPSILON_S)
        overlaps = []
        while at < len(input_slots) and input_slots[at]["start"] < slot["end"] - EPSILON_S:
            overlaps.append(at)
            at += 1
        exact = [index for index in overlaps if _same_slot(input_slots[index], slot)]
        origins = {value for index in overlaps for value in input_slots[index].get("all_origins", ())}
        slot["all_origins"] = tuple(sorted(origins))
        if len(exact) != 1 or len(overlaps) != 1:
            output.extend(_row(voice, origins, reason=reason) for voice in slot["voices"])
            continue
        previous = grouped[exact[0]]
        joined = join_unique_voices([row["voice"] for row in previous], slot["voices"])
        for index, voice in enumerate(slot["voices"]):
            parent = int(joined["target_to_source"][index])
            if parent < 0:
                output.append(_row(voice, origins, reason="ambiguous_slot_voice"))
            else:
                row = previous[parent]
                output.append(_row(voice, row["origins"], row["native_index"], row["reason"]))
    return output


def _profile_labels(arrays: dict, summary: dict, prefix: str, rows: Sequence[dict], reference: Mapping) -> None:
    native_masks = np.asarray(reference["note_native_masks"])
    native_chord_masks = np.asarray(reference["chord_native_masks"])
    native_mapping = np.asarray(reference["note_to_chord"])
    trusted = np.asarray(reference["chord_trusted"])
    native_members = defaultdict(set)
    for index, chord_index in enumerate(native_mapping):
        native_members[int(chord_index)].add(index)
    indices = np.asarray([row["native_index"] for row in rows], dtype=np.int64)
    duration = np.zeros(len(rows), dtype=np.bool_)
    valid = indices >= 0
    duration[valid] = native_masks[indices[valid], 1]
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["voice"].source_index].append(index)
    group_members = list(groups.values())
    singleton = duration.copy()
    whole = np.full(len(group_members), -1, dtype=np.int64)
    for group_index, members in enumerate(group_members):
        if len(members) != 1:
            singleton[members] = False
        mapped = indices[members]
        if np.any(mapped < 0) or len(np.unique(mapped)) != len(mapped):
            continue
        chord_ids = np.unique(native_mapping[mapped])
        if len(chord_ids) == 1:
            chord_index = int(chord_ids[0])
            if set(mapped.tolist()) == native_members[chord_index]:
                whole[group_index] = chord_index
    ioi = np.zeros(len(group_members), dtype=np.bool_)
    for index in range(1, len(group_members)):
        previous, current = int(whole[index - 1]), int(whole[index])
        if previous >= 0 and current == previous + 1:
            ioi[index] = bool(trusted[previous] and trusted[current] and native_chord_masks[current, 1])
    arrays[f"{prefix}_duration_candidate_mask"] = duration
    arrays[f"{prefix}_singleton_duration_candidate_mask"] = singleton
    arrays[f"{prefix}_group_element_index"] = np.asarray(list(groups), dtype=np.int64)
    arrays[f"{prefix}_native_chord_index"] = whole
    arrays[f"{prefix}_ioi_candidate_mask"] = ioi
    arrays[f"{prefix}_ioi_native_index"] = np.where(ioi, whole, -1)
    summary["profiles"][prefix.removeprefix("profile_")] = {
        "voice_count": len(rows), "duration_candidate_count": int(duration.sum()),
        "singleton_duration_candidate_count": int(singleton.sum()),
        "group_count": len(group_members), "whole_native_chord_count": int((whole >= 0).sum()),
        "consecutive_native_ioi_candidate_count": int(ioi.sum())}


def build_sample_game_mapping(*, score_bytes: bytes, production_read_result: MidiReadResult,
                              timing_reference: Mapping[str, np.ndarray], profiles: Sequence[GameProfile],
                              options: AdaptOptions = AdaptOptions(),
                              playback_settings: PlaybackSettings = PlaybackSettings(),
                              min_playable_ms: float = MIN_GAME_NOTE_MS) -> dict:
    """Return NPZ-ready arrays and JSON-safe coverage for mechanical/free-play.

    ``native_note_masks``/``native_chord_masks`` are immutable-source copies.
    Profile IOI arrays are sounding-group indexed; note arrays are voice
    indexed.  All candidates remain diagnostics, not executable training masks.
    Caller owns archive/reference/implementation hash binding and persistence.
    """
    parsed = parse_midi_bytes(score_bytes)
    song = production_read_result.song
    note_count = len(parsed.notes)
    reference = timing_reference
    note_masks = np.asarray(reference["note_native_masks"])
    chord_masks = np.asarray(reference["chord_native_masks"])
    mapping = np.asarray(reference["note_to_chord"])
    if note_masks.shape != (note_count, 6) or note_masks.dtype.kind != "b":
        raise ValueError("native note masks have wrong shape or dtype")
    if chord_masks.ndim != 2 or chord_masks.shape[1] != 2 or chord_masks.dtype.kind != "b":
        raise ValueError("native chord masks have wrong shape or dtype")
    trusted = np.asarray(reference["chord_trusted"])
    if trusted.shape != (len(chord_masks),) or trusted.dtype.kind != "b":
        raise ValueError("native chord trusted flags have wrong shape or dtype")
    ticks = sorted({note.start_tick for note in parsed.notes})
    tick_to_chord = {tick: index for index, tick in enumerate(ticks)}
    expected_mapping = np.asarray([tick_to_chord[note.start_tick] for note in parsed.notes])
    if mapping.dtype.kind not in "iu" or not np.array_equal(mapping, expected_mapping) or len(chord_masks) != len(ticks):
        raise ValueError("native note/chord ordering differs from score parser")
    durations = np.asarray(reference["note_score_duration_seconds"])
    starts = np.asarray(reference["chord_score_seconds"])
    if (durations.shape != (note_count,) or starts.shape != (len(ticks),)
            or not np.isfinite(durations).all() or not np.isfinite(starts).all()
            or not np.allclose(durations, [note.duration_s for note in parsed.notes], rtol=0, atol=EPSILON_S)
            or not np.allclose(starts[mapping], [note.start_s for note in parsed.notes], rtol=0, atol=EPSILON_S)):
        raise ValueError("native timing reference differs from score bytes")
    ids = [profile.id for profile in profiles]
    if len(set(ids)) != len(ids) or not ids or not set(ids).issubset({"default", "delta_force_harmonica"}):
        raise ValueError("profiles must be distinct default/Delta game profiles")
    source_voices = [Voice(note.start_s, note.end_s, note.pitch, index) for index, note in enumerate(song.notes)]
    # The training parser retains zero-duration notes, unlike the production
    # reader. Keep their original native indices/masks but never match them to
    # a positive interval merely because it falls inside the endpoint epsilon.
    training_voices = [Voice(note.start_s, note.end_s, note.pitch, index)
                       for index, note in enumerate(parsed.notes) if note.end_s > note.start_s]
    joined = join_unique_voices(training_voices, source_voices)
    source_rows = []
    for index, voice in enumerate(source_voices):
        native = int(joined["target_to_source"][index])
        native = training_voices[native].source_index if native >= 0 else -1
        source_rows.append(_row(voice, (index,), native, str(joined["target_reason"][index])))
    native_to_source = np.full(note_count, -1, dtype=np.int64)
    for index, voice in enumerate(training_voices):
        native_to_source[voice.source_index] = joined["source_to_target"][index]
    arrays = {"native_note_masks": note_masks.copy(), "native_chord_masks": chord_masks.copy(),
              "native_note_to_chord": mapping.copy(),
              "native_to_source_note_index": native_to_source}
    summary = {"schema_version": 1, "kind": "source_game_mapping", "diagnostic_only": True,
               "native_masks_modified": False, "training_applied": False,
               "performance_targets_emitted": False, "epsilon_s": EPSILON_S, "options": asdict(options),
               "scenario": "free_play", "baseline": "mechanical", "min_playable_ms": min_playable_ms,
               "ioi_index_domain": "profile sounding groups; no deleted intermediate native chord allowed",
               "duration_scope": "native-label provenance only; singleton mask excludes shared chord holds",
               "stages": {}, "profiles": {}}
    _add_stage(arrays, summary, "source", source_rows)
    adapted = adapt_source_song(song, options)
    import_slots, _ = _trace_import(song, options, adapted)
    import_candidates = []
    for slot in import_slots:
        reasons = slot["reasons"]
        for voice in slot["voices"]:
            # Recover IDs by their order in public storage, not by folded MIDI
            # pitch lookup, which could alias an unrelated original note.
            item = adapted.notes[voice.source_index]
            note_ids = [note_id for note_id in item["notes"]
                        if note_id_to_midi(note_id) + int(item.get("semitone", 0)) == voice.pitch]
            origins = tuple(index for note_id in note_ids for index in slot["origins"].get(note_id, ()))
            import_candidates.append((voice, origins, reasons))
    # One batch is essential both for reciprocal uniqueness across all output
    # voices and to avoid rebuilding the complete source index for each note.
    matched = join_unique_voices(source_voices, [row[0] for row in import_candidates],
                                 allowed_source_members=[row[1] for row in import_candidates])
    import_rows = []
    for index, (voice, origins, reasons) in enumerate(import_candidates):
        original = int(matched["target_to_source"][index])
        if reasons:
            import_rows.append(_row(voice, origins, reason="|".join(sorted(reasons))))
        elif original >= 0 and source_rows[original]["native_index"] >= 0:
            import_rows.append(_row(voice, origins, source_rows[original]["native_index"], "exact"))
        else:
            why = source_rows[original]["reason"] if original >= 0 else str(matched["target_reason"][index])
            import_rows.append(_row(voice, origins, reason=why))
    _add_stage(arrays, summary, "import", import_rows)
    prepared = prepare_score(adapted.notes, transpose=0, fold_octaves=True, bpm=adapted.bpm,
                             min_playable_ms=min_playable_ms)
    prepared_slots = _slots(prepared.notes, adapted.bpm)
    prepared_rows = _propagate_slots(import_slots, import_rows, prepared_slots, reason="transport_slot_changed")
    _add_stage(arrays, summary, "prepared", prepared_rows)
    for profile in profiles:
        if profile.legacy_keymap is not None:
            logical, _, _ = _legacy_projection(prepared.notes, adapted.bpm, profile, playback_settings, rules=False)
        else:
            logical, _, _ = _delta_projection(prepared.notes, adapted.bpm, profile, playback_settings, "free_play", rules=False)
        logical_groups = defaultdict(list)
        for voice in logical:
            logical_groups[voice.source_index].append(voice)
        profile_slots = []
        for index, slot in enumerate(prepared_slots):
            voices = logical_groups[index]
            profile_slots.append({"start": slot["start"], "end": slot["end"], "voices": voices,
                                  "pitches": tuple(sorted(voice.pitch for voice in voices))})
        rows = _propagate_slots(prepared_slots, prepared_rows, profile_slots, reason="profile_slot_changed")
        prefix = f"profile_{profile.id}"
        _add_stage(arrays, summary, prefix, rows)
        _profile_labels(arrays, summary, prefix, rows, reference)
    return {"arrays": arrays, "summary": summary}
