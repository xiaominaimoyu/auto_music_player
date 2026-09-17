"""Pure playback preparation shared by GUI controls and tests.

This module changes score data only.  It never schedules input and therefore
keeps transport choices above the existing Profile -> IR -> Compiler ->
EventPlayer boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.score_io import midi_to_note_id, note_id_to_midi
from core.score_model import require_valid

_BLACK_TO_NATURAL = {1: 0, 3: 2, 6: 5, 8: 7, 10: 9}
_NATURAL_PCS = {0, 2, 4, 5, 7, 9, 11}
_MIDI_MIN, _MIDI_MAX = 48, 83


@dataclass(frozen=True)
class TransportDegradation:
    index: int
    requested: str
    actual: str
    reason: str

    def __str__(self):
        return f"元素 {self.index + 1}: {self.requested} → {self.actual} ({self.reason})"


@dataclass
class PreparedScore:
    notes: list[dict]
    source_start: int
    source_end: int
    degradations: list[TransportDegradation] = field(default_factory=list)


def _fold_pitch(pitch: int) -> tuple[int, bool]:
    original = pitch
    while pitch < _MIDI_MIN:
        pitch += 12
    while pitch > _MIDI_MAX:
        pitch -= 12
    return pitch, pitch != original


def _pitch_to_storage(pitch: int) -> tuple[str, int]:
    pc = pitch % 12
    semitone = 0
    natural_pitch = pitch
    if pc not in _NATURAL_PCS:
        natural_pc = _BLACK_TO_NATURAL[pc]
        natural_pitch = pitch - (pc - natural_pc)
        semitone = 1
    note_id = midi_to_note_id(natural_pitch)
    if note_id is None:
        raise ValueError(f"MIDI 音高 {pitch} 无法映射到 C3-B5")
    return note_id, semitone


def prepare_score(
    notes,
    *,
    start_index: int = 0,
    end_index: int | None = None,
    transpose: int = 0,
    fold_octaves: bool = True,
) -> PreparedScore:
    """Slice and transpose canonical storage without touching playback code.

    ``end_index`` is exclusive.  A mixed-accidental chord cannot be represented
    by the frozen storage schema (one ``semitone`` flag per element), so it is
    reduced deterministically to its highest pitch with a degradation record.
    """
    if not isinstance(notes, list):
        raise ValueError("乐谱音符必须是列表")
    total = len(notes)
    if isinstance(start_index, bool) or not isinstance(start_index, int):
        raise ValueError("片段起点必须是整数")
    if end_index is None:
        end_index = total
    if isinstance(end_index, bool) or not isinstance(end_index, int):
        raise ValueError("片段终点必须是整数")
    if not 0 <= start_index <= end_index <= total:
        raise ValueError(f"片段范围必须满足 0 ≤ 起点 ≤ 终点 ≤ {total}")
    if isinstance(transpose, bool) or not isinstance(transpose, int) or not -24 <= transpose <= 24:
        raise ValueError("移调必须是 -24 到 24 的整数半音")

    output = []
    degradations = []
    for absolute_index, source in enumerate(notes[start_index:end_index], start=start_index):
        ids = list(source.get("notes") or [])
        duration = source.get("dur")
        if not ids or transpose == 0:
            item = {"notes": ids, "dur": duration}
            if source.get("semitone"):
                item["semitone"] = int(source["semitone"])
            output.append(item)
            continue

        current_semitone = int(source.get("semitone", 0))
        mapped = []
        folded_count = 0
        for note_id in ids:
            requested_pitch = note_id_to_midi(note_id) + current_semitone + transpose
            pitch = requested_pitch
            if not _MIDI_MIN <= pitch <= _MIDI_MAX:
                if not fold_octaves:
                    mapped = []
                    degradations.append(
                        TransportDegradation(
                            absolute_index,
                            f"{note_id}{transpose:+d}",
                            "休止",
                            "移调后超出可演奏范围",
                        )
                    )
                    break
                pitch, folded = _fold_pitch(pitch)
                folded_count += int(folded)
            mapped_id, mapped_semitone = _pitch_to_storage(pitch)
            mapped.append((pitch, mapped_id, mapped_semitone))

        if not mapped:
            output.append({"notes": [], "dur": duration})
            continue
        if folded_count:
            degradations.append(
                TransportDegradation(
                    absolute_index,
                    f"移调 {transpose:+d}",
                    "八度折叠",
                    f"{folded_count} 个音超出范围",
                )
            )

        semitones = {item[2] for item in mapped}
        if len(mapped) > 1 and len(semitones) > 1:
            winner = max(mapped, key=lambda item: item[0])
            output.append(
                {
                    "notes": [winner[1]],
                    "dur": duration,
                    **({"semitone": 1} if winner[2] else {}),
                }
            )
            degradations.append(
                TransportDegradation(
                    absolute_index,
                    "混合升半音和弦",
                    winner[1] + ("#" if winner[2] else ""),
                    "存储格式每个和弦只能共享一个半音状态",
                )
            )
            continue

        item = {"notes": [entry[1] for entry in mapped], "dur": duration}
        if next(iter(semitones)):
            item["semitone"] = 1
        output.append(item)

    require_valid(output)
    return PreparedScore(output, start_index, end_index, degradations)


def normalize_transport_preferences(settings, total_notes: int) -> dict:
    """Validate persisted JSON and return a complete, safe preference object."""
    source = settings if isinstance(settings, dict) else {}
    bpm = source.get("bpm", 100)
    transpose = source.get("transpose", 0)
    segment = source.get("segment", [0, total_notes])
    try:
        bpm = int(bpm)
        transpose = int(transpose)
        start, end = int(segment[0]), int(segment[1])
    except (TypeError, ValueError, IndexError):
        bpm, transpose, start, end = 100, 0, 0, total_notes
    if not 30 <= bpm <= 300:
        bpm = 100
    if not -24 <= transpose <= 24:
        transpose = 0
    if not 0 <= start <= end <= total_notes:
        start, end = 0, total_notes
    return {
        "version": 1,
        "bpm": bpm,
        "transpose": transpose,
        "segment": [start, end],
    }
