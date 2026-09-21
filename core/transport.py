"""Pure playback preparation shared by GUI controls and tests.

This module changes score data only.  It never schedules input and therefore
keeps transport choices above the existing Profile -> IR -> Compiler ->
EventPlayer boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.score_io import midi_to_note_id, note_id_to_midi
from core.score_model import MAX_DUR_BEATS, require_valid

_BLACK_TO_NATURAL = {1: 0, 3: 2, 6: 5, 8: 7, 10: 9}
_NATURAL_PCS = {0, 2, 4, 5, 7, 9, 11}
_MIDI_MIN, _MIDI_MAX = 48, 83
MIN_GAME_NOTE_MS = 60.0


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


def _stabilize_short_elements(
    notes: list[dict],
    *,
    bpm: float,
    minimum_ms: float,
    source_start: int,
) -> tuple[list[dict], list[TransportDegradation]]:
    """Absorb timing fragments that a game cannot sample reliably.

    MIDI overlap boundaries can produce sub-millisecond notes and rests.  Each
    removed fragment donates its duration to a retained neighbor, so cleanup
    never shortens or lengthens the musical timeline.
    """

    if minimum_ms <= 0 or len(notes) < 2:
        return notes, []
    beat_ms = 60000.0 / bpm
    long_indices = [
        index
        for index, item in enumerate(notes)
        if float(item["dur"]) * beat_ms + 1e-9 >= minimum_ms
    ]
    if not long_indices:
        anchor = max(range(len(notes)), key=lambda index: float(notes[index]["dur"]))
        kept = dict(notes[anchor])
        kept["notes"] = list(kept.get("notes") or [])
        remaining = sum(float(item["dur"]) for item in notes)
        collapsed = []
        while remaining > 1e-9:
            item = dict(kept)
            item["notes"] = list(kept["notes"])
            item["dur"] = min(remaining, MAX_DUR_BEATS)
            collapsed.append(item)
            remaining -= item["dur"]
        degradations = [
            TransportDegradation(
                source_start + index,
                f"{float(item['dur']) * beat_ms:.1f}ms 片段",
                "并入最长片段",
                f"低于 {minimum_ms:g}ms 游戏可演奏下限",
            )
            for index, item in enumerate(notes)
            if index != anchor
        ]
        return collapsed, degradations

    output: list[dict] = []
    pending_duration = 0.0
    degradations: list[TransportDegradation] = []
    for index, source in enumerate(notes):
        item = dict(source)
        item["notes"] = list(source.get("notes") or [])
        duration = float(item["dur"])
        duration_ms = duration * beat_ms
        if duration_ms + 1e-9 < minimum_ms:
            degradations.append(
                TransportDegradation(
                    source_start + index,
                    f"{duration_ms:.1f}ms 片段",
                    "并入相邻片段",
                    f"低于 {minimum_ms:g}ms 游戏可演奏下限",
                )
            )
            if output:
                output[-1]["dur"] = float(output[-1]["dur"]) + duration
            else:
                pending_duration += duration
            continue
        if pending_duration:
            item["dur"] = duration + pending_duration
            pending_duration = 0.0
        output.append(item)

    if pending_duration and output:
        output[-1]["dur"] = float(output[-1]["dur"]) + pending_duration
    return output, degradations


def prepare_score(
    notes,
    *,
    start_index: int = 0,
    end_index: int | None = None,
    transpose: int = 0,
    fold_octaves: bool = True,
    bpm: float = 100,
    min_playable_ms: float = 0.0,
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
    if isinstance(bpm, bool) or not isinstance(bpm, (int, float)) or not 1 <= float(bpm) <= 300:
        raise ValueError("播放 BPM 必须在 1 到 300 之间")
    if (
        isinstance(min_playable_ms, bool)
        or not isinstance(min_playable_ms, (int, float))
        or not 0 <= float(min_playable_ms) <= 1000
    ):
        raise ValueError("最短可演奏时值必须在 0 到 1000 毫秒之间")

    output = []
    degradations = []
    for absolute_index, source in enumerate(notes[start_index:end_index], start=start_index):
        ids = list(source.get("notes") or [])
        unique_ids = list(dict.fromkeys(ids))
        if len(unique_ids) != len(ids):
            degradations.append(
                TransportDegradation(
                    absolute_index,
                    f"和弦 {len(ids)} 个音",
                    f"去重为 {len(unique_ids)} 个音",
                    "重复按键会造成无效重触发",
                )
            )
        ids = unique_ids
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

    output, timing_degradations = _stabilize_short_elements(
        output,
        bpm=float(bpm),
        minimum_ms=float(min_playable_ms),
        source_start=start_index,
    )
    degradations.extend(timing_degradations)
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
