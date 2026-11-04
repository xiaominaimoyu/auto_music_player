"""Shared musical timing for the 21-key player and its audio preview (milliseconds)."""

from dataclasses import dataclass

from core.humanize import timing_parts
from core.timing import slot_hold_and_advance


@dataclass(frozen=True)
class DirectSlot:
    cursor_ms: float
    onset_ms: float
    hold_ms: float
    advance_ms: float


def direct_slots(notes, bpm, hold_ratio, gap_ms, timings, *, additive, min_gap_ms,
                 keymap, start_index=0):
    beat_ms = 60000.0 / max(1, bpm)
    gap = max(0.0, gap_ms)
    cursor = 0.0
    previous_release = None
    slots = []
    for index, note in enumerate(notes):
        dur = float(note["dur"]) * beat_ms
        offset, ratio, pull = timing_parts(timings[index]) if timings is not None else (0, None, 0)
        onset = cursor + offset
        if note["notes"]:
            followed = index + 1 < len(notes) and bool(notes[index + 1]["notes"])
            hold, advance = slot_hold_and_advance(
                dur, hold_ratio if ratio is None else ratio, gap,
                additive=additive, followed_by_attack=followed,
            )
            hold = max(0.0, hold - pull)
            if index >= start_index and any(keymap.key_for(nid) for nid in note["notes"]):
                if previous_release is not None:
                    onset = max(onset, previous_release + min_gap_ms)
                previous_release = onset + hold
        else:
            hold, advance = 0.0, dur + gap if additive else dur
        slots.append(DirectSlot(cursor, onset, hold, advance))
        cursor += advance
    return slots


def strum_order(keys, strum_ms, index):
    """Return the order of attacks and the interval between them in ms."""
    if strum_ms <= 0 or len(keys) < 2:
        return keys, 0.0
    order = keys[::-1] if index % 2 else keys
    return order, min(25.0, max(8.0, strum_ms)) / (len(keys) - 1)
