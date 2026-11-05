"""Project planned game inputs onto a local MIDI instrument."""

from core.compiler import DEFAULT_MODIFIER_BUTTONS
from core.direct_timing import direct_slots, strum_order
from core.humanize import plan_timings
from core.keyboard_driver import KeyboardDriver
from core.score_io import note_id_to_midi


def compiled_preview_events(events, profile):
    """Read the compiled keyboard/mouse sequence, including physical pitch overrides."""
    direct = {key: note_id_to_midi(nid) for nid, key in profile.pitch_direct_overrides.items()}
    pitches = {key: note_id_to_midi(f"mid_{degree}")
               for degree, key in enumerate(profile.pitch_keys, 1)}
    buttons = profile.modifier_buttons or {mod.value: key for mod, key in DEFAULT_MODIFIER_BUTTONS.items()}
    shifts = {key: {"lower": -12, "higher": 12, "semitone": 1}[mod] for mod, key in buttons.items()}
    active_button = None
    sounding = {}
    result = []
    for event in events:
        if event.device == "mouse":
            if event.action == "down":
                active_button = event.key
            elif event.key == active_button:
                active_button = None
            continue
        if event.source_index is None:  # e.g. NPC submit key
            continue
        if event.action == "down":
            pitch = direct.get(event.key)
            if pitch is None and event.key in pitches:
                pitch = pitches[event.key] + shifts.get(active_button, 0)
            if pitch is None:  # 未知音键:与真实演奏一样,这个音不会发声
                continue
            sounding[event.key] = pitch
            result.append((event.t_ms, pitch, True))
        elif event.key in sounding:
            result.append((event.t_ms, sounding.pop(event.key), False))
    return result


def compiled_preview_duration(events, timeline_ms=0.0):
    """Compiled event list carries no explicit tail; use the last event or caller's tail."""
    last = max((float(event.t_ms) for event in events), default=0.0)
    return max(last, float(timeline_ms or 0.0))


def direct_preview_events(notes, bpm, keymap, *, hold_ratio, gap_ms, timing, humanize, seed):
    """Use the 21-key player's note slots, strum order and release timing."""
    timings = plan_timings(notes, humanize, seed=seed) if humanize is not None else None
    slots = direct_slots(notes, bpm, hold_ratio, gap_ms, timings, additive=timing.additive_gap,
                         min_gap_ms=humanize.min_gap_ms if humanize else 0.0, keymap=keymap)
    result = []
    available_ms = 0.0
    driver_step = 0.0 if timing.batch_chords else KeyboardDriver.CHORD_INTERVAL_MS
    for i, (note, slot) in enumerate(zip(notes, slots)):
        pitches = [note_id_to_midi(nid) for nid in note["notes"] if keymap.key_for(nid)]
        if not pitches:
            continue
        order, step = strum_order(pitches, humanize.strum_ms if humanize else 0, i)
        step = step or driver_step
        onset = max(available_ms, slot.onset_ms)
        for index, pitch in enumerate(order):
            result.append((onset + index * step, pitch, True))
        release = onset + (len(order) - 1) * step + slot.hold_ms
        for index, pitch in enumerate(pitches):
            result.append((release + index * driver_step, pitch, False))
        available_ms = result[-1][0]
    duration = max(available_ms, slots[-1].cursor_ms + slots[-1].advance_ms) if slots else 0.0
    return result, duration
