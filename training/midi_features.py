"""Deterministic MIDI-to-expression feature extraction.

The module intentionally stops at dataset construction.  It preserves piano
velocity and sustain-pedal supervision, while the current desktop player is
only expected to consume timing and key-hold predictions later.
"""

from __future__ import annotations

import bisect
import io
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import mido
import numpy as np


NOTE_FEATURE_NAMES = (
    "pitch_centered",
    "pitch_class_sin",
    "pitch_class_cos",
    "score_onset_beats",
    "score_duration_log1p",
    "beat_phase_sin",
    "beat_phase_cos",
    "bar_phase_sin",
    "bar_phase_cos",
    "chord_size_log1p",
    "rank_in_chord",
    "pitch_span",
    "local_density_log1p",
    "previous_chord_ioi_log1p",
    "next_chord_ioi_log1p",
)

NOTE_TARGET_NAMES = (
    "micro_offset_ms",
    "log_key_duration_ratio",
    "velocity_0_1",
    "velocity_robust_z",
    "pedal_at_onset",
    "pedal_coverage",
)

CHORD_FEATURE_NAMES = (
    "score_onset_beats",
    "chord_size_log1p",
    "pitch_centered_mean",
    "pitch_span",
    "beat_phase_sin",
    "beat_phase_cos",
    "bar_phase_sin",
    "bar_phase_cos",
    "previous_chord_ioi_log1p",
    "next_chord_ioi_log1p",
    "local_density_log1p",
)

CHORD_TARGET_NAMES = (
    "onset_delta_ms",
    "log_ioi_ratio",
)

MAX_SUPERVISED_ONSET_DELTA_MS = 2_000.0


class MidiFeatureError(ValueError):
    """Raised when a MIDI/alignment triplet is unsafe to train on."""


@dataclass(frozen=True)
class MidiNote:
    start_tick: int
    end_tick: int
    start_s: float
    end_s: float
    pitch: int
    velocity: int
    channel: int
    source_order: int
    ticks_per_beat: int

    @property
    def onset_beats(self) -> float:
        return self.start_tick / self.ticks_per_beat

    @property
    def duration_beats(self) -> float:
        return max(0, self.end_tick - self.start_tick) / self.ticks_per_beat

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass(frozen=True)
class PedalEvent:
    tick: int
    time_s: float
    value: int
    channel: int


@dataclass(frozen=True)
class TimeSignatureEvent:
    tick: int
    numerator: int
    denominator: int


@dataclass(frozen=True)
class ParsedMidi:
    ticks_per_beat: int
    notes: tuple[MidiNote, ...]
    pedal_events: tuple[PedalEvent, ...]
    time_signatures: tuple[TimeSignatureEvent, ...]
    duration_s: float


@dataclass(frozen=True)
class PerformanceArrays:
    note_features: np.ndarray
    note_targets: np.ndarray
    note_masks: np.ndarray
    chord_features: np.ndarray
    chord_targets: np.ndarray
    chord_masks: np.ndarray
    note_to_chord: np.ndarray
    diagnostics: dict[str, float | int | bool]


def parse_midi_bytes(data: bytes) -> ParsedMidi:
    """Parse note, tempo, sustain and time-signature data in a stable order."""

    try:
        midi = mido.MidiFile(file=io.BytesIO(data))
        merged = mido.merge_tracks(midi.tracks)
    except Exception as exc:  # mido exposes several parser exception types
        raise MidiFeatureError(f"MIDI parse failed: {exc}") from exc

    if int(midi.type) not in {0, 1}:
        raise MidiFeatureError(f"unsupported asynchronous MIDI type: {midi.type}")
    ticks_per_beat = int(midi.ticks_per_beat)
    if not 1 <= ticks_per_beat <= 0x7FFF:
        raise MidiFeatureError("MIDI must use a positive PPQ time division, not SMPTE")

    tempo = 500_000
    absolute_tick = 0
    absolute_s = 0.0
    source_order = 0
    active: dict[tuple[int, int], deque[tuple[int, float, int, int]]] = defaultdict(deque)
    notes: list[MidiNote] = []
    pedal_events: list[PedalEvent] = []
    signatures: list[TimeSignatureEvent] = [TimeSignatureEvent(0, 4, 4)]

    for message in merged:
        delta_tick = int(message.time)
        if delta_tick < 0:
            raise MidiFeatureError("MIDI contains a negative delta time")
        absolute_s += mido.tick2second(delta_tick, ticks_per_beat, tempo)
        absolute_tick += delta_tick

        if message.type == "set_tempo":
            tempo = int(message.tempo)
            if tempo <= 0:
                raise MidiFeatureError("MIDI tempo must be positive")
            continue
        if message.type == "time_signature":
            numerator = int(message.numerator)
            denominator = int(message.denominator)
            if numerator > 0 and denominator > 0:
                event = TimeSignatureEvent(absolute_tick, numerator, denominator)
                if signatures and signatures[-1].tick == absolute_tick:
                    signatures[-1] = event
                else:
                    signatures.append(event)
            continue
        if message.type == "control_change" and int(message.control) == 64:
            pedal_events.append(
                PedalEvent(
                    absolute_tick,
                    float(absolute_s),
                    int(message.value),
                    int(message.channel),
                )
            )
            continue

        is_on = message.type == "note_on" and int(message.velocity) > 0
        is_off = message.type == "note_off" or (
            message.type == "note_on" and int(message.velocity) == 0
        )
        if is_on:
            source_order += 1
            active[(int(message.channel), int(message.note))].append(
                (absolute_tick, float(absolute_s), int(message.velocity), source_order)
            )
        elif is_off:
            key = (int(message.channel), int(message.note))
            if not active[key]:
                raise MidiFeatureError(
                    f"MIDI note_off has no note_on: channel={key[0]} pitch={key[1]}"
                )
            start_tick, start_s, velocity, order = active[key].popleft()
            notes.append(
                MidiNote(
                    start_tick=start_tick,
                    end_tick=max(start_tick, absolute_tick),
                    start_s=start_s,
                    end_s=max(start_s, float(absolute_s)),
                    pitch=key[1],
                    velocity=velocity,
                    channel=key[0],
                    source_order=order,
                    ticks_per_beat=ticks_per_beat,
                )
            )

    dangling = sum(len(queue) for queue in active.values())
    if dangling:
        raise MidiFeatureError(f"MIDI contains {dangling} unterminated notes")
    if not notes:
        raise MidiFeatureError("MIDI contains no completed notes")

    notes.sort(
        key=lambda note: (
            note.start_tick,
            note.pitch,
            note.end_tick,
            note.channel,
            note.source_order,
        )
    )
    # Python's sort is stable, so equal-time messages keep their order from
    # the merged MIDI stream.  Sorting by value here would invert valid
    # same-tick CC64 transitions.
    pedal_events.sort(key=lambda event: (event.time_s, event.channel))
    signatures.sort(key=lambda event: event.tick)
    return ParsedMidi(
        ticks_per_beat=ticks_per_beat,
        notes=tuple(notes),
        pedal_events=tuple(pedal_events),
        time_signatures=tuple(signatures),
        duration_s=max(note.end_s for note in notes),
    )


class _PedalTimeline:
    def __init__(self, events: Iterable[PedalEvent]):
        grouped: dict[int, list[PedalEvent]] = defaultdict(list)
        for event in events:
            grouped[event.channel].append(event)
        self._events = {
            channel: tuple(sorted(values, key=lambda event: event.time_s))
            for channel, values in grouped.items()
        }
        self._times = {
            channel: tuple(event.time_s for event in values)
            for channel, values in self._events.items()
        }

    @property
    def available(self) -> bool:
        return bool(self._events)

    def available_for(self, channel: int) -> bool:
        return bool(self._events.get(channel))

    def value_at(self, channel: int, at_s: float) -> int:
        events = self._events.get(channel, ())
        times = self._times.get(channel, ())
        index = bisect.bisect_right(times, at_s) - 1
        return events[index].value if index >= 0 else 0

    def coverage(self, channel: int, start_s: float, end_s: float) -> float:
        if end_s <= start_s:
            return 0.0
        events = self._events.get(channel, ())
        times = self._times.get(channel, ())
        if not events:
            return 0.0
        cursor = start_s
        value = self.value_at(channel, start_s)
        held_s = 0.0
        index = bisect.bisect_right(times, start_s)
        while index < len(events) and events[index].time_s < end_s:
            boundary = events[index].time_s
            if value >= 64:
                held_s += max(0.0, boundary - cursor)
            cursor = boundary
            value = events[index].value
            index += 1
        if value >= 64:
            held_s += max(0.0, end_s - cursor)
        return min(1.0, max(0.0, held_s / (end_s - start_s)))


def _bar_phase(note: MidiNote, signatures: Sequence[TimeSignatureEvent]) -> float:
    ticks = [event.tick for event in signatures]
    index = max(0, bisect.bisect_right(ticks, note.start_tick) - 1)
    signature = signatures[index]
    beats_from_signature = (note.start_tick - signature.tick) / note.ticks_per_beat
    bar_beats = signature.numerator * (4.0 / signature.denominator)
    if bar_beats <= 0:
        return 0.0
    return (beats_from_signature % bar_beats) / bar_beats


def _robust_linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if not len(x):
        raise MidiFeatureError("aligned performance has no supervised chord for tempo fit")
    if len(x) < 2 or float(np.ptp(x)) <= 1e-9:
        return 1.0, float(y[0] - x[0])
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    if mad > 1e-9:
        keep = np.abs(residual - median) <= 6.0 * 1.4826 * mad
        if int(keep.sum()) >= 2 and float(np.ptp(x[keep])) > 1e-9:
            slope, intercept = np.polyfit(x[keep], y[keep], 1)
    if not math.isfinite(float(slope)) or float(slope) <= 0:
        raise MidiFeatureError("aligned performance has a non-positive tempo scale")
    return float(slope), float(intercept)


def _local_density(chord_beats: np.ndarray, window_beats: float = 2.0) -> np.ndarray:
    result = np.empty(len(chord_beats), dtype=np.float64)
    left = 0
    right = 0
    for index, onset in enumerate(chord_beats):
        while left < len(chord_beats) and chord_beats[left] < onset - window_beats:
            left += 1
        while right < len(chord_beats) and chord_beats[right] <= onset + window_beats:
            right += 1
        result[index] = right - left
    return result


def build_aligned_features(
    score: ParsedMidi,
    performance: ParsedMidi,
    perf_idx: np.ndarray,
    interpolated: np.ndarray,
) -> PerformanceArrays:
    """Build aligned note/chord arrays without using interpolated notes as labels."""

    perf_idx_raw = np.asarray(perf_idx)
    interpolated_raw = np.asarray(interpolated)
    if perf_idx_raw.dtype.kind not in {"i", "u"}:
        raise MidiFeatureError("alignment perf_idx must use an integer dtype")
    if interpolated_raw.dtype.kind != "b":
        raise MidiFeatureError("alignment interpolated must use a boolean dtype")
    perf_idx = perf_idx_raw.astype(np.int64, copy=False)
    interpolated = interpolated_raw.astype(np.bool_, copy=False)
    note_count = len(score.notes)
    if perf_idx.ndim != 1 or interpolated.ndim != 1:
        raise MidiFeatureError("alignment arrays must be one-dimensional")
    if len(perf_idx) != note_count or len(interpolated) != note_count:
        raise MidiFeatureError(
            "alignment length does not match score note count: "
            f"score={note_count} perf_idx={len(perf_idx)} interpolated={len(interpolated)}"
        )
    if np.any(perf_idx < 0) or np.any(perf_idx >= len(performance.notes)):
        raise MidiFeatureError("alignment contains an out-of-range performance index")
    if len(np.unique(perf_idx)) != len(perf_idx):
        raise MidiFeatureError("alignment contains duplicate performance indices")

    aligned = tuple(performance.notes[int(index)] for index in perf_idx)
    mismatch = sum(
        score_note.pitch != perf_note.pitch
        for score_note, perf_note in zip(score.notes, aligned)
    )
    if mismatch:
        raise MidiFeatureError(f"alignment contains {mismatch} pitch mismatches")

    chord_ticks = np.array(sorted({note.start_tick for note in score.notes}), dtype=np.int64)
    tick_to_chord = {int(tick): index for index, tick in enumerate(chord_ticks)}
    note_to_chord = np.array(
        [tick_to_chord[note.start_tick] for note in score.notes], dtype=np.int32
    )
    chord_count = len(chord_ticks)
    chord_members = [np.flatnonzero(note_to_chord == index) for index in range(chord_count)]
    chord_score_s = np.array(
        [score.notes[int(members[0])].start_s for members in chord_members],
        dtype=np.float64,
    )
    chord_beats = np.array(
        [score.notes[int(members[0])].onset_beats for members in chord_members],
        dtype=np.float64,
    )
    chord_perf_s = np.empty(chord_count, dtype=np.float64)
    chord_real_ratio = np.empty(chord_count, dtype=np.float64)
    for chord_index, members in enumerate(chord_members):
        real_members = members[~interpolated[members]]
        selected = real_members if len(real_members) else members
        chord_perf_s[chord_index] = float(
            np.median([aligned[int(index)].start_s for index in selected])
        )
        chord_real_ratio[chord_index] = len(real_members) / len(members)

    trusted_chords = chord_real_ratio >= 0.5
    tempo_scale, tempo_intercept = _robust_linear_fit(
        chord_score_s[trusted_chords],
        chord_perf_s[trusted_chords],
    )
    if not 0.20 <= tempo_scale <= 5.0:
        raise MidiFeatureError(f"implausible global tempo scale: {tempo_scale:.6f}")
    expected_chord_s = tempo_scale * chord_score_s + tempo_intercept
    chord_onset_delta_ms = (chord_perf_s - expected_chord_s) * 1000.0

    score_ioi_s = np.diff(chord_score_s, prepend=chord_score_s[0])
    perf_ioi_s = np.diff(chord_perf_s, prepend=chord_perf_s[0])
    previous_ioi_beats = np.diff(chord_beats, prepend=chord_beats[0])
    next_ioi_beats = np.diff(chord_beats, append=chord_beats[-1])
    log_ioi_ratio = np.zeros(chord_count, dtype=np.float64)
    valid_ioi = (score_ioi_s > 1e-6) & (perf_ioi_s > 1e-6)
    log_ioi_ratio[valid_ioi] = np.log(
        perf_ioi_s[valid_ioi] / (score_ioi_s[valid_ioi] * tempo_scale)
    )

    real_velocities = np.array(
        [note.velocity for note, is_interp in zip(aligned, interpolated) if not is_interp],
        dtype=np.float64,
    )
    velocity_median = float(np.median(real_velocities)) if len(real_velocities) else 64.0
    if len(real_velocities):
        velocity_q1, velocity_q3 = np.quantile(real_velocities, [0.25, 0.75])
        velocity_iqr = max(1.0, float(velocity_q3 - velocity_q1))
    else:
        velocity_iqr = 1.0

    pedal = _PedalTimeline(performance.pedal_events)
    density = _local_density(chord_beats)
    note_features = np.zeros((note_count, len(NOTE_FEATURE_NAMES)), dtype=np.float32)
    note_targets = np.zeros((note_count, len(NOTE_TARGET_NAMES)), dtype=np.float32)
    note_masks = np.zeros((note_count, len(NOTE_TARGET_NAMES)), dtype=np.bool_)

    for note_index, (score_note, perf_note) in enumerate(zip(score.notes, aligned)):
        chord_index = int(note_to_chord[note_index])
        members = chord_members[chord_index]
        pitches = np.array([score.notes[int(member)].pitch for member in members])
        sorted_members = sorted(
            (int(member) for member in members), key=lambda member: score.notes[member].pitch
        )
        rank = sorted_members.index(note_index)
        rank_norm = rank / max(1, len(sorted_members) - 1)
        beat_phase = score_note.onset_beats % 1.0
        bar_phase = _bar_phase(score_note, score.time_signatures)
        pitch_class_phase = 2.0 * math.pi * (score_note.pitch % 12) / 12.0
        note_features[note_index] = (
            (score_note.pitch - 60.0) / 24.0,
            math.sin(pitch_class_phase),
            math.cos(pitch_class_phase),
            score_note.onset_beats,
            math.log1p(score_note.duration_beats),
            math.sin(2.0 * math.pi * beat_phase),
            math.cos(2.0 * math.pi * beat_phase),
            math.sin(2.0 * math.pi * bar_phase),
            math.cos(2.0 * math.pi * bar_phase),
            math.log1p(len(members)),
            rank_norm,
            (float(pitches.max()) - float(pitches.min())) / 48.0,
            math.log1p(density[chord_index]),
            math.log1p(max(0.0, previous_ioi_beats[chord_index])),
            math.log1p(max(0.0, next_ioi_beats[chord_index])),
        )
        score_duration_scaled = max(1e-4, score_note.duration_s * tempo_scale)
        perf_duration = max(1e-4, perf_note.duration_s)
        note_targets[note_index] = (
            (perf_note.start_s - chord_perf_s[chord_index]) * 1000.0,
            math.log(perf_duration / score_duration_scaled),
            perf_note.velocity / 127.0,
            (perf_note.velocity - velocity_median) / velocity_iqr,
            pedal.value_at(perf_note.channel, perf_note.start_s) / 127.0,
            pedal.coverage(perf_note.channel, perf_note.start_s, perf_note.end_s),
        )
        if not interpolated[note_index]:
            note_masks[note_index, :4] = True
            if pedal.available_for(perf_note.channel):
                note_masks[note_index, 4:] = True

    chord_features = np.zeros((chord_count, len(CHORD_FEATURE_NAMES)), dtype=np.float32)
    chord_targets = np.zeros((chord_count, len(CHORD_TARGET_NAMES)), dtype=np.float32)
    chord_masks = np.zeros((chord_count, len(CHORD_TARGET_NAMES)), dtype=np.bool_)
    for chord_index, members in enumerate(chord_members):
        score_note = score.notes[int(members[0])]
        pitches = np.array([score.notes[int(member)].pitch for member in members])
        beat_phase = score_note.onset_beats % 1.0
        bar_phase = _bar_phase(score_note, score.time_signatures)
        chord_features[chord_index] = (
            score_note.onset_beats,
            math.log1p(len(members)),
            (float(pitches.mean()) - 60.0) / 24.0,
            (float(pitches.max()) - float(pitches.min())) / 48.0,
            math.sin(2.0 * math.pi * beat_phase),
            math.cos(2.0 * math.pi * beat_phase),
            math.sin(2.0 * math.pi * bar_phase),
            math.cos(2.0 * math.pi * bar_phase),
            math.log1p(max(0.0, previous_ioi_beats[chord_index])),
            math.log1p(max(0.0, next_ioi_beats[chord_index])),
            math.log1p(density[chord_index]),
        )
        chord_targets[chord_index] = (
            chord_onset_delta_ms[chord_index],
            log_ioi_ratio[chord_index],
        )
        enough_real = bool(trusted_chords[chord_index])
        # Long-form tempo drift and pauses are represented by log IOI.  Keep
        # absolute onset residual only as a bounded auxiliary target so a few
        # multi-second sections cannot dominate its loss/normalizer.
        chord_masks[chord_index, 0] = (
            enough_real
            and abs(chord_onset_delta_ms[chord_index])
            <= MAX_SUPERVISED_ONSET_DELTA_MS
        )
        chord_masks[chord_index, 1] = (
            enough_real
            and chord_index > 0
            and bool(trusted_chords[chord_index - 1])
            and bool(valid_ioi[chord_index])
        )

    finite_arrays = (note_features, note_targets, chord_features, chord_targets)
    if not all(np.isfinite(array).all() for array in finite_arrays):
        raise MidiFeatureError("generated feature arrays contain NaN or infinity")

    diagnostics: dict[str, float | int | bool] = {
        "note_count": note_count,
        "chord_count": chord_count,
        "interpolated_count": int(interpolated.sum()),
        "interpolation_rate": float(interpolated.mean()),
        "supervised_note_count": int((~interpolated).sum()),
        "supervised_chord_count": int(chord_masks[:, 0].sum()),
        "onset_delta_aux_masked_count": int(
            (trusted_chords & (np.abs(chord_onset_delta_ms) > MAX_SUPERVISED_ONSET_DELTA_MS)).sum()
        ),
        "global_tempo_scale": tempo_scale,
        "tempo_intercept_s": tempo_intercept,
        "has_pedal": pedal.available,
        "onset_delta_abs_p95_ms": float(
            np.quantile(np.abs(chord_onset_delta_ms[trusted_chords]), 0.95)
        ),
        "micro_offset_abs_p95_ms": float(
            np.quantile(np.abs(note_targets[~interpolated, 0]), 0.95)
        ),
    }
    return PerformanceArrays(
        note_features=note_features,
        note_targets=note_targets,
        note_masks=note_masks,
        chord_features=chord_features,
        chord_targets=chord_targets,
        chord_masks=chord_masks,
        note_to_chord=note_to_chord,
        diagnostics=diagnostics,
    )


__all__ = [
    "CHORD_FEATURE_NAMES",
    "CHORD_TARGET_NAMES",
    "MidiFeatureError",
    "MAX_SUPERVISED_ONSET_DELTA_MS",
    "MidiNote",
    "NOTE_FEATURE_NAMES",
    "NOTE_TARGET_NAMES",
    "ParsedMidi",
    "PerformanceArrays",
    "build_aligned_features",
    "parse_midi_bytes",
]
