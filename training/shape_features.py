"""Score-only note features derived from geometry that already lives in the shards.

Why this exists: the duration channel is the one place where the model still sits above the
cross-performance noise floor, and the diagnosis says it is underfitting rather than
overfitting -- 14x more parameters did not lower the in-sample error either.  The remaining
question is what the inputs cannot describe.  Measured against the shape profile different
performers agree on (``tools/analyze_shape_explanatory_power.py``, both splits), the strongest
quantity is *relative* note length -- this note's score duration against its neighbours --
which the shipped ``score_duration_log1p`` does not express: it carries absolute length only.

Everything here is computed from columns the dataset builder already wrote, so it needs no
rebuild of the 1,238 shards: ``pitch_centered`` is ``(pitch - 60) / 24`` (recoverable because
MIDI pitches are integers and the grid is far coarser than float32 precision), column 3 holds
raw ``score_onset_beats`` and column 4 ``log1p(score_duration_beats)``.  No target column is
read, so these features cannot leak the label.

Two measured exclusions belong to this history, because both look useful on paper:

* ``rest_after_log1p`` is structurally constant at zero -- PianoCoRe writes every score note
  to run at least until the next onset (measured on the balanced validation set: 0 of 11,816
  notes have a positive rest, 100% are held past the next onset, median overrun 0.4975
  beats), so it can carry no signal;
* ``next_note_onset_gap_log1p`` correlates |r| ~ 0.005 with the shared shape profile.

The block is versioned by its column names, not by a counter: a run records
``note_shape_feature_names`` in its config and inference rebuilds exactly those columns in
that order, so widening the block later cannot silently break an older checkpoint.
"""

from __future__ import annotations

import numpy as np

PITCH_CENTER_COLUMN = 0
ONSET_BEATS_COLUMN = 3
DURATION_LOG1P_COLUMN = 4
BEAT_TOLERANCE = 1e-6
LOCAL_WINDOW = 4

# the shipped block, in the order written into note_features
SHAPE_FEATURE_NAMES = ("duration_versus_local_mean_log", "voices_sounding_at_offset_log1p",
                       "pitch_repeats_previous_chord", "concurrent_notes_at_onset_log1p",
                       "same_pitch_previous_span_fraction", "same_pitch_next_span_fraction")


def _recover_pitches(note_features: np.ndarray) -> np.ndarray:
    pitches = np.rint(60.0 + 24.0 * np.asarray(note_features[:, PITCH_CENTER_COLUMN], dtype=np.float64))
    if not np.isfinite(pitches).all():
        raise ValueError("note features carry non-finite pitches")
    return pitches.astype(np.int64)


def _same_pitch_gaps(onsets: np.ndarray, pitches: np.ndarray, span: float) -> tuple[np.ndarray, np.ndarray]:
    """Beat distance to the previous and next note on the same key, in fractions of ``span``.

    A key that is not struck again inside the piece keeps the default of one full span, so
    the column stays finite and "no other occurrence" reads identically for a short minuet
    and a long ballad.
    """
    gaps = np.full((len(onsets), 2), span, dtype=np.float64)
    order = np.lexsort((onsets, pitches))
    last_seen: dict[int, int] = {}
    for index in order:
        pitch = int(pitches[index])
        if pitch in last_seen:
            gaps[index, 0] = onsets[index] - onsets[last_seen[pitch]]
        last_seen[pitch] = index
    next_seen: dict[int, int] = {}
    for index in reversed(order):
        pitch = int(pitches[index])
        if pitch in next_seen:
            gaps[index, 1] = onsets[next_seen[pitch]] - onsets[index]
        next_seen[pitch] = index
    normalised = np.maximum(gaps, 0.0) / max(span, 1e-6)
    return normalised[:, 0], normalised[:, 1]


def _sounding_count(starts: np.ndarray, ends: np.ndarray, instants: np.ndarray) -> np.ndarray:
    """How many notes are sounding strictly inside each instant, excluding the one ending there.

    Counted as #{onset < t} - #{offset <= t}.  The note whose offset *is* t is already removed
    by the second term, so nothing subtracts it again -- doing so would push the count of an
    isolated note negative and turn ``log1p`` into ``-inf``.

    The tolerance is not cosmetic: a duration survives the shard round trip as
    ``expm1(float32(log1p(2)))``, i.e. a few nanobeats late, so without it a note that ends
    exactly when the next one begins -- the common case in keyed music -- would be called
    overlapping by rounding noise instead of by the score.
    """
    sorted_starts = np.sort(starts)
    sorted_ends = np.sort(ends)
    started = np.searchsorted(sorted_starts, instants - BEAT_TOLERANCE, side="right")
    ended = np.searchsorted(sorted_ends, instants + BEAT_TOLERANCE, side="right")
    return np.maximum(started - ended, 0).astype(np.float64)


def _chord_members(note_to_chord: np.ndarray) -> dict[int, np.ndarray]:
    chords = np.unique(note_to_chord)
    return {int(chord): np.flatnonzero(note_to_chord == chord) for chord in chords}


def _repeats_previous_chord(pitches: np.ndarray, members: dict[int, np.ndarray]) -> np.ndarray:
    flags = np.zeros(len(pitches), dtype=np.float64)
    previous: set[int] = set()
    for chord in sorted(members):
        current = {int(pitches[index]) for index in members[chord]}
        for index in members[chord]:
            flags[index] = 1.0 if int(pitches[index]) in previous else 0.0
        previous = current
    return flags


def _versus_local_mean(durations: np.ndarray) -> np.ndarray:
    count = len(durations)
    padded = np.concatenate((np.repeat(durations[0], LOCAL_WINDOW), durations,
                             np.repeat(durations[-1], LOCAL_WINDOW)))
    window = np.convolve(padded, np.ones(2 * LOCAL_WINDOW + 1), mode="valid") / (2 * LOCAL_WINDOW + 1)
    return np.log(np.maximum(durations, 1e-4) / np.maximum(window, 1e-4))


def derive_shape_features(note_features: np.ndarray, chord_onset_beats: np.ndarray,
                          note_to_chord: np.ndarray,
                          names: tuple[str, ...] = SHAPE_FEATURE_NAMES) -> np.ndarray:
    """Build the requested shape columns from raw (pre-normalisation) score geometry only."""
    features = np.asarray(note_features, dtype=np.float64)
    positions = np.asarray(chord_onset_beats, dtype=np.float64)
    if features.ndim != 2 or positions.ndim != 1 or not len(features) or not len(positions):
        raise ValueError("shape features need non-empty note and chord position arrays")
    if features.shape[1] <= DURATION_LOG1P_COLUMN:
        raise ValueError(f"note features must carry at least {DURATION_LOG1P_COLUMN + 1} columns "
                         f"(pitch, onset beats, duration log1p); got width {features.shape[1]}")
    onsets = features[:, ONSET_BEATS_COLUMN]
    durations = np.expm1(features[:, DURATION_LOG1P_COLUMN])
    if not np.isfinite(durations).all():
        raise ValueError("note score durations are not finite")
    offsets = onsets + np.maximum(durations, 0.0)
    span = max(float(positions.max()) - float(positions[0]), 1e-6)
    pitches = _recover_pitches(features)
    previous_gap, next_gap = _same_pitch_gaps(onsets, pitches, span)
    columns = {
        "duration_versus_local_mean_log": _versus_local_mean(durations),
        "voices_sounding_at_offset_log1p": np.log1p(_sounding_count(onsets, offsets, offsets)),
        "concurrent_notes_at_onset_log1p": np.log1p(_sounding_count(onsets, offsets, onsets)),
        "pitch_repeats_previous_chord": _repeats_previous_chord(pitches, _chord_members(
            np.asarray(note_to_chord, dtype=np.int64))),
        "same_pitch_previous_span_fraction": np.log1p(previous_gap),
        "same_pitch_next_span_fraction": np.log1p(next_gap),
    }
    unknown = [name for name in names if name not in columns]
    if unknown:
        raise ValueError(f"unknown shape features: {unknown}")
    return np.column_stack([np.asarray(columns[name], dtype=np.float64) for name in names]).astype(np.float32)


__all__ = ["SHAPE_FEATURE_NAMES", "derive_shape_features"]
