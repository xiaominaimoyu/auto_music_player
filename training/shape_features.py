"""Score-only note features derived from geometry that already lives in the shards.

Why this exists: the duration channel is the one place where the model still sits above the
cross-performance noise floor (measured ``shape_mae`` 0.507 against a per-note floor of
0.425), and the diagnosis says it is underfitting rather than overfitting -- 14x more
parameters did not lower the in-sample error either.  The standing hypothesis is that the
inputs cannot describe *voice continuity*: nothing in the shipped feature list tells the
model when the same key is struck again, or how many other notes are still sounding when
this one ends, and both are natural reasons a performer lets a note run longer or shorter
than the score says.

Everything here is computed from columns the dataset builder already wrote, so it needs no
rebuild of the 1,238 shards: ``pitch_centered`` is ``(pitch - 60) / 24`` (recoverable because
MIDI pitches are integers and the grid is far coarser than float32 precision), column 3 holds
raw ``score_onset_beats`` and column 4 ``log1p(score_duration_beats)``.  No target column is
read, so these features cannot leak the label -- including for the piece whose label they are
meant to explain.

The gaps are expressed as a fraction of the piece's own beat span so they stay scale-free and
behave the same way under ``relative_position`` normalisation as the positions do.  When a key
is not struck again inside the piece the gap is reported as the whole span (the value is then
identical to "the next same-pitch note lies beyond this piece"), which keeps the feature
finite without inventing a magic constant.
"""

from __future__ import annotations

import numpy as np

PITCH_CENTER_COLUMN = 0
ONSET_BEATS_COLUMN = 3
DURATION_LOG1P_COLUMN = 4
SHAPE_FEATURE_NAMES = ("same_pitch_previous_span_fraction", "same_pitch_next_span_fraction",
                       "voices_sounding_at_offset_log1p")


def _recover_pitches(note_features: np.ndarray) -> np.ndarray:
    pitches = np.rint(60.0 + 24.0 * np.asarray(note_features[:, PITCH_CENTER_COLUMN], dtype=np.float64))
    if not np.isfinite(pitches).all():
        raise ValueError("note features carry non-finite pitches")
    return pitches.astype(np.int64)


def _same_pitch_gaps(onsets: np.ndarray, pitches: np.ndarray, span: float) -> np.ndarray:
    """Per note, the beat distance to the previous and next note on the same key.

    A key that is not struck again inside the piece keeps the default of one full span, so the
    column stays finite and "no next occurrence" reads the same way for every piece length.
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
    return np.maximum(gaps, 0.0) / max(span, 1e-6)


BEAT_TOLERANCE = 1e-6


def _voices_at_offset(onsets: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """How many *other* notes are still sounding at the instant this note's score ends.

    Counted as #{onset < t} - #{offset <= t} over sorted endpoints, which excludes the note
    itself for free: at its own offset it has already "ended" by that definition.

    A tolerance is not cosmetic here.  Durations survive the shard round trip as
    ``expm1(float32(log1p(duration)))``, so a note that ends exactly when the next one begins
    ends up a few nanobeats *after* it, and every adjacent pair -- the common case in keyed
    music -- would be classified as overlapping by rounding noise instead of by the score.
    Ties are therefore resolved outward by ``BEAT_TOLERANCE`` beats, which is orders of
    magnitude below the smallest real event spacing in these shards.
    """
    sorted_onsets = np.sort(onsets)
    sorted_offsets = np.sort(offsets)
    started = np.searchsorted(sorted_onsets, offsets - BEAT_TOLERANCE, side="right")
    ended = np.searchsorted(sorted_offsets, offsets + BEAT_TOLERANCE, side="right")
    return np.maximum(started - ended, 0).astype(np.float64)


def derive_shape_features(note_features: np.ndarray, chord_onset_beats: np.ndarray) -> np.ndarray:
    """Build the shape columns from raw (pre-normalisation) score geometry only."""
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
    gaps = _same_pitch_gaps(onsets, _recover_pitches(features), span)
    sounding = _voices_at_offset(onsets, offsets)
    return np.column_stack((np.log1p(gaps[:, 0]), np.log1p(gaps[:, 1]),
                            np.log1p(sounding))).astype(np.float32)
