"""Project the shipping rule humanizer into label units, with no model involved.

Why this exists: ``playback_adoption.POLICY`` requires a model to improve on three
baselines before the player may use it -- ``identity``, ``rules`` and
``zero_model_adapter``.  In label units ``identity`` and ``zero_model_adapter`` are the
same prediction (all zeros), which the piece-level bootstrap has reported all along;
``rules`` -- the humanizer that actually ships today -- has never been measured against
the sequence model.  That gap decides the adoption question on its own: a model that
loses to the rules would be replacing working behaviour with worse behaviour, whatever
its validation loss says.

What is reproduced.  ``core.humanize.build_plan`` is called as the app calls it, with the
app's own parameters, and its output goes through the compiler's chained anti-overlap
lower bound (``t >= previous release + min_gap_ms``, ``core/compiler.py:277``), because an
offset that is not clamped would press two keys at once and is therefore not something the
player can ever emit.  Everything is computed in the locked score-timeline units supplied
by the timing reference -- score seconds, not performance seconds -- which is the same
``fixed_score`` calibration the adoption gate itself uses.

What is deliberately not reproduced, and which way each gap cuts.  These are disclosed
because they change the reading of any number produced here:

* **One element is the interval until the next one.**  In the shipping storage format an
  element's ``dur`` is how long it occupies the timeline before the next event, and the hold
  tier and the breath rule both read it as note density.  PianoCoRe writes each note's
  *notated* length instead, and typically runs it past the next onset, so using the written
  length as the element duration is a category error: the compiler's anti-overlap chain then
  pushes every later onset further, the delay accumulates down the piece, and a measured
  ``rules`` onset error of ~22 seconds on this corpus is exactly what that mistake produces.
  The projection therefore uses the gap to the next chord onset, with the piece's tail filled
  by the last chord's longest written voice, and expresses the duration prediction as
  ``log(hold x gap / written length)`` so the label's own denominator survives.
* **The leap rule cannot fire on this corpus.**  ``_pitch_value`` parses only the game's
  21-key degree names (``low``/``mid``/``high`` + ``1..7``); a full-keyboard MIDI pitch is
  unparseable, and unparseable means no extra delay.  Inventing a semitone threshold would
  be writing a new rule, not measuring the shipping one, so ``leap_rule_coverage`` reports
  how *much* would have fired instead, and the baseline is credited with slightly less
  shaping than it would get in the game.
* **No rest elements.**  PianoCoRe writes every note at least up to the next onset, so
  ``prev_was_rest`` never becomes true and the breath rule fires only after long elements.
  That removes shaping, and it removes it from the baseline, not from the model.
* **No ``gap_ms``, ``settle_ms`` or ``max_hold_ms``.**  Those are profile-specific latency
  and cap constants of the game path, not musical shaping; ``gap_ms`` in particular would
  stretch every interval by a constant the score never had.

Two realizations are reported because the rules are stochastic and the choice matters.
``rules_session`` averages the absolute error over independent sessions -- it is what a
player actually hears, and it is the POLICY counterpart.  ``rules_centred`` averages the
predictions first, which removes most of the jitter and the within-tier hold spread and so
is the *stronger* baseline (Jensen: ``E|e| >= |E[p] - y|``).  Beating only the first while
losing to the second is not an adoption argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from core.humanize import HumanizeParams, build_plan

# a placeholder note id that `core.humanize._pitch_value` cannot parse, which is exactly how
# the shipping rule behaves on a full-keyboard pitch: the element sounds, the leap rule stays
# silent.  An empty list would have been read as a rest instead.
UNPARSEABLE_NOTE_ID = "midi_unknown"


@dataclass(frozen=True)
class RuleTimings:
    """Per-session onset offsets and hold ratios, already clamped for anti-overlap.

    ``offset_ms`` and ``hold`` are ``(sessions, chords)``: session ``s`` is one playback the
    app could produce, and row ``i`` is the net shift of that element's onset away from the
    score timeline after the chained lower bound.
    """

    offset_ms: np.ndarray
    hold: np.ndarray

    @property
    def sessions(self) -> int:
        return int(self.offset_ms.shape[0])


def element_gaps(chord_onset_beats: Sequence[float], note_dur_beats: Sequence[float],
                 note_to_chord: Sequence[int]) -> np.ndarray:
    """One element per chord: how long it holds the timeline before the next element arrives.

    This is the game's own meaning of an element's ``dur``, and the humanizer's hold tiers and
    breath threshold read it as note density.  The final element has no successor, so it keeps
    the longest written voice of its own chord -- the only length in the score that says when
    that sound actually stops.
    """
    onsets = np.asarray(chord_onset_beats, dtype=np.float64)
    durations = np.asarray(note_dur_beats, dtype=np.float64)
    mapping = np.asarray(note_to_chord, dtype=np.int64)
    if onsets.ndim != 1 or not len(onsets):
        raise ValueError("chord onset beats must be a non-empty vector")
    if durations.ndim != 1 or mapping.shape != durations.shape:
        raise ValueError("note durations must pair with the note-to-chord mapping")
    if np.any(mapping < 0) or np.any(mapping >= len(onsets)):
        raise ValueError("note_to_chord escapes the chord range")
    if len(onsets) > 1 and np.any(np.diff(onsets) <= 0):
        raise ValueError("chord onsets must be strictly increasing")
    gaps = np.diff(onsets)[:len(onsets) - 1] if len(onsets) > 1 else np.array([], dtype=np.float64)
    last = int(mapping[-1])
    tail = durations[mapping == last]
    if not len(tail):
        raise ValueError("the final chord has no notes")
    gaps = np.append(gaps, float(tail.max()))
    if np.any(gaps <= 0) or not np.isfinite(gaps).all():
        raise ValueError("element gaps must be positive and finite")
    return gaps


PHRASE_LONG_DUR_BEATS = 2.0
SHORT_ELEMENT_MAX_BEATS = 0.5


def tier_of(element_dur_beats: float) -> str:
    """Which hold band the shipping humanizer would pick for this element.

    The thresholds are the module's own -- half a beat and ``PHRASE_LONG_DUR`` -- reproduced
    here so a fitted rule can be attributed to the same bands the player would use, rather
    than to a partition only the fitter understands.
    """
    beats = float(element_dur_beats)
    if beats <= SHORT_ELEMENT_MAX_BEATS:
        return "short_hold"
    if beats < PHRASE_LONG_DUR_BEATS:
        return "mid_hold"
    return "long_hold"


def rule_timings(chord_score_seconds: Sequence[float], element_dur_beats: Sequence[float],
                 *, seconds_per_beat: float, params: HumanizeParams | None = None,
                 sessions: int = 8, seed: int = 20261003) -> RuleTimings:
    """Run the shipping humanizer ``sessions`` times and clamp each session's timeline.

    Durations reach the humanizer in **beats**, because that is the unit its tier and
    phrase-boundary thresholds are written in (``short_hold`` below half a beat, breath after
    two beats).  They re-enter the timeline in seconds through ``seconds_per_beat``, which is
    the locked score reference tempo -- the same ``fixed_score`` calibration the adoption gate
    uses, never the recording's own tempo.  It is needed only to know whether the anti-overlap
    bound can bite: a held element whose written length in milliseconds already exceeds the
    gap to the next onset forces the next key later.
    """
    if sessions < 1:
        raise ValueError("sessions must be at least 1")
    tempo = float(seconds_per_beat)
    if not np.isfinite(tempo) or tempo <= 0:
        raise ValueError("seconds_per_beat must be finite and positive")
    nominal = np.asarray(chord_score_seconds, dtype=np.float64)
    beats = np.asarray(element_dur_beats, dtype=np.float64)
    if nominal.ndim != 1 or nominal.shape != beats.shape or not len(nominal):
        raise ValueError("chord score seconds and element durations must be one aligned vector")
    if np.any(np.diff(nominal) < 0.0) or nominal[0] < 0:
        raise ValueError("score timeline must be non-decreasing and start at or after zero")
    if not np.isfinite(nominal).all() or not np.isfinite(beats).all() or np.any(beats <= 0):
        raise ValueError("score timeline and element durations must be finite and positive")
    active = params or HumanizeParams()
    elements = [{"notes": [UNPARSEABLE_NOTE_ID], "dur": float(beat)} for beat in beats]
    offsets = np.zeros((int(sessions), len(nominal)), dtype=np.float64)
    holds = np.zeros_like(offsets)
    for session in range(int(sessions)):
        plan = build_plan(elements, active, seed=int(seed) + int(session))
        raw_offset = np.asarray([float(offset) for offset, _ratio in plan.timings], dtype=np.float64)
        hold = np.asarray([float(ratio) if ratio is not None else 1.0 for _offset, ratio in plan.timings],
                          dtype=np.float64)
        applied = _clamp_timeline(nominal, beats * tempo, raw_offset, hold, active)
        offsets[session] = applied
        holds[session] = hold
    return RuleTimings(offsets, holds)


def _clamp_timeline(nominal_seconds: np.ndarray, durations_seconds: np.ndarray,
                    offsets_ms: np.ndarray, holds: np.ndarray, params: HumanizeParams) -> np.ndarray:
    """``t = max(nominal + offset, previous release + min_gap)``, in milliseconds.

    Mirrors ``core/compiler.py`` for the note branch.  Rests and modifier-button settle are
    absent here by design (see the module docstring), so the only bound is the keyboard one.
    """
    applied = np.zeros(len(nominal_seconds), dtype=np.float64)
    previous_release = None
    minimum_gap = float(params.min_gap_ms)
    for index in range(len(nominal_seconds)):
        nominal_ms = float(nominal_seconds[index]) * 1000.0
        candidate = max(0.0, nominal_ms + float(offsets_ms[index]))
        if previous_release is not None:
            candidate = max(candidate, previous_release + minimum_gap)
        applied[index] = candidate - nominal_ms
        previous_release = candidate + float(holds[index]) * float(durations_seconds[index]) * 1000.0
    return applied


def onset_predictions(timings: RuleTimings) -> np.ndarray:
    """Per-chord onset shift in milliseconds -- the unit ``onset_delta_ms`` is labelled in."""
    return timings.offset_ms


def ioi_log_ratio_predictions(timings: RuleTimings, chord_score_seconds: Sequence[float]) -> np.ndarray:
    """Per-chord ``log(applied IOI / score IOI)`` produced by the shifted onsets.

    An interval the projection would make non-positive is **missing coverage**, reported as
    ``nan`` -- the same convention the adoption pipeline uses -- because the player could not
    emit it and scoring it as zero error would flatter the baseline.  The first chord has no
    preceding interval and is always ``nan``.
    """
    nominal = np.asarray(chord_score_seconds, dtype=np.float64)
    if nominal.ndim != 1 or nominal.shape[0] != timings.offset_ms.shape[1]:
        raise ValueError("score timeline length differs from the rule timings")
    score_ioi = np.diff(nominal, prepend=nominal[0])
    offset_delta = np.diff(timings.offset_ms, axis=1, prepend=timings.offset_ms[:, :1]) / 1000.0
    applied_ioi = score_ioi[None, :] + offset_delta
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log(applied_ioi / np.where(score_ioi > 0, score_ioi, np.nan))
    ratio[~np.isfinite(ratio)] = np.nan
    ratio[:, 0] = np.nan
    ratio[applied_ioi <= 0] = np.nan
    return ratio


def duration_log_ratio_predictions(timings: RuleTimings, note_to_chord: Sequence[int],
                                   element_dur_seconds: Sequence[float],
                                   note_written_seconds: Sequence[float]) -> np.ndarray:
    """Per-note ``log(hold x element gap / written length)``: the rule's answer to the label.

    The rule shortens a note to ``hold`` of the time its element occupies, while the label
    divides the played length by the note's **written** length.  In this corpus those two
    denominators differ by the legato overrun -- notes are notated past the next onset -- so
    the gap has to be converted back, or the baseline would be credited with a shrink it never
    applies.  Nothing here uses a performance measurement.
    """
    mapping = np.asarray(note_to_chord, dtype=np.int64)
    gaps = np.asarray(element_dur_seconds, dtype=np.float64)
    written = np.asarray(note_written_seconds, dtype=np.float64)
    if mapping.ndim != 1 or written.shape != mapping.shape:
        raise ValueError("note written durations must pair with the note-to-chord mapping")
    if gaps.shape[0] != timings.hold.shape[1]:
        raise ValueError("element durations must match the rule timings")
    if np.any(written <= 0) or not np.isfinite(written).all():
        raise ValueError("written note durations must be positive and finite")
    applied = timings.hold[:, mapping] * gaps[mapping]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(applied / written[None, :])


def leap_rule_coverage(pitch_sequences: Sequence[Sequence[float]], *, threshold: int = 4) -> dict[str, float]:
    """How often the leap rule *would* fire if full-keyboard pitches had degree names.

    Reported so nobody reads the baseline as "the app has no leap shaping" without knowing
    how much of it is being lost to the naming mismatch.  ``threshold`` is the module's own
    ``LEAP_THRESHOLD``, whose unit is the 21-key degree grid; applying it to semitones is a
    rough but stated surrogate, and the number here is a coverage statistic, never a
    prediction.
    """
    fires = 0
    pairs = 0
    for pitches in pitch_sequences:
        previous = None
        for pitch in pitches:
            if previous is not None:
                pairs += 1
                if abs(float(pitch) - float(previous)) >= threshold:
                    fires += 1
            previous = pitch
    return {"consecutive_pairs": pairs, "would_fire": fires,
            "coverage": float(fires / pairs) if pairs else 0.0}


__all__ = ["RuleTimings", "UNPARSEABLE_NOTE_ID", "duration_log_ratio_predictions",
           "element_gaps", "ioi_log_ratio_predictions", "leap_rule_coverage",
           "onset_predictions", "rule_timings"]
