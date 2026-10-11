"""Offline, score-only timing/hold listening candidates; never sends game input.

The supplied miniatures are original procedural studies, not redistributed
PianoCoRe data. Their identity-derived arrays are inference features only, not
human supervision. Piano velocity and pedal are deliberately equal across a
pair so this first listening study isolates timing and key-hold changes.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import wave

import mido
import numpy as np

from training.midi_features import build_aligned_features, parse_midi_bytes


@dataclass(frozen=True)
class ListeningNote:
    pitch: int
    start_s: float
    end_s: float
    velocity: int = 76


def _validate_plan(notes: list[ListeningNote]) -> None:
    if not notes:
        raise ValueError("empty note plan")
    for note in notes:
        if (not 0 <= note.pitch <= 127 or not 1 <= note.velocity <= 127
                or not np.isfinite([note.start_s, note.end_s]).all()
                or note.start_s < 0 or note.end_s <= note.start_s):
            raise ValueError("invalid note plan")


def limit_retrigger_overlap(notes: list[ListeningNote]) -> list[ListeningNote]:
    """Avoid an older note-off cancelling a new voice of the same MIDI pitch."""
    result = list(notes)
    by_pitch: dict[int, list[int]] = {}
    for index, note in enumerate(result):
        by_pitch.setdefault(note.pitch, []).append(index)
    for indices in by_pitch.values():
        indices.sort(key=lambda i: result[i].start_s)
        for current, following in zip(indices, indices[1:]):
            note, next_note = result[current], result[following]
            if next_note.start_s <= note.start_s:
                raise ValueError("simultaneous duplicate pitch cannot be retriggered safely")
            end = min(note.end_s, next_note.start_s)
            result[current] = ListeningNote(note.pitch, note.start_s, end, note.velocity)
    _validate_plan(result)
    return result


def expression_plan(score, note_to_chord: np.ndarray,
                    note_predictions: np.ndarray | None = None,
                    chord_predictions: np.ndarray | None = None,
                    *, strength: float = 0.5) -> tuple[list[ListeningNote], dict]:
    """Reconstruct with score tempo only (scale 1); apply disclosed safety bounds.

    IOI is integrated, not added as independent absolute onset residuals. No
    performance-derived global fit, performer velocity statistics, or target
    arrays are consulted here. Auxiliary absolute onset predictions are unused.
    """
    if not 0 <= strength <= 1:
        raise ValueError("strength must be in [0,1]")
    mapping = np.asarray(note_to_chord)
    chord_ticks = sorted({note.start_tick for note in score.notes})
    count = len(chord_ticks)
    if mapping.shape != (len(score.notes),) or mapping.dtype.kind not in "iu":
        raise ValueError("invalid note-to-chord mapping")
    expected = {tick: i for i, tick in enumerate(chord_ticks)}
    if any(mapping[i] != expected[n.start_tick] for i, n in enumerate(score.notes)):
        raise ValueError("mapping does not match score")
    if (note_predictions is None) != (chord_predictions is None):
        raise ValueError("both prediction heads are required")
    predicted = note_predictions is not None
    ny = np.zeros((len(score.notes), 6)) if not predicted else np.asarray(note_predictions)
    cy = np.zeros((count, 2)) if not predicted else np.asarray(chord_predictions)
    if ny.shape != (len(score.notes), 6) or cy.shape != (count, 2):
        raise ValueError("prediction shape mismatch")
    if not np.isfinite(ny).all() or not np.isfinite(cy).all():
        raise ValueError("non-finite prediction")
    chord_s = np.array([next(n.start_s for n in score.notes if n.start_tick == tick)
                        for tick in chord_ticks])
    score_ioi = np.diff(chord_s, prepend=chord_s[0])
    adjusted = score_ioi * np.exp(np.clip(cy[:, 1], -0.25, 0.25) * strength)
    onset = chord_s[0] + np.cumsum(adjusted)
    micro = np.clip(ny[:, 0], -30, 30) * strength / 1000
    durations = np.array([n.duration_s for n in score.notes])
    duration_ratio = np.exp(np.clip(ny[:, 1], np.log(0.4), np.log(1.6)) * strength)
    notes = []
    for i, note in enumerate(score.notes):
        start = max(0.0, float(onset[mapping[i]] + micro[i]))
        hold = max(0.04, float(durations[i] * duration_ratio[i]))
        notes.append(ListeningNote(note.pitch, start, start + hold))
    notes = limit_retrigger_overlap(notes)
    return notes, {
        "anchor": "fixed_score_tempo", "global_scale": 1.0,
        "strength": strength, "velocity": 76, "pedal": "off_for_both",
        "micro_limit_ms": 30, "raw_log_ioi_limit": 0.25,
        "raw_duration_ratio_limits": [0.4, 1.6], "hold_floor_s": 0.04,
        "note_count": len(notes), "uses_oracle_performance": False,
        "raw_micro_clipped": int(np.sum(np.abs(ny[:, 0]) > 30)),
        "raw_ioi_clipped": int(np.sum(np.abs(cy[:, 1]) > 0.25)),
        "raw_duration_clipped": int(np.sum((ny[:, 1] < np.log(0.4))
                                             | (ny[:, 1] > np.log(1.6)))),
    }


def plan_midi_bytes(notes: list[ListeningNote]) -> bytes:
    _validate_plan(notes)
    midi = mido.MidiFile(ticks_per_beat=1000)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=1000000, time=0))
    track.append(mido.Message("program_change", program=0, time=0))
    events = []
    for note in notes:
        start, end = round(note.start_s * 1000), round(note.end_s * 1000)
        end = max(start + 1, end)
        events.extend([(start, 1, note.pitch, note.velocity), (end, 0, note.pitch, 0)])
    previous = 0
    for tick, kind, pitch, velocity in sorted(events):
        track.append(mido.Message("note_on" if kind else "note_off", note=pitch,
                                  velocity=velocity, time=tick - previous))
        previous = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    buffer = io.BytesIO()
    midi.save(file=buffer)
    return buffer.getvalue()


def render_plan(notes: list[ListeningNote], soundfont: Path,
                *, sample_rate: int = 22050, total_s: float | None = None) -> np.ndarray:
    """Generate stereo float samples without opening an audio device."""
    import tinysoundfont

    _validate_plan(notes)
    synth = tinysoundfont.Synth(gain=-12, samplerate=sample_rate)
    sfid = synth.sfload(str(soundfont))
    synth.program_select(0, sfid, 0, 0)
    synth.control_change(0, 64, 0)
    frames = round((total_s if total_s is not None else max(n.end_s for n in notes) + 1.5)
                   * sample_rate)
    if frames <= max(round(n.end_s * sample_rate) for n in notes):
        raise ValueError("render duration omits a note-off")
    output = np.empty((frames, 2), dtype=np.float32)
    events = []
    for note in notes:
        events.extend([(round(note.start_s * sample_rate), 1, note.pitch, note.velocity),
                       (round(note.end_s * sample_rate), 0, note.pitch, 0)])
    position = 0
    for frame, kind, pitch, velocity in sorted(events) + [(frames, -1, 0, 0)]:
        if frame > position:
            audio = np.frombuffer(synth.generate_simple(frame - position), dtype=np.float32)
            output[position:frame] = audio.reshape(-1, 2)
            position = frame
        if kind == 1:
            synth.noteon(0, pitch, velocity)
        elif kind == 0:
            synth.noteoff(0, pitch)
    synth.sfunload(sfid)
    if not np.isfinite(output).all():
        raise ValueError("renderer produced non-finite audio")
    return output


def write_audio_pair(left: np.ndarray, right: np.ndarray, left_path: Path,
                     right_path: Path, *, sample_rate: int = 22050) -> dict:
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 2:
        raise ValueError("audio pair shape mismatch")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("non-finite audio")
    peak = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))))
    if peak <= 1e-8:
        raise ValueError("silent audio pair")
    gain = 0.8 / peak  # same pair gain, never independent loudness normalization
    for audio, path in ((left, left_path), (right, right_path)):
        if path.exists():
            raise ValueError(f"refusing to overwrite {path}")
        pcm = np.rint(np.clip(audio * gain, -1, 1) * 32767).astype("<i2")
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm.tobytes())
    return {"shared_gain": gain, "pre_gain_peak": peak, "sample_rate": sample_rate,
            "frames": len(left), "clipped_samples": 0}


def original_study_midi(study: int) -> bytes:
    """Three original eight-bar miniatures: arpeggio, chords, variable tempo."""
    if study not in (0, 1, 2):
        raise ValueError("unknown original study")
    notes = []
    progression = [(48, 60, 64, 67), (45, 57, 60, 64), (53, 60, 65, 69), (43, 59, 62, 67)]
    for bar in range(8):
        root, a, b, c = progression[bar % 4]
        onset = bar * 4
        notes.append((onset, 3.5, root))
        for step in range(8):
            chord = [a, b, c][step % 3]
            if study == 1:
                if step % 2:
                    continue
                for pitch in (a, b, c):
                    notes.append((onset + step * 0.5, 0.85, pitch))
            else:
                notes.append((onset + step * 0.5, 0.42, chord + (12 if step >= 4 else 0)))
    ppq = 480
    midi = mido.MidiFile(ticks_per_beat=ppq)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    events = [(0, -2, mido.MetaMessage("time_signature", numerator=4, denominator=4)),
              (0, -1, mido.MetaMessage("set_tempo", tempo=500000 if study != 1 else 600000))]
    if study == 2:
        events.extend([(8 * ppq, -1, mido.MetaMessage("set_tempo", tempo=650000)),
                       (20 * ppq, -1, mido.MetaMessage("set_tempo", tempo=450000))])
    for beat, hold, pitch in notes:
        events.extend([(round(beat * ppq), 1, mido.Message("note_on", note=pitch, velocity=76)),
                       (round((beat + hold) * ppq), 0, mido.Message("note_off", note=pitch))])
    previous = 0
    for tick, _, message in sorted(events, key=lambda row: (row[0], row[1])):
        track.append(message.copy(time=tick - previous))
        previous = tick
    buffer = io.BytesIO()
    midi.save(file=buffer)
    return buffer.getvalue()


def score_inference_arrays(score) -> dict[str, np.ndarray]:
    # Targets returned by this identity call are discarded, not stored or learned.
    features = build_aligned_features(score, score, np.arange(len(score.notes)),
                                      np.zeros(len(score.notes), dtype=np.bool_))
    return {"note_features": features.note_features, "chord_features": features.chord_features,
            "note_to_chord": features.note_to_chord,
            "note_targets": np.zeros_like(features.note_targets),
            "chord_targets": np.zeros_like(features.chord_targets),
            "note_masks": np.zeros_like(features.note_masks),
            "chord_masks": np.zeros_like(features.chord_masks)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_candidates(*, checkpoint: Path, soundfont: Path, output: Path,
                      strength: float = 0.5) -> dict:
    from training.audit_sequence_expression import load_model
    from training.sequence_expression import predict_sample
    from training.tempo_reference import score_timing_reference

    if output.exists():
        raise ValueError("render output must be a new directory")
    model, config = load_model(checkpoint, "cpu")
    if not config.get("timing_reference_sha256"):
        raise ValueError("listening requires a score-tempo-conditioned checkpoint")
    output.mkdir(parents=True)
    (output / "_INCOMPLETE").touch()
    spec = {"study_id": "amp-expression-timing-hold-20261003",
            "study_version": "1.0", "consent_version": "research-share-v1",
            "render_config": {"engine": "tinysoundfont-0.3.7", "soundfont_sha256": sha256(soundfont),
                              "sample_rate": 22050, "instrument": "GeneralUser-GS-2.0.3 Grand Piano",
                              "pair_common_gain": True, "study_domain": "original_local_dev",
                              "evaluated_controls": ["timing", "key_hold"], "velocity": 76,
                              "pedal": "off_for_both", "global_scale": 1.0,
                              "model_checkpoint_sha256": sha256(checkpoint),
                              "production_approved": False}, "trials": []}
    details = []
    for index in range(3):
        score_data = original_study_midi(index)
        (output / f"study_{index + 1}_score.mid").write_bytes(score_data)
        score = parse_midi_bytes(score_data)
        arrays = score_inference_arrays(score)
        reference = score_timing_reference(score_data)
        ny, cy = predict_sample(model, arrays, config=config, timing_reference=reference)
        baseline, baseline_meta = expression_plan(score, arrays["note_to_chord"])
        candidate, candidate_meta = expression_plan(score, arrays["note_to_chord"], ny, cy, strength=strength)
        total_s = max(max(n.end_s for n in baseline), max(n.end_s for n in candidate)) + 1.5
        a = render_plan(baseline, soundfont, total_s=total_s)
        b = render_plan(candidate, soundfont, total_s=total_s)
        left_path = output / f"study_{index + 1}_identity.wav"
        right_path = output / f"study_{index + 1}_candidate.wav"
        audio = write_audio_pair(a, b, left_path, right_path)
        (output / f"study_{index + 1}_identity.mid").write_bytes(plan_midi_bytes(baseline))
        (output / f"study_{index + 1}_candidate.mid").write_bytes(plan_midi_bytes(candidate))
        np.savez_compressed(output / f"study_{index + 1}_raw_predictions.npz", note_predictions=ny,
                            chord_predictions=cy)
        spec["trials"].append({"piece_group": f"original-miniature-{index + 1}", "source_split": "local_dev",
                               "excerpt_id": f"eight-bars-{index + 1}",
                               "baseline": {"candidate_id": "identity-fixed-score", "wav_path": left_path.name},
                               "candidate": {"candidate_id": f"{config['architecture']}-bounded-strength-{strength}",
                                             "wav_path": right_path.name}})
        details.append({"study": index + 1, "baseline": baseline_meta, "candidate": candidate_meta,
                        "audio": audio, "pcm_identical": bool(np.array_equal(a, b)),
                        "human_training_labels_created": 0})
    (output / "pairs.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "render_summary.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    integrity = {p.name: sha256(p) for p in output.iterdir() if p.is_file() and not p.name.startswith("_")}
    (output / "integrity.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    (output / "_INCOMPLETE").unlink()
    (output / "_SUCCESS").touch()
    return {"output": str(output), "trials": 3, "audio_files": 6, "human_preferences": 0}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--soundfont", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--strength", type=float, default=0.5)
    args = parser.parse_args(argv)
    result = render_candidates(**vars(args))
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
