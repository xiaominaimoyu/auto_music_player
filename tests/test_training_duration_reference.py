import io

import mido
import numpy as np
import pytest

from training.audit_duration_reference import (
    DurationReferenceError, _sample_review, audit_duration_reference,
    evaluate_duration_prediction, load_duration_reference, make_duration_reference,
    reconstruct_key_duration_seconds,
)
from training.tempo_reference import build_timing_reference, make_timing_reference
from tests.test_training_tempo_reference import _dataset_fixture


def _midi(*, duration=240, tempo=500_000, pedal=(), chords=False):
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    events = [(0, 0, mido.MetaMessage("set_tempo", tempo=tempo))]
    for tick, value in pedal:
        events.append((tick, 1, mido.Message("control_change", control=64, value=value)))
    for index in range(4):
        for pitch in ((60, 64) if chords else (60,)):
            events.extend(((index * 480, 3, mido.Message("note_on", note=pitch, velocity=70)),
                           (index * 480 + duration, 4 if duration == 0 else 2, mido.Message("note_off", note=pitch))))
    previous = 0
    for tick, _, message in sorted(events, key=lambda event: event[:2]):
        message.time = tick - previous
        previous = tick
        track.append(message)
    out = io.BytesIO()
    midi.save(file=out)
    return out.getvalue()


def _reference(*, duration=240, pedal=(), chords=False, interpolated=None):
    score, perf = _midi(chords=chords), _midi(duration=duration, tempo=600_000, pedal=pedal, chords=chords)
    count = 8 if chords else 4
    idx = np.arange(count)
    interp = np.zeros(count, dtype=bool) if interpolated is None else interpolated
    timing = make_timing_reference(score, perf, idx, interp)
    return make_duration_reference(score, perf, idx, interp, timing)


def test_original_native_duration_roundtrip_and_explicit_fixed_score():
    reference = _reference()
    result = evaluate_duration_prediction(reference, reference["note_log_key_duration_ratio"],
                                          predicted_onset_seconds=reference["note_performance_onset_seconds"])
    assert result["key_duration"]["count"] == 4
    assert result["key_duration"]["max_ms"] < .0001
    assert result["key_end"]["max_ms"] < .0001
    assert result["uses_performance_calibration"] is True
    assert result["deployment_evidence"] is False
    fixed = evaluate_duration_prediction(reference, np.zeros(4), calibration="fixed_score", fixed_scale=1.0)
    assert fixed["key_duration"]["mae_ms"] == pytest.approx(50)
    assert fixed["key_end"] == {"count": 0, "mae_ms": None, "p95_ms": None, "max_ms": None}
    with pytest.raises(DurationReferenceError, match="explicit fixed_scale"):
        reconstruct_key_duration_seconds(reference, np.zeros(4), calibration="fixed_score")


def test_no_pedal_is_unknown_not_zero_extension():
    reference = _reference()
    assert np.isnan(reference["note_sustain_end_seconds"]).all()
    assert not reference["note_sustain_end_observed"].any()
    review = _sample_review(reference)
    assert review["sustain_known_count"] == 0
    assert review["sustain_unknown_count"] == 4
    assert review["native_singleton_micro_count"] == 4
    assert review["native_singleton_nonzero_micro_count"] == 0


def test_sustain_release_separate_from_key_up_and_reattack_flag():
    reference = _reference(pedal=((0, 127), (960, 0)))
    assert reference["note_performance_key_end_seconds"][0] == pytest.approx(.3)
    assert reference["note_sustain_end_seconds"][0] == pytest.approx(1.2)
    assert reference["note_same_pitch_sustain_reattack"][0]
    assert not reference["note_same_pitch_key_overlap"][0]
    assert reference["note_sustain_end_observed"].all()
    assert _sample_review(reference)["sustain_extended_count"] == 2


def test_missing_pedal_release_is_censored_and_not_fabricated():
    reference = _reference(pedal=((0, 127),))
    assert reference["note_sustain_end_censored"].all()
    assert np.isnan(reference["note_sustain_end_seconds"]).all()
    assert not reference["note_sustain_end_observed"].any()


def test_boundary_pedal_release_and_zero_duration_floor_not_repaired():
    reference = _reference(duration=0, pedal=((0, 127), (480, 0)))
    result = evaluate_duration_prediction(reference, reference["note_log_key_duration_ratio"])
    assert result["key_duration"]["mae_ms"] == pytest.approx(.1, abs=1e-7)
    assert reference["note_performance_duration_floored"].all()
    assert reference["note_native_masks"][:, 1].all()
    assert reference["note_pedal_boundary_at_keyoff"][:2].all()


def test_native_interpolation_and_missing_projected_notes_excluded_with_coverage():
    reference = _reference(interpolated=np.array([False, True, False, False]))
    predicted = np.array([.3, np.nan, .3, np.nan])
    result = evaluate_duration_prediction(reference, predicted_duration_seconds=predicted,
        prediction_valid=np.array([True, False, True, False]), calibration="fixed_score", fixed_scale=1)
    assert result["native_duration_count"] == 3
    assert result["missing_prediction_count"] == 1
    assert result["key_duration"]["count"] == 2
    assert not reference["note_native_masks"][1].any()


def test_multi_note_micro_is_not_singleton_and_key_overlap_is_indicator_only():
    reference = _reference(duration=720, chords=True)
    review = _sample_review(reference)
    assert review["native_singleton_micro_count"] == 0
    assert review["ambiguity_flags"]["note_same_pitch_key_overlap"] == 6
    assert reference["note_native_masks"][:, 1].all()
    assert review["score_explicit_tempo_event_count"] == 1


def test_bad_prediction_and_mismatched_native_binding_rejected():
    reference = _reference()
    with pytest.raises(DurationReferenceError, match="exactly one"):
        evaluate_duration_prediction(reference)
    with pytest.raises(DurationReferenceError, match="finite and positive"):
        reconstruct_key_duration_seconds(reference, np.full(4, 1000.0))
    with pytest.raises(DurationReferenceError, match="boolean"):
        evaluate_duration_prediction(reference, np.zeros(4), prediction_valid=np.ones(4))
    score, perf = _midi(), _midi(tempo=600_000)
    timing = make_timing_reference(score, perf, np.arange(4), np.zeros(4, dtype=bool))
    timing["note_native_masks"][0, 1] = False
    with pytest.raises(DurationReferenceError, match="native labels differ"):
        make_duration_reference(score, perf, np.arange(4), np.zeros(4, dtype=bool), timing)


def test_audit_original_source_and_output_integrity_binding(tmp_path):
    dataset, archive, ids = _dataset_fixture(tmp_path)
    timing = tmp_path / "timing"
    build_timing_reference(dataset_root=dataset, archive_path=archive, sample_id_files=[ids], output_dir=timing)
    output = tmp_path / "duration"
    report = audit_duration_reference(timing_reference=timing, output_dir=output)
    assert report["split_counts"] == {"train": 1}
    loaded = load_duration_reference(output)
    assert loaded["sample-one"]["note_performance_index"].tolist() == list(range(6))
    with pytest.raises(DurationReferenceError, match="new or empty"):
        audit_duration_reference(timing_reference=timing, output_dir=output)
    sample_path = output / report["samples"][0]["path"]
    sample_path.write_bytes(sample_path.read_bytes() + b"tampered")
    with pytest.raises(DurationReferenceError, match="sample hash mismatch"):
        load_duration_reference(output)
