import hashlib
import io
import json
import zipfile

import mido
import numpy as np
import pytest

from training.midi_features import build_aligned_features, parse_midi_bytes
from training.pianocore_dataset import ZIP_PREFIX, write_deterministic_npz
from training.tempo_reference import (
    TimingReferenceError,
    build_timing_reference,
    evaluate_timing_prediction,
    load_timing_reference,
    make_timing_reference,
    reconstruct_ioi_seconds,
    score_timing_reference,
)


def _midi(count=69, *, tempos=((0, 500_000),), signatures=(), duration_ticks=240):
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    events = [(tick, 0, mido.MetaMessage("set_tempo", tempo=value)) for tick, value in tempos]
    events.extend((tick, 1, mido.MetaMessage("time_signature", numerator=num, denominator=den))
                  for tick, num, den in signatures)
    for index in range(count):
        events.extend(((index * 480, 3, mido.Message("note_on", note=60 + index % 12, velocity=70)),
                       (index * 480 + duration_ticks, 2, mido.Message("note_off", note=60 + index % 12))))
    previous = 0
    for tick, _, message in sorted(events, key=lambda value: value[:2]):
        message.time = tick - previous
        previous = tick
        track.append(message)
    buffer = io.BytesIO()
    midi.save(file=buffer)
    return buffer.getvalue()


def _reference(count=69, **score_kwargs):
    return make_timing_reference(_midi(count, **score_kwargs), _midi(count, tempos=((0, 600_000),)),
                                 np.arange(count), np.zeros(count, dtype=bool))


def test_score_tempo_condition_resolves_same_features_different_labels():
    first, changing, performance = _midi(6), _midi(6, tempos=((0, 500_000), (960, 1_000_000))), _midi(6)
    def features(data):
        return build_aligned_features(parse_midi_bytes(data), parse_midi_bytes(performance),
                                      np.arange(6), np.zeros(6, dtype=bool))
    baseline, altered = features(first), features(changing)
    assert np.array_equal(baseline.chord_features, altered.chord_features)
    assert not np.allclose(baseline.chord_targets[:, 1], altered.chord_targets[:, 1])
    a, b = score_timing_reference(first), score_timing_reference(changing)
    assert not np.allclose(a["chord_tempo_log_ratio"], b["chord_tempo_log_ratio"])
    assert not np.allclose(a["note_duration_tempo_log_ratio"], b["note_duration_tempo_log_ratio"])
    assert "global_tempo_scale" not in b
    assert "chord_performance_seconds" not in b


def test_seconds_integration_and_native_ioi_round_trip():
    reference = _reference(8, tempos=((0, 500_000), (960, 1_000_000)), duration_ticks=720)
    np.testing.assert_allclose(reference["chord_score_seconds"], [0, .5, 1, 2, 3, 4, 5, 6])
    assert reference["note_score_duration_seconds"][1] == pytest.approx(1.0)
    predicted = reconstruct_ioi_seconds(reference, reference["chord_log_ioi_ratio"])
    actual = np.diff(reference["chord_performance_seconds"], prepend=0)
    np.testing.assert_allclose(predicted[1:], actual[1:], atol=1e-7)


def test_native_ioi_has_true_one_four_sixteen_bar_metrics():
    reference = _reference()
    report = evaluate_timing_prediction(reference, reference["chord_log_ioi_ratio"])
    assert report["uses_performance_calibration"] is True
    assert report["deployment_evidence"] is False
    assert report["ioi_mae_ms"] < .0001
    assert report["bar_drift"]["1"]["window_count"] == 17
    assert report["bar_drift"]["4"]["window_count"] == 14
    assert report["bar_drift"]["16"]["window_count"] == 2
    assert report["bar_drift"]["16"]["mae_ms"] < .01


def test_untrusted_window_is_skipped_and_no_later_onset_accumulates():
    reference = _reference()
    reference["chord_native_masks"][2, 1] = False
    report = evaluate_timing_prediction(reference, reference["chord_log_ioi_ratio"])
    assert report["aligned_onset_count"] == 1
    assert report["bar_drift"]["1"]["window_count"] == 16
    assert report["bar_drift"]["16"]["window_count"] == 1


def test_meter_changes_redundant_events_and_partial_bar_policy():
    reference = _reference(17, signatures=((0, 4, 4), (960, 4, 4), (1920, 3, 4), (4800, 6, 8)))
    assert reference["chord_bar_index"][4] == 1
    assert reference["chord_bar_index"][7] == 2
    assert reference["chord_bar_index"][10] == 3
    np.testing.assert_array_equal(reference["partial_bar_indices"], [])
    partial = _reference(17, signatures=((0, 4, 4), (960, 3, 4)))
    np.testing.assert_array_equal(partial["partial_bar_indices"], [0])
    assert partial["chord_bar_index"][2] == 1
    report = evaluate_timing_prediction(partial, partial["chord_log_ioi_ratio"])
    assert report["bar_drift"]["1"]["skipped_window_count"] >= 2


def test_calibration_requires_explicit_deployment_scale_and_finite_ioi():
    reference = _reference(6)
    fixed = evaluate_timing_prediction(reference, np.zeros(6), calibration="fixed_score", fixed_scale=1.0)
    assert fixed["uses_performance_calibration"] is False
    assert fixed["ioi_mae_ms"] == pytest.approx(100.0)
    with pytest.raises(TimingReferenceError, match="explicit fixed_scale"):
        reconstruct_ioi_seconds(reference, np.zeros(6), calibration="fixed_score")
    with pytest.raises(TimingReferenceError, match="finite and positive"):
        reconstruct_ioi_seconds(reference, np.full(6, 1000.0))


def _dataset_fixture(tmp_path, split="train"):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    score, performance = _midi(6), _midi(6, tempos=((0, 600_000),))
    idx, interpolated = np.arange(6), np.zeros(6, dtype=bool)
    alignment = io.BytesIO()
    np.savez_compressed(alignment, perf_idx=idx, interpolated=interpolated)
    arrays = build_aligned_features(parse_midi_bytes(score), parse_midi_bytes(performance), idx, interpolated)
    shard_data = {name: getattr(arrays, name) for name in ("note_features", "note_targets", "note_masks",
                                                          "chord_features", "chord_targets", "chord_masks", "note_to_chord")}
    shard_data.update(sample_ids=np.array(["sample-one"]), sample_note_offsets=np.array([0, 6]), sample_chord_offsets=np.array([0, 6]))
    shard_hash = write_deterministic_npz(dataset / "part.npz", shard_data)
    row = {"id": "sample-one", "split": split, "official_split": split, "piece_group": "one-work",
           "performance_dataset": "demo", "note_count": 6, "chord_count": 6, "sample_index": 0,
           "note_offset": 0, "chord_offset": 0, "shard_path": "part.npz", "shard_sha256": shard_hash}
    archive = tmp_path / "refined.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for prefix, value in (("score_midi", score), ("performance_midi", performance), ("alignment", alignment.getvalue())):
            relative = f"data/{prefix}.bin"
            row[f"{prefix}_path"] = relative
            row[f"{prefix}_sha256"] = hashlib.sha256(value).hexdigest()
            handle.writestr(f"{ZIP_PREFIX}/{relative}", value)
    (dataset / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    selection = tmp_path / "ids.txt"
    selection.write_text("sample-one\n", encoding="utf-8")
    return dataset, archive, selection


def test_build_load_source_index_and_hash_binding(tmp_path):
    dataset, archive, selection = _dataset_fixture(tmp_path)
    output = tmp_path / "reference"
    summary = build_timing_reference(dataset_root=dataset, archive_path=archive,
                                     sample_id_files=[selection], output_dir=output)
    assert summary["split_counts"] == {"train": 1}
    loaded = load_timing_reference(output)
    assert len(loaded["sample-one"]["note_to_chord"]) == 6
    sample_path = output / summary["samples"][0]["path"]
    sample_path.write_bytes(sample_path.read_bytes() + b"modified")
    with pytest.raises(TimingReferenceError, match="sample hash mismatch"):
        load_timing_reference(output)


def test_reject_test_before_creating_any_output(tmp_path):
    dataset, archive, selection = _dataset_fixture(tmp_path, "test")
    output = tmp_path / "reference"
    with pytest.raises(TimingReferenceError, match="test/unknown split forbidden"):
        build_timing_reference(dataset_root=dataset, archive_path=archive,
                               sample_id_files=[selection], output_dir=output)
    assert not output.exists()


def test_summary_and_source_hash_changes_rejected(tmp_path):
    dataset, archive, selection = _dataset_fixture(tmp_path)
    output = tmp_path / "reference"
    build_timing_reference(dataset_root=dataset, archive_path=archive,
                           sample_id_files=[selection], output_dir=output)
    (dataset / "part.npz").write_bytes((dataset / "part.npz").read_bytes() + b"modified")
    with pytest.raises(TimingReferenceError, match="source shard hash mismatch"):
        load_timing_reference(output)
    (output / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(TimingReferenceError, match="summary hash mismatch"):
        load_timing_reference(output)
