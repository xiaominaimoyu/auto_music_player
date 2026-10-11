import hashlib
import io
import json
import zipfile

import mido
import numpy as np
import pytest

import training.audit_bar_supervision as audit_module
from training.audit_bar_supervision import audit_bar_supervision
from training.bar_supervision import (
    BarSupervisionError, NORMALIZED_TARGET_DOMAIN, build_bar_spans,
    decompose_bar_targets, evaluate_normalized_timing, span_coverage,
    supervised_span_eligibility, validate_horizons,
)
from training.midi_features import build_aligned_features, parse_midi_bytes
from training.pianocore_dataset import ZIP_PREFIX, write_deterministic_npz
from training.tempo_reference import TimingReferenceError, build_timing_reference, score_timing_reference


def _geometry(count=69, bar_beats=4):
    beats = np.arange(count, dtype=np.float64)
    return {"chord_score_seconds": beats * .5, "chord_score_beats": beats,
            "chord_bar_index": (beats // bar_beats).astype(np.int64),
            "chord_bar_phase": beats % bar_beats / bar_beats,
            "partial_bar_indices": np.empty(0, dtype=np.int64)}


def _masks(count):
    native = np.ones(count, dtype=bool)
    native[0] = False
    return native, np.ones(count, dtype=bool)


def _midi(count=69, tempo=500_000, signatures=()):
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    events = [(0, 0, mido.MetaMessage("set_tempo", tempo=tempo))]
    events.extend((tick, 1, mido.MetaMessage("time_signature", numerator=num, denominator=den))
                  for tick, num, den in signatures)
    for index in range(count):
        events.extend(((index * 480, 3, mido.Message("note_on", note=60 + index % 12, velocity=70)),
                       (index * 480 + 240, 2, mido.Message("note_off", note=60 + index % 12))))
    previous = 0
    for tick, _, message in sorted(events, key=lambda item: item[:2]):
        message.time = tick - previous
        previous = tick
        track.append(message)
    buffer = io.BytesIO()
    midi.save(file=buffer)
    return buffer.getvalue()


def test_exact_real_bar_horizons_and_three_four():
    spans = build_bar_spans(_geometry())
    native, trusted = _masks(spans.chord_count)
    coverage = span_coverage(spans, native, trusted)
    assert [coverage["horizons"][str(h)]["eligible_window_count"] for h in (1, 4, 16)] == [17, 14, 2]
    assert set(spans.ends[spans.horizons == 16] - spans.starts[spans.horizons == 16]) == {64}
    three = build_bar_spans(_geometry(49, 3))
    assert set(three.ends[three.horizons == 16] - three.starts[three.horizons == 16]) == {48}
    assert coverage["pickup_status"].startswith("unknown")
    assert spans.starts.flags.writeable is False


def test_missing_boundary_is_unavailable_not_interpolated():
    reference = _geometry(13)
    for key in tuple(reference):
        if key != "partial_bar_indices":
            reference[key] = np.delete(reference[key], 4)
    spans = build_bar_spans(reference, (1,))
    assert spans.boundary_count == 3
    assert spans.missing_end_counts["1"] == 2
    assert len(spans.starts) == 1
    np.testing.assert_array_equal(spans.starts, [7])
    np.testing.assert_array_equal(spans.ends, [11])


def test_meter_changes_partial_bars_and_redundant_signatures():
    clean = score_timing_reference(_midi(17, signatures=((0, 4, 4), (960, 4, 4), (1920, 3, 4), (4800, 6, 8))))
    spans = build_bar_spans(clean, (1,))
    assert spans.partial_bar_count == 0
    np.testing.assert_array_equal(spans.ends - spans.starts, [4, 3, 3, 3, 3])
    partial = build_bar_spans(score_timing_reference(_midi(17, signatures=((0, 4, 4), (960, 3, 4)))), (1,))
    assert partial.partial_excluded_counts["1"] == 1
    assert 0 not in partial.starts


def test_only_score_allowlist_is_read_and_tempo_changes_preserve_bar_endpoints():
    original = _geometry()
    first = build_bar_spans(original)
    changed = {**original, "global_tempo_scale": np.nan,
               "chord_performance_seconds": np.array([np.inf]), "chord_log_ioi_ratio": "invalid",
               "chord_native_masks": "invalid", "chord_trusted": object()}
    second = build_bar_spans(changed)
    np.testing.assert_array_equal(first.starts, second.starts)
    np.testing.assert_array_equal(first.ends, second.ends)
    changed["chord_score_seconds"] = np.cumsum(np.r_[0., np.linspace(.3, 1.2, 68)])
    third = build_bar_spans(changed)
    np.testing.assert_array_equal(first.ends, third.ends)
    assert not np.array_equal(first.score_ioi_seconds, third.score_ioi_seconds)


def test_native_invalid_gap_and_untrusted_start_rejections_are_disjoint():
    spans = build_bar_spans(_geometry(13), (1,))
    native, trusted = _masks(13)
    native[2] = False
    trusted[4] = False
    result = supervised_span_eligibility(spans, native, trusted)
    assert result["eligible"].tolist() == [False, False, True]
    assert result["untrusted_start"].tolist() == [False, True, False]
    assert result["invalid_ioi_after_trusted_start"].tolist() == [True, False, False]
    assert native[2] == np.bool_(False)


def test_core_ownership_and_max_chords_do_not_mask_long_windows():
    spans = build_bar_spans(_geometry(193))
    native, trusted = _masks(193)
    report = span_coverage(spans, native, trusted, core_size=128)
    long = report["horizons"]["16"]
    assert long["eligible_window_count"] == 33
    assert long["same_core_window_count"] == 17
    assert long["cross_core_window_count"] == 16
    limited = span_coverage(spans, native, trusted, core_size=128, max_chords=64)
    assert limited["horizons"]["16"]["max_chords_excluded_window_count"] == 33
    retained = span_coverage(spans, native, trusted, max_chords=65)
    assert retained["horizons"]["16"]["max_chords_excluded_window_count"] == 0
    assert native.sum() == 192


@pytest.mark.parametrize("horizons", [(), (1, 1), (True,), (1.0,), (0,), (-1,), (np.nan,), None])
def test_invalid_horizons_fail_closed(horizons):
    with pytest.raises(BarSupervisionError):
        validate_horizons(horizons)


@pytest.mark.parametrize("name,value", [
    ("chord_score_seconds", [0., np.nan]), ("chord_score_seconds", [0., 0.]),
    ("chord_score_seconds", [0., -1.]), ("chord_score_seconds", np.zeros((2, 1))),
    ("chord_score_beats", np.arange(5)), ("chord_bar_phase", np.full(69, np.inf)),
    ("chord_bar_phase", np.ones(69)), ("chord_bar_index", np.arange(69, dtype=float)),
    ("partial_bar_indices", np.array([0, 0])), ("partial_bar_indices", np.array([-1])),
])
def test_malformed_geometry_fails_closed(name, value):
    reference = _geometry()
    reference[name] = value
    with pytest.raises(BarSupervisionError):
        build_bar_spans(reference)


def test_missing_geometry_empty_and_duplicate_coordinates_fail():
    reference = _geometry()
    del reference["chord_bar_phase"]
    with pytest.raises(BarSupervisionError, match="missing score"):
        build_bar_spans(reference)
    with pytest.raises(BarSupervisionError):
        build_bar_spans(_geometry(0))
    reference = _geometry()
    reference["chord_bar_phase"][1] = 0
    with pytest.raises(BarSupervisionError):
        build_bar_spans(reference)


@pytest.mark.parametrize("option", [dict(core_size=0), dict(core_size=True), dict(core_size=128.0),
                                   dict(max_chords=-1), dict(max_chords=True), dict(max_chords=64.0)])
def test_capacity_parameters_are_strict_integers(option):
    spans = build_bar_spans(_geometry())
    with pytest.raises(BarSupervisionError):
        span_coverage(spans, *_masks(69), **option)


def test_masks_require_boolean_chord_shape():
    spans = build_bar_spans(_geometry())
    native, trusted = _masks(69)
    for first, second in ((native.astype(int), trusted), (native[:-1], trusted), (native, trusted[:-1])):
        with pytest.raises(BarSupervisionError):
            supervised_span_eligibility(spans, first, second)


def test_bar_target_decomposition_reconstructs_and_normalizes_physical_seconds():
    spans = build_bar_spans(_geometry(13))
    target = np.linspace(-.6, .8, 13)
    native, trusted = _masks(13)
    report = decompose_bar_targets(spans, native, trusted, target)
    assert report["bar_count"] == 3
    assert report["covered_interval_count"] == 12
    assert np.isnan(report["local_log_ioi_ratio"][0])
    assert report["normalization_max_relative_error"] < 1e-14
    assert report["reconstruction_max_error_seconds"] < 1e-14
    for k, (start, end) in enumerate(zip(report["bar_starts"], report["bar_ends"])):
        index = slice(start + 1, end + 1)
        b = report["bar_tempo_log_ratio"][k]
        local = report["local_log_ioi_ratio"][index]
        np.testing.assert_allclose(b + local, target[index])
        assert np.sum(spans.score_ioi_seconds[index] * np.exp(local)) == pytest.approx(np.sum(spans.score_ioi_seconds[index]))
    assert report["slow_curve_fitted"] is False


def test_decomposition_does_not_fill_invalid_or_missing_bars():
    spans = build_bar_spans(_geometry(13))
    native, trusted = _masks(13)
    native[2] = False
    report = decompose_bar_targets(spans, native, trusted, np.zeros(13))
    assert report["bar_count"] == 2
    assert not report["local_available"][:5].any()
    with pytest.raises(BarSupervisionError, match="one-bar"):
        decompose_bar_targets(build_bar_spans(_geometry(13), (4,)), native, trusted, np.zeros(13))


def test_explicit_normalized_domain_checks_human_without_calibrating_predictions():
    spans = build_bar_spans(_geometry(69))
    native, trusted = _masks(69)
    target = np.full(69, np.log(1.1))
    human = spans.score_ioi_seconds * 1.1 * 1.2
    exact = evaluate_normalized_timing(spans, target, target, native, trusted,
                                      human_ioi_seconds=human, global_tempo_scale=1.2)
    identity = evaluate_normalized_timing(spans, np.zeros(69), target, native, trusted)
    assert exact["domain"] == NORMALIZED_TARGET_DOMAIN
    assert exact["ioi_mae_ms"] == 0
    assert exact["native_target_roundtrip_max_error_seconds"] < 1e-15
    assert identity["ioi_mae_ms"] == pytest.approx(50)
    assert identity["bar_drift"]["16"]["mae_ms"] == pytest.approx(3200)
    assert exact["uses_prediction_performance_calibration"] is False
    assert exact["adoption_gate_changed"] is False


def test_no_eligible_coverage_is_null_not_zero():
    spans = build_bar_spans(_geometry(3))
    native = np.zeros(3, dtype=bool)
    trusted = np.ones(3, dtype=bool)
    report = evaluate_normalized_timing(spans, np.zeros(3), np.zeros(3), native, trusted)
    assert report["ioi_mae_ms"] is None
    assert report["bar_drift"]["1"] == {"window_count": 0, "mae_ms": None, "p95_ms": None}
    decomposition = decompose_bar_targets(spans, native, trusted, np.zeros(3))
    assert decomposition["normalization_max_relative_error"] is None


@pytest.mark.parametrize("ratio", [np.full(69, np.nan), np.full(68, 0.), np.full(69, 1000.), np.full(69, -1000.)])
def test_nonfinite_shapes_and_exponent_over_underflow_rejected(ratio):
    spans = build_bar_spans(_geometry())
    with pytest.raises(BarSupervisionError):
        evaluate_normalized_timing(spans, ratio, np.zeros(69), *_masks(69))


@pytest.mark.parametrize("options", [dict(global_tempo_scale=1), dict(human_ioi_seconds=np.ones(69)),
                                   dict(human_ioi_seconds=np.ones(69), global_tempo_scale=0),
                                   dict(human_ioi_seconds=np.full(69, np.nan), global_tempo_scale=1)])
def test_optional_human_calibration_is_strict_and_paired(options):
    spans = build_bar_spans(_geometry())
    with pytest.raises(BarSupervisionError):
        evaluate_normalized_timing(spans, np.zeros(69), np.zeros(69), *_masks(69), **options)


def _timing_fixture(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    score, performance = _midi(69), _midi(69, tempo=600_000)
    idx, interpolated = np.arange(69), np.zeros(69, dtype=bool)
    alignment = io.BytesIO()
    np.savez_compressed(alignment, perf_idx=idx, interpolated=interpolated)
    arrays = build_aligned_features(parse_midi_bytes(score), parse_midi_bytes(performance), idx, interpolated)
    values = {name: getattr(arrays, name) for name in ("note_features", "note_targets", "note_masks",
              "chord_features", "chord_targets", "chord_masks", "note_to_chord")}
    values.update(sample_ids=np.array(["demo"]), sample_note_offsets=np.array([0, 69]), sample_chord_offsets=np.array([0, 69]))
    digest = write_deterministic_npz(dataset / "part.npz", values)
    row = {"id": "demo", "split": "train", "official_split": "train", "piece_group": "demo-piece",
           "performance_dataset": "demo-source", "note_count": 69, "chord_count": 69, "sample_index": 0,
           "note_offset": 0, "chord_offset": 0, "shard_path": "part.npz", "shard_sha256": digest}
    archive = tmp_path / "refined.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for prefix, value in (("score_midi", score), ("performance_midi", performance), ("alignment", alignment.getvalue())):
            path = f"data/{prefix}.bin"
            row[f"{prefix}_path"] = path
            row[f"{prefix}_sha256"] = hashlib.sha256(value).hexdigest()
            handle.writestr(f"{ZIP_PREFIX}/{path}", value)
    (dataset / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    selection = tmp_path / "selection.txt"
    selection.write_text("demo\n", encoding="utf-8")
    reference = tmp_path / "timing"
    build_timing_reference(dataset_root=dataset, archive_path=archive, sample_id_files=[selection], output_dir=reference)
    return reference


def test_audit_writes_sealed_source_bound_diagnostics_not_training_inputs(tmp_path):
    reference = _timing_fixture(tmp_path)
    output = tmp_path / "audit"
    report = audit_bar_supervision(timing_reference=reference, output_dir=output, max_chords=65)
    digest = hashlib.sha256((output / "summary.json").read_bytes()).hexdigest()
    assert (output / "_SUCCESS").read_text().strip() == digest
    assert not (output / "_INCOMPLETE").exists()
    assert report["implementation_hashes"]["bar_supervision.py"]
    assert report["model_input"] is False
    assert report["training_changed"] is False
    assert report["by_split"]["train"]["horizons"]["16"]["eligible_window_count"] == 2
    sample = report["samples"][0]
    path = output / sample["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sample["sha256"]
    with np.load(path, allow_pickle=False) as arrays:
        assert arrays["local_available"].sum() == 68
        assert np.isnan(arrays["local_log_ioi_ratio"][0])
    with pytest.raises(BarSupervisionError, match="new or empty"):
        audit_bar_supervision(timing_reference=reference, output_dir=output)
    with pytest.raises(BarSupervisionError, match="inside the timing"):
        audit_bar_supervision(timing_reference=reference, output_dir=reference / "bad")


def test_audit_rejects_tampered_source_and_writes_nothing(tmp_path):
    reference = _timing_fixture(tmp_path)
    source = json.loads((reference / "summary.json").read_text(encoding="utf-8"))
    (reference / source["samples"][0]["path"]).write_bytes(b"tampered")
    output = tmp_path / "audit"
    with pytest.raises(TimingReferenceError, match="sample hash mismatch"):
        audit_bar_supervision(timing_reference=reference, output_dir=output)
    assert not output.exists()


def test_cli_parsing_and_sealed_output(tmp_path, capsys):
    reference = _timing_fixture(tmp_path)
    output = tmp_path / "cli-audit"
    assert audit_module.main(["--timing-reference", str(reference), "--output-dir", str(output),
                              "--core-size", "128", "--max-chords", "65"]) == 0
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["sample_count"] == 1
    assert (output / "_SUCCESS").read_text().strip() == hashlib.sha256((output / "summary.json").read_bytes()).hexdigest()


@pytest.mark.parametrize("change", ["code", "summary", "sample", "manifest"])
def test_binding_changes_leave_incomplete_not_success(tmp_path, monkeypatch, change):
    reference = _timing_fixture(tmp_path)
    output = tmp_path / "audit"
    source = json.loads((reference / "summary.json").read_text(encoding="utf-8"))
    original_write = audit_module.write_deterministic_npz
    if change == "code":
        original_hash = audit_module._sha256
        calls = 0

        def changed_hash(path):
            nonlocal calls
            if path.name == "bar_supervision.py":
                calls += 1
                if calls > 1:
                    return "changed-code-hash"
            return original_hash(path)

        monkeypatch.setattr(audit_module, "_sha256", changed_hash)
    else:
        def changed_write(path, arrays):
            result = original_write(path, arrays)
            target = {"summary": reference / "summary.json",
                      "sample": reference / source["samples"][0]["path"],
                      "manifest": reference.parent / "dataset" / "manifest.jsonl"}[change]
            target.write_bytes(target.read_bytes() + b" ")
            return result

        monkeypatch.setattr(audit_module, "write_deterministic_npz", changed_write)
    with pytest.raises(BarSupervisionError, match="changed during audit"):
        audit_bar_supervision(timing_reference=reference, output_dir=output)
    assert (output / "_INCOMPLETE").is_file()
    assert not (output / "_SUCCESS").exists()
    assert not (output / "summary.json").exists()


def test_audit_rejects_source_dataset_output(tmp_path):
    reference = _timing_fixture(tmp_path)
    output = reference.parent / "dataset" / "illegal-audit"
    with pytest.raises(BarSupervisionError, match="inside the source dataset"):
        audit_bar_supervision(timing_reference=reference, output_dir=output)
    assert not output.exists()


def test_test_split_cannot_become_diagnostic_input(tmp_path):
    reference = _timing_fixture(tmp_path)
    manifest = reference.parent / "dataset" / "manifest.jsonl"
    row = json.loads(manifest.read_text(encoding="utf-8"))
    row["official_split"] = "test"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    summary_path = reference / "summary.json"
    source = json.loads(summary_path.read_text(encoding="utf-8"))
    source["dataset_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    summary_path.write_text(json.dumps(source), encoding="utf-8")
    (reference / "_SUCCESS").write_text(hashlib.sha256(summary_path.read_bytes()).hexdigest() + "\n", encoding="ascii")
    output = tmp_path / "audit"
    with pytest.raises(TimingReferenceError, match="test reference forbidden"):
        audit_bar_supervision(timing_reference=reference, output_dir=output)
    assert not output.exists()
