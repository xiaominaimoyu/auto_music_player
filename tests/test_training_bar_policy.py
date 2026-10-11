from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.audit_sequence_expression import attach_checkpoint_bar_geometry, bar_layout_coverage
from training.benchmark_sequence_inference import benchmark
from training.sequence_expression import (
    SequenceSample, attach_bar_geometry, collate_windows, fit_input_normalizers, make_model, make_windows,
    make_windows_from_config, predict_sample, prepare_sample,
)
from training.train_expression import TrainingRunError, _sha256
from training.train_sequence_expression import _run_epoch, train, write_integrity

BAR_MAX = 16
BAR_CONTEXT = 2


def _geometry_reference(bar_sizes, beat_seconds=0.5):
    seconds, beats, bars, phases = [], [], [], []
    for index, size in enumerate(bar_sizes):
        for position in range(size):
            bars.append(index)
            phases.append(position / size)
            beats.append(4 * index + 4 * position / size)
            seconds.append((4 * index + 4 * position / size) * beat_seconds)
    count = len(seconds)
    return {"score_seconds_per_beat_reference": beat_seconds,
            "chord_tempo_log_ratio": np.zeros(count), "note_duration_tempo_log_ratio": np.zeros(count * 2),
            "chord_score_seconds": np.asarray(seconds, dtype=np.float64),
            "chord_score_beats": np.asarray(beats, dtype=np.float64),
            "chord_bar_index": np.asarray(bars, dtype=np.int64), "chord_bar_phase": np.asarray(phases),
            "partial_bar_indices": np.empty(0, dtype=np.int64),
            # These performance-derived fields must never reach the planner.
            "global_tempo_scale": 997.0, "chord_performance_seconds": np.arange(count) * 3.0}


def _arrays(chords):
    mapping = np.repeat(np.arange(chords), 2)
    note = np.zeros((len(mapping), 15), dtype=np.float32)
    chord = np.zeros((chords, 11), dtype=np.float32)
    note[:, 3] = mapping / 2.0
    chord[:, 0] = np.arange(chords) / 2.0
    targets = np.zeros((chords, 2), dtype=np.float32)
    targets[1:, 1] = np.log(1.1)
    masks = np.ones((chords, 2), dtype=bool)
    masks[0, 1] = False
    return {"note_features": note, "chord_features": chord, "note_to_chord": mapping,
            "note_targets": np.zeros((len(mapping), 6), dtype=np.float32),
            "note_masks": np.ones((len(mapping), 6), dtype=bool),
            "chord_targets": targets, "chord_masks": masks}


def _sample(bar_sizes=(4, 4, 4, 4, 4, 4), horizons=(1, 3), chords=24):
    reference = _geometry_reference(bar_sizes)
    return prepare_sample("bar-work", _arrays(chords), timing_reference=reference), reference, horizons


def _config(sample, **extra):
    config = {"architecture": "mlp", "hidden_dim": 8, "dropout": 0.0, "core_size": 12, "context_size": BAR_CONTEXT,
              "batch_size": 4, "target_shuffle": False, "relative_position": True,
              "note_target_means": [0.0] * 6, "note_target_scales": [1.0] * 6,
              "chord_target_means": [0.0] * 2, "chord_target_scales": [1.0] * 2,
              "onset_weight": 0.0, "micro_weight": 0.0, "duration_weight": 0.0, "velocity_weight": 0.0,
              "pedal_weight": 0.0, "ioi_weight": 1.0, "cumulative_weight": 0.0,
              **fit_input_normalizers([sample])}
    config.update(extra)
    return config


def _bar_config(sample, **extra):
    return _config(sample, window_policy="bar", max_window_chords=BAR_MAX, bar_horizons=[1, 3], **extra)


def _naive_retained(sample, windows):
    rows = np.asarray(sample.bar_span_rows, dtype=np.int64)
    return sorted((int(window.core_start), sorted(map(tuple, rows[(rows[:, 0] >= window.core_start)
                                                                  & (rows[:, 1] < window.core_end)].tolist())))
                  for window in windows)


def test_none_max_window_chords_falls_back_to_the_planner_default():
    sample, reference, horizons = _sample()
    prepared = attach_bar_geometry(sample, reference, horizons=horizons)
    recorded = _bar_config(prepared)
    recorded["max_window_chords"] = None
    explicit = _bar_config(prepared)
    explicit["max_window_chords"] = 512
    assert make_windows_from_config(prepared, recorded) == make_windows_from_config(prepared, explicit)


def test_default_policy_reproduces_the_legacy_fixed_layout_exactly():
    sample, _, _ = _sample()
    config = _config(sample)
    assert make_windows_from_config(sample, config) == list(make_windows(sample, 12, BAR_CONTEXT))
    batch = collate_windows(make_windows_from_config(sample, config), config)
    assert not set(batch) & {"bar_span_rows", "bar_span_mask", "bar_ioi_mask", "score_ioi_seconds"}


def test_bar_policy_tiles_cores_once_and_bounds_every_window():
    sample, reference, horizons = _sample()
    prepared = attach_bar_geometry(sample, reference, horizons=horizons)
    windows = make_windows_from_config(prepared, _bar_config(prepared))
    capacity = BAR_MAX - 2 * BAR_CONTEXT
    owners = np.zeros(len(prepared.chord_features), dtype=np.int64)
    for window in windows:
        owners[window.core_start:window.core_end] += 1
        assert 0 < window.core_end - window.core_start <= capacity
        assert window.end - window.start <= BAR_MAX
        assert window.start == max(0, window.core_start - BAR_CONTEXT)
        assert window.end == min(len(prepared.chord_features), window.core_end + BAR_CONTEXT)
    assert np.all(owners == 1)
    assert [window.core_start for window in windows] == [0, 12]


def test_spans_ending_on_a_core_boundary_are_excluded_never_truncated():
    sample, reference, horizons = _sample()
    prepared = attach_bar_geometry(sample, reference, horizons=horizons)
    windows = make_windows_from_config(prepared, _bar_config(prepared))
    assert _naive_retained(prepared, windows) == [(0, [(0, 4), (4, 8)]), (12, [(12, 16), (16, 20)])]
    rows = np.asarray(prepared.bar_span_rows, dtype=np.int64)
    kinds = np.asarray(prepared.bar_span_horizons, dtype=np.int64)
    for window in windows:
        stored = np.asarray(window.bar_span_rows, dtype=np.int64)
        assert stored.ndim == 2 and stored.shape[1] == 2
        assert len(window.bar_span_horizons) == len(stored)
        keep = (rows[:, 0] >= window.core_start) & (rows[:, 1] < window.core_end)
        assert np.array_equal(stored, rows[keep])
        assert np.array_equal(np.asarray(window.bar_span_horizons, dtype=np.int64), kinds[keep])
    assert all(np.all(np.asarray(window.bar_span_horizons) == 1) for window in windows)


def test_planning_never_consumes_labels_or_performance_fields():
    sample, reference, horizons = _sample()
    prepared = attach_bar_geometry(sample, reference, horizons=horizons)
    hostile = dict(reference, global_tempo_scale=4.0, chord_performance_seconds=np.arange(24) * 99.0,
                  chord_native_masks=np.zeros((24, 2), dtype=bool), note_native_masks=np.zeros((48, 6), dtype=bool))
    assert np.array_equal(attach_bar_geometry(sample, hostile, horizons=horizons).bar_span_rows, prepared.bar_span_rows)
    flipped = prepare_sample("bar-work", dict(_arrays(24), chord_masks=np.zeros((24, 2), dtype=bool)),
                             timing_reference=reference)
    assert np.array_equal(attach_bar_geometry(flipped, reference, horizons=horizons).bar_span_rows,
                          prepared.bar_span_rows)


def test_geometry_rejects_a_clock_that_disagrees_with_the_sample():
    sample, reference, horizons = _sample()
    slowed = dict(reference, chord_score_seconds=reference["chord_score_seconds"] * 2.0)
    with pytest.raises(TrainingRunError, match="score clock differs"):
        attach_bar_geometry(sample, slowed, horizons=horizons)
    no_clock = prepare_sample("bare", _arrays(24))
    with pytest.raises(TrainingRunError, match="requires pure score IOI seconds"):
        attach_bar_geometry(no_clock, reference, horizons=horizons)


def test_bar_policy_without_geometry_fails_instead_of_falling_back_to_128():
    sample, _, _ = _sample()
    with pytest.raises(TrainingRunError, match="requires score-only bar geometry"):
        make_windows_from_config(sample, _bar_config(sample))
    with pytest.raises(TrainingRunError, match="requires score-only bar geometry"):
        predict_sample(make_model(_bar_config(sample)), sample, config=_bar_config(sample))
    with pytest.raises(TrainingRunError, match="unknown window policy"):
        make_windows_from_config(sample, _config(sample, window_policy="bars"))


def test_collate_carries_window_local_spans_and_only_native_valid_ones():
    sample, reference, horizons = _sample()
    arrays = _arrays(24)
    arrays["chord_masks"] = arrays["chord_masks"].copy()
    arrays["chord_masks"][3, 1] = False
    prepared = attach_bar_geometry(prepare_sample("bar-work", arrays, timing_reference=reference),
                                   reference, horizons=horizons)
    windows = make_windows_from_config(prepared, _bar_config(prepared))
    batch = collate_windows(windows, _bar_config(prepared, bar_tempo_weight=1.0))
    rows = batch["bar_span_rows"].numpy()
    mask = batch["bar_span_mask"].numpy()
    candidates = batch["bar_span_candidates"].numpy()
    widths = np.array([window.end - window.start for window in windows])
    for index, window in enumerate(windows):
        local = rows[index][:int(candidates[index].sum())]
        assert np.all(local >= 0) and np.all(local < widths[index])
        assert np.all(local[:, 1] - local[:, 0] >= 1)
        assert np.array_equal(local + window.start, np.asarray(window.bar_span_rows, dtype=np.int64))
    assert mask[candidates].sum() == candidates.sum() - 1
    ioi = batch["bar_ioi_mask"].numpy()
    for index, window in enumerate(windows):
        cstart, cend = window.core_start - window.start, window.core_end - window.start
        assert not ioi[index, :cstart + 1].any()
        assert np.array_equal(ioi[index, cstart + 1:cend],
                              prepared.chord_masks[window.core_start + 1:window.core_end, 1])
        assert not ioi[index, cend:].any()


def test_epoch_reports_bar_counts_and_backpropagates_into_the_log_ioi_head():
    sample, reference, horizons = _sample()
    prepared = attach_bar_geometry(sample, reference, horizons=horizons)
    config = _bar_config(prepared, bar_elapsed_weight=0.5, bar_tempo_weight=1.0, bar_local_weight=0.25)
    model = make_model(config)
    metrics = _run_epoch(model, [prepared], config, device=torch.device("cpu"), optimizer=None, seed=1)
    assert metrics["window_policy"] == "bar"
    assert metrics["bar_supervised_steps"] > 0 and metrics["bar_loss"] is not None
    totals = metrics["bar_horizon_counts"]
    # "candidate" spans are the ones a core fully contains; the span ending on a core
    # boundary was already dropped by the planner and surfaces in audit coverage instead.
    assert totals["1"] == {**totals["1"], "candidate_spans": 4, "eligible_spans": 4}
    assert totals["3"]["candidate_spans"] == 0 and totals["3"]["eligible_spans"] == 0
    assert all(entry["candidate_spans"] >= entry["eligible_spans"] for entry in totals.values())
    assert totals["1"]["interval_rows"] > 0
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    stepped = _run_epoch(model, [prepared], config, device=torch.device("cpu"), optimizer=optimizer, seed=1)
    assert stepped["bar_loss"] is not None
    grads = {name: parameter.grad for name, parameter in model.named_parameters()}
    assert any(parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
               for parameter in model.parameters())


def test_zero_bar_weights_keep_the_legacy_metric_shape_and_skip_span_tensors():
    sample, reference, horizons = _sample()
    prepared = attach_bar_geometry(sample, reference, horizons=horizons)
    config = _bar_config(prepared)
    model = make_model(config)
    metrics = _run_epoch(model, [prepared], config, device=torch.device("cpu"), optimizer=None, seed=1)
    assert metrics["bar_loss"] is None and metrics["bar_supervised_steps"] == 0
    assert set(metrics["bar_component_losses"]) == {"bar_elapsed_weight", "bar_tempo_weight", "bar_local_weight"}
    assert all(value is None for value in metrics["bar_component_losses"].values())


@pytest.mark.parametrize(("kwargs", "message"), (
    ({"window_policy": "fixed", "bar_tempo_weight": 1.0}, "bar loss weights require window_policy=bar"),
    ({"window_policy": "bar", "bar_tempo_weight": 1.0, "target_shuffle": True}, "target shuffle with bar supervision"),
    ({"window_policy": "sideways"}, "window_policy must be fixed or bar"),
    ({"window_policy": "bar", "max_window_chords": 4}, "must exceed twice the context size"),
    ({"window_policy": "bar", "bar_elapsed_weight": -1.0}, "bar loss weights must be finite and nonnegative"),
))
def test_train_rejects_unsafe_bar_configurations_before_reading_data(kwargs, message):
    with pytest.raises(TrainingRunError, match=message):
        train(dataset="missing-dataset", output="missing-output", architecture="mlp",
              train_sample_id_file="missing-train", validation_sample_id_file="missing-validation",
              **kwargs)


def test_bar_policy_requires_the_pure_score_timing_reference():
    with pytest.raises(TrainingRunError, match="requires a pure-score timing reference"):
        train(dataset="missing-dataset", output="missing-output", architecture="mlp",
              train_sample_id_file="missing-train", validation_sample_id_file="missing-validation",
              window_policy="bar", max_window_chords=512)


def test_audit_geometry_helpers_passthrough_and_report_coverage():
    sample, reference, horizons = _sample()
    prepared = attach_bar_geometry(sample, reference, horizons=horizons)
    assert attach_checkpoint_bar_geometry([sample], None, {"window_policy": "fixed"}) == [sample]
    attached = attach_checkpoint_bar_geometry([sample], {"bar-work": reference},
                                              {"window_policy": "bar", "bar_horizons": list(horizons),
                                               "context_size": BAR_CONTEXT, "max_window_chords": BAR_MAX})
    assert np.array_equal(attached[0].bar_span_rows, prepared.bar_span_rows)
    with pytest.raises(TrainingRunError, match="unknown window policy"):
        attach_checkpoint_bar_geometry([sample], None, {"window_policy": "sideways"})
    coverage = bar_layout_coverage(attached, _bar_config(prepared))
    assert coverage["window_policy"] == "bar" and coverage["core_count"] == 2
    assert coverage["horizons"]["1"]["geometry_spans"] == 5
    assert coverage["horizons"]["1"]["retained_spans"] == 4
    assert coverage["horizons"]["1"]["eligible_spans"] == 4
    assert coverage["horizons"]["3"]["retained_spans"] == 0


def _benchmark_fixture(tmp_path, config_extra, arrays_extra):
    audit, run, output = tmp_path / "audit", tmp_path / "run", tmp_path / "out"
    audit.mkdir(parents=True)
    run.mkdir(parents=True)
    checkpoint = run / "checkpoint_best.pt"
    checkpoint.write_bytes(b"local-test-artifact")
    np.savez_compressed(audit / "score.npz", note_features=np.zeros((24, 17), np.float32),
                        chord_features=np.zeros((24, 13), np.float32), note_to_chord=np.repeat(np.arange(24), 2),
                        **arrays_extra)
    (audit / "report.json").write_text(json.dumps({"checkpoint_sha256": _sha256(checkpoint),
                                                  "prediction_files": {"sample": "score.npz"}}))
    write_integrity(audit)
    write_integrity(run)
    config = {"architecture": "gru", **config_extra}
    return audit, checkpoint, output, config


def test_benchmark_requires_bar_geometry_and_keeps_the_planned_layout(tmp_path):
    sample, reference, horizons = _sample()
    prepared = attach_bar_geometry(sample, reference, horizons=horizons)
    extra = {"bar_span_rows": prepared.bar_span_rows, "bar_span_horizons": prepared.bar_span_horizons,
             "bar_boundary_indices": prepared.bar_boundary_indices,
             "bar_requested_horizons": prepared.bar_requested_horizons}
    config_extra = {"window_policy": "bar", "context_size": BAR_CONTEXT, "max_window_chords": BAR_MAX,
                    "bar_horizons": list(horizons)}
    audit, checkpoint, output, config = _benchmark_fixture(tmp_path, config_extra, extra)
    seen = []

    def prediction(model, loaded, **kwargs):
        assert loaded.bar_span_rows is not None and loaded.score_ioi_seconds is None
        seen.append(loaded)
        return np.zeros((24, 6)), np.zeros((24, 2))

    threads = torch.get_num_threads()
    try:
        with patch("training.benchmark_sequence_inference.load_model", return_value=(object(), config)), \
             patch("training.benchmark_sequence_inference.verify_inventory"), \
             patch("training.benchmark_sequence_inference.predict_sample", side_effect=prediction):
            result = benchmark(audit=audit, checkpoint=checkpoint, output=output, cpu_threads=2)
    finally:
        torch.set_num_threads(threads)
    assert result["window_policy"] == "bar" and result["bar_geometry_sample_count"] == 1
    assert result["planned_core_count_smallest_sample"] == 2
    assert result["fixed128_core_count_smallest_sample"] == 1

    audit2, checkpoint2, output2, _ = _benchmark_fixture(tmp_path / "missing", config_extra, {})
    with patch("training.benchmark_sequence_inference.load_model", return_value=(object(), config)), \
         patch("training.benchmark_sequence_inference.verify_inventory"):
        with pytest.raises(ValueError, match="lacks score-only geometry"):
            benchmark(audit=audit2, checkpoint=checkpoint2, output=output2, cpu_threads=2)
