from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from tools.bootstrap_sequence_vs_baselines import (RULE_BASELINES, bootstrap_difference, collapse_to_pieces,
                                                   main, rule_projection_errors)
from training.train_expression import TrainingRunError


def _with_rule_errors(errors: dict, channel: str, sample_ids, session: float, centred: float) -> dict:
    for sample_id in sample_ids:
        errors[(channel, RULE_BASELINES[0], sample_id)] = session
        errors[(channel, RULE_BASELINES[1], sample_id)] = centred
    return errors


def test_pieces_carry_equal_weight_no_matter_how_many_rows_they_have():
    supervised = {"short": np.array([1.0, 1.0]), "long": np.array([3.0] * 1000)}
    errors = _with_rule_errors({("ch", "runA", "short"): 1.0, ("ch", "runA", "long"): 3.0},
                               "ch", ("short", "long"), 2.0, 2.0)
    identities = {"short": ("ASAP", "pieceA"), "long": ("ASAP", "pieceB")}
    pieces = collapse_to_pieces("ch", supervised, errors, ["runA"], identities)
    assert list(pieces) == ["ASAP::pieceA", "ASAP::pieceB"]
    # row weighting would have said 2.996; piece equal weighting says 2.0
    assert float(np.mean([piece["model"] for piece in pieces.values()])) == pytest.approx(2.0)


def test_a_missing_rules_error_is_refused_rather_than_dropped_from_a_paired_test():
    # every predictor has to be measured on the same pieces, or "paired" stops meaning anything
    supervised = {"r1": np.array([1.0, 2.0])}
    errors = {("ch", "runA", "r1"): 0.5}
    with pytest.raises(TrainingRunError, match="missing rules_session"):
        collapse_to_pieces("ch", supervised, errors, ["runA"], {"r1": ("ASAP", "g")})


def test_seeds_and_recordings_inside_one_piece_are_averaged_before_comparison():
    supervised = {"r1": np.array([2.0, 2.0]), "r2": np.array([4.0, 4.0])}
    errors = _with_rule_errors({("ch", "seedA", "r1"): 1.0, ("ch", "seedB", "r1"): 3.0,
                                ("ch", "seedA", "r2"): 5.0, ("ch", "seedB", "r2"): 7.0},
                               "ch", ("r1", "r2"), 1.0, 0.5)
    identities = {"r1": ("ATEPP", "shared"), "r2": ("ATEPP", "shared")}
    pieces = collapse_to_pieces("ch", supervised, errors, ["seedA", "seedB"], identities)
    only = pieces["ATEPP::shared"]
    # seeds first: r1 -> 2.0, r2 -> 6.0; then the two recordings of the piece are equal weight
    assert only["model"] == pytest.approx(4.0)
    assert only["recordings"] == 2 and only["rows"] == 4
    assert only["predict_zero"] == pytest.approx(3.0)   # mean(|2|, |4|)
    assert only["predict_recording_median"] == pytest.approx(0.0)  # every recording is constant
    assert only[RULE_BASELINES[0]] == pytest.approx(1.0)


def test_a_missing_seed_is_simply_left_out_rather_than_guessing():
    supervised = {"r1": np.array([1.0, 2.0])}
    errors = _with_rule_errors({("ch", "seedA", "r1"): 0.5}, "ch", ("r1",), 0.4, 0.3)
    pieces = collapse_to_pieces("ch", supervised, errors, ["seedA", "seedGone"], {"r1": ("ASAP", "g")})
    assert pieces["ASAP::g"]["model"] == pytest.approx(0.5)
    assert pieces["ASAP::g"]["recordings"] == 1


def test_paired_difference_is_negative_when_the_model_is_better():
    pairs = [(1.0, 2.0)] * 8
    report = bootstrap_difference(pairs, draws=200, seed=7)
    assert report["mean_paired_difference"] == pytest.approx(-1.0)
    assert report["interval_high"] < 0.0 and report["supports_improvement"] is True
    assert report["probability_model_worse"] == pytest.approx(0.0)


def test_the_interval_crossing_zero_does_not_support_improvement():
    pairs = [(-1.0, 0.0), (1.0, 0.0)] * 6
    report = bootstrap_difference(pairs, draws=400, seed=11)
    assert report["mean_paired_difference"] == pytest.approx(0.0)
    assert report["interval_low"] <= 0.0 <= report["interval_high"]
    assert report["supports_improvement"] is False


def test_bootstrap_is_reproducible_for_a_registered_seed():
    # differences must vary, otherwise resampling cannot produce a spread at all
    pairs = [(float(value % 5) - 2.0, 0.0) for value in range(12)]
    first = bootstrap_difference(pairs, draws=300, seed=20261003)
    second = bootstrap_difference(pairs, draws=300, seed=20261003)
    other = bootstrap_difference(pairs, draws=300, seed=999)
    assert first == second
    assert first["interval_low"] != other["interval_low"]
    assert first["interval_low"] < first["mean_paired_difference"] < first["interval_high"]


def test_fewer_than_five_pieces_reports_no_interval():
    report = bootstrap_difference([(1.0, 2.0)] * 4, draws=200, seed=1)
    assert report["pieces"] == 4 and "note" in report and "interval_low" not in report


def test_the_locked_test_split_and_stupid_draw_counts_are_refused(tmp_path: Path):
    base = ["--dataset", str(tmp_path), "--timing-reference", str(tmp_path), "--sample-id-file", str(tmp_path),
            "--output", str(tmp_path / "out"), "--runs", str(tmp_path)]
    with pytest.raises(SystemExit, match="locked test"):
        main(base + ["--split", "test"])
    with pytest.raises(SystemExit, match="draws"):
        main(base + ["--draws", "10"])


def test_the_report_is_sealed_and_names_the_strongest_baseline(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch):
    module = "tools.bootstrap_sequence_vs_baselines"
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    rows = []
    for index in range(6):
        rows.append(json.dumps({"id": f"s{index}", "performance_dataset": "ASAP",
                                "piece_group": f"piece{index}"}))
    (dataset / "manifest.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    ids = tmp_path / "ids.txt"
    ids.write_text("\n".join(f"s{index}" for index in range(6)) + "\n", encoding="utf-8")

    class _Sample:
        """Two supervised rows per recording, values v and v+2, so no baseline is degenerate."""

        def __init__(self, sample_id: str, value: float) -> None:
            self.sample_id = sample_id
            self.chord_masks = np.ones((2, 2), dtype=bool)
            # the first chord has no preceding interval, exactly as the real native masks say
            self.chord_masks[0, 1] = False
            self.chord_targets = np.array([[value, value + 2.0], [value + 2.0, value]], dtype=np.float32)
            self.note_masks = np.ones((2, 6), dtype=bool)
            self.note_targets = np.zeros((2, 6), dtype=np.float32)
            self.note_targets[:, 1] = [value, value + 2.0]
            self.note_features = np.zeros((2, 15), dtype=np.float32)
            self.note_features[:, 0] = [0.0, 0.25]
            self.chord_features = np.zeros((2, 3), dtype=np.float32)
            self.note_to_chord = np.array([0, 1], dtype=np.int32)

    samples = [_Sample(f"s{index}", float(index + 1)) for index in range(6)]
    monkeypatch.setattr(f"{module}.load_selected_samples", lambda *args, **kwargs: samples)
    monkeypatch.setattr(f"{module}.load_timing_reference",
                        lambda path: {sample.sample_id: _reference_row([0.0, 1.0], [1.0, 1.0], [0, 1])
                                      for sample in samples})

    def _fake_load_model(path, device):
        config = {"note_target_means": [0.0] * 6, "note_target_scales": [1.0] * 6,
                  "chord_target_means": [0.0, 0.0], "chord_target_scales": [1.0, 1.0],
                  "window_policy": "fixed"}
        return object(), config

    monkeypatch.setattr(f"{module}.load_model", _fake_load_model)
    # predict every supervised label as zero: the model then equals the predict-zero baseline
    monkeypatch.setattr(f"{module}.predict_sample",
                        lambda model, sample, config, device: (np.zeros((2, 6)), np.zeros((2, 2))))
    reference = tmp_path / "ref"
    reference.mkdir()
    (reference / "summary.json").write_text("{}", encoding="utf-8")
    run = tmp_path / "runA"
    run.mkdir()
    output = tmp_path / "out"
    assert main(["--dataset", str(dataset), "--timing-reference", str(reference),
                 "--sample-id-file", str(ids), "--output", str(output), "--runs", str(run),
                 "--draws", "200"]) == 0
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    onset = report["channels"]["onset_delta_ms"]
    assert report["adoption_evidence"] is False and onset["pieces"] == 6 and onset["rows"] == 12
    table = onset["piece_equal_weight_mae"]
    # each piece holds v and v+2: a per-recording constant costs exactly 1.0, while predicting
    # zero costs the mean of v+1 over v=1..6, which is also what this degenerate model does
    assert table["predict_recording_median"] == pytest.approx(1.0)
    assert table["model"] == pytest.approx(table["predict_zero"]) == pytest.approx(4.5)
    assert onset["strongest_baseline"] == "predict_recording_median"
    assert onset["baseline_is_degenerate"] is False
    assert onset["improvement_over_strongest_baseline"] == pytest.approx(1.0 - 4.5 / 1.0)
    assert onset["paired_bootstrap"]["predict_recording_median"]["probability_model_worse"] == pytest.approx(1.0)
    assert onset["gate"]["passed_numeric_gate"] is False
    # the shipped rules are now measured in the same pieces, in the same units
    assert set(table) == {"model", "predict_zero", "predict_recording_median", *RULE_BASELINES}
    assert all(table[baseline] > 0.0 for baseline in RULE_BASELINES)
    assert table["rules_session"] >= table["rules_centred"] - 1e-9  # Jensen: averaging errors >= error of mean
    probe = onset["policy_gate_probe"]
    assert probe["required_relative_improvement"] == pytest.approx(0.10)
    assert probe["measured_improvement_over_rules"] == pytest.approx(1.0 - 4.5 / table["rules_session"])
    assert probe["required_baselines"] == ["identity", "rules", "zero_model_adapter"]
    report_rules = report["rules_projection"]
    assert report_rules["sessions_per_recording"] == 8 and report_rules["leap_rule"]["fires_on_this_corpus"] is False
    assert report_rules["supervised_rows"]["log_ioi_ratio"] == 6
    assert (output / "_SUCCESS").read_text(encoding="ascii").strip()


def _projection_sample(chord_count: int, onsets: list[float], note_beats: list[float],
                       labels: list[float], *, first_ioi_supervised: bool = False):
    class _S:
        pass

    sample = _S()
    sample.sample_id = "s"
    sample.chord_features = np.zeros((chord_count, 3), dtype=np.float32)
    sample.note_features = np.zeros((len(note_beats), 15), dtype=np.float32)
    sample.note_to_chord = np.arange(len(note_beats), dtype=np.int32)
    sample.chord_masks = np.ones((chord_count, 2), dtype=bool)
    if not first_ioi_supervised:
        sample.chord_masks[0, 1] = False
    sample.chord_targets = np.zeros((chord_count, 2), dtype=np.float32)
    sample.note_masks = np.ones((len(note_beats), 6), dtype=bool)
    sample.note_targets = np.zeros((len(note_beats), 6), dtype=np.float32)
    sample.note_targets[:, 1] = labels
    return sample


def _reference_row(chord_beats, note_beats, mapping, *, spb: float = 0.5) -> dict:
    """A locked score timeline in the shape the timing reference sidecar has."""
    beats = np.asarray(chord_beats, dtype=np.float64)
    written = np.asarray(note_beats, dtype=np.float64)
    return {"chord_score_seconds": beats * spb, "chord_score_beats": beats,
            "note_score_duration_beats": written, "note_score_duration_seconds": written * spb,
            "note_to_chord": np.asarray(mapping, dtype=np.int32),
            "score_seconds_per_beat_reference": np.asarray(spb)}


def test_the_rule_projection_is_measured_on_the_supervised_rows_of_each_channel():
    # two elements one beat apart at 0.5 s per beat, written one beat long each, and the
    # duration label says every real note was 10% shorter than written
    sample = _projection_sample(2, [0.0, 0.5], [1.0, 1.0], [math.log(0.9), math.log(0.9)])
    row = _reference_row([0.0, 1.0], [1.0, 1.0], [0, 1])
    supervised = {"log_key_duration_ratio": np.array([math.log(0.9), math.log(0.9)]),
                  "onset_delta_ms": np.array([0.0, 0.0]), "log_ioi_ratio": np.array([0.0])}
    observed = rule_projection_errors(sample, row, supervised, sessions=6, seed=1, duration_centring=False)
    again = rule_projection_errors(sample, row, supervised, sessions=6, seed=1, duration_centring=False)
    assert observed == again
    duration = observed["log_key_duration_ratio"]
    # a one-beat element lands in the mid tier, so the rule holds it between 0.70 and 0.85 of
    # what the score wrote, and the label says the performer kept 0.9: the error is therefore
    # bounded by log(0.9/0.85) and log(0.9/0.70), and the rule cannot close it by construction
    assert math.log(0.9 / 0.85) <= duration["rules_session"] <= math.log(0.9 / 0.70)
    assert duration["rules_centred"] <= duration["rules_session"]  # Jensen: mean of errors >= error of mean
    assert duration["rules_rows"] == 2.0 and duration["rules_missing_rows"] == 0.0
    assert set(observed) == {"log_key_duration_ratio", "onset_delta_ms", "log_ioi_ratio"}
    assert observed["log_ioi_ratio"]["rules_rows"] == 1.0


def test_a_projection_that_cannot_reach_the_labels_is_refused_rather_than_reported():
    # a single-chord recording has no emittable interval at all: silence there would be a lie
    sample = _projection_sample(1, [0.0], [1.0], [0.0])
    row = _reference_row([0.0], [1.0], [0])
    supervised = {"log_key_duration_ratio": np.array([0.0]), "onset_delta_ms": np.array([0.0]),
                  "log_ioi_ratio": np.array([0.0])}
    with pytest.raises(TrainingRunError, match="no emittable rule prediction"):
        rule_projection_errors(sample, row, supervised, sessions=2, seed=1, duration_centring=False)


def test_a_count_mismatch_between_reference_and_sample_is_refused():
    sample = _projection_sample(2, [0.0, 0.5], [1.0, 1.0], [0.0, 0.0])
    row = _reference_row([0.0, 0.5, 1.0], [1.0, 1.0], [0, 1])
    with pytest.raises(TrainingRunError, match="chord count differs"):
        rule_projection_errors(sample, row, {"log_key_duration_ratio": np.array([0.0, 0.0])},
                               sessions=2, seed=1, duration_centring=False)
