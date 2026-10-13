from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from core.humanize import HumanizeParams
from tools.fit_humanize_hold_bands import (candidate_params, fit_hold_bands, hold_demand,
                                           main)
from tools.bootstrap_sequence_vs_baselines import TrainingRunError
from training.rules_baseline import tier_of


def _reference(**overrides) -> dict:
    row = {"chord_score_beats": np.array([0.0, 1.0, 3.0]),
           "chord_score_seconds": np.array([0.0, 0.5, 1.5]),
           "note_score_duration_beats": np.array([1.0, 1.0, 2.0]),
           "note_score_duration_seconds": np.array([0.5, 0.5, 1.0]),
           "note_to_chord": np.array([0, 1, 2], dtype=np.int32),
           "score_seconds_per_beat_reference": np.asarray(0.5)}
    row.update(overrides)
    return row


def test_tier_of_reproduces_the_humanizer_bands_exactly():
    assert tier_of(0.5) == "short_hold"          # "<= 0.5" is short
    assert tier_of(0.500001) == "mid_hold"
    assert tier_of(1.999) == "mid_hold"
    assert tier_of(2.0) == "long_hold"           # PHRASE_LONG_DUR is ">= 2"


def test_hold_demand_is_the_played_length_over_the_span_the_element_occupies():
    # gaps of 1, 2 and 2 beats at 0.5 s/beat -> spans 0.5, 1.0, 1.0 s against written
    # 0.5, 0.5, 1.0 s, and labels of 0.9x, 0.5x and 1.0x the written length
    demand, bands = hold_demand(_reference(), np.array([math.log(0.9), math.log(0.5), math.log(1.0)]))
    assert demand.tolist() == pytest.approx([0.9, 0.25, 1.0])
    assert bands.tolist() == ["mid_hold", "long_hold", "long_hold"]


def test_hold_demand_drops_rows_it_cannot_interpret():
    # a label of -50 is finite and positive once exponentiated, so it survives and is exactly
    # why the fit uses medians; only a non-finite or non-positive demand is unusable at all
    demand, bands = hold_demand(_reference(), np.array([math.log(0.9), math.nan, math.log(1.0)]))
    assert demand.tolist() == pytest.approx([0.9, 1.0]) and len(bands) == 2
    assert hold_demand(_reference(), np.array([0.0, math.log(0.9), math.log(1.0)]))[0].size == 3


def test_unsupervised_notes_are_dropped_before_the_fit_and_bad_ones_are_refused():
    # row 2 would demand an absurd hold, but it carries no trusted label, so it never reaches
    # the fit at all; a supervised row with no written duration is a data fault and is refused
    labels = np.array([math.log(0.9), 12.0, math.log(1.0)])
    demand, bands = hold_demand(_reference(), labels, np.array([True, False, True]))
    assert demand.tolist() == pytest.approx([0.9, 1.0])
    assert bands.tolist() == ["mid_hold", "long_hold"]
    with pytest.raises(TrainingRunError, match="mask must match"):
        hold_demand(_reference(), labels, np.array([True, True]))
    with pytest.raises(TrainingRunError, match="positive written duration"):
        hold_demand(_reference(note_score_duration_seconds=np.array([0.0, 0.5, 1.0])), labels)


def test_the_median_keeps_an_alignment_artifact_from_moving_a_shipping_constant():
    fitted = fit_hold_bands([(np.array([0.9, 1e-22, 0.95]), np.array(["mid_hold"] * 3))], band_width=0.06)
    assert fitted["mid_hold"]["observations"] == 3
    assert fitted["mid_hold"]["median_hold"] == pytest.approx(0.9)
    assert fitted["mid_hold"]["clipped_hold"] == pytest.approx(0.9)


def test_hold_demand_refuses_an_impossible_span():
    with pytest.raises(TrainingRunError, match="reference tempo"):
        hold_demand(_reference(score_seconds_per_beat_reference=np.asarray(0.0)), np.zeros(3))
    with pytest.raises(TrainingRunError, match="share one length"):
        hold_demand(_reference(), np.array([0.0, 0.0]))


def test_bands_are_fitted_from_medians_and_clipped_at_the_longest_playable_length():
    fitted = fit_hold_bands([(np.array([0.9, 1.4, 0.1]), np.array(["mid_hold", "long_hold", "short_hold"]))],
                            band_width=0.06)
    assert fitted["mid_hold"]["median_hold"] == pytest.approx(0.9)
    assert fitted["mid_hold"]["clipped_hold"] == pytest.approx(0.9)
    assert fitted["long_hold"]["median_hold"] == pytest.approx(1.4)
    assert fitted["long_hold"]["clipped_hold"] == pytest.approx(1.0)   # cannot exceed its own span
    assert fitted["short_hold"]["observations"] == 1


def test_a_band_with_no_observations_keeps_the_shipping_value_instead_of_inventing_one():
    shipping = HumanizeParams()
    fitted = fit_hold_bands([(np.array([0.8]), np.array(["mid_hold"]))], shipping=shipping)
    assert fitted["short_hold"]["kept_shipping_value"] is True
    assert fitted["short_hold"]["median_hold"] is None
    assert tuple(fitted["short_hold"]["shipping_band"]) == tuple(shipping.short_hold)
    with pytest.raises(ValueError, match="band_width"):
        fit_hold_bands([], band_width=0.0)


def test_candidate_bands_are_centred_on_the_fit_and_stay_legal():
    fitted = fit_hold_bands([(np.array([0.95, 0.2, 1.4]),
                             np.array(["mid_hold", "short_hold", "long_hold"]))], band_width=0.06)
    params = candidate_params(fitted, band_width=0.06)
    assert params.mid_hold == pytest.approx((0.89, 1.0))          # clipped at 1.0 from 0.95 -> 0.89..1.01
    assert params.short_hold == pytest.approx((0.14, 0.26))
    assert params.long_hold == pytest.approx((0.94, 1.0))         # median 1.4 clipped to 1.0
    assert 0 < params.short_hold[0] <= params.short_hold[1] <= 1.0
    assert 0 < params.mid_hold[0] <= params.mid_hold[1] <= 1.0
    # untouched knobs stay exactly as the app ships them
    assert params.jitter_ms == HumanizeParams().jitter_ms
    assert params.breath_ms == HumanizeParams().breath_ms


class _Sample:
    def __init__(self, sample_id: str, labels: list[float], holds: list[float]) -> None:
        notes = len(labels)
        self.sample_id = sample_id
        self.note_features = np.zeros((notes, 15), dtype=np.float32)
        self.chord_features = np.zeros((notes, 3), dtype=np.float32)
        self.note_to_chord = np.arange(notes, dtype=np.int32)
        self.note_masks = np.ones((notes, 6), dtype=bool)
        self.note_targets = np.zeros((notes, 6), dtype=np.float32)
        self.note_targets[:, 1] = labels
        self.chord_masks = np.ones((notes, 2), dtype=bool)
        self.chord_masks[0, 1] = False
        self.chord_targets = np.zeros((notes, 2), dtype=np.float32)
        self.chord_targets[:, 0] = [0.0] * notes
        self.chord_targets[:, 1] = holds


def test_the_tool_fits_on_the_train_list_and_scores_the_candidate_on_the_development_list(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = "tools.fit_humanize_hold_bands"
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    rows = [json.dumps({"id": "t1", "split": "train", "performance_dataset": "ASAP", "piece_group": "g1"}),
            json.dumps({"id": "v1", "split": "validation", "performance_dataset": "ASAP", "piece_group": "g2"}),
            json.dumps({"id": "v2", "split": "validation", "performance_dataset": "ASAP", "piece_group": "g3"})]
    (dataset / "manifest.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    fit_file = tmp_path / "train.txt"
    fit_file.write_text("t1\n", encoding="utf-8")
    evaluate_file = tmp_path / "validation.txt"
    evaluate_file.write_text("v1\nv2\n", encoding="utf-8")
    # the report fingerprints the reference summary, so a readable file has to exist even when
    # the reference itself is faked
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")

    def _reference_row(notes: int, span_beats: float) -> dict:
        onsets = np.arange(notes, dtype=np.float64) * span_beats
        written = np.full(notes, span_beats, dtype=np.float64)
        seconds = np.asarray(0.5)
        return {"chord_score_beats": onsets, "chord_score_seconds": onsets * float(seconds),
                "note_score_duration_beats": written, "note_score_duration_seconds": written * float(seconds),
                "note_to_chord": np.arange(notes, dtype=np.int32),
                "score_seconds_per_beat_reference": seconds}

    references = {"t1": _reference_row(4, 1.0), "v1": _reference_row(4, 1.0), "v2": _reference_row(4, 1.0)}
    # the fit recording asks for 0.95 of the span on every note, so the fitted mid band moves
    # from the shipping 0.70-0.85 to roughly 0.89-1.00, and the validation labels ask for the
    # same 0.95 -- the candidate should therefore beat both the score and the shipping rule
    monkeypatch.setattr(f"{module}.load_timing_reference", lambda path: references)
    monkeypatch.setattr(f"{module}._load_sample_id_file",
                        lambda path: ["t1"] if "train" in path.name else ["v1", "v2"])
    fit_sample = _Sample("t1", [math.log(0.95)] * 4, [0.0] * 4)
    monkeypatch.setattr(f"{module}.load_selected_samples",
                        lambda root, split, ids, **kwargs: [fit_sample] if split == "train"
                        else [_Sample("v1", [math.log(0.95)] * 4, [0.0] * 4),
                              _Sample("v2", [math.log(0.95)] * 4, [0.0] * 4)])
    output = tmp_path / "out"
    assert main(["--dataset", str(dataset), "--timing-reference", str(tmp_path),
                 "--fit-id-file", str(fit_file), "--sample-id-file", str(evaluate_file),
                 "--output", str(output), "--sessions", "4", "--draws", "200"]) == 0
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert report["fitted_bands"]["mid_hold"]["median_hold"] == pytest.approx(0.95)
    assert report["fit_supervised_notes"] == 4 and report["adoption_evidence"] is False
    duration = report["channels"]["log_key_duration_ratio"]
    table = duration["piece_equal_weight_mae"]
    assert table["model"] < table["predict_zero"] < table["rules_session"]
    assert duration["improvement_over_identity"] > 0.05
    assert report["verdict"]["log_key_duration_ratio"]["reaches_policy_fraction"] is True
    assert report["model_column_means"]


def test_the_locked_test_split_is_refused_for_both_roles(tmp_path: Path):
    base = ["--dataset", str(tmp_path), "--timing-reference", str(tmp_path), "--fit-id-file", str(tmp_path),
            "--sample-id-file", str(tmp_path), "--output", str(tmp_path / "out")]
    with pytest.raises(SystemExit, match="locked test"):
        main(base + ["--fit-split", "test"])
    with pytest.raises(SystemExit, match="locked test"):
        main(base + ["--evaluate-split", "test"])
