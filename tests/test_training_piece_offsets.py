from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from tools.audit_piece_offset_reproducibility import icc1, main, performance_offsets


class _FakeSample:
    """One performance: chord onsets, a constant IOI the caller can set, note durations."""

    def __init__(self, sample_id: str, onsets: list[float], durations: list[float],
                 ioi: float = 0.2) -> None:
        self.sample_id = sample_id
        self.chord_targets = np.column_stack([onsets, [ioi] * len(onsets)]).astype(np.float32)
        self.chord_masks = np.ones((len(onsets), 2), dtype=bool)
        self.note_targets = np.zeros((len(durations), 6), dtype=np.float32)
        self.note_targets[:, 1] = durations
        self.note_masks = np.ones((len(durations), 6), dtype=bool)


def test_icc1_matches_the_one_way_definition_by_hand():
    # three pieces of two offsets each: MSB=200, MSW=2, k_mean=2 -> (200-2)/(200+2)
    report = icc1([np.array([0.0, 2.0]), np.array([10.0, 12.0]), np.array([20.0, 22.0])])
    assert report["icc1"] == pytest.approx(198.0 / 202.0)
    assert report["pieces"] == 3 and report["offsets"] == 6
    assert report["between_piece_variance"] == pytest.approx(200.0)
    assert report["within_piece_variance"] == pytest.approx(2.0)


def test_icc1_goes_negative_when_pieces_are_indistinguishable():
    report = icc1([np.array([0.0, 10.0]), np.array([0.0, 10.0]), np.array([0.0, 10.0])])
    assert report["between_piece_variance"] == pytest.approx(0.0)
    assert report["icc1"] == pytest.approx(-1.0)
    assert icc1([np.array([1.0, 2.0])]) == {"note": "fewer than two pieces with two or more offsets"}


def test_icc1_handles_groups_of_different_sizes():
    # sizes 3/2/3: MSB=300.75, MSW=3.6, k_mean=8/3 -> (300.75-3.6)/(300.75+(8/3-1)*3.6)
    report = icc1([np.array([0.0, 2.0, 4.0]), np.array([10.0, 12.0]), np.array([20.0, 22.0, 24.0])])
    assert report["pieces"] == 3 and report["offsets"] == 8
    assert report["between_piece_variance"] == pytest.approx(300.75)
    assert report["within_piece_variance"] == pytest.approx(3.6)
    assert report["icc1"] == pytest.approx(297.15 / 306.75)
    # total SS splits exactly: between 601.5 + within 18 over 7 degrees of freedom
    assert report["offset_sd_overall"] == pytest.approx(float(np.sqrt((601.5 + 18.0) / 7.0)))


def test_icc1_ignores_singletons_without_shifting_the_group_sizes():
    # the singleton must not be counted, and dropping it must not misalign the sizes vector
    report = icc1([np.array([5.0]), np.array([0.0, 2.0]), np.array([10.0, 12.0]), np.array([20.0, 22.0])])
    assert report["pieces"] == 3
    assert report["icc1"] == pytest.approx(198.0 / 202.0)


def test_performance_offsets_use_only_supervised_rows():
    sample = _FakeSample("s", [0.0, 1.0, 2.0], [0.4, 0.6, 0.8])
    sample.chord_masks[2, 0] = False
    sample.note_masks[0, 1] = False
    offsets = performance_offsets(sample)
    onset_value, onset_rows = offsets["onset_delta_ms"]
    assert onset_value == pytest.approx(0.5) and onset_rows == 2
    ioi_value, ioi_rows = offsets["log_ioi_ratio"]
    assert ioi_value == pytest.approx(0.2) and ioi_rows == 3
    duration_value, duration_rows = offsets["log_key_duration_ratio"]
    assert duration_value == pytest.approx(0.7) and duration_rows == 2


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    root.mkdir()
    rows = []
    for piece in ("pieceA", "pieceB"):
        for index, onsets in enumerate(([0.0, 2.0], [10.0, 12.0]), start=1):
            rows.append(json.dumps({"id": f"{piece}{index}", "split": "train", "piece_group": piece,
                                    "performer": f"performer{index}", "chord_count": len(onsets),
                                    "qc_status": "accepted"}))
    (root / "manifest.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


def test_main_reports_one_offset_per_performer_and_seals_the_artifact(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch):
    dataset = _dataset(tmp_path)
    store = {"pieceA1": _FakeSample("pieceA1", [0.0, 2.0], [0.0, 2.0], ioi=0.2),
             "pieceA2": _FakeSample("pieceA2", [10.0, 12.0], [10.0, 12.0], ioi=0.2),
             "pieceB1": _FakeSample("pieceB1", [0.0, 2.0], [0.0, 2.0], ioi=0.8),
             "pieceB2": _FakeSample("pieceB2", [10.0, 12.0], [10.0, 12.0], ioi=0.8)}
    monkeypatch.setattr("tools.audit_piece_offset_reproducibility.load_selected_samples",
                        lambda root, split, sample_ids: [store[sample_id] for sample_id in sorted(sample_ids)])
    output = tmp_path / "out"
    assert main(["--dataset", str(dataset), "--split", "train", "--output", str(output),
                 "--minimum-offset-rows", "1", "--groups-per-load", "2"]) == 0
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert report["status"] == "complete" and report["trained"] is False
    assert report["candidate_groups"] == 2 and report["groups_loaded"] == 2
    onset = report["channels"]["onset_delta_ms"]
    assert onset["pieces_measured"] == 2 and onset["offset_pairs"] == 2
    assert onset["median_abs_offset_pair_difference"] == pytest.approx(10.0)
    # identical within-piece spread and no between-piece spread pushes ICC(1) to its floor of -1
    assert onset["icc1"] == pytest.approx(-1.0)
    # the IOI offset repeats exactly across performers, so between-piece variance is all the variance
    assert report["channels"]["log_ioi_ratio"]["icc1"] == pytest.approx(1.0)
    assert report["channels"]["log_key_duration_ratio"]["icc1"] == pytest.approx(-1.0)
    assert (output / "_SUCCESS").read_text(encoding="ascii").strip()
