from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from tools.audit_cross_performance_reproducibility import (
    CHANNELS,
    PairBuffer,
    centre_labels,
    chord_labels,
    main,
    note_labels,
    score_groups,
    select_performances,
    summarize_channel,
)


def _row(sample_id: str, group: str, performer: str, *, split: str = "train",
         chords: int = 2, qc: str = "accepted") -> str:
    return json.dumps({"id": sample_id, "split": split, "piece_group": group, "performer": performer,
                       "chord_count": chords, "qc_status": qc})


class _FakeSample:
    """Two chords, three notes; duration means come out as [1.0, 0.25]."""

    def __init__(self, sample_id: str, onsets: list[float], iois: list[float]) -> None:
        self.sample_id = sample_id
        self.chord_targets = np.column_stack([onsets, iois]).astype(np.float32)
        self.chord_masks = np.ones((2, 2), dtype=bool)
        self.chord_features = np.column_stack([[0.0, 1.0], np.zeros(2)]).astype(np.float32)
        self.note_targets = np.zeros((3, 6), dtype=np.float32)
        self.note_targets[:, 1] = [0.5, 1.5, 0.25]
        self.note_masks = np.ones((3, 6), dtype=bool)
        self.note_to_chord = np.array([0, 0, 1], dtype=np.int32)


def test_select_performances_takes_one_recording_per_distinct_performer():
    rows = [{"performer": "Bob", "id": "b2"}, {"performer": "Ann", "id": "a1"},
            {"performer": "Bob", "id": "b1"}, {"performer": "", "id": "u1"},
            {"performer": "", "id": "u2"}, {"performer": "Cara", "id": "c1"}]
    # buckets: '' -> u1, Ann -> a1, Bob -> b1, Cara -> c1, in that sorted order
    assert select_performances(rows, 4) == ["u1", "a1", "b1", "c1"]
    assert select_performances(rows, 2) == ["u1", "a1"]
    # unnamed recordings collapse into one bucket, so they can never pair with each other
    assert "u2" not in select_performances(rows, 10)
    assert select_performances(rows, 4) == select_performances(list(reversed(rows)), 4)


def test_score_groups_keeps_only_multi_performance_equal_length_accepted_rows(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join([
        _row("g1-a", "g1", "Ann"), _row("g1-b", "g1", "Bob"),
        _row("g2-a", "g2", "Ann"), _row("g2-b", "g2", "Bob", chords=3),
        _row("g3-a", "g3", "Ann"),
        _row("g4-a", "g4", "Ann"), _row("g4-b", "g4", "Bob", split="validation"),
        _row("g5-a", "g5", "Ann"), _row("g5-b", "g5", "Bob", qc="rejected"),
    ]) + "\n", encoding="utf-8")
    groups = score_groups(manifest, "train")
    assert [group for group, _rows in groups] == ["g1"]
    assert [row["id"] for row in groups[0][1]] == ["g1-a", "g1-b"]


def _buffer(pairs: list[tuple[list[float], list[float]]]) -> PairBuffer:
    buffer = PairBuffer()
    for left, right in pairs:
        buffer.add(np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64))
    return buffer


def test_summarize_channel_reports_scatter_and_the_per_performance_floor():
    first = [100.0, 200.0, 100.0, 200.0, 110.0, 190.0]
    second = [110.0, 190.0, 105.0, 205.0, 105.0, 205.0]
    channel = summarize_channel(_buffer([(first, second)]), minimum_positions=6)
    assert channel["pair_positions"] == 6
    assert channel["median_abs_cross_performance_difference"] == pytest.approx(7.5)
    assert channel["p90_abs_cross_performance_difference"] == pytest.approx(float(np.percentile(
        [10, 10, 5, 5, 5, 15], 90)))
    assert channel["cross_performance_rms"] == pytest.approx(float(np.sqrt(np.mean(np.square([10, 10, 5, 5, 5, 15])))))
    assert channel["implied_per_performance_rms"] == pytest.approx(channel["cross_performance_rms"] / np.sqrt(2.0))
    assert channel["pearson_correlation"] == pytest.approx(float(np.corrcoef(first, second)[0, 1]))
    assert channel["split_half_reliability_estimate"] == pytest.approx(
        2.0 * channel["pearson_correlation"] / (1.0 + channel["pearson_correlation"]))


def test_summarize_channel_handles_constant_labels_and_thin_samples():
    constant = summarize_channel(_buffer([([1.0, 1.0], [1.0, 1.0])]), minimum_positions=2)
    assert constant["cross_performance_rms"] == 0.0
    assert constant["pearson_correlation"] is None
    assert constant["split_half_reliability_estimate"] is None
    thin = summarize_channel(_buffer([([1.0], [2.0])]), minimum_positions=1000)
    assert thin["pair_positions"] == 1
    assert "note" in thin and "cross_performance_rms" not in thin


def test_chord_labels_leaves_unsupervised_positions_nan():
    sample = _FakeSample("s", [10.0, 20.0], [0.1, -0.1])
    sample.chord_masks[1, 0] = False
    labels = chord_labels(sample, sample.note_to_chord)
    assert labels["onset_delta_ms"][0] == pytest.approx(10.0)
    assert np.isnan(labels["onset_delta_ms"][1])
    assert labels["log_ioi_ratio"].tolist() == pytest.approx([0.1, -0.1])
    assert labels["log_key_duration_ratio"].tolist() == pytest.approx([1.0, 0.25], abs=1e-6)
    assert labels["score_onset_beats"].tolist() == pytest.approx([0.0, 1.0])


def _write_dataset(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "manifest.jsonl").write_text("\n".join([
        _row("s1", "g1", "Ann"), _row("s2", "g1", "Bob"), _row("s3", "g1", "Cara"),
    ]) + "\n", encoding="utf-8")


def _fake_samples() -> dict[str, _FakeSample]:
    return {"s1": _FakeSample("s1", [100.0, 200.0], [0.1, -0.1]),
            "s2": _FakeSample("s2", [110.0, 190.0], [0.1, -0.1]),
            "s3": _FakeSample("s3", [105.0, 205.0], [0.1, -0.1])}


def test_main_seals_a_three_performer_pair_sweep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    output = tmp_path / "out"
    store = _fake_samples()
    monkeypatch.setattr("tools.audit_cross_performance_reproducibility.load_selected_samples",
                        lambda root, split, sample_ids: [store[sample_id] for sample_id in sorted(sample_ids)])
    assert main(["--dataset", str(dataset), "--split", "train", "--output", str(output),
                 "--minimum-pair-positions", "6"]) == 0
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert report["status"] == "complete" and report["trained"] is False
    assert report["eligible_groups"] == 1 and report["groups_used"] == 1
    assert report["pairs_compared"] == 3
    assert report["groups_skipped_for_one_distinct_performer"] == 0
    onset = report["channels"]["onset_delta_ms"]
    assert onset["pair_positions"] == 6
    assert onset["median_abs_cross_performance_difference"] == pytest.approx(7.5)
    assert report["channels"]["log_ioi_ratio"]["cross_performance_rms"] == 0.0
    assert (output / "_SUCCESS").read_text(encoding="ascii").strip()


def test_main_refuses_an_unbounded_pair_volume(tmp_path: Path):
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    with pytest.raises(SystemExit, match="estimated"):
        main(["--dataset", str(dataset), "--output", str(tmp_path / "out"), "--max-pair-positions", "1"])


def test_main_refuses_to_reuse_an_output_directory(tmp_path: Path):
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    output = tmp_path / "out"
    output.mkdir()
    (output / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        main(["--dataset", str(dataset), "--output", str(output)])


def test_centre_labels_removes_the_level_and_leaves_missing_positions_missing():
    labels = {"onset_delta_ms": np.array([100.0, 200.0, 300.0, np.nan]),
              "log_ioi_ratio": np.array([0.1, 0.2, 0.3, 0.4]),
              "log_key_duration_ratio": np.full(4, np.nan),
              "score_onset_beats": np.array([0.0, 1.0, 2.0, 3.0])}
    centred = centre_labels(labels, CHANNELS)
    assert centred["onset_delta_ms"][:3].tolist() == pytest.approx([-100.0, 0.0, 100.0])
    assert np.isnan(centred["onset_delta_ms"][3])
    assert centred["log_ioi_ratio"].tolist() == pytest.approx([-0.15, -0.05, 0.05, 0.15])
    # a channel with nothing supervised stays all-NaN instead of becoming NaN minus NaN
    assert np.all(np.isnan(centred["log_key_duration_ratio"]))
    assert centred["score_onset_beats"].tolist() == [0.0, 1.0, 2.0, 3.0]


class _FakeNoteSample:
    """Four notes: duration col 1, micro col 0, velocity col 3, onset beats in feature col 3."""

    def __init__(self, sample_id: str, durations: list[float], micro: list[float]) -> None:
        self.sample_id = sample_id
        self.note_targets = np.zeros((len(durations), 6), dtype=np.float32)
        self.note_targets[:, 1] = durations
        self.note_targets[:, 0] = micro
        self.note_targets[:, 3] = [0.5] * len(durations)
        self.note_masks = np.ones((len(durations), 6), dtype=bool)
        self.note_features = np.zeros((len(durations), 15), dtype=np.float32)
        self.note_features[:, 3] = np.arange(len(durations), dtype=np.float32)
        self.chord_features = np.zeros((len(durations), 11), dtype=np.float32)
        self.chord_targets = np.zeros((len(durations), 2), dtype=np.float32)
        self.chord_masks = np.ones((len(durations), 2), dtype=bool)
        self.note_to_chord = np.arange(len(durations), dtype=np.int32)


def test_note_labels_are_per_note_and_masked_to_nan():
    sample = _FakeNoteSample("s", [1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
    sample.note_masks[1, 1] = False
    labels = note_labels(sample)
    assert np.isnan(labels["log_key_duration_ratio"][1])
    assert labels["log_key_duration_ratio"][[0, 2, 3]].tolist() == pytest.approx([1.0, 3.0, 4.0])
    assert labels["micro_offset_ms"].tolist() == pytest.approx([10.0, 20.0, 30.0, 40.0])
    assert labels["velocity_robust_z"].tolist() == pytest.approx([0.5, 0.5, 0.5, 0.5])
    assert labels["score_onset_beats"].tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0])


def test_centring_compares_profiles_so_a_constant_shift_disappears(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.jsonl").write_text("\n".join([
        json.dumps({"id": "s1", "split": "train", "piece_group": "g1", "performer": "Ann",
                    "chord_count": 2, "qc_status": "accepted"}),
        json.dumps({"id": "s2", "split": "train", "piece_group": "g1", "performer": "Bob",
                    "chord_count": 2, "qc_status": "accepted"}),
    ]) + "\n", encoding="utf-8")
    # the second recording is the first one played 50 ms later everywhere: a level, not a profile
    store = {"s1": _FakeSample("s1", [100.0, 200.0], [0.1, -0.1]),
             "s2": _FakeSample("s2", [150.0, 250.0], [0.1, -0.1])}
    monkeypatch.setattr("tools.audit_cross_performance_reproducibility.load_selected_samples",
                        lambda root, split, sample_ids: [store[sample_id] for sample_id in sorted(sample_ids)])
    for name, arguments in (("raw", []), ("centred", ["--centre-per-performance"])):
        output = tmp_path / name
        assert main(["--dataset", str(dataset), "--output", str(output),
                     "--minimum-pair-positions", "2"] + arguments) == 0
        report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert report["granularity"] == "chord"
        channel = report["channels"]["onset_delta_ms"]
        assert channel["cross_performance_rms"] == pytest.approx(50.0 if name == "raw" else 0.0)


def test_note_granularity_pairs_duration_row_by_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.jsonl").write_text("\n".join([
        json.dumps({"id": "n1", "split": "train", "piece_group": "g1", "performer": "Ann",
                    "chord_count": 4, "qc_status": "accepted"}),
        json.dumps({"id": "n2", "split": "train", "piece_group": "g1", "performer": "Bob",
                    "chord_count": 4, "qc_status": "accepted"}),
    ]) + "\n", encoding="utf-8")
    # same profile, different level: per note the durations are (0,1,2,3) then shifted by 10
    store = {"n1": _FakeNoteSample("n1", [1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]),
             "n2": _FakeNoteSample("n2", [11.0, 12.0, 13.0, 14.0], [0.0, 0.0, 0.0, 0.0])}
    monkeypatch.setattr("tools.audit_cross_performance_reproducibility.load_selected_samples",
                        lambda root, split, sample_ids: [store[sample_id] for sample_id in sorted(sample_ids)])
    output = tmp_path / "note"
    assert main(["--dataset", str(dataset), "--output", str(output), "--granularity", "note",
                 "--minimum-pair-positions", "4", "--centre-per-performance"]) == 0
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert report["granularity"] == "note"
    assert set(report["channels"]) == {"log_key_duration_ratio", "micro_offset_ms", "velocity_robust_z"}
    duration = report["channels"]["log_key_duration_ratio"]
    assert duration["pair_positions"] == 4
    assert duration["cross_performance_rms"] == pytest.approx(0.0)
    assert report["channels"]["micro_offset_ms"]["cross_performance_rms"] == pytest.approx(0.0)
