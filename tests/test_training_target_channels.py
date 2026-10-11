from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from tools.audit_target_channels import CHANNELS, per_channel
from training.midi_features import CHORD_TARGET_NAMES, NOTE_TARGET_NAMES


def _row(sample_id: str, values: dict[tuple[str, int], np.ndarray]) -> dict[str, object]:
    channel_values = {name: np.asarray(values.get((name, column), [0.0]), dtype=np.float64)
                      for _kind, name, column in CHANNELS}
    return {"id": sample_id, "source": "ASAP", "piece_group": f"group-{sample_id}",
            "channel_values": channel_values,
            "channel_mass": {key: float(np.abs(array).sum()) for key, array in channel_values.items()}}


def test_channel_map_matches_the_shipped_target_orders():
    for kind, name, column in CHANNELS:
        names = CHORD_TARGET_NAMES if kind == "chord" else NOTE_TARGET_NAMES
        assert names[column] == name


def test_identity_and_centered_error_measure_the_piece_level_offset():
    rows = [
        # every onset in the piece sits 300 ms late: predict-zero costs 300 ms, centering costs nothing
        _row("offset-only", {("onset_delta_ms", 0): np.array([300.0, 300.0, 300.0, 300.0])}),
        # symmetric scatter around zero: centering cannot help
        _row("scatter-only", {("onset_delta_ms", 0): np.array([-200.0, 200.0, -100.0, 100.0])}),
    ]
    channel = per_channel(rows)["onset_delta_ms"]
    assert channel["identity_baseline_mae"] == pytest.approx((300 * 4 + 150 * 4) / 8)
    assert channel["per_sample_median_centered_mae"] == pytest.approx((0 + 150 * 4) / 8)
    assert channel["sample_count"] == 2
    assert channel["supervised_row_count"] == 8
    assert 0.0 < channel["median_shift_share"] < 1.0
    assert channel["error_mass_in_top_10pct_samples"] == pytest.approx(1200 / 1800)


def test_concentration_is_per_channel_not_a_shared_total():
    values = {("onset_delta_ms", 0): np.array([0.0, 0.0, 1000.0]),
              ("log_ioi_ratio", 1): np.array([1.0, 1.0, 1.0]),
              ("micro_offset_ms", 0): np.array([0.0, 0.0, 0.0]),
              ("log_key_duration_ratio", 1): np.array([0.0, 0.0, 0.0]),
              ("velocity_robust_z", 3): np.array([0.0, 0.0, 0.0])}
    heavy = _row("heavy", values)
    light = _row("light", {key: np.zeros(3) for key in values})
    report = per_channel([heavy, light])
    assert report["onset_delta_ms"]["error_mass_in_top_10pct_samples"] == pytest.approx(1.0)
    assert report["log_ioi_ratio"]["error_mass_in_top_10pct_samples"] == pytest.approx(1.0)
    assert report["micro_offset_ms"]["error_mass_in_top_10pct_samples"] is None
    assert report["micro_offset_ms"]["identity_baseline_mae"] == 0.0


class _FakeSample:
    sample_id = "sample-a"

    def __init__(self) -> None:
        self.chord_targets = np.array([[40.0, 0.1], [-20.0, -0.1], [10.0, 0.2]], dtype=np.float32)
        self.chord_masks = np.ones((3, 2), dtype=bool)
        self.note_targets = np.zeros((3, 6), dtype=np.float32)
        self.note_targets[:, 1] = [0.5, -0.5, 0.25]
        self.note_masks = np.ones((3, 6), dtype=bool)
        self.note_masks[1, 1] = False


def test_main_writes_a_sealed_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.jsonl").write_text(json.dumps({"id": "sample-a", "performance_dataset": "ASAP",
                                                       "piece_group": "group-a"}) + "\n", encoding="utf-8")
    ids = tmp_path / "ids.txt"
    ids.write_text("sample-a\n", encoding="utf-8")
    output = tmp_path / "out"
    monkeypatch.setattr("tools.audit_target_channels.load_selected_samples",
                        lambda root, split, sample_ids: [_FakeSample()])
    from tools.audit_target_channels import main
    assert main(["--dataset", str(dataset), "--sample-id-file", str(ids), "--split", "validation",
                 "--output", str(output)]) == 0
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert report["status"] == "complete" and report["sample_count"] == 1
    channel = report["channels"]["onset_delta_ms"]
    assert channel["supervised_row_count"] == 3
    assert channel["identity_baseline_mae"] == pytest.approx(np.abs([40.0, -20.0, 10.0]).mean())
    assert report["channels"]["log_key_duration_ratio"]["supervised_row_count"] == 2
    assert (output / "_SUCCESS").read_text(encoding="ascii").strip()
