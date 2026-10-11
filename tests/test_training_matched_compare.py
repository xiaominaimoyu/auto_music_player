from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from tools.compare_matched_sequence_runs import main as compare_main
from training.train_expression import TrainingRunError


def _run(directory: Path, policy: str, seed: int, point_losses: list[float], ioi_maes: list[float],
         onset_maes: list[float], bar_components: dict[str, float] | None,
         onset_weight: float = 0.1) -> None:
    directory.mkdir(parents=True)
    config = {"pipeline": "auto_music_player.expression_sequence_pilot", "architecture": "gru",
              "window_policy": policy, "seed": seed, "onset_weight": onset_weight, "parameter_count": 52872, "epochs": len(point_losses),
              "bar_elapsed_weight": 1.0 if bar_components else 0.0,
              "bar_tempo_weight": 1.0 if bar_components else 0.0,
              "bar_local_weight": 1.0 if bar_components else 0.0,
              "max_window_chords": 512 if policy == "bar" else None,
              "bar_horizons": [1, 4, 16] if policy == "bar" else None,
              "train_sample_id_file_sha256": "train-sha", "validation_sample_id_file_sha256": "validation-sha",
              "timing_reference_sha256": "reference-sha", "implementation_sha256": {"bar_windows.py": "sha"}}
    (directory / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    lines = []
    for epoch, (point, ioi, onset) in enumerate(zip(point_losses, ioi_maes, onset_maes), start=1):
        total = point + (sum(bar_components.values()) if bar_components else 0.0)
        validation = {"loss": total, "point_loss": point, "bar_loss": (sum(bar_components.values()) if bar_components else None),
                      "bar_component_losses": bar_components, "target_mae": {"log_ioi_ratio": ioi,
                                                                             "log_key_duration_ratio": 0.6,
                                                                             "micro_offset_ms": 6.0,
                                                                             "onset_delta_ms": onset}}
        lines.append(json.dumps({"epoch": epoch, "train": {"loss": point - 0.01}, "validation": validation}))
    (directory / "metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (directory / "_SUCCESS").write_text("ok\n", encoding="ascii")


def test_paired_delta_averages_across_seeds_instead_of_trusting_one(tmp_path: Path):
    fixed = tmp_path / "fixed_seed1"
    bar = tmp_path / "bar_seed1"
    fixed2 = tmp_path / "fixed_seed2"
    bar2 = tmp_path / "bar_seed2"
    _run(fixed, "fixed", 20261003, [0.33, 0.32], [0.180, 0.176], [583.0, 583.0], None)
    _run(bar, "bar", 20261003, [0.34, 0.31], [0.185, 0.174], [590.0, 588.0], {"bar_elapsed_weight": 0.007,
                                                                             "bar_tempo_weight": 0.007,
                                                                             "bar_local_weight": 0.018})
    _run(fixed2, "fixed", 20261004, [0.33, 0.32], [0.180, 0.176], [583.0, 583.0], None)
    _run(bar2, "bar", 20261004, [0.35, 0.34], [0.186, 0.180], [591.0, 590.0], {"bar_elapsed_weight": 0.007,
                                                                              "bar_tempo_weight": 0.007,
                                                                              "bar_local_weight": 0.018})
    output = tmp_path / "compare"
    assert compare_main(["--runs", str(fixed), str(bar), str(fixed2), str(bar2), "--output", str(output)]) == 0
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert set(report["paired_by_seed"]) == {"20261003", "20261004"}
    # seed 1 improves (-0.01) while seed 2 regresses (+0.02): the mean must not hide the spread.
    epoch_two = report["paired_mean_delta"]["2"]
    assert epoch_two["validation_point_loss_mean"] == pytest.approx(0.005)
    assert epoch_two["validation_point_loss_spread"] == pytest.approx(0.03)
    assert report["paired_mean_delta"]["2"]["validation_mae_mean"]["log_ioi_ratio"] == pytest.approx(0.001)
    assert report["paired_mean_delta"]["2"]["validation_mae_mean"]["onset_delta_ms"] == pytest.approx(6.0)
    assert report["paired_mean_delta"]["2"]["validation_mae_mean"]["log_key_duration_ratio"] == pytest.approx(0.0)
    assert report["paired_by_seed"]["20261003"]["per_epoch"]["2"]["validation_point_loss"] == pytest.approx(-0.01)
    assert report["paired_by_seed"]["20261004"]["per_epoch"]["2"]["validation_point_loss"] == pytest.approx(0.02)
    assert report["interpretation"]["adoption_gate_changed"] is False
    assert report["interpretation"]["single_seed_warning"]


def test_refuses_existing_output_and_incomplete_runs(tmp_path: Path):
    run = tmp_path / "run"
    _run(run, "fixed", 1, [0.3], [0.18], [583.0], None)
    output = tmp_path / "compare"
    output.mkdir()
    (output / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        compare_main(["--runs", str(run), "--output", str(output)])
    (run / "_SUCCESS").unlink()
    with pytest.raises(TrainingRunError, match="did not complete"):
        compare_main(["--runs", str(run), "--output", str(tmp_path / "other")])


def test_pairing_can_switch_to_any_run_config_key(tmp_path: Path):
    baseline, candidate = tmp_path / "onset01", tmp_path / "onset00"
    baseline2, candidate2 = tmp_path / "onset01_b", tmp_path / "onset00_b"
    _run(baseline, "fixed", 11, [0.33, 0.32], [0.180, 0.176], [583.0, 583.0], None, onset_weight=0.1)
    _run(candidate, "fixed", 11, [0.33, 0.315], [0.180, 0.171], [583.0, 583.0], None, onset_weight=0.0)
    _run(baseline2, "fixed", 12, [0.33, 0.32], [0.180, 0.176], [583.0, 583.0], None, onset_weight=0.1)
    _run(candidate2, "fixed", 12, [0.33, 0.318], [0.180, 0.173], [583.0, 583.0], None, onset_weight=0.0)
    output = tmp_path / "compare"
    assert compare_main(["--runs", str(baseline), str(candidate), str(baseline2), str(candidate2),
                         "--pair-by", "onset_weight", "--baseline-value", "0.1", "--candidate-value", "0.0",
                         "--output", str(output)]) == 0
    report = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert report["pair_key"] == "onset_weight"
    assert set(report["paired_by_seed"]) == {"11", "12"}
    assert all(run["arm"] in {"0.1", "0.0"} for run in report["runs"])
    assert report["paired_mean_delta"]["2"]["validation_point_loss_mean"] == pytest.approx(-0.0035)


def test_duplicate_policy_for_one_seed_is_rejected_not_silently_overwritten(tmp_path: Path):
    fixed = tmp_path / "fixed"
    first = tmp_path / "bar_a"
    second = tmp_path / "bar_b"
    _run(fixed, "fixed", 7, [0.33], [0.18], [583.0], None)
    _run(first, "bar", 7, [0.34], [0.185], [590.0], {"bar_elapsed_weight": 0.007, "bar_tempo_weight": 0.007,
                                                     "bar_local_weight": 0.018})
    _run(second, "bar", 7, [0.31], [0.174], [588.0], {"bar_elapsed_weight": 0.007, "bar_tempo_weight": 0.007,
                                                      "bar_local_weight": 0.002})
    with pytest.raises(TrainingRunError, match=r"duplicate window_policy=bar run for seed 7"):
        compare_main(["--runs", str(fixed), str(first), str(second), "--output", str(tmp_path / "compare")])
