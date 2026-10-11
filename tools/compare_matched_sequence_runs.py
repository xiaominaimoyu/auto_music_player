"""Compare matched sequence runs (for example fixed versus bar policy) side by side.

Reads completed run directories and reports per-epoch validation numbers plus the
matched-epoch delta against the first listed run.  It never changes a gate: smaller
validation loss here is a training signal, not evidence for player adoption, and a
single seed cannot support a piece-level claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from training.train_expression import TrainingRunError, _sha256, _write_json

TRACKED_MAE = ("log_ioi_ratio", "log_key_duration_ratio", "micro_offset_ms", "onset_delta_ms")


def read_run(directory: Path) -> dict[str, object]:
    config_path, metrics_path = directory / "run_config.json", directory / "metrics.jsonl"
    if not config_path.is_file() or not metrics_path.is_file():
        raise TrainingRunError(f"run is missing config or metrics: {directory}")
    if not (directory / "_SUCCESS").is_file() or (directory / "_INCOMPLETE").exists():
        raise TrainingRunError(f"run did not complete: {directory}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    epochs = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {"directory": str(directory), "config": config, "epochs": epochs}


def summarize(run: dict[str, object], pair_by: str) -> dict[str, object]:
    config, epochs = run["config"], run["epochs"]
    rows = []
    for record in epochs:
        validation = record["validation"]
        row = {"epoch": record["epoch"], "train_loss": record["train"]["loss"], "validation_loss": validation["loss"],
               "validation_point_loss": validation["point_loss"], "validation_bar_loss": validation["bar_loss"],
               "validation_bar_components": validation.get("bar_component_losses"),
               "validation_mae": {name: validation["target_mae"].get(name) for name in TRACKED_MAE}}
        rows.append(row)
    best_index = int(np.argmin([row["validation_loss"] for row in rows]))
    return {"label": Path(run["directory"]).name, "architecture": config["architecture"],
            "window_policy": config.get("window_policy", "fixed"), "seed": config.get("seed"),
            "pair_key": pair_by, "arm": str(config.get(pair_by)),
            "parameter_count": config.get("parameter_count"), "epochs_requested": config.get("epochs"),
            "epochs_recorded": len(rows), "best_epoch": rows[best_index]["epoch"],
            "best_validation_loss": rows[best_index]["validation_loss"],
            "final_validation_loss": rows[-1]["validation_loss"],
            "best_validation_point_loss": rows[best_index]["validation_point_loss"],
            "best_validation_mae": rows[best_index]["validation_mae"],
            "final_validation_mae": rows[-1]["validation_mae"],
            "bar_weights": {key: config.get(key) for key in ("bar_elapsed_weight", "bar_tempo_weight", "bar_local_weight")},
            "max_window_chords": config.get("max_window_chords"), "bar_horizons": config.get("bar_horizons"),
            "train_sample_id_file_sha256": config.get("train_sample_id_file_sha256"),
            "validation_sample_id_file_sha256": config.get("validation_sample_id_file_sha256"),
            "timing_reference_sha256": config.get("timing_reference_sha256"),
            "implementation_sha256": config.get("implementation_sha256"), "epochs": rows}


def matched_delta(summary: dict[str, object], baseline: dict[str, object]) -> dict[str, object]:
    """Per-epoch deltas against the baseline run at the same epoch number."""
    by_epoch = {row["epoch"]: row for row in baseline["epochs"]}
    deltas = {}
    for row in summary["epochs"]:
        reference = by_epoch.get(row["epoch"])
        if reference is None:
            continue
        deltas[str(row["epoch"])] = {
            "validation_loss": row["validation_loss"] - reference["validation_loss"],
            "validation_point_loss": row["validation_point_loss"] - reference["validation_point_loss"],
            "validation_mae": {name: ((row["validation_mae"].get(name) or 0.0)
                                      - (reference["validation_mae"].get(name) or 0.0)) for name in TRACKED_MAE},
        }
    return deltas


def paired_by_seed(runs: list[dict[str, object]], baseline_value: str,
                   candidate_value: str) -> tuple[dict[str, object], dict[str, object]]:
    """Pair arms that share a seed, then average the paired deltas.

    A single seed cannot separate the knob under test from run-to-run noise, so the
    per-seed delta is the unit of evidence and the mean across seeds is the summary.
    """
    by_key: dict[tuple[object, str], dict[str, object]] = {}
    for run in runs:
        key = (run["seed"], run["arm"])
        if key in by_key:
            raise TrainingRunError(f"duplicate {run['pair_key']}={run['arm']} run for seed {run['seed']}: "
                                   f"{by_key[key]['label']} and {run['label']}; pair one baseline with one "
                                   f"candidate per invocation")
        by_key[key] = run
    pairs: dict[str, object] = {}
    accumulated: dict[str, dict[str, list[float]]] = {}
    for seed in sorted({seed for seed, _ in by_key}, key=str):
        baseline, candidate = by_key.get((seed, baseline_value)), by_key.get((seed, candidate_value))
        if baseline is None or candidate is None:
            continue
        per_epoch = matched_delta(candidate, baseline)
        pairs[str(seed)] = {"baseline": baseline["label"], "candidate": candidate["label"],
                            "baseline_best_validation_point_loss": baseline["best_validation_point_loss"],
                            "candidate_best_validation_point_loss": candidate["best_validation_point_loss"],
                            "per_epoch": per_epoch}
        for epoch, values in per_epoch.items():
            bucket = accumulated.setdefault(epoch, {"validation_point_loss": [], "validation_mae": {}})
            bucket["validation_point_loss"].append(values["validation_point_loss"])
            for name, delta in values["validation_mae"].items():
                bucket["validation_mae"].setdefault(name, []).append(delta)
    mean = {epoch: {"validation_point_loss_mean": float(np.mean(values["validation_point_loss"])),
                    "validation_point_loss_spread": (float(np.max(values["validation_point_loss"])
                                                       - np.min(values["validation_point_loss"]))
                    if len(values["validation_point_loss"]) > 1 else 0.0),
                    "validation_mae_mean": {name: float(np.mean(deltas)) for name, deltas in values["validation_mae"].items()}}
            for epoch, values in sorted(accumulated.items(), key=lambda item: int(item[0]))}
    return pairs, mean


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="run directories, baseline first")
    parser.add_argument("--pair-by", default="window_policy",
                        help="run_config key whose values name the two arms, e.g. onset_weight")
    parser.add_argument("--baseline-value", default="fixed")
    parser.add_argument("--candidate-value", default="bar")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.output.exists() and any(arguments.output.iterdir()):
        raise SystemExit(f"refusing to overwrite existing output: {arguments.output}")
    runs = [summarize(read_run(directory), arguments.pair_by) for directory in arguments.runs]
    baseline = runs[0]
    pairs, paired_mean = paired_by_seed(runs, arguments.baseline_value, arguments.candidate_value)
    report = {"kind": "matched_sequence_run_comparison", "status": "complete", "baseline": baseline["label"],
              "pair_key": arguments.pair_by, "baseline_value": arguments.baseline_value,
              "candidate_value": arguments.candidate_value,
              "runs": runs, "delta_vs_baseline": {run["label"]: matched_delta(run, baseline) for run in runs[1:]},
              "paired_by_seed": pairs, "paired_mean_delta": paired_mean,
              "interpretation": {"loss_direction": "lower validation loss is a training signal only",
                                 "adoption_gate_changed": False, "blind_listening_performed": False,
                                 "human_ab_evidence": False, "deployment_evidence": False,
                                 "single_seed_warning": "one seed cannot support a piece-level or bootstrap claim; "
                                                       "repeat seeds and run piece-level paired bootstrap before any "
                                                       "adoption discussion"}}
    arguments.output.mkdir(parents=True, exist_ok=True)
    _write_json(arguments.output / "summary.json", report)
    digest = _sha256(arguments.output / "summary.json")
    (arguments.output / "_SUCCESS").write_text(digest + "\n", encoding="ascii")
    table = [{"label": run["label"], "policy": run["window_policy"], "best_epoch": run["best_epoch"],
              "best_val_loss": round(run["best_validation_loss"], 5),
              "best_mae": {name: (round(value, 5) if value is not None else None)
                          for name, value in run["best_validation_mae"].items()}} for run in runs]
    print(json.dumps({"output": str(arguments.output), "summary_sha256": digest, "runs": table,
                      "note": "total loss includes the bar term; compare point loss and MAE columns instead",
                      "paired_seed_count": len(pairs),
                      "paired_mean_point_loss_delta": {epoch: round(values["validation_point_loss_mean"], 5)
                                                       for epoch, values in paired_mean.items()},
                      "paired_mean_mae_delta": {epoch: {name: round(value, 5) for name, value in
                                                        values["validation_mae_mean"].items()}
                                                for epoch, values in paired_mean.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
