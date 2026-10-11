"""Score the bar auxiliary objective on checkpoints that were trained with and without it.

The bar losses are functions of the same per-chord log-IOI outputs the point loss already
supervises, so an arm trained on them can only help if it actually lowers the aggregate.
This tool loads several completed sequence runs, evaluates every checkpoint on one shared
locked validation set under an identical bar probe (weights ~0 so the probe cannot change
the epoch average), and reports each checkpoint's auxiliary objective side by side.

It trains nothing and never touches the player.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from training.bar_windows import DEFAULT_MAX_WINDOW_CHORDS
from training.audit_sequence_expression import load_model
from training.sequence_expression import attach_bar_geometry, load_selected_samples
from training.train_expression import TrainingRunError, _load_sample_id_file, _sha256, _write_json
from training.train_sequence_expression import _run_epoch
from training.tempo_reference import load_timing_reference

PROBE_WEIGHTS = 1e-9
BAR_HORIZONS = (1, 4, 16)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--timing-reference", type=Path, required=True)
    parser.add_argument("--sample-id-file", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="completed run directories")
    parser.add_argument("--checkpoint", default="checkpoint_best.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--cpu-threads", type=int, default=4)
    arguments = parser.parse_args(argv)
    if arguments.output.exists() and any(arguments.output.iterdir()):
        raise SystemExit(f"refusing to overwrite existing output: {arguments.output}")
    torch.set_num_threads(arguments.cpu_threads)
    dataset_root = arguments.dataset.resolve()
    sample_ids = frozenset(str(value) for value in _load_sample_id_file(arguments.sample_id_file))
    reference = load_timing_reference(arguments.timing_reference)
    loaded = load_selected_samples(dataset_root, arguments.split, sample_ids, timing_reference=reference,
                                   relative_position=True)
    samples = [attach_bar_geometry(sample, reference[str(sample.sample_id)], horizons=BAR_HORIZONS)
               for sample in loaded]
    if not samples:
        raise TrainingRunError("no samples selected for the bar objective probe")
    results = []
    for run_directory in arguments.runs:
        checkpoint = (run_directory / arguments.checkpoint).resolve()
        model, config = load_model(checkpoint, "cpu")
        probe = dict(config)
        probe.update({"window_policy": "bar",
                      "max_window_chords": int(config.get("max_window_chords") or DEFAULT_MAX_WINDOW_CHORDS),
                      "bar_horizons": list(BAR_HORIZONS), "target_shuffle": False, "cumulative_weight": 0.0,
                      **{name: PROBE_WEIGHTS for name in ("bar_elapsed_weight", "bar_tempo_weight",
                                                          "bar_local_weight")}})
        metrics = _run_epoch(model, samples, probe, device=torch.device("cpu"), optimizer=None, seed=arguments.seed)
        results.append({"run": run_directory.name, "checkpoint": str(checkpoint),
                        "checkpoint_sha256": _sha256(checkpoint), "trained_window_policy": config.get("window_policy"),
                        "trained_bar_weights": {name: config.get(name) for name in
                                                ("bar_elapsed_weight", "bar_tempo_weight", "bar_local_weight")},
                        "auxiliary_objective": metrics["bar_component_losses"],
                        "point_loss": metrics["point_loss"],
                        "validation_mae": {name: metrics["target_mae"].get(name) for name in
                                           ("log_ioi_ratio", "log_key_duration_ratio", "onset_delta_ms")},
                        "eligible_span_counts": {key: value["eligible_spans"]
                                                 for key, value in metrics["bar_horizon_counts"].items()}})
    report = {"kind": "bar_objective_cross_evaluation", "status": "complete", "trained": False,
              "probe_weight": PROBE_WEIGHTS,
              "probe_note": "every checkpoint is scored with the same near-zero weights, so the reported auxiliary "
                            "values are measurements and not a training signal",
              "dataset": str(dataset_root), "split": arguments.split, "sample_count": len(samples),
              "chord_count": sum(len(sample.chord_features) for sample in samples),
              "sample_id_file": str(arguments.sample_id_file.resolve()),
              "sample_id_file_sha256": _sha256(arguments.sample_id_file),
              "timing_reference_sha256": _sha256(arguments.timing_reference / "summary.json"),
              "seed": arguments.seed, "cpu_threads": arguments.cpu_threads, "results": results,
              "reading": {"lowest_is_best": "compare auxiliary_objective across runs; a run trained on the bar loss "
                                            "that is not lower than a run that never saw it means the objective was "
                                            "not actually fitted",
                          "adoption_evidence": False, "model_quality_claim": False}}
    arguments.output.mkdir(parents=True, exist_ok=True)
    _write_json(arguments.output / "summary.json", report)
    digest = _sha256(arguments.output / "summary.json")
    (arguments.output / "_SUCCESS").write_text(digest + "\n", encoding="ascii")
    print(json.dumps({"output": str(arguments.output), "summary_sha256": digest,
                      "runs": [{r["run"]: r["auxiliary_objective"]} for r in results]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
