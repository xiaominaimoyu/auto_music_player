"""Piece-level paired bootstrap of the sequence model against its simple baselines.

The registered numeric gate for this line is not "validation loss went down": it is
``timing``/``duration`` improving by at least 5% over the strongest simple baseline, with a
piece-level 95% bootstrap interval that supports the improvement and no source regressing by
more than 2%.  Piece-level matters because one recording can contribute thousands of chords
while another contributes a few hundred -- row weighting would let a handful of long
recordings decide the verdict, and it hides the fact that recordings of the same piece share
everything a score-only model is able to predict.

Protocol (fixed here, matching the older flat-row pipeline's convention):
* the resampling unit is ``performance_dataset::piece_group``;
* inside a piece, recordings are averaged, then seeds are averaged, so seed noise cannot
  inflate the sample size;
* pieces carry equal weight inside a draw;
* 2,000 draws with a registered seed, percentile interval at 2.5/97.5;
* fewer than five pieces and no interval is reported at all.

Baselines: ``predict_zero`` (play the score as written) and ``predict_recording_median`` (one
constant per recording, the most a per-recording offset could explain).  The stronger of the
two is what the 5% requirement is measured against.  Nothing trains, and the locked test
split is refused.
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
import torch

from training.audit_sequence_expression import load_model
from training.sequence_expression import checkpoint_preparation_flags, load_selected_samples, predict_sample
from training.train_expression import TrainingRunError, _load_sample_id_file, _sha256, _write_json
from training.tempo_reference import load_timing_reference

# (channel, row kind, column) -- the same map tools/audit_target_channels.py uses
CHANNELS = (("onset_delta_ms", "chord", 0), ("log_ioi_ratio", "chord", 1),
            ("log_key_duration_ratio", "note", 1))
BASELINES = ("predict_zero", "predict_recording_median")
MINIMUM_PIECES = 5


def bootstrap_difference(pairs: list[tuple[float, float]], draws: int, seed: int) -> dict[str, object]:
    """Percentile interval for the paired difference (model minus baseline) across pieces."""
    if len(pairs) < MINIMUM_PIECES:
        return {"pieces": len(pairs), "note": f"fewer than {MINIMUM_PIECES} pieces, no interval reported"}
    differences = np.asarray([(model - baseline) for model, baseline in pairs], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(draws, differences.size))
    means = differences[indices].mean(axis=1)
    point = float(differences.mean())
    low, high = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return {"pieces": int(differences.size), "draws": draws, "seed": seed,
            "mean_paired_difference": point, "interval_low": low, "interval_high": high,
            "probability_model_worse": float((means > 0).mean()),
            "supports_improvement": bool(point < 0.0 and high < 0.0)}


def collapse_to_pieces(channel: str, supervised: dict[str, np.ndarray],
                       errors: dict[tuple[str, str, str], float], run_names: list[str],
                       identities: dict[str, tuple[str, str]]) -> dict[str, dict[str, float]]:
    """One row per ``source::piece_group``: recordings averaged, then seeds averaged.

    Averaging happens twice before anything is compared, so neither a long recording nor a
    lucky seed can carry more weight than a piece of music deserves.
    """
    buckets: dict[str, dict[str, list[float]]] = {}
    for sample_id, label in supervised.items():
        source, group = identities[sample_id]
        model_values = [errors[(channel, run, sample_id)] for run in run_names
                        if (channel, run, sample_id) in errors]
        if not model_values:
            continue
        bucket = buckets.setdefault(f"{source}::{group}", {"model": [], "predict_zero": [],
                                                           "predict_recording_median": [], "recordings": [],
                                                           "rows": []})
        bucket["model"].append(float(np.mean(model_values)))
        bucket["predict_zero"].append(float(np.abs(label).mean()))
        bucket["predict_recording_median"].append(float(np.abs(label - np.median(label)).mean()))
        bucket["recordings"].append(1.0)
        bucket["rows"].append(float(label.size))
    pieces: dict[str, dict[str, float]] = {}
    for piece, data in sorted(buckets.items()):
        merged = {key: float(np.mean(data[key])) for key in ("model", "predict_zero", "predict_recording_median")}
        # counts add up, errors average: mixing the two is how a long recording would otherwise
        # quietly buy itself extra weight inside its own piece
        merged["recordings"] = int(len(data["model"]))
        merged["rows"] = int(np.sum(data["rows"]))
        pieces[piece] = merged
    return pieces


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--timing-reference", type=Path, required=True)
    parser.add_argument("--sample-id-file", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="one directory per seed")
    parser.add_argument("--checkpoint", default="checkpoint_best.pt")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261003)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.output.exists() and any(arguments.output.iterdir()):
        raise SystemExit(f"refusing to overwrite existing output: {arguments.output}")
    if arguments.split == "test":
        raise SystemExit("the locked test split may only be read once the model is frozen")
    if arguments.draws < 100:
        raise SystemExit("draws must be at least 100")
    dataset_root = arguments.dataset.resolve()
    sample_ids = frozenset(str(value) for value in _load_sample_id_file(arguments.sample_id_file))
    identities: dict[str, tuple[str, str]] = {}
    with (dataset_root / "manifest.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["id"])
            if sample_id in sample_ids:
                identities[sample_id] = (str(row.get("performance_dataset") or "unknown"),
                                         str(row.get("piece_group") or sample_id))
    if set(identities) != set(sample_ids):
        raise TrainingRunError("dataset manifest does not cover every selected sample")
    reference = load_timing_reference(arguments.timing_reference)
    device = torch.device(arguments.device if torch.cuda.is_available() or arguments.device == "cpu" else "cpu")

    def _collect_labels(rows) -> dict[str, dict[str, np.ndarray]]:
        """Supervised label column per (channel, recording)."""
        collected: dict[str, dict[str, np.ndarray]] = {name: {} for name, _kind, _column in CHANNELS}
        for sample in rows:
            for name, kind, column in CHANNELS:
                mask = np.asarray(getattr(sample, f"{kind}_masks")[:, column], dtype=bool)
                label = np.asarray(getattr(sample, f"{kind}_targets")[:, column], dtype=np.float64)[mask]
                if label.size:
                    collected[name][sample.sample_id] = label
        return collected

    # one preparation per distinct flag combination: labels and inputs must be in the units
    # the checkpoint was trained in, or the arm comparison silently mixes unit systems
    prepared: dict[tuple[bool, bool, bool], list] = {}
    labels_by_flags: dict[tuple[bool, bool, bool], dict[str, dict[str, np.ndarray]]] = {}
    sample_count = 0
    # (channel, run name, sample id) -> mean absolute error of that run on that sample
    errors: dict[tuple[str, str, str], float] = {}
    for run_directory in arguments.runs:
        checkpoint = (run_directory / arguments.checkpoint).resolve()
        model, config = load_model(checkpoint, device)
        flags = checkpoint_preparation_flags(config)
        if flags not in prepared:
            prepared[flags] = load_selected_samples(dataset_root, arguments.split, sample_ids,
                                                    timing_reference=reference, relative_position=flags[2],
                                                    duration_centering=flags[0], shape_features=flags[1])
            if not prepared[flags]:
                raise TrainingRunError("no samples selected for the bootstrap")
            labels_by_flags[flags] = _collect_labels(prepared[flags])
            sample_count = len(prepared[flags])
        samples = prepared[flags]
        supervised = labels_by_flags[flags]
        for sample in samples:
            note_output, chord_output = predict_sample(model, sample, config=config, device=device)
            arrays = {"chord": chord_output, "note": note_output}
            for name, kind, column in CHANNELS:
                label = supervised[name].get(sample.sample_id)
                if label is None:
                    continue
                prediction = np.asarray(arrays[kind][:, column], dtype=np.float64)[
                    np.asarray(getattr(sample, f"{kind}_masks")[:, column], dtype=bool)]
                errors[(name, run_directory.name, sample.sample_id)] = float(np.abs(prediction - label).mean())
        del model

    run_names = [run.name for run in arguments.runs]
    channels: dict[str, object] = {}
    for name, _kind, _column in CHANNELS:
        pieces = collapse_to_pieces(name, supervised[name], errors, run_names, identities)
        summary: dict[str, object] = {"pieces": len(pieces),
                                      "recordings": int(sum(value["recordings"] for value in pieces.values())),
                                      "rows": int(sum(value["rows"] for value in pieces.values()))}
        if not pieces:
            summary["note"] = "no supervised rows for this channel"
            channels[name] = summary
            continue
        model_mae = float(np.mean([value["model"] for value in pieces.values()]))
        baseline_mae = {baseline: float(np.mean([value[baseline] for value in pieces.values()]))
                       for baseline in BASELINES}
        strongest = min(baseline_mae, key=lambda baseline: baseline_mae[baseline])
        comparisons = {baseline: bootstrap_difference([(value["model"], value[baseline]) for value in pieces.values()],
                                                      arguments.draws, arguments.bootstrap_seed)
                       for baseline in BASELINES}
        improvement = 1.0 - model_mae / baseline_mae[strongest] if baseline_mae[strongest] > 0 else None
        summary.update({
            "piece_equal_weight_mae": {"model": model_mae, **baseline_mae},
            "strongest_baseline": strongest,
            # a baseline of exactly zero means every recording was one row long: "predict its own
            # median" then reproduces it trivially, and no relative improvement can be defined
            "baseline_is_degenerate": baseline_mae[strongest] <= 0.0,
            "improvement_over_strongest_baseline": None if improvement is None else float(improvement),
            "paired_bootstrap": comparisons,
            "gate": {"required_relative_improvement": 0.05,
                     "relative_improvement_reached": improvement is not None and improvement >= 0.05,
                     "interval_supports_improvement": bool(comparisons[strongest].get("supports_improvement", False)),
                     "passed_numeric_gate": bool(improvement is not None and improvement >= 0.05
                                                 and comparisons[strongest].get("supports_improvement", False))},
        })
        channels[name] = summary

    report = {"kind": "piece_level_paired_bootstrap_vs_baselines", "status": "complete", "trained": False,
              "dataset": str(dataset_root), "split": arguments.split, "device": str(device),
              "runs": run_names, "sample_count": sample_count,
              "sample_id_file": str(arguments.sample_id_file.resolve()),
              "sample_id_file_sha256": _sha256(arguments.sample_id_file),
              "timing_reference_sha256": _sha256(arguments.timing_reference / "summary.json"),
              "protocol": {"unit": "performance_dataset::piece_group", "recordings_and_seeds_averaged_per_piece": True,
                           "piece_weighting": "equal", "draws": arguments.draws,
                           "bootstrap_seed": arguments.bootstrap_seed, "interval": [2.5, 97.5],
                           "sign": "a negative paired difference means the model errs less than the baseline"},
              "channels": channels,
              "reading": {"adoption_evidence": False,
                          "note": "passing this numeric gate is necessary but not sufficient: player adoption also "
                                  "requires human A/B preferences and the compiler/game expressibility check, with "
                                  "the rule-based path kept as the fallback"},
              "model_inputs_modified": False, "adoption_evidence": False}
    arguments.output.mkdir(parents=True, exist_ok=True)
    _write_json(arguments.output / "summary.json", report)
    digest = _sha256(arguments.output / "summary.json")
    (arguments.output / "_SUCCESS").write_text(digest + "\n", encoding="ascii")
    print(json.dumps({"output": str(arguments.output), "summary_sha256": digest,
                      "channels": {name: {key: value[key] for key in
                                         ("pieces", "piece_equal_weight_mae", "strongest_baseline",
                                          "improvement_over_strongest_baseline", "gate") if key in value}
                                   for name, value in channels.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
