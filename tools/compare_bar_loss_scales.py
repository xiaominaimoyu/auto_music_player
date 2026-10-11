"""CPU scale comparison for the bar auxiliary losses; this script trains nothing.

Reuses a completed sequence run's weights and train-fitted normalizers, takes the first
N sample ids in sorted order, and runs evaluation epochs (``optimizer=None``) under the
fixed and bar policies with all three bar weights set to 1.0 so each component is
reported at its own natural magnitude.  The point of the artifact is scale: how large
elapsed/tempo/local are next to the point loss, so a pilot can pick weights without
letting bar supervision swallow it.
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
from training.sequence_expression import (
    attach_bar_geometry, load_selected_samples, make_windows_from_config,
)
from training.train_expression import _load_sample_id_file, _sha256, _write_json
from training.train_sequence_expression import _run_epoch
from training.tempo_reference import load_timing_reference

BAR_WEIGHT_NAMES = ("bar_elapsed_weight", "bar_tempo_weight", "bar_local_weight")


def window_profile(samples, config) -> dict[str, float]:
    widths, cores = [], 0
    for sample in samples:
        windows = make_windows_from_config(sample, config)
        cores += len(windows)
        widths += [window.end - window.start for window in windows]
    return {"core_count": cores, "mean_core_chords": round(np.mean(widths), 2) if widths else 0,
            "maximum_window_chords": max(widths) if widths else 0}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--timing-reference", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--sample-id-file", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--cpu-threads", type=int, default=4)
    arguments = parser.parse_args(argv)
    if arguments.output.exists() and any(arguments.output.iterdir()):
        raise SystemExit(f"refusing to overwrite existing output: {arguments.output}")
    if arguments.count <= 0:
        raise SystemExit("sample count must be positive")
    torch.set_num_threads(arguments.cpu_threads)
    model, config = load_model(arguments.source_checkpoint, "cpu")
    reference = load_timing_reference(arguments.timing_reference)
    all_ids = _load_sample_id_file(arguments.sample_id_file)
    selected = frozenset(sorted(str(value) for value in all_ids)[:arguments.count])
    common = {"architecture": config["architecture"], "hidden_dim": config["hidden_dim"],
              "dropout": config["dropout"], "relative_position": bool(config["relative_position"]),
              "core_size": config["core_size"], "context_size": config["context_size"],
              "batch_size": config["batch_size"], "target_shuffle": False,
              "note_input_means": config["note_input_means"], "note_input_scales": config["note_input_scales"],
              "chord_input_means": config["chord_input_means"], "chord_input_scales": config["chord_input_scales"],
              "note_target_means": config["note_target_means"], "note_target_scales": config["note_target_scales"],
              "chord_target_means": config["chord_target_means"], "chord_target_scales": config["chord_target_scales"],
              "onset_weight": config["onset_weight"], "duration_weight": config["duration_weight"],
              "velocity_weight": config["velocity_weight"], "pedal_weight": config["pedal_weight"],
              "ioi_weight": config["ioi_weight"], "micro_weight": config.get("micro_weight", 1.0),
              "cumulative_weight": 0.0}
    fixed_config = dict(common, window_policy="fixed")
    bar_config = dict(common, window_policy="bar", max_window_chords=512, bar_horizons=[1, 4, 16],
                      **{name: 1.0 for name in BAR_WEIGHT_NAMES})
    fixed_samples = load_selected_samples(arguments.dataset.resolve(), arguments.split, selected,
                                          timing_reference=reference,
                                          relative_position=bool(config["relative_position"]))
    bar_samples = [attach_bar_geometry(sample, reference[str(sample.sample_id)], horizons=(1, 4, 16))
                   for sample in fixed_samples]
    device = torch.device("cpu")
    fixed_metrics = _run_epoch(model, fixed_samples, fixed_config, device=device, optimizer=None, seed=arguments.seed)
    bar_metrics = _run_epoch(model, bar_samples, bar_config, device=device, optimizer=None, seed=arguments.seed)
    ioi_names = [name for name in bar_metrics["target_mae"] if name.endswith("ioi_ratio") or "ioi" in name]
    scales = {name: bar_metrics["bar_component_losses"][name] for name in BAR_WEIGHT_NAMES}
    point = bar_metrics["point_loss"]
    report = {"kind": "bar_loss_scale_comparison", "status": "complete", "trained": False,
              "model_quality_evaluated": False, "adoption_evidence": False,
              "dataset": str(arguments.dataset.resolve()), "source_checkpoint": str(arguments.source_checkpoint.resolve()),
              "source_checkpoint_sha256": _sha256(arguments.source_checkpoint),
              "timing_reference": str(arguments.timing_reference.resolve()),
              "timing_reference_sha256": _sha256(arguments.timing_reference / "summary.json"),
              "sample_id_file": str(arguments.sample_id_file.resolve()),
              "sample_id_file_sha256": _sha256(arguments.sample_id_file),
              "split": arguments.split, "requested_count": arguments.count,
              "loaded_sample_count": len(fixed_samples),
              "chord_count": sum(len(sample.chord_features) for sample in fixed_samples),
              "seed": arguments.seed, "cpu_threads": arguments.cpu_threads,
              "weights_used_for_this_probe": {name: 1.0 for name in BAR_WEIGHT_NAMES},
              "fixed_policy": {"loss": fixed_metrics["loss"], "point_loss": fixed_metrics["point_loss"],
                               "window_profile": window_profile(fixed_samples, fixed_config),
                               "target_mae_ioi": {name: fixed_metrics["target_mae"][name] for name in ioi_names}},
              "bar_policy": {"loss": bar_metrics["loss"], "point_loss": bar_metrics["point_loss"],
                             "window_profile": window_profile(bar_samples, bar_config),
                             "bar_loss": bar_metrics["bar_loss"], "bar_components": scales,
                             "bar_supervised_steps": bar_metrics["bar_supervised_steps"],
                             "bar_horizon_counts": bar_metrics["bar_horizon_counts"],
                             "bar_local_rows_excluded_count": bar_metrics["bar_local_rows_excluded_count"],
                             "target_mae_ioi": {name: bar_metrics["target_mae"][name] for name in ioi_names}},
              "relative_to_point_loss": {name: (None if not point else round(value / point, 4))
                                         for name, value in scales.items()},
              "suggested_weights_at_share": {
                  str(share): {name: (None if not value else round(share * point / value, 4))
                               for name, value in scales.items()}
                  for share in (0.05, 0.10, 0.25)},
              "interpretation": {"point_loss_compare": "fixed and bar point losses are NOT interchangeable: "
                                                       "different window counts and padding change the masked mean",
                                 "purpose": "bar component magnitudes only; weights still need a pilot and "
                                            "piece-level validation before any adoption decision"}}
    arguments.output.mkdir(parents=True, exist_ok=True)
    _write_json(arguments.output / "summary.json", report)
    digest = _sha256(arguments.output / "summary.json")
    (arguments.output / "_SUCCESS").write_text(digest + "\n", encoding="ascii")
    print(json.dumps({"output": str(arguments.output), "summary_sha256": digest,
                      "bar_components": scales, "point_loss": point,
                      "relative_to_point_loss": report["relative_to_point_loss"],
                      "suggested": report["suggested_weights_at_share"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
