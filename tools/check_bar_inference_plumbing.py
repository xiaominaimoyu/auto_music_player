"""End-to-end plumbing check for the bar window policy; performs no training.

Reuses an existing sequence run's weights and train-fitted normalizers, overrides only
the window policy and the balanced development bindings, then drives the real
``audit_sequence_expression`` and ``benchmark_sequence_inference`` over 50 locked
validation samples.  Outputs prove that score-only geometry survives the audit NPZ,
that integrity binds it, and that zero-target CPU inference keeps the planned layout;
they say nothing about model quality, listening preference, or player adoption.
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

from training.audit_sequence_expression import audit
from training.benchmark_sequence_inference import benchmark
from training.train_expression import TrainingRunError, _sha256, _write_json
from training.train_sequence_expression import PIPELINE_NAME, reference_hash, write_integrity

SOURCE_KEYS = ("architecture", "hidden_dim", "dropout", "relative_position", "note_input_means",
               "note_input_scales", "chord_input_means", "chord_input_scales", "note_target_means",
               "note_target_scales", "chord_target_means", "chord_target_scales", "dataset_success_sha256")


def build_run(root: Path, source_payload: dict, policy: str, bindings: dict) -> Path:
    run = root / policy
    if run.exists() and any(run.iterdir()):
        raise TrainingRunError(f"refusing to overwrite existing run: {run}")
    run.mkdir(parents=True)
    config = {key: source_payload["run_config"][key] for key in SOURCE_KEYS}
    config.update(bindings)
    config.update({"pipeline": PIPELINE_NAME, "pipeline_version": 1, "dataset_root": bindings["dataset_root"],
                   "window_policy": policy, "context_size": bindings["context_size"],
                   "max_window_chords": bindings["max_window_chords"], "bar_horizons": bindings["bar_horizons"],
                   "target_shuffle": False, "epochs": 0, "best_epoch": 0,
                   "plumbing_check": True, "trained_by_this_script": False,
                   "weights": f"copied from {bindings['source_checkpoint']}",
                   "core_size": bindings["core_size"], "batch_size": bindings["batch_size"]})
    _write_json(run / "run_config.json", config)
    checkpoint = run / "checkpoint_plumbing.pt"
    torch.save({"pipeline": PIPELINE_NAME, "pipeline_version": 1, "model": source_payload["model"],
                "epoch": 0, "run_config": config}, checkpoint)
    write_integrity(run)
    return run


def rebuild_layout_from_audit(audit_root: Path) -> dict[str, int]:
    """Re-plan windows from the audit NPZ geometry alone; labels stay unread."""
    from training.bar_supervision import BarSpans
    from training.bar_windows import plan_bar_windows

    fields = ("bar_span_rows", "bar_span_horizons", "bar_boundary_indices", "bar_requested_horizons")
    cores = chords = fixed_cores = widest = 0
    samples = 0
    for path in sorted((audit_root / "predictions").glob("*.npz")):
        with np.load(path, allow_pickle=False) as stored:
            if any(name not in stored for name in fields):
                continue
            rows, kinds = stored["bar_span_rows"], stored["bar_span_horizons"]
            bounds, requested = stored["bar_boundary_indices"], stored["bar_requested_horizons"]
            count = len(stored["chord_features"])
        spans = BarSpans(count, rows[:, 0], rows[:, 1], kinds, np.zeros(count),
                         tuple(int(value) for value in requested), len(bounds), 0, {}, {})
        windows = plan_bar_windows(count, bounds, spans, 16, 512)
        cores += len(windows)
        chords += count
        fixed_cores += -(-count // 128)
        widest = max([widest] + [window.end - window.start for window in windows])
        samples += 1
    return {"sample_count": samples, "core_count": cores, "chord_count": chords,
            "fixed128_core_count": fixed_cores, "maximum_window_chords": widest}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True,
                        help="completed sequence run's checkpoint_best.pt; weights are copied, never trained")
    parser.add_argument("--timing-reference", type=Path, required=True)
    parser.add_argument("--validation-sample-id-file", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", choices=("fixed", "bar"), default=["fixed", "bar"])
    parser.add_argument("--context-size", type=int, default=16)
    parser.add_argument("--max-window-chords", type=int, default=512)
    parser.add_argument("--core-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu-threads", type=int, default=4)
    arguments = parser.parse_args(argv)
    if arguments.work_root.exists() and any(arguments.work_root.iterdir()):
        raise SystemExit(f"refusing to overwrite existing work root: {arguments.work_root}")
    dataset_root = arguments.dataset.resolve()
    source = arguments.source_checkpoint.resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("pipeline") != PIPELINE_NAME:
        raise SystemExit("source checkpoint is not a sequence-pilot artifact")
    bindings = {"dataset_root": str(dataset_root), "source_checkpoint": str(source),
                "context_size": arguments.context_size, "max_window_chords": arguments.max_window_chords,
                "bar_horizons": [1, 4, 16], "core_size": arguments.core_size, "batch_size": arguments.batch_size,
                "timing_reference": str(arguments.timing_reference.resolve()),
                "timing_reference_sha256": reference_hash(arguments.timing_reference),
                "validation_sample_id_file": str(arguments.validation_sample_id_file.resolve()),
                "validation_sample_id_file_sha256": _sha256(arguments.validation_sample_id_file.resolve())}
    results = {}
    for policy in arguments.policies:
        run = build_run(arguments.work_root / "runs", payload, policy, bindings)
        checkpoint = run / "checkpoint_plumbing.pt"
        report = audit(dataset=dataset_root, checkpoint=checkpoint, output=arguments.work_root / f"audit_{policy}",
                       sample_id_file=arguments.validation_sample_id_file,
                       validation_cache=arguments.validation_cache, timing_reference=arguments.timing_reference,
                       device="cpu", seed=20261006)
        timing = benchmark(audit=arguments.work_root / f"audit_{policy}", checkpoint=checkpoint,
                           output=arguments.work_root / f"benchmark_{policy}", cpu_threads=arguments.cpu_threads)
        layout = report["bar_layout"]
        rebuilt = rebuild_layout_from_audit(arguments.work_root / f"audit_{policy}")
        results[policy] = {"audit_status": report["status"], "sample_count": report["sample_id_count"],
                           "window_policy_in_report": report["window_policy"], "core_count": layout["core_count"],
                           "chord_count": layout["chord_count"], "horizons": layout["horizons"],
                           "layout_rebuilt_from_audit_npz": rebuilt,
                           "audit_and_npz_layout_match": (rebuilt["core_count"] == layout["core_count"]
                                                          and rebuilt["chord_count"] == layout["chord_count"]),
                           "benchmark": {key: timing[key] for key in
                                         ("median_seconds", "p95_seconds", "maximum_seconds", "sample_count",
                                          "planned_core_count_smallest_sample",
                                          "fixed128_core_count_smallest_sample", "bitwise_repeat_passed",
                                          "bar_geometry_sample_count")},
                           "report_sha256": _sha256(arguments.work_root / f"audit_{policy}" / "report.json")}
    summary = {"kind": "bar_inference_plumbing_check", "status": "complete", "trained": False,
               "model_quality_evaluated": False, "adoption_evidence": False,
               "note": "untrained-by-this-script weights reused only to exercise the real audit/benchmark path",
               "dataset": str(dataset_root), "source_checkpoint": str(source),
               "timing_reference": bindings["timing_reference"],
               "timing_reference_sha256": bindings["timing_reference_sha256"],
               "validation_sample_id_file": bindings["validation_sample_id_file"],
               "validation_sample_id_file_sha256": bindings["validation_sample_id_file_sha256"],
               "validation_cache": str(arguments.validation_cache.resolve()),
               "validation_cache_sha256": _sha256(arguments.validation_cache),
               "context_size": arguments.context_size, "max_window_chords": arguments.max_window_chords,
               "cpu_threads": arguments.cpu_threads, "results": results,
               "implementation_sha256": {name: _sha256(PROJECT_ROOT / "training" / name)
                                         for name in ("bar_windows.py", "bar_objective.py", "bar_supervision.py",
                                                      "sequence_expression.py", "train_sequence_expression.py",
                                                      "audit_sequence_expression.py",
                                                      "benchmark_sequence_inference.py")}}
    output = arguments.work_root
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "summary.json", summary)
    print(json.dumps({"summary": str(output / "summary.json"),
                      "summary_sha256": _sha256(output / "summary.json"),
                      "benchmark": {policy: results[policy]["benchmark"] for policy in results}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
