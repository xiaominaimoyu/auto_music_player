"""Audit locked validation sequence pilots and export raw per-work predictions."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.audit_expression import BOUNDED_TARGETS, IDENTITY_TARGETS, _Stats, _bootstrap
from training.bar_supervision import DEFAULT_BAR_HORIZONS
from training.midi_features import CHORD_TARGET_NAMES, NOTE_TARGET_NAMES
from training.sequence_expression import (
    SequenceExpressionModel, attach_bar_geometry, checkpoint_preparation_flags, load_selected_samples,
    make_model, make_windows_from_config, predict_sample,
)
from training.train_expression import TrainingRunError, _load_sample_id_file, _load_validation_cache, _resolve_device, _sha256, _write_json
from training.train_sequence_expression import PIPELINE_NAME, load_reference, reference_hash, write_integrity
from training.validate_pianocore_dataset import validate_dataset


def load_model(checkpoint: str | Path, device: str | torch.device = "cpu") -> tuple[SequenceExpressionModel, dict[str, Any]]:
    """Load this pipeline's own local checkpoint, returning prediction config."""
    checkpoint_path = Path(checkpoint).resolve()
    run_root = checkpoint_path.parent
    success, integrity = run_root / "_SUCCESS", run_root / "integrity.json"
    if not success.is_file() or not integrity.is_file() or (run_root / "_INCOMPLETE").exists():
        raise TrainingRunError("checkpoint must belong to a completed local sequence run")
    if success.read_text(encoding="ascii").strip() != _sha256(integrity):
        raise TrainingRunError("checkpoint run integrity fingerprint mismatch")
    inventory = json.loads(integrity.read_text(encoding="utf-8"))
    if inventory.get("pipeline") != PIPELINE_NAME:
        raise TrainingRunError("checkpoint run has an unexpected pipeline")
    entry = next((row for row in inventory.get("files", [])
                  if row.get("path") == checkpoint_path.name), None)
    if entry is None or entry.get("sha256") != _sha256(checkpoint_path):
        raise TrainingRunError("checkpoint is missing from or differs from run integrity")
    selected_device = torch.device(device)
    payload = torch.load(checkpoint_path, map_location=selected_device, weights_only=False)
    if payload.get("pipeline") != PIPELINE_NAME or not isinstance(payload.get("run_config"), dict):
        raise TrainingRunError("checkpoint is not a sequence-pilot artifact")
    config = payload["run_config"]
    model = make_model(config).to(selected_device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, config


def _bar_horizons(config: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(value) for value in config.get("bar_horizons", DEFAULT_BAR_HORIZONS))


def attach_checkpoint_bar_geometry(samples, reference, config):
    """Rebuild the checkpoint's score-only geometry so inference cuts the same windows."""
    policy = str(config.get("window_policy", "fixed"))
    if policy == "fixed":
        return samples
    if policy != "bar":
        raise TrainingRunError(f"checkpoint has an unknown window policy: {policy}")
    if reference is None:
        raise TrainingRunError("bar window policy requires the locked timing reference")
    horizons = _bar_horizons(config)
    return [attach_bar_geometry(sample, reference[str(sample.sample_id)], horizons=horizons) for sample in samples]


def bar_layout_coverage(samples, config) -> dict[str, Any]:
    """Report spans kept by the planned cores per horizon; labels only filter rows."""
    horizons = _bar_horizons(config)
    summary = {str(horizon): {"geometry_spans": 0, "retained_spans": 0, "eligible_spans": 0,
                              "interval_rows": 0} for horizon in horizons}
    core_count = chord_count = 0
    for sample in samples:
        if sample.bar_span_rows is None or not len(sample.bar_span_rows):
            continue
        chord_count += len(sample.chord_features)
        rows = np.asarray(sample.bar_span_rows, dtype=np.int64)
        kinds = np.asarray(sample.bar_span_horizons, dtype=np.int64)
        invalid = np.r_[0, np.cumsum(~sample.chord_masks[:, 1], dtype=np.int64)]
        for horizon in horizons:
            summary[str(horizon)]["geometry_spans"] += int((kinds == horizon).sum())
        for window in make_windows_from_config(sample, config):
            core_count += 1
            retained = np.asarray(window.bar_span_rows, dtype=np.int64).reshape(-1, 2)
            retained_kinds = np.asarray(window.bar_span_horizons, dtype=np.int64)
            if not len(retained):
                continue
            eligible = (invalid[retained[:, 1] + 1] - invalid[retained[:, 0] + 1]) == 0
            widths = retained[:, 1] - retained[:, 0]
            for horizon in horizons:
                key = str(horizon)
                selected = retained_kinds == horizon
                summary[key]["retained_spans"] += int(selected.sum())
                summary[key]["eligible_spans"] += int((selected & eligible).sum())
                summary[key]["interval_rows"] += int(widths[selected & eligible].sum())
    return {"window_policy": str(config.get("window_policy", "fixed")), "sample_count": len(samples),
            "horizons": summary, "core_count": core_count, "chord_count": chord_count,
            "eligibility_rule": "every interval row of the span is native-valid; masks never move a core cut",
            "model_input": False, "deployment_evidence": False}


def audit(*, dataset: str | Path, checkpoint: str | Path, output: str | Path,
          sample_id_file: str | Path, validation_cache: str | Path | None = None,
          timing_reference: str | Path | None = None, device: str = "auto",
          seed: int = 20261003) -> dict[str, Any]:
    dataset_root, checkpoint_path, output_root = Path(dataset).resolve(), Path(checkpoint).resolve(), Path(output).resolve()
    sample_path = Path(sample_id_file).resolve()
    sample_ids = _load_sample_id_file(sample_path)
    selected_device = _resolve_device(device)
    model, config = load_model(checkpoint_path, selected_device)
    if config.get("dataset_success_sha256") != _sha256(dataset_root / "_SUCCESS"):
        raise TrainingRunError("checkpoint dataset fingerprint mismatch")
    if config.get("validation_sample_id_file_sha256") != _sha256(sample_path):
        raise TrainingRunError("audit selection must match locked checkpoint validation IDs")
    report = (_load_validation_cache(dataset_root, validation_cache)
              if validation_cache is not None else validate_dataset(dataset_root))
    reference_path = timing_reference if timing_reference is not None else config.get("timing_reference")
    if reference_hash(reference_path) != config.get("timing_reference_sha256"):
        raise TrainingRunError("audit timing reference does not match training contract")
    reference = load_reference(reference_path)
    centre_duration, derive_shape, _relative = checkpoint_preparation_flags(config)
    samples = load_selected_samples(dataset_root, "validation", sample_ids,
                                    timing_reference=reference, relative_position=bool(config["relative_position"]),
                                    duration_centering=centre_duration, shape_features=derive_shape)
    samples = attach_checkpoint_bar_geometry(samples, reference, config)
    identities: dict[str, tuple[str, str]] = {}
    with (dataset_root / "manifest.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["id"])
            if sample_id in sample_ids:
                source = str(row.get("performance_dataset") or "unknown")
                identities[sample_id] = (source, f"{source}::{row.get('piece_group') or sample_id}")
    if output_root.exists() and any(output_root.iterdir()):
        raise TrainingRunError(f"audit output must be empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "_INCOMPLETE").write_text("sequence audit in progress\n", encoding="ascii")
    predictions_root = output_root / "predictions"
    predictions_root.mkdir()
    rng = np.random.default_rng(seed)
    stats: dict[str, _Stats] = {}
    source_stats: dict[str, _Stats] = {}
    work_parts: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
    prediction_files: dict[str, str] = {}
    timing_metrics: dict[str, Any] = {}
    for sample in samples:
        note_prediction, chord_prediction = predict_sample(model, sample, config=config, device=selected_device)
        source, work_key = identities[sample.sample_id]
        filename = re.sub(r"[^A-Za-z0-9_.-]", "_", sample.sample_id) + ".npz"
        path = predictions_root / filename
        if path.exists():
            raise TrainingRunError("sample prediction filenames collide")
        # Score-only geometry so the CPU benchmark can rebuild this run's exact
        # window layout without targets, masks, or a performance clock.
        geometry = {} if sample.bar_span_rows is None else {
            "bar_span_rows": sample.bar_span_rows, "bar_span_horizons": sample.bar_span_horizons,
            "bar_boundary_indices": sample.bar_boundary_indices,
            "bar_requested_horizons": sample.bar_requested_horizons}
        np.savez_compressed(path, sample_id=np.asarray(sample.sample_id), work_key=np.asarray(work_key),
                            note_prediction=note_prediction, chord_prediction=chord_prediction,
                            note_targets=sample.note_targets, chord_targets=sample.chord_targets,
                            note_masks=sample.note_masks, chord_masks=sample.chord_masks,
                            note_features=sample.note_features, chord_features=sample.chord_features,
                            note_to_chord=sample.note_to_chord, **geometry)
        prediction_files[sample.sample_id] = path.relative_to(output_root).as_posix()
        for kind, prediction, names in (("note", note_prediction, NOTE_TARGET_NAMES), ("chord", chord_prediction, CHORD_TARGET_NAMES)):
            target = getattr(sample, f"{kind}_targets")
            masks = getattr(sample, f"{kind}_masks")
            means = np.asarray(config[f"{kind}_target_means"])
            for index, name in enumerate(names):
                valid = masks[:, index]
                if not valid.any():
                    continue
                baseline_name = "identity" if name in IDENTITY_TARGETS else "mean"
                baseline = np.zeros(int(valid.sum())) if baseline_name == "identity" else np.full(int(valid.sum()), means[index])
                for model_name, pred in (("model", prediction[valid, index]), (baseline_name, baseline)):
                    key = f"{model_name}:{name}"
                    target_values = target[valid, index]
                    stats.setdefault(key, _Stats(sample_rate=1.0, rng=rng)).add(pred, target_values)
                    source_key = f"{key}:{source}"
                    source_stats.setdefault(source_key, _Stats(sample_rate=1.0, rng=rng)).add(pred, target_values)
                    if name in BOUNDED_TARGETS:
                        stats[key].add_bounds(pred)
                        source_stats[source_key].add_bounds(pred)
                    work_parts[(key, work_key)].append((float(np.abs(pred - target_values).sum()), int(valid.sum())))
        if reference is not None:
            from training.tempo_reference import evaluate_timing_prediction
            ref = reference[sample.sample_id]
            timing_metrics[sample.sample_id] = {
                model_name: {mode: evaluate_timing_prediction(ref, values, calibration=mode, fixed_scale=1.0)
                             for mode in ("oracle_global", "fixed_score")}
                for model_name, values in (("model", chord_prediction[:, 1]), ("identity", np.zeros(len(chord_prediction))))
            }
    work_values: dict[str, dict[str, float]] = defaultdict(dict)
    for (key, work_key), parts in work_parts.items():
        work_values[key][work_key] = sum(value for value, _ in parts) / sum(count for _, count in parts)
    result: dict[str, Any] = {
        "status": "complete", "pipeline": PIPELINE_NAME, "split": "validation",
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": _sha256(checkpoint_path),
        "dataset_root": str(dataset_root), "dataset_validation": report,
        "architecture": config["architecture"], "target_shuffle_training": bool(config["target_shuffle"]),
        "sample_id_file": str(sample_path), "sample_id_file_sha256": _sha256(sample_path),
        "sample_id_count": len(sample_ids), "work_unit": "performance_dataset::piece_group",
        "work_aggregation": "within-work event weighted MAE; across-work equal weighting",
        "metrics": {key: value.report() for key, value in sorted(stats.items())},
        "source_metrics": {key: value.report() for key, value in sorted(source_stats.items())},
        "work_equal": {key: _bootstrap(value.values(), seed=seed) for key, value in sorted(work_values.items())},
        "work_values": dict(work_values), "prediction_files": prediction_files,
        "timing_reference_sha256": reference_hash(reference_path), "timing_metrics": timing_metrics,
        "timing_evaluation_modes": {
            "oracle_global": "ground-truth fitted scale used only to evaluate expressive residuals; not inference tempo recovery",
            "fixed_score": "deployment-style score tempo with fixed scale 1; no performance-derived input",
        },
        "device": str(selected_device), "seed": seed,
        "window_policy": str(config.get("window_policy", "fixed")),
        "max_window_chords": config.get("max_window_chords"),
        "bar_horizons": list(_bar_horizons(config)),
        "bar_layout": bar_layout_coverage(samples, config),
    }
    _write_json(output_root / "report.json", result)
    (output_root / "report.json.sha256").write_text(_sha256(output_root / "report.json") + "\n", encoding="ascii")
    write_integrity(output_root)
    (output_root / "_INCOMPLETE").unlink()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "checkpoint", "sample-id-file"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--output", "--output-dir", dest="output", required=True)
    parser.add_argument("--validation-cache")
    parser.add_argument("--timing-reference")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=20261003)
    args = vars(parser.parse_args(argv))
    try:
        result = audit(**args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"sequence audit failed: {exc}\n")
    print(json.dumps({key: result[key] for key in ("status", "architecture", "sample_id_count")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
