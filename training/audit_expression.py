"""Reproducible, source-aware audit for an expression checkpoint.

Unlike the historical evaluator, this command keeps full-split sums, uses a
deterministic global p95 sample, reports source and work-equal metrics, and
stores bootstrap intervals over works.  It is read-only with respect to the
dataset and checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from training.midi_features import CHORD_TARGET_NAMES, NOTE_TARGET_NAMES
from training.train_expression import (
    ExpressionMLP,
    TrainingRunError,
    _load_sample_id_file,
    _load_validation_cache,
    _normalizer_vectors,
    _prepare_features,
    _resolve_device,
    _selected_shard_paths,
    _select_sample_arrays,
    _shard_paths,
    _validate_sample_ids,
)
from training.validate_pianocore_dataset import validate_dataset


IDENTITY_TARGETS = {
    "micro_offset_ms",
    "log_key_duration_ratio",
    "onset_delta_ms",
    "log_ioi_ratio",
}
BOUNDED_TARGETS = {
    "velocity_0_1",
    "pedal_at_onset",
    "pedal_coverage",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _Stats:
    def __init__(self, *, sample_rate: float, rng: np.random.Generator) -> None:
        self.n = 0
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.pred_sum = 0.0
        self.target_sum = 0.0
        self.pred_sq_sum = 0.0
        self.target_sq_sum = 0.0
        self.cross_sum = 0.0
        self.bias_sum = 0.0
        self.out_of_range = 0
        self.sample: list[float] = []
        self.sample_rate = float(sample_rate)
        self.rng = rng

    def add(self, prediction: np.ndarray, target: np.ndarray) -> None:
        prediction = np.asarray(prediction, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if prediction.shape != target.shape:
            raise TrainingRunError("audit prediction/target shape mismatch")
        if not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise TrainingRunError("audit encountered a non-finite prediction or target")
        error = prediction - target
        self.n += int(len(error))
        self.abs_sum += float(np.abs(error).sum())
        self.sq_sum += float(np.square(error).sum())
        self.pred_sum += float(prediction.sum())
        self.target_sum += float(target.sum())
        self.pred_sq_sum += float(np.square(prediction).sum())
        self.target_sq_sum += float(np.square(target).sum())
        self.cross_sum += float((prediction * target).sum())
        self.bias_sum += float(error.sum())
        if self.sample_rate:
            selected = np.flatnonzero(self.rng.random(len(error)) < self.sample_rate)
            self.sample.extend(np.abs(error[selected]).tolist())

    def add_bounds(self, prediction: np.ndarray) -> None:
        self.out_of_range += int(((prediction < 0.0) | (prediction > 1.0)).sum())

    def report(self) -> dict[str, float | int | None]:
        if not self.n:
            return {"n": 0, "mae": None, "rmse": None, "bias": None, "r2": None, "correlation": None, "prediction_std": None, "target_std": None, "p95": None, "p95_sample_n": 0, "out_of_range_fraction": None}
        n = float(self.n)
        pred_mean = self.pred_sum / n
        target_mean = self.target_sum / n
        pred_var = max(0.0, self.pred_sq_sum / n - pred_mean * pred_mean)
        target_var = max(0.0, self.target_sq_sum / n - target_mean * target_mean)
        covariance = self.cross_sum / n - pred_mean * target_mean
        return {
            "n": self.n,
            "mae": self.abs_sum / n,
            "rmse": math.sqrt(self.sq_sum / n),
            "bias": self.bias_sum / n,
            "r2": 1.0 - self.sq_sum / (n * target_var) if target_var else None,
            "correlation": covariance / math.sqrt(pred_var * target_var) if pred_var and target_var else None,
            "prediction_std": math.sqrt(pred_var),
            "target_std": math.sqrt(target_var),
            "p95": float(np.quantile(self.sample, 0.95)) if self.sample else None,
            "p95_sample_n": len(self.sample),
            "out_of_range_fraction": self.out_of_range / n,
        }


def _load_model(checkpoint: Path, device: torch.device) -> tuple[ExpressionMLP, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = payload.get("run_config")
    if not isinstance(config, dict):
        raise TrainingRunError("checkpoint is missing run_config")
    model = ExpressionMLP(
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, config


def _array_config(config: dict[str, Any], key: str, size: int, default: float) -> np.ndarray:
    value = config.get(key)
    if value is None:
        return np.full(size, default, dtype=np.float32)
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise TrainingRunError(f"checkpoint has invalid {key}")
    return result


def _source_name(path: Path) -> str:
    name = path.parent.name.removeprefix("source=")
    return name.split("--", 1)[0]


def _source_work_indices(
    arrays: dict[str, np.ndarray],
    kind: str,
    work_key_by_sample: dict[str, str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    offset_name = "sample_note_offsets" if kind == "note" else "sample_chord_offsets"
    offsets = arrays[offset_name]
    sample_ids = [str(value) for value in arrays["sample_ids"].tolist()]
    ids = [
        work_key_by_sample.get(sample_id, sample_id)
        if work_key_by_sample is not None
        else sample_id
        for sample_id in sample_ids
    ]
    return np.repeat(np.asarray(ids, dtype=object), np.diff(offsets)), sample_ids


def _add_work_values(
    work_values: dict[tuple[str, str, str], list[float]],
    *,
    model_name: str,
    target_name: str,
    work_keys: np.ndarray,
    error: np.ndarray,
    mask: np.ndarray,
) -> None:
    if not len(work_keys):
        return
    # sample_ids are expanded from offsets, therefore equal work keys are
    # contiguous.  Segment once instead of comparing every row with every
    # work (which becomes quadratic on the full test split).
    boundaries = np.flatnonzero(
        np.r_[True, work_keys[1:] != work_keys[:-1], True]
    )
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        selected = mask[start:end]
        if selected.any():
            work_values[(model_name, target_name, str(work_keys[start]))].append(
                float(np.abs(error[start:end][selected]).mean())
            )


def _bootstrap(values: Iterable[float], *, seed: int, rounds: int = 1000) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"n_works": 0, "mean": None, "lower_95": None, "upper_95": None}
    rng = np.random.default_rng(seed)
    samples = array[rng.integers(0, len(array), size=(rounds, len(array)))].mean(axis=1)
    return {
        "n_works": int(len(array)),
        "mean": float(array.mean()),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
    }


def audit(
    *,
    dataset: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    split: str = "test",
    device: str = "auto",
    batch_size: int = 32768,
    sample_rate: float = 0.002,
    seed: int = 20261003,
    sample_id_file: str | Path | None = None,
    validation_cache: str | Path | None = None,
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise TrainingRunError("audit split must be validation or test")
    if batch_size <= 0 or not 0.0 <= sample_rate <= 1.0:
        raise TrainingRunError("batch_size must be positive and sample_rate must be in [0, 1]")
    dataset_root = Path(dataset).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    selected_sample_ids = None
    sample_id_path: Path | None = None
    if sample_id_file is not None:
        sample_id_path = Path(sample_id_file).resolve()
        selected_sample_ids = _load_sample_id_file(sample_id_path)
        _validate_sample_ids(dataset_root, selected_sample_ids, "audit sample ID file")
        # This also verifies that every requested ID belongs to the requested split.
        _selected_shard_paths(dataset_root, split, selected_sample_ids)
    validation_report = (
        _load_validation_cache(dataset_root, validation_cache)
        if validation_cache is not None
        else validate_dataset(dataset_root)
    )
    selected_device = _resolve_device(device)
    model, config = _load_model(checkpoint_path, selected_device)
    work_key_by_sample: dict[str, str] = {}
    with (dataset_root / "manifest.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["id"])
            source_name = str(row.get("performance_dataset") or "unknown")
            piece_group = str(row.get("piece_group") or sample_id)
            work_key_by_sample[sample_id] = f"{source_name}::{piece_group}"
    normalizer = json.loads((dataset_root / "normalizer.json").read_text(encoding="utf-8"))
    note_mean, note_scale = _normalizer_vectors(normalizer, NOTE_TARGET_NAMES, "note_targets")
    chord_mean, chord_scale = _normalizer_vectors(normalizer, CHORD_TARGET_NAMES, "chord_targets")
    relative_position = bool(config.get("relative_position", False))
    # The target and feature dimensions are independent; infer exact feature
    # lengths from the model when evaluating legacy checkpoints.
    note_input_mean = _array_config(config, "note_input_means", int(model.note_net[0].in_features), 0.0)
    note_input_scale = _array_config(config, "note_input_scales", int(model.note_net[0].in_features), 1.0)
    chord_input_mean = _array_config(config, "chord_input_means", int(model.chord_net[0].in_features), 0.0)
    chord_input_scale = _array_config(config, "chord_input_scales", int(model.chord_net[0].in_features), 1.0)
    target_names = {"note": NOTE_TARGET_NAMES, "chord": CHORD_TARGET_NAMES}
    normalizers = {"note": (note_mean, note_scale), "chord": (chord_mean, chord_scale)}
    models = {"model": model}
    global_stats: dict[tuple[str, str], _Stats] = {}
    source_stats: dict[tuple[str, str, str], _Stats] = {}
    work_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    rng = np.random.default_rng(seed)
    paths = _selected_shard_paths(dataset_root, split, selected_sample_ids)
    if paths is None:
        paths = _shard_paths(dataset_root, split)
    with torch.inference_mode():
        for path in paths:
            source = _source_name(path)
            with np.load(path, allow_pickle=False) as loaded:
                arrays = {name: loaded[name] for name in loaded.files}
            if selected_sample_ids is not None:
                arrays = _select_sample_arrays(arrays, selected_sample_ids)
                if arrays is None:
                    continue
            for kind in ("note", "chord"):
                names = target_names[kind]
                mean, scale = normalizers[kind]
                input_mean = note_input_mean if kind == "note" else chord_input_mean
                input_scale = note_input_scale if kind == "note" else chord_input_scale
                features = _prepare_features(
                    arrays,
                    kind=kind,
                    relative_position=relative_position,
                    means=input_mean,
                    scales=input_scale,
                )
                targets = arrays[f"{kind}_targets"]
                masks = arrays[f"{kind}_masks"].astype(bool, copy=False)
                work_keys, _ = _source_work_indices(
                    arrays, kind, work_key_by_sample
                )
                prediction_parts: list[np.ndarray] = []
                for start in range(0, len(features), batch_size):
                    tensor = torch.from_numpy(features[start : start + batch_size]).to(selected_device)
                    output_tensor = (model.forward_note(tensor) if kind == "note" else model.forward_chord(tensor)).cpu().numpy()
                    prediction_parts.append(output_tensor)
                prediction = np.concatenate(prediction_parts, axis=0) * scale + mean
                # Shards store raw targets; only the model output is in the
                # train-normalized target space.
                target = np.asarray(targets, dtype=np.float64)
                for index, name in enumerate(names):
                    valid = masks[:, index]
                    if not valid.any():
                        continue
                    pred_values = prediction[valid, index]
                    target_values = target[valid, index]
                    key = ("model", name)
                    global_stats.setdefault(key, _Stats(sample_rate=sample_rate, rng=rng)).add(pred_values, target_values)
                    source_key = ("model", name, source)
                    source_stats.setdefault(source_key, _Stats(sample_rate=0.0, rng=rng)).add(pred_values, target_values)
                    _add_work_values(
                        work_values,
                        model_name="model",
                        target_name=name,
                        work_keys=work_keys[valid],
                        error=pred_values - target_values,
                        mask=np.ones(len(pred_values), dtype=bool),
                    )
                    if name in BOUNDED_TARGETS:
                        global_stats[key].add_bounds(pred_values)
                        source_stats[source_key].add_bounds(pred_values)
                    if name in IDENTITY_TARGETS:
                        identity = np.zeros_like(target_values)
                        identity_key = ("identity", name)
                        global_stats.setdefault(identity_key, _Stats(sample_rate=sample_rate, rng=rng)).add(identity, target_values)
                        identity_source_key = ("identity", name, source)
                        source_stats.setdefault(identity_source_key, _Stats(sample_rate=0.0, rng=rng)).add(identity, target_values)
                        _add_work_values(
                            work_values,
                            model_name="identity",
                            target_name=name,
                            work_keys=work_keys[valid],
                            error=identity - target_values,
                            mask=np.ones(len(identity), dtype=bool),
                        )
                    if name not in BOUNDED_TARGETS and name not in IDENTITY_TARGETS:
                        # A train-fitted constant is the correct simple baseline.
                        mean_values = np.full_like(target_values, mean[index])
                        mean_key = ("mean", name)
                        global_stats.setdefault(mean_key, _Stats(sample_rate=sample_rate, rng=rng)).add(mean_values, target_values)
                        mean_source_key = ("mean", name, source)
                        source_stats.setdefault(mean_source_key, _Stats(sample_rate=0.0, rng=rng)).add(mean_values, target_values)
                        _add_work_values(
                            work_values,
                            model_name="mean",
                            target_name=name,
                            work_keys=work_keys[valid],
                            error=mean_values - target_values,
                            mask=np.ones(len(mean_values), dtype=bool),
                        )
    metrics = {f"{model_name}:{name}": stats.report() for (model_name, name), stats in sorted(global_stats.items())}
    source_metrics = {
        f"{model_name}:{name}:{source}": stats.report()
        for (model_name, name, source), stats in sorted(source_stats.items())
    }
    work_equal: dict[str, dict[str, float | int | None]] = {}
    for (model_name, name), _ in sorted(global_stats.items()):
        values = [np.mean(parts) for (m, target_name, _), parts in work_values.items() if m == model_name and target_name == name]
        work_equal[f"{model_name}:{name}"] = _bootstrap(values, seed=seed + len(work_equal))
    selected_work_values: dict[str, dict[str, float]] | None = None
    if selected_sample_ids is not None:
        selected_work_values = {}
        for (model_name, name, work_key), parts in sorted(work_values.items()):
            selected_work_values.setdefault(f"{model_name}:{name}", {})[
                str(work_key)
            ] = float(np.mean(parts))
    result = {
        "schema_version": 1,
        "status": "complete",
        "dataset": str(dataset_root),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "split": split,
        "device": str(selected_device),
        "sample_rate": sample_rate,
        "work_unit": "performance_dataset::piece_group",
        "dataset_validation": validation_report,
        "metrics": metrics,
        "source_metrics": source_metrics,
        "work_equal_bootstrap": work_equal,
    }
    if sample_id_path is not None:
        result["sample_id_file"] = str(sample_id_path)
        result["sample_id_file_sha256"] = _sha256(sample_id_path)
        result["sample_id_count"] = len(selected_sample_ids or ())
        result["work_values"] = selected_work_values or {}
    if validation_cache is not None:
        result["validation_cache"] = str(Path(validation_cache).resolve())
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--sample-rate", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument(
        "--sample-id-file",
        default=None,
        help="optional newline-delimited sample IDs; emits per-work values for this subset",
    )
    parser.add_argument(
        "--validation-cache",
        default=None,
        help="optional cache from a prior valid dataset validation",
    )
    args = parser.parse_args(argv)
    try:
        result = audit(
            dataset=args.dataset,
            checkpoint=args.checkpoint,
            output=args.output,
            split=args.split,
            device=args.device,
            batch_size=args.batch_size,
            sample_rate=args.sample_rate,
            seed=args.seed,
            sample_id_file=args.sample_id_file,
            validation_cache=args.validation_cache,
        )
    except (TrainingRunError, RuntimeError, ValueError, OSError) as exc:
        print(f"audit failed: {exc}")
        return 1
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit", "main"]
