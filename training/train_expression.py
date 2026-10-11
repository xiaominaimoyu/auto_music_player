"""Train the first offline expression baseline on validated PianoCoRe shards.

This is intentionally a small, auditable baseline rather than the final
sequence model.  It keeps note-level and chord-level heads separate, consumes
only train-fitted target statistics, respects every supervision mask, and
never touches the desktop playback path.  A run leaves ``_INCOMPLETE`` until
the last checkpoint, metrics, and integrity file have been written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise RuntimeError(
        "training.train_expression requires the separate PyTorch training environment"
    ) from exc

from training.midi_features import (
    CHORD_FEATURE_NAMES,
    CHORD_TARGET_NAMES,
    NOTE_FEATURE_NAMES,
    NOTE_TARGET_NAMES,
)
from training.quality_mask import (
    QualityMaskData,
    QualityMaskError,
    apply_quality_mask,
    load_quality_mask,
)
from training.validate_pianocore_dataset import validate_dataset


PIPELINE_NAME = "auto_music_player.expression_baseline"
PIPELINE_VERSION = 1
SHARD_ARRAY_NAMES = {
    "chord_features",
    "chord_masks",
    "chord_targets",
    "note_features",
    "note_masks",
    "note_targets",
}
DEFAULT_SEED = 20261003


class TrainingRunError(RuntimeError):
    """Raised when a training run cannot satisfy its artifact contract."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_canonical_json(value) + b"\n")


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _shard_paths(dataset_root: Path, split: str) -> list[Path]:
    paths = sorted(
        (
            path
            for path in dataset_root.glob(f"shards/split={split}/source=*/part-*.npz")
            if path.is_file()
        ),
        key=lambda path: path.relative_to(dataset_root).as_posix(),
    )
    if not paths:
        raise TrainingRunError(f"validated dataset has no {split!r} shards")
    return paths


def _iter_shard_arrays(
    dataset_root: Path,
    split: str,
    *,
    max_shards: int | None = None,
    shuffle_shards: bool = False,
    seed: int = 0,
    paths: Sequence[Path] | None = None,
) -> Iterator[tuple[Path, dict[str, np.ndarray]]]:
    selected_paths = list(paths) if paths is not None else _shard_paths(dataset_root, split)
    if shuffle_shards:
        np.random.default_rng(seed).shuffle(selected_paths)
    if max_shards is not None:
        selected_paths = selected_paths[:max_shards]
    for path in selected_paths:
        with np.load(path, allow_pickle=False) as loaded:
            if not SHARD_ARRAY_NAMES.issubset(set(loaded.files)):
                raise TrainingRunError(
                    f"{path.name}: shard is missing training arrays after validation"
                )
            arrays = {
                name: loaded[name]
                for name in loaded.files
                if name in SHARD_ARRAY_NAMES or name.startswith("sample_")
            }
        yield path, arrays


def _load_sample_id_file(path: str | os.PathLike[str]) -> frozenset[str]:
    sample_path = Path(path).resolve()
    if not sample_path.is_file():
        raise TrainingRunError(f"sample ID file does not exist: {sample_path}")
    values = frozenset(
        line.strip()
        for line in sample_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not values:
        raise TrainingRunError(f"sample ID file is empty: {sample_path}")
    return values


def _validate_sample_ids(dataset_root: Path, sample_ids: frozenset[str], label: str) -> None:
    remaining = set(sample_ids)
    with (dataset_root / "manifest.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not remaining:
                break
            if line.strip():
                remaining.discard(str(json.loads(line)["id"]))
    if remaining:
        raise TrainingRunError(
            f"{label} contains sample IDs absent from dataset: "
            + ", ".join(sorted(remaining)[:3])
        )


def _selected_shard_paths(
    dataset_root: Path, split: str, sample_ids: frozenset[str] | None
) -> list[Path] | None:
    if sample_ids is None:
        return None
    paths: set[Path] = set()
    remaining = set(sample_ids)
    with (dataset_root / "manifest.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") == split and str(row.get("id")) in sample_ids:
                paths.add(dataset_root / str(row["shard_path"]))
                remaining.discard(str(row["id"]))
    if remaining:
        raise TrainingRunError(
            f"sample IDs for {split} do not occur in that split: "
            + ", ".join(sorted(remaining)[:3])
        )
    return sorted(paths, key=lambda path: path.relative_to(dataset_root).as_posix())


def _load_validation_cache(dataset_root: Path, path: str | os.PathLike[str]) -> dict[str, Any]:
    cache_path = Path(path).resolve()
    try:
        cache = _load_json(cache_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingRunError(f"cannot read validation cache: {cache_path}") from exc
    if not isinstance(cache, dict):
        raise TrainingRunError("validation cache must be an object")
    report = cache.get("validation_report")
    if not isinstance(report, dict) or report.get("status") != "valid":
        raise TrainingRunError("validation cache does not contain a valid report")
    if Path(str(report.get("dataset_root"))).resolve() != dataset_root:
        raise TrainingRunError("validation cache dataset_root does not match dataset")
    success_path = dataset_root / "_SUCCESS"
    if cache.get("dataset_success_sha256") != _sha256(success_path):
        raise TrainingRunError("validation cache does not match dataset _SUCCESS")
    return report


def _select_sample_arrays(
    arrays: Mapping[str, np.ndarray], sample_ids: frozenset[str]
) -> dict[str, np.ndarray] | None:
    """Select complete ragged works from a shard for a diagnostic pilot."""

    stored_ids = np.asarray(arrays["sample_ids"])
    selected = [
        index
        for index, value in enumerate(stored_ids)
        if str(value) in sample_ids
    ]
    if not selected:
        return None
    result = dict(arrays)
    result["sample_ids"] = stored_ids[selected]
    selected_note_offsets = np.asarray(arrays["sample_note_offsets"], dtype=np.int64)
    selected_chord_offsets = np.asarray(arrays["sample_chord_offsets"], dtype=np.int64)
    note_lengths = [
        int(selected_note_offsets[index + 1] - selected_note_offsets[index])
        for index in selected
    ]
    chord_lengths = [
        int(selected_chord_offsets[index + 1] - selected_chord_offsets[index])
        for index in selected
    ]
    result["sample_note_offsets"] = np.concatenate(
        ([0], np.cumsum(note_lengths, dtype=np.int64))
    )
    result["sample_chord_offsets"] = np.concatenate(
        ([0], np.cumsum(chord_lengths, dtype=np.int64))
    )
    for name in ("note_features", "note_targets", "note_masks"):
        result[name] = np.concatenate(
            [
                arrays[name][
                    selected_note_offsets[index] : selected_note_offsets[index + 1]
                ]
                for index in selected
            ],
            axis=0,
        )
    for name in ("chord_features", "chord_targets", "chord_masks"):
        result[name] = np.concatenate(
            [
                arrays[name][
                    selected_chord_offsets[index] : selected_chord_offsets[index + 1]
                ]
                for index in selected
            ],
            axis=0,
        )
    if "note_to_chord" in arrays:
        note_parts: list[np.ndarray] = []
        chord_cursor = 0
        for index, chord_length in zip(selected, chord_lengths):
            note_start, note_end = (
                selected_note_offsets[index],
                selected_note_offsets[index + 1],
            )
            chord_start = selected_chord_offsets[index]
            note_parts.append(
                np.asarray(arrays["note_to_chord"][note_start:note_end], dtype=np.int64)
                - int(chord_start)
                + chord_cursor
            )
            chord_cursor += chord_length
        result["note_to_chord"] = np.concatenate(note_parts, axis=0)
    return result


def _relative_position_features(
    features: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    *,
    kind: str,
) -> np.ndarray:
    """Replace absolute score onset with a per-work [0, 1] position."""

    result = np.asarray(features, dtype=np.float32).copy()
    offset_name = "sample_note_offsets" if kind == "note" else "sample_chord_offsets"
    onset_index = 3 if kind == "note" else 0
    offsets = arrays.get(offset_name)
    if offsets is None:
        raise TrainingRunError(f"shard is missing {offset_name} for relative positions")
    for start, end in zip(offsets[:-1], offsets[1:]):
        start_i, end_i = int(start), int(end)
        if end_i <= start_i:
            continue
        values = result[start_i:end_i, onset_index]
        lo = float(values[0])
        hi = float(values[-1])
        span = hi - lo
        if span > 1e-6:
            result[start_i:end_i, onset_index] = (values - lo) / span
        else:
            result[start_i:end_i, onset_index] = 0.0
    return result


def _prepare_features(
    arrays: Mapping[str, np.ndarray],
    *,
    kind: str,
    relative_position: bool,
    means: np.ndarray | None,
    scales: np.ndarray | None,
) -> np.ndarray:
    if kind == "note":
        features = arrays["note_features"]
    elif kind == "chord":
        features = arrays["chord_features"]
    else:  # pragma: no cover - internal misuse guard
        raise ValueError(kind)
    result = (
        _relative_position_features(features, arrays, kind=kind)
        if relative_position
        else np.asarray(features, dtype=np.float32)
    )
    if means is not None and scales is not None:
        result = (result - means) / scales
    return np.asarray(result, dtype=np.float32)


def _feature_normalizer_vectors(
    dataset_root: Path,
    *,
    kind: str,
    relative_position: bool,
    max_shards: int | None = None,
    sample_ids: frozenset[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit input statistics on the train split only."""

    total = 0
    sums: np.ndarray | None = None
    squares: np.ndarray | None = None
    for _, arrays in _iter_shard_arrays(
        dataset_root,
        "train",
        max_shards=max_shards,
        paths=_selected_shard_paths(dataset_root, "train", sample_ids),
    ):
        if sample_ids is not None:
            selected_arrays = _select_sample_arrays(arrays, sample_ids)
            if selected_arrays is None:
                continue
            arrays = selected_arrays
        features = _prepare_features(
            arrays,
            kind=kind,
            relative_position=relative_position,
            means=None,
            scales=None,
        ).astype(np.float64, copy=False)
        if sums is None:
            sums = np.zeros(features.shape[1], dtype=np.float64)
            squares = np.zeros(features.shape[1], dtype=np.float64)
        sums += features.sum(axis=0)
        squares += np.square(features).sum(axis=0)
        total += len(features)
    if not total or sums is None or squares is None:
        raise TrainingRunError(f"cannot fit {kind} input normalizer on empty train split")
    mean = sums / total
    variance = np.maximum(squares / total - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _normalizer_vectors(
    normalizer: Mapping[str, Any], names: tuple[str, ...], group: str
) -> tuple[np.ndarray, np.ndarray]:
    values = normalizer.get(group)
    if not isinstance(values, dict):
        raise TrainingRunError(f"normalizer is missing {group}")
    means: list[float] = []
    scales: list[float] = []
    for name in names:
        entry = values.get(name)
        if not isinstance(entry, dict):
            raise TrainingRunError(f"normalizer is missing {group}.{name}")
        mean = float(entry.get("mean", 0.0))
        std = float(entry.get("std", 1.0))
        if not math.isfinite(mean) or not math.isfinite(std):
            raise TrainingRunError(f"normalizer {group}.{name} is non-finite")
        means.append(mean)
        scales.append(max(std, 1e-6))
    return np.asarray(means, dtype=np.float32), np.asarray(scales, dtype=np.float32)


def _batch_iterator(
    arrays: Mapping[str, np.ndarray],
    *,
    kind: str,
    batch_size: int,
    shuffle: bool,
    rng: np.random.Generator,
    relative_position: bool = False,
    feature_means: np.ndarray | None = None,
    feature_scales: np.ndarray | None = None,
    target_shuffle: bool = False,
) -> Iterator[tuple[Tensor, Tensor, Tensor, Tensor]]:
    if kind == "note":
        features = _prepare_features(
            arrays,
            kind=kind,
            relative_position=relative_position,
            means=feature_means,
            scales=feature_scales,
        )
        targets = arrays["note_targets"]
        masks = arrays["note_masks"]
        quality_weights = arrays.get(
            "note_quality_weights", np.ones(masks.shape, dtype=np.float32)
        )
    elif kind == "chord":
        features = _prepare_features(
            arrays,
            kind=kind,
            relative_position=relative_position,
            means=feature_means,
            scales=feature_scales,
        )
        targets = arrays["chord_targets"]
        masks = arrays["chord_masks"]
        quality_weights = arrays.get(
            "chord_quality_weights", np.ones(masks.shape, dtype=np.float32)
        )
    else:  # pragma: no cover - internal misuse guard
        raise ValueError(kind)
    indices = np.arange(len(features), dtype=np.int64)
    if shuffle:
        rng.shuffle(indices)
    target_indices = np.arange(len(features), dtype=np.int64)
    if target_shuffle:
        rng.shuffle(target_indices)
    for start in range(0, len(indices), batch_size):
        index = indices[start : start + batch_size]
        target_index = target_indices[index] if target_shuffle else index
        yield (
            torch.from_numpy(np.asarray(features[index], dtype=np.float32)),
            torch.from_numpy(np.asarray(targets[target_index], dtype=np.float32)),
            torch.from_numpy(np.asarray(masks[target_index], dtype=np.bool_)),
            torch.from_numpy(
                np.asarray(quality_weights[target_index], dtype=np.float32)
            ),
        )


class ExpressionMLP(nn.Module):
    """Compact baseline with independent note and chord expression heads."""

    def __init__(
        self,
        note_feature_dim: int = len(NOTE_FEATURE_NAMES),
        note_target_dim: int = len(NOTE_TARGET_NAMES),
        chord_feature_dim: int = len(CHORD_FEATURE_NAMES),
        chord_target_dim: int = len(CHORD_TARGET_NAMES),
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or not 0.0 <= dropout < 1.0:
            raise ValueError("hidden_dim must be positive and dropout must be in [0, 1)")
        self.note_net = self._make_head(note_feature_dim, note_target_dim, hidden_dim, dropout)
        self.chord_net = self._make_head(
            chord_feature_dim, chord_target_dim, hidden_dim, dropout
        )

    @staticmethod
    def _make_head(
        feature_dim: int, target_dim: int, hidden_dim: int, dropout: float
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward_note(self, features: Tensor) -> Tensor:
        return self.note_net(features)

    def forward_chord(self, features: Tensor) -> Tensor:
        return self.chord_net(features)


def _masked_huber(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    weights: Tensor,
) -> tuple[Tensor, int]:
    per_value = F.smooth_l1_loss(prediction, target, reduction="none", beta=1.0)
    weighted_mask = mask.to(dtype=per_value.dtype) * weights
    denominator = weighted_mask.sum().clamp_min(1.0)
    return (per_value * weighted_mask).sum() / denominator, int(mask.sum().item())


@dataclass
class MetricAccumulator:
    loss_sum: float = 0.0
    loss_weight: float = 0.0
    abs_sum: dict[str, float] = field(default_factory=dict)
    sq_sum: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    samples: dict[str, list[float]] = field(default_factory=dict)
    sample_limit: int = 200_000

    def add_loss(self, loss: Tensor, weight: float) -> None:
        self.loss_sum += float(loss.detach().item()) * max(1, weight)
        self.loss_weight += max(1, weight)

    def add_targets(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor,
        names: tuple[str, ...],
        means: Tensor,
        scales: Tensor,
    ) -> None:
        prediction = prediction.detach() * scales + means
        target = target.detach() * scales + means
        error = (prediction - target).abs().cpu().numpy()
        mask_np = mask.detach().cpu().numpy().astype(bool)
        for index, name in enumerate(names):
            values = error[mask_np[:, index], index].astype(np.float64, copy=False)
            if not len(values):
                continue
            self.abs_sum[name] = self.abs_sum.get(name, 0.0) + float(values.sum())
            self.sq_sum[name] = self.sq_sum.get(name, 0.0) + float(
                np.square(values).sum()
            )
            self.counts[name] = self.counts.get(name, 0) + int(len(values))
            sample_values = self.samples.setdefault(name, [])
            remaining = self.sample_limit - len(sample_values)
            if remaining > 0:
                sample_values.extend(values[:remaining].tolist())

    def report(self, prefix: str = "") -> dict[str, float | int]:
        result: dict[str, float | int] = {
            f"{prefix}loss": self.loss_sum / max(1.0, self.loss_weight),
        }
        for name in sorted(self.counts):
            count = self.counts[name]
            values = np.asarray(self.samples.get(name, ()), dtype=np.float64)
            result[f"{prefix}{name}_mae"] = self.abs_sum[name] / count
            result[f"{prefix}{name}_rmse"] = math.sqrt(self.sq_sum[name] / count)
            result[f"{prefix}{name}_count"] = count
            if len(values):
                result[f"{prefix}{name}_p95"] = float(np.quantile(values, 0.95))
        return result


def _run_epoch(
    model: ExpressionMLP,
    dataset_root: Path,
    split: str,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    batch_size: int,
    note_means: Tensor,
    note_scales: Tensor,
    chord_means: Tensor,
    chord_scales: Tensor,
    note_weights: Tensor,
    chord_weights: Tensor,
    seed: int,
    max_shards: int | None = None,
    max_steps: int | None = None,
    relative_position: bool = False,
    note_input_means: np.ndarray | None = None,
    note_input_scales: np.ndarray | None = None,
    chord_input_means: np.ndarray | None = None,
    chord_input_scales: np.ndarray | None = None,
    shuffle_shards: bool = False,
    target_shuffle: bool = False,
    quality_mask: Mapping[str, Mapping[str, Any]] | None = None,
    quality_mask_splits: frozenset[str] = frozenset(),
    selected_sample_ids: frozenset[str] | None = None,
) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    note_metrics = MetricAccumulator()
    chord_metrics = MetricAccumulator()
    quality_weight_sum = 0.0
    quality_affected_target_count = 0
    rng = np.random.default_rng(seed)
    step_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for _, arrays in _iter_shard_arrays(
            dataset_root,
            split,
            max_shards=max_shards,
            shuffle_shards=training and shuffle_shards,
            seed=seed,
            paths=_selected_shard_paths(dataset_root, split, selected_sample_ids),
        ):
            if selected_sample_ids is not None:
                selected_arrays = _select_sample_arrays(arrays, selected_sample_ids)
                if selected_arrays is None:
                    continue
                arrays = selected_arrays
            if quality_mask is not None and split in quality_mask_splits:
                arrays = dict(arrays)
                arrays["note_masks"], arrays["note_quality_weights"] = apply_quality_mask(
                    arrays, kind="note", decisions=quality_mask
                )
                arrays["chord_masks"], arrays["chord_quality_weights"] = apply_quality_mask(
                    arrays, kind="chord", decisions=quality_mask
                )
            for kind in ("note", "chord"):
                iterator = _batch_iterator(
                    arrays,
                    kind=kind,
                    batch_size=batch_size,
                    shuffle=training,
                    rng=rng,
                    relative_position=relative_position,
                    feature_means=(
                        note_input_means if kind == "note" else chord_input_means
                    ),
                    feature_scales=(
                        note_input_scales if kind == "note" else chord_input_scales
                    ),
                    target_shuffle=training and target_shuffle,
                )
                for features, targets, masks, quality_weights in iterator:
                    if max_steps is not None and step_count >= max_steps:
                        break
                    features = features.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    quality_weights = quality_weights.to(device, non_blocking=True)
                    if kind == "note":
                        means, scales, weights = note_means, note_scales, note_weights
                        prediction = model.forward_note(features)
                        metrics = note_metrics
                        names = NOTE_TARGET_NAMES
                    else:
                        means, scales, weights = chord_means, chord_scales, chord_weights
                        prediction = model.forward_chord(features)
                        metrics = chord_metrics
                        names = CHORD_TARGET_NAMES
                    normalized_target = (targets - means) / scales
                    loss, valid_count = _masked_huber(
                        prediction,
                        normalized_target,
                        masks,
                        weights * quality_weights,
                    )
                    valid_quality_weights = quality_weights[masks]
                    weighted_count = float(valid_quality_weights.sum().item())
                    quality_weight_sum += weighted_count
                    quality_affected_target_count += int(
                        (valid_quality_weights < 1.0).sum().item()
                    )
                    if training:
                        assert optimizer is not None
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    metrics.add_loss(loss, weighted_count)
                    metrics.add_targets(
                        prediction,
                        normalized_target,
                        masks,
                        names,
                        means,
                        scales,
                    )
                    step_count += 1
                if max_steps is not None and step_count >= max_steps:
                    break
            if max_steps is not None and step_count >= max_steps:
                break
    result = {"steps": step_count}
    result.update(note_metrics.report("note_"))
    result.update(chord_metrics.report("chord_"))
    note_weight = note_metrics.loss_weight
    chord_weight = chord_metrics.loss_weight
    total_weight = note_weight + chord_weight
    result["loss"] = (
        note_metrics.loss_sum + chord_metrics.loss_sum
    ) / max(1.0, total_weight)
    result["valid_target_count"] = sum(note_metrics.counts.values()) + sum(
        chord_metrics.counts.values()
    )
    result["quality_weight_sum"] = quality_weight_sum
    result["quality_affected_target_count"] = quality_affected_target_count
    return result


def _resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise TrainingRunError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _save_checkpoint(
    path: Path,
    model: ExpressionMLP,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    run_config: Mapping[str, Any],
    train_metrics: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
) -> None:
    torch.save(
        {
            "pipeline": PIPELINE_NAME,
            "pipeline_version": PIPELINE_VERSION,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "run_config": dict(run_config),
            "train_metrics": dict(train_metrics),
            "validation_metrics": dict(validation_metrics),
        },
        path,
    )


def _write_integrity(output_root: Path) -> None:
    files = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name not in {"_INCOMPLETE", "_SUCCESS"}
    )
    inventory = {
        "pipeline": PIPELINE_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "files": [
            {
                "path": path.relative_to(output_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ],
    }
    _write_json(output_root / "integrity.json", inventory)
    digest = _sha256(output_root / "integrity.json")
    (output_root / "_SUCCESS").write_text(digest + "\n", encoding="ascii")


def train(
    *,
    dataset: str | os.PathLike[str],
    output: str | os.PathLike[str],
    device: str = "auto",
    epochs: int = 20,
    batch_size: int = 2048,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 128,
    dropout: float = 0.0,
    onset_weight: float = 0.1,
    duration_weight: float = 1.0,
    velocity_weight: float = 1.0,
    pedal_weight: float = 1.0,
    ioi_weight: float = 1.0,
    input_normalize: bool = False,
    relative_position: bool = False,
    shuffle_shards: bool = False,
    target_shuffle: bool = False,
    seed: int = DEFAULT_SEED,
    patience: int = 8,
    max_shards: int | None = None,
    max_steps_per_epoch: int | None = None,
    quality_mask: str | os.PathLike[str] | None = None,
    train_sample_id_file: str | os.PathLike[str] | None = None,
    validation_sample_id_file: str | os.PathLike[str] | None = None,
    validation_cache: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    if epochs <= 0 or batch_size <= 0 or hidden_dim <= 0 or patience <= 0:
        raise TrainingRunError("epochs, batch_size, hidden_dim, and patience must be positive")
    if (
        learning_rate <= 0
        or weight_decay < 0
        or onset_weight <= 0
        or duration_weight <= 0
        or velocity_weight <= 0
        or pedal_weight <= 0
        or ioi_weight <= 0
    ):
        raise TrainingRunError(
            "learning_rate and target weights must be positive; weight_decay non-negative"
        )
    dataset_root = Path(dataset).resolve()
    train_sample_ids: frozenset[str] | None = None
    train_sample_id_file_path: Path | None = None
    if train_sample_id_file is not None:
        train_sample_id_file_path = Path(train_sample_id_file).resolve()
        train_sample_ids = _load_sample_id_file(train_sample_id_file_path)
    validation_sample_ids: frozenset[str] | None = None
    validation_sample_id_file_path: Path | None = None
    if validation_sample_id_file is not None:
        validation_sample_id_file_path = Path(validation_sample_id_file).resolve()
        validation_sample_ids = _load_sample_id_file(validation_sample_id_file_path)
    quality_mask_data: QualityMaskData | None = None
    if quality_mask is not None:
        try:
            quality_mask_data = load_quality_mask(quality_mask)
        except QualityMaskError as exc:
            raise TrainingRunError(str(exc)) from exc
        _validate_sample_ids(dataset_root, frozenset(quality_mask_data.decisions), "quality mask")
    if train_sample_ids is not None:
        _validate_sample_ids(dataset_root, train_sample_ids, "train sample ID file")
    if validation_sample_ids is not None:
        _validate_sample_ids(dataset_root, validation_sample_ids, "validation sample ID file")
    output_root = Path(output).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise TrainingRunError(f"training output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    incomplete = output_root / "_INCOMPLETE"
    incomplete.write_text("training run in progress\n", encoding="utf-8")

    validation_report = (
        _load_validation_cache(dataset_root, validation_cache)
        if validation_cache is not None
        else validate_dataset(dataset_root)
    )
    normalizer = _load_json(dataset_root / "normalizer.json")
    note_mean_np, note_scale_np = _normalizer_vectors(
        normalizer, NOTE_TARGET_NAMES, "note_targets"
    )
    chord_mean_np, chord_scale_np = _normalizer_vectors(
        normalizer, CHORD_TARGET_NAMES, "chord_targets"
    )
    if input_normalize:
        note_input_mean_np, note_input_scale_np = _feature_normalizer_vectors(
            dataset_root,
            kind="note",
            relative_position=relative_position,
            max_shards=max_shards,
            sample_ids=train_sample_ids,
        )
        chord_input_mean_np, chord_input_scale_np = _feature_normalizer_vectors(
            dataset_root,
            kind="chord",
            relative_position=relative_position,
            max_shards=max_shards,
            sample_ids=train_sample_ids,
        )
    else:
        note_input_mean_np = np.zeros(len(NOTE_FEATURE_NAMES), dtype=np.float32)
        note_input_scale_np = np.ones(len(NOTE_FEATURE_NAMES), dtype=np.float32)
        chord_input_mean_np = np.zeros(len(CHORD_FEATURE_NAMES), dtype=np.float32)
        chord_input_scale_np = np.ones(len(CHORD_FEATURE_NAMES), dtype=np.float32)
    selected_device = _resolve_device(device)
    _seed_everything(seed)
    model = ExpressionMLP(hidden_dim=hidden_dim, dropout=dropout).to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    note_means = torch.from_numpy(note_mean_np).to(selected_device)
    note_scales = torch.from_numpy(note_scale_np).to(selected_device)
    chord_means = torch.from_numpy(chord_mean_np).to(selected_device)
    chord_scales = torch.from_numpy(chord_scale_np).to(selected_device)
    note_weights = torch.tensor(
        [1.0, duration_weight, velocity_weight, velocity_weight, pedal_weight, pedal_weight],
        dtype=torch.float32,
        device=selected_device,
    )
    chord_weights = torch.tensor(
        [onset_weight, ioi_weight], dtype=torch.float32, device=selected_device
    )

    run_config: dict[str, Any] = {
        "pipeline": PIPELINE_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "dataset_root": str(dataset_root),
        "dataset_validation": validation_report,
        "device": str(selected_device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": (
            torch.cuda.get_device_name(selected_device)
            if selected_device.type == "cuda"
            else None
        ),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "onset_weight": onset_weight,
        "duration_weight": duration_weight,
        "velocity_weight": velocity_weight,
        "pedal_weight": pedal_weight,
        "ioi_weight": ioi_weight,
        "input_normalize": input_normalize,
        "relative_position": relative_position,
        "shuffle_shards": shuffle_shards,
        "target_shuffle": target_shuffle,
        "note_input_means": note_input_mean_np.tolist(),
        "note_input_scales": note_input_scale_np.tolist(),
        "chord_input_means": chord_input_mean_np.tolist(),
        "chord_input_scales": chord_input_scale_np.tolist(),
        "seed": seed,
        "patience": patience,
        "max_shards": max_shards,
        "max_steps_per_epoch": max_steps_per_epoch,
        "quality_mask_path": quality_mask_data.path if quality_mask_data else None,
        "quality_mask_sha256": quality_mask_data.sha256 if quality_mask_data else None,
        "quality_mask_policy_version": (
            quality_mask_data.policy_version if quality_mask_data else None
        ),
        "quality_mask_row_count": quality_mask_data.row_count if quality_mask_data else 0,
        "quality_mask_reviewed_count": (
            quality_mask_data.reviewed_count if quality_mask_data else 0
        ),
        "quality_mask_applied_splits": ["train"] if quality_mask_data else [],
        "train_sample_id_file": (
            str(train_sample_id_file_path) if train_sample_id_file_path else None
        ),
        "train_sample_id_file_sha256": (
            _sha256(train_sample_id_file_path)
            if train_sample_id_file_path is not None
            else None
        ),
        "train_sample_id_count": len(train_sample_ids) if train_sample_ids else 0,
        "validation_sample_id_file": (
            str(validation_sample_id_file_path)
            if validation_sample_id_file_path
            else None
        ),
        "validation_sample_id_file_sha256": (
            _sha256(validation_sample_id_file_path)
            if validation_sample_id_file_path is not None
            else None
        ),
        "validation_sample_id_count": (
            len(validation_sample_ids) if validation_sample_ids else 0
        ),
        "dataset_validation_cache": (
            str(Path(validation_cache).resolve()) if validation_cache is not None else None
        ),
        "dataset_validation_cache_sha256": (
            _sha256(Path(validation_cache).resolve())
            if validation_cache is not None
            else None
        ),
        "note_features": list(NOTE_FEATURE_NAMES),
        "note_targets": list(NOTE_TARGET_NAMES),
        "chord_features": list(CHORD_FEATURE_NAMES),
        "chord_targets": list(CHORD_TARGET_NAMES),
    }
    _write_json(output_root / "run_config.json", run_config)

    metrics_path = output_root / "metrics.jsonl"
    best_validation = math.inf
    best_epoch = 0
    stale_epochs = 0
    try:
        with metrics_path.open("w", encoding="utf-8") as metrics_handle:
            for epoch in range(1, epochs + 1):
                train_metrics = _run_epoch(
                    model,
                    dataset_root,
                    "train",
                    optimizer=optimizer,
                    device=selected_device,
                    batch_size=batch_size,
                    note_means=note_means,
                    note_scales=note_scales,
                    chord_means=chord_means,
                    chord_scales=chord_scales,
                    note_weights=note_weights,
                    chord_weights=chord_weights,
                    seed=seed + epoch,
                    max_shards=max_shards,
                    max_steps=max_steps_per_epoch,
                    relative_position=relative_position,
                    note_input_means=note_input_mean_np,
                    note_input_scales=note_input_scale_np,
                    chord_input_means=chord_input_mean_np,
                    chord_input_scales=chord_input_scale_np,
                    shuffle_shards=shuffle_shards,
                    target_shuffle=target_shuffle,
                    quality_mask=(quality_mask_data.decisions if quality_mask_data else None),
                    quality_mask_splits=frozenset({"train"}),
                    selected_sample_ids=train_sample_ids,
                )
                validation_metrics = _run_epoch(
                    model,
                    dataset_root,
                    "validation",
                    optimizer=None,
                    device=selected_device,
                    batch_size=batch_size,
                    note_means=note_means,
                    note_scales=note_scales,
                    chord_means=chord_means,
                    chord_scales=chord_scales,
                    note_weights=note_weights,
                    chord_weights=chord_weights,
                    seed=seed,
                    max_shards=max_shards,
                    max_steps=max_steps_per_epoch,
                    relative_position=relative_position,
                    note_input_means=note_input_mean_np,
                    note_input_scales=note_input_scale_np,
                    chord_input_means=chord_input_mean_np,
                    chord_input_scales=chord_input_scale_np,
                    quality_mask=None,
                    selected_sample_ids=validation_sample_ids,
                )
                record = {
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": validation_metrics,
                }
                metrics_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                metrics_handle.flush()
                print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
                _save_checkpoint(
                    output_root / "checkpoint_last.pt",
                    model,
                    optimizer,
                    epoch,
                    run_config,
                    train_metrics,
                    validation_metrics,
                )
                validation_loss = float(validation_metrics["loss"])
                if validation_loss < best_validation:
                    best_validation = validation_loss
                    best_epoch = epoch
                    stale_epochs = 0
                    _save_checkpoint(
                        output_root / "checkpoint_best.pt",
                        model,
                        optimizer,
                        epoch,
                        run_config,
                        train_metrics,
                        validation_metrics,
                    )
                else:
                    stale_epochs += 1
                    if stale_epochs >= patience:
                        break
        run_config["best_epoch"] = best_epoch
        run_config["best_validation_loss"] = best_validation
        _write_json(output_root / "run_config.json", run_config)
        _write_integrity(output_root)
        incomplete.unlink(missing_ok=True)
        return {
            "status": "complete",
            "output": str(output_root),
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation,
            "device": str(selected_device),
        }
    except Exception:
        # Keep the marker and partial checkpoints for diagnosis/resume tooling.
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the offline expression MLP on a validated PianoCoRe dataset."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--onset-weight", type=float, default=0.1)
    parser.add_argument("--duration-weight", type=float, default=1.0)
    parser.add_argument("--velocity-weight", type=float, default=1.0)
    parser.add_argument("--pedal-weight", type=float, default=1.0)
    parser.add_argument("--ioi-weight", type=float, default=1.0)
    parser.add_argument("--input-normalize", action="store_true")
    parser.add_argument("--relative-position", action="store_true")
    parser.add_argument("--shuffle-shards", action="store_true")
    parser.add_argument(
        "--target-shuffle",
        action="store_true",
        help="shuffle train targets as a negative-control experiment",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument(
        "--quality-mask",
        default=None,
        help="JSONL quality sidecar; decisions are applied to train supervision only",
    )
    parser.add_argument(
        "--train-sample-id-file",
        default=None,
        help="optional newline-delimited sample IDs for a diagnostic train subset",
    )
    parser.add_argument(
        "--validation-sample-id-file",
        default=None,
        help="optional newline-delimited sample IDs for a diagnostic validation subset",
    )
    parser.add_argument(
        "--validation-cache",
        default=None,
        help="optional cache from a prior valid dataset validation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = train(
            dataset=args.dataset,
            output=args.output,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            onset_weight=args.onset_weight,
            duration_weight=args.duration_weight,
            velocity_weight=args.velocity_weight,
            pedal_weight=args.pedal_weight,
            ioi_weight=args.ioi_weight,
            input_normalize=args.input_normalize,
            relative_position=args.relative_position,
            shuffle_shards=args.shuffle_shards,
            target_shuffle=args.target_shuffle,
            seed=args.seed,
            patience=args.patience,
            max_shards=args.max_shards,
            max_steps_per_epoch=args.max_steps_per_epoch,
            quality_mask=args.quality_mask,
            train_sample_id_file=args.train_sample_id_file,
            validation_sample_id_file=args.validation_sample_id_file,
            validation_cache=args.validation_cache,
        )
    except (TrainingRunError, RuntimeError, ValueError, OSError) as exc:
        print(f"training failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ExpressionMLP", "TrainingRunError", "main", "train"]
