"""Train locked-sample offline sequence pilots without touching playback."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from training.bar_objective import true_bar_objective
from training.bar_supervision import DEFAULT_BAR_HORIZONS, validate_horizons
from training.bar_windows import DEFAULT_MAX_WINDOW_CHORDS
from training.midi_features import CHORD_TARGET_NAMES, NOTE_TARGET_NAMES
from training.piece_groups import group_membership, score_groups
from training.rhythm_objective import DEFAULT_CUMULATIVE_SPANS, cumulative_ioi_loss, validate_spans
from training.shape_features import SHAPE_FEATURE_NAMES
from training.shared_profile import (DURATION_COLUMN as SHARED_DURATION_COLUMN, apply_shared_targets,
                                     profile_coverage, shared_duration_targets)
from training.sequence_expression import (
    BAR_WEIGHT_KEYS, BAR_WEIGHT_TO_COMPONENT, SequenceExpressionModel, SequenceSample, attach_bar_geometry,
    bar_weight_total, chord_input_names, collate_windows, fit_input_normalizers, load_selected_samples,
    make_model, make_windows, make_windows_from_config, note_input_names, shuffle_training_targets, to_device,
)
from training.train_expression import (
    TrainingRunError, _load_json, _load_sample_id_file, _load_validation_cache,
    _normalizer_vectors, _resolve_device, _seed_everything, _sha256, _write_json,
)
from training.validate_pianocore_dataset import validate_dataset


PIPELINE_NAME = "auto_music_player.expression_sequence_pilot"
PIPELINE_VERSION = 1


def load_reference(path: str | Path | None) -> Mapping[str, Mapping[str, Any]] | None:
    if path is None:
        return None
    from training.tempo_reference import load_timing_reference
    return load_timing_reference(path)


def reference_hash(path: str | Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path).resolve()
    if resolved.is_file():
        return _sha256(resolved)
    for name in ("_SUCCESS", "summary.json", "manifest.jsonl"):
        candidate = resolved / name
        if candidate.is_file():
            return _sha256(candidate)
    raise TrainingRunError("timing reference has no hashable artifact contract")


def _run_epoch(model: SequenceExpressionModel, samples: Sequence[SequenceSample], config: Mapping[str, Any], *,
               device: torch.device, optimizer: torch.optim.Optimizer | None, seed: int,
               max_steps: int | None = None) -> dict[str, Any]:
    training = optimizer is not None
    cumulative_weight = float(config.get("cumulative_weight", 0.0))
    if cumulative_weight > 0 and config.get("target_shuffle", False):
        raise TrainingRunError("target shuffle with cumulative supervision is unsupported")
    cumulative_spans = validate_spans(config.get("cumulative_spans", DEFAULT_CUMULATIVE_SPANS))
    bar_weights = {name: float(config.get(name, 0.0)) for name in BAR_WEIGHT_KEYS}
    bar_weight = sum(bar_weights.values())
    if bar_weight > 0 and config.get("target_shuffle", False):
        raise TrainingRunError("target shuffle with bar supervision is unsupported")
    bar_horizons = validate_horizons(config.get("bar_horizons", DEFAULT_BAR_HORIZONS))
    model.train(training)
    rng = np.random.default_rng(seed)
    windows = []
    for sample in samples:
        if training and config.get("target_shuffle", False):
            sample = shuffle_training_targets(sample, rng)
        windows.extend(make_windows_from_config(sample, config))
    if training:
        rng.shuffle(windows)
    names = {"note": NOTE_TARGET_NAMES, "chord": CHORD_TARGET_NAMES}
    means = {kind: torch.tensor(config[f"{kind}_target_means"], device=device) for kind in names}
    scales = {kind: torch.tensor(config[f"{kind}_target_scales"], device=device) for kind in names}
    weights = {
        "note": torch.tensor([float(config.get("micro_weight", 1.0)), float(config["duration_weight"]), float(config["velocity_weight"]),
                               float(config["velocity_weight"]), float(config["pedal_weight"]),
                               float(config["pedal_weight"])], device=device),
        "chord": torch.tensor([float(config["onset_weight"]), float(config["ioi_weight"])], device=device),
    }
    totals = {kind: np.zeros(len(target_names), dtype=np.float64) for kind, target_names in names.items()}
    counts = {kind: np.zeros(len(target_names), dtype=np.int64) for kind, target_names in names.items()}
    loss_sum, loss_weight, steps = 0.0, 0.0, 0
    point_loss_sum, point_weight_sum = 0.0, 0.0
    cumulative_loss_sum, cumulative_steps = 0.0, 0
    cumulative_span_counts = {str(span): 0 for span in cumulative_spans}
    cumulative_span_errors = {str(span): 0.0 for span in cumulative_spans}
    cumulative_clipped_count = 0
    bar_loss_sum, bar_steps = 0.0, 0
    bar_component_sums = {name: 0.0 for name in BAR_WEIGHT_KEYS}
    bar_horizon_counts = {str(horizon): {"candidate_spans": 0, "eligible_spans": 0, "interval_rows": 0,
                                         "clipped_rows": 0} for horizon in bar_horizons}
    bar_local_rows_excluded = 0
    bar_spans_without_local_rows = 0
    batch_size = int(config["batch_size"])
    with torch.set_grad_enabled(training):
        for start in range(0, len(windows), batch_size):
            if max_steps is not None and steps >= max_steps:
                break
            batch = to_device(collate_windows(windows[start:start + batch_size], config), device)
            predictions = model(batch)
            numerator = torch.zeros((), device=device)
            denominator = torch.zeros((), device=device)
            for kind, prediction in zip(("note", "chord"), predictions):
                target = batch[f"{kind}_targets"]
                mask = batch[f"{kind}_masks"]
                normalized = (target - means[kind]) / scales[kind]
                per_target = F.smooth_l1_loss(prediction, normalized, reduction="none")
                weighted_mask = mask.to(prediction.dtype) * weights[kind]
                numerator = numerator + (per_target * weighted_mask).sum()
                denominator = denominator + weighted_mask.sum()
                raw_error = torch.abs(prediction.detach() * scales[kind] + means[kind] - target)
                dims = tuple(range(mask.ndim - 1))
                totals[kind] += (raw_error * mask).sum(dim=dims).cpu().numpy()
                counts[kind] += mask.sum(dim=dims).cpu().numpy()
            weight = float(denominator.detach().item())
            cumulative = torch.zeros((), device=device)
            cumulative_count = 0
            if cumulative_weight > 0:
                chord_raw = predictions[1] * scales["chord"] + means["chord"]
                cumulative, cumulative_metrics = cumulative_ioi_loss(
                    chord_raw[..., 1], batch["chord_targets"][..., 1], batch["score_ioi_seconds"],
                    batch["cumulative_ioi_mask"], spans=cumulative_spans,
                    log_ratio_clip=float(config.get("cumulative_log_ratio_clip", 4.0)),
                )
                cumulative_count = cumulative_metrics["valid_window_count"]
                cumulative_clipped_count += cumulative_metrics["prediction_clipped_count"]
                for key, count in cumulative_metrics["span_counts"].items():
                    cumulative_span_counts[key] += count
                    if count:
                        cumulative_span_errors[key] += cumulative_metrics["span_relative_mae"][key] * count
            bar_total = torch.zeros((), device=device)
            bar_eligible = 0
            components: dict[str, Tensor] = {}
            if bar_weight > 0:
                chord_raw = predictions[1] * scales["chord"] + means["chord"]
                components, bar_metrics = true_bar_objective(
                    chord_raw[..., 1], batch["chord_targets"][..., 1], batch["score_ioi_seconds"],
                    batch["bar_span_rows"], batch["bar_span_mask"], batch["bar_span_horizons"],
                    horizons=bar_horizons, log_ratio_clip=float(config.get("bar_log_ratio_clip", 4.0)),
                    minimum_elapsed_s=float(config.get("bar_minimum_elapsed_s", 1e-6)),
                    bar_ioi_mask=batch["bar_ioi_mask"],
                )
                bar_total = sum(bar_weights[name] * components[BAR_WEIGHT_TO_COMPONENT[name]]
                                for name in BAR_WEIGHT_KEYS)
                bar_eligible = int(bar_metrics["valid_span_count"])
                bar_local_rows_excluded += int(bar_metrics["local_rows_excluded_by_mask_count"])
                bar_spans_without_local_rows += int(bar_metrics["spans_without_local_row_count"])
                for horizon in bar_horizons:
                    entry = bar_horizon_counts[str(horizon)]
                    selected = batch["bar_span_horizons"] == horizon
                    entry["candidate_spans"] += int((selected & batch["bar_span_candidates"]).sum().item())
                    entry["eligible_spans"] += int((selected & batch["bar_span_mask"]).sum().item())
                    reported = bar_metrics["horizons"][str(horizon)]
                    entry["interval_rows"] += int(reported["local_interval_count"])
                    entry["clipped_rows"] += int(reported["prediction_clipped_row_count"])
            if weight <= 0 and not cumulative_count and not bar_eligible:
                continue
            point_loss = numerator / denominator if weight > 0 else numerator * 0.0
            loss = point_loss + cumulative_weight * cumulative + bar_total
            if not torch.isfinite(loss):
                raise TrainingRunError("non-finite sequence training loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            # Historical point-only runs retain exactly the old weighting.
            # Supervision-only batches use their eligible window/span count instead.
            optimization_weight = weight if weight > 0 else float(cumulative_count + bar_eligible)
            loss_sum += float(loss.detach().item()) * optimization_weight
            loss_weight += optimization_weight
            point_loss_sum += float(numerator.detach().item())
            point_weight_sum += weight
            if cumulative_count:
                cumulative_loss_sum += float(cumulative.detach().item())
                cumulative_steps += 1
            if bar_eligible:
                bar_loss_sum += float(bar_total.detach().item())
                bar_steps += 1
                for name in BAR_WEIGHT_KEYS:
                    bar_component_sums[name] += float(components[BAR_WEIGHT_TO_COMPONENT[name]].detach().item())
            steps += 1
    if not steps:
        raise TrainingRunError("sequence epoch has no supervised batches")
    return {
        "loss": loss_sum / loss_weight, "steps": steps, "weighted_target_count": point_weight_sum,
        "point_loss": point_loss_sum / point_weight_sum if point_weight_sum else None,
        "cumulative_loss": cumulative_loss_sum / cumulative_steps if cumulative_steps else None,
        "cumulative_span_counts": cumulative_span_counts,
        "cumulative_span_relative_mae": {
            key: cumulative_span_errors[key] / count if count else None
            for key, count in cumulative_span_counts.items()
        },
        "cumulative_prediction_clipped_count": cumulative_clipped_count,
        "window_policy": str(config.get("window_policy", "fixed")),
        "bar_loss": bar_loss_sum / bar_steps if bar_steps else None,
        "bar_component_losses": {name: bar_component_sums[name] / bar_steps if bar_steps else None
                                 for name in BAR_WEIGHT_KEYS},
        "bar_supervised_steps": bar_steps,
        "bar_horizon_counts": bar_horizon_counts,
        "bar_local_rows_excluded_count": bar_local_rows_excluded,
        "bar_spans_without_local_rows_count": bar_spans_without_local_rows,
        "target_mae": {name: float(totals[kind][index] / counts[kind][index]) if counts[kind][index] else None
                       for kind, target_names in names.items() for index, name in enumerate(target_names)},
        "target_counts": {name: int(counts[kind][index])
                          for kind, target_names in names.items() for index, name in enumerate(target_names)},
        "target_shuffle_applied": bool(training and config.get("target_shuffle", False)),
    }


def write_integrity(output_root: Path) -> None:
    paths = sorted(path for path in output_root.rglob("*") if path.is_file()
                   and path.name not in {"_INCOMPLETE", "_SUCCESS", "integrity.json"})
    inventory = {"pipeline": PIPELINE_NAME, "pipeline_version": PIPELINE_VERSION,
                 "files": [{"path": path.relative_to(output_root).as_posix(),
                            "size_bytes": path.stat().st_size, "sha256": _sha256(path)} for path in paths]}
    _write_json(output_root / "integrity.json", inventory)
    (output_root / "_SUCCESS").write_text(_sha256(output_root / "integrity.json") + "\n", encoding="ascii")


def train(*, dataset: str | Path, output: str | Path, architecture: str,
          train_sample_id_file: str | Path, validation_sample_id_file: str | Path,
          timing_reference: str | Path | None = None, validation_cache: str | Path | None = None,
          device: str = "auto", epochs: int = 5, batch_size: int = 16, core_size: int = 128,
          context_size: int = 16, hidden_dim: int = 64, dropout: float = 0.1,
          learning_rate: float = 1e-3, weight_decay: float = 1e-4, seed: int = 20261003,
          patience: int = 5, target_shuffle: bool = False, onset_weight: float = 0.1,
          duration_weight: float = 1.0, velocity_weight: float = 1.0,
          pedal_weight: float = 1.0, ioi_weight: float = 1.0,
          micro_weight: float = 1.0, cumulative_weight: float = 0.0,
          cumulative_spans: Sequence[int] = DEFAULT_CUMULATIVE_SPANS,
          cumulative_log_ratio_clip: float = 4.0,
          window_policy: str = "fixed", max_window_chords: int = DEFAULT_MAX_WINDOW_CHORDS,
          bar_horizons: Sequence[int] = DEFAULT_BAR_HORIZONS,
          bar_elapsed_weight: float = 0.0, bar_tempo_weight: float = 0.0, bar_local_weight: float = 0.0,
          bar_log_ratio_clip: float = 4.0, bar_minimum_elapsed_s: float = 1e-6,
          max_steps_per_epoch: int | None = None, cpu_threads: int | None = None,
          duration_centering: bool = False, shape_features: bool = False,
          duration_target: str = "label") -> dict[str, Any]:
    if architecture not in {"mlp", "gru"}:
        raise TrainingRunError("architecture must be mlp or gru")
    if duration_target not in {"label", "shared_profile"}:
        raise TrainingRunError("duration_target must be label or shared_profile")
    if duration_target == "shared_profile" and not duration_centering:
        raise TrainingRunError("the shared profile is defined on median-centred labels; enabling it "
                               "without --duration-centering would reintroduce the per-recording level")
    if min(epochs, batch_size, core_size, hidden_dim, patience) <= 0 or context_size < 0:
        raise TrainingRunError("invalid positive dimensions or negative context")
    if not 0 <= dropout < 1 or learning_rate <= 0 or weight_decay < 0:
        raise TrainingRunError("invalid optimizer/dropout settings")
    if window_policy not in {"fixed", "bar"}:
        raise TrainingRunError("window_policy must be fixed or bar")
    bar_horizons = validate_horizons(bar_horizons)
    for name, value in (("bar log ratio clip", bar_log_ratio_clip), ("bar minimum elapsed seconds", bar_minimum_elapsed_s)):
        if not math.isfinite(value) or value <= 0:
            raise TrainingRunError(f"{name} must be finite and positive")
    if any(not math.isfinite(value) or value < 0
           for value in (bar_elapsed_weight, bar_tempo_weight, bar_local_weight)):
        raise TrainingRunError("bar loss weights must be finite and nonnegative")
    loss_weights = (micro_weight, onset_weight, duration_weight, velocity_weight, pedal_weight,
                    ioi_weight, cumulative_weight, bar_elapsed_weight, bar_tempo_weight, bar_local_weight)
    if any(not math.isfinite(value) or value < 0 for value in loss_weights) or not any(loss_weights):
        raise TrainingRunError("loss weights must be finite, nonnegative, with at least one enabled")
    cumulative_spans = validate_spans(cumulative_spans)
    if not math.isfinite(cumulative_log_ratio_clip) or cumulative_log_ratio_clip <= 0:
        raise TrainingRunError("cumulative log ratio clip must be finite and positive")
    if cumulative_weight > 0 and timing_reference is None:
        raise TrainingRunError("cumulative loss requires an explicit pure-score timing reference")
    if cumulative_weight > 0 and target_shuffle:
        raise TrainingRunError("target shuffle with cumulative supervision is unsupported")
    bar_enabled = (bar_elapsed_weight + bar_tempo_weight + bar_local_weight) > 0
    if bar_enabled and window_policy != "bar":
        raise TrainingRunError("bar loss weights require window_policy=bar")
    if bar_enabled and target_shuffle:
        raise TrainingRunError("target shuffle with bar supervision is unsupported")
    if window_policy == "bar" and max_window_chords <= 2 * context_size:
        raise TrainingRunError("max window chords must exceed twice the context size")
    # Checked last: every other bar mistake is more specific than a missing input.
    if window_policy == "bar" and timing_reference is None:
        raise TrainingRunError("window_policy=bar requires a pure-score timing reference")
    if max_steps_per_epoch is not None and max_steps_per_epoch <= 0:
        raise TrainingRunError("max_steps_per_epoch must be positive")
    if cpu_threads is not None:
        if cpu_threads <= 0:
            raise TrainingRunError("cpu_threads must be positive")
        torch.set_num_threads(cpu_threads)
    dataset_root, output_root = Path(dataset).resolve(), Path(output).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise TrainingRunError(f"sequence output must be empty: {output_root}")
    train_path, validation_path = Path(train_sample_id_file).resolve(), Path(validation_sample_id_file).resolve()
    train_ids, validation_ids = _load_sample_id_file(train_path), _load_sample_id_file(validation_path)
    if train_ids & validation_ids:
        raise TrainingRunError("train/validation sample IDs overlap")
    validation_report = (_load_validation_cache(dataset_root, validation_cache)
                         if validation_cache is not None else validate_dataset(dataset_root))
    reference = load_reference(timing_reference)
    # one boolean at the CLI, but the run records the exact column block it used, so inference
    # rebuilds this checkpoint's width even after the shipped block grows
    shape_names = SHAPE_FEATURE_NAMES if shape_features else ()
    train_samples = load_selected_samples(dataset_root, "train", train_ids, timing_reference=reference,
                                          duration_centering=duration_centering, shape_feature_names=shape_names)
    validation_samples = load_selected_samples(dataset_root, "validation", validation_ids, timing_reference=reference,
                                               duration_centering=duration_centering, shape_feature_names=shape_names)
    shared_profile: dict[str, Any] = {}
    if duration_target == "shared_profile":
        # The profile averages only over the *training* recordings of each piece: building it from
        # any split the model is scored on would move the answer into the question.  Validation
        # samples deliberately keep their own median-centred labels, so early stopping, the
        # decomposition and the bootstrap all still judge the model against real performances.
        membership = group_membership(score_groups(dataset_root / "manifest.jsonl", "train"), train_ids)
        targets = shared_duration_targets(train_samples, membership)
        train_samples = apply_shared_targets(train_samples, targets)
        shared_profile = profile_coverage(targets)
        if not shared_profile["shared_recordings"]:
            # A development selection that holds one recording per piece makes this target
            # unreachable: every group is "alone in this split", the override changes nothing, and
            # the run would silently be a duplicate of the centred control while reporting itself
            # as the shared-profile arm.  That is exactly how an experiment measures nothing and
            # looks like it did, so it is refused here rather than noticed later.
            raise TrainingRunError(
                "shared duration target matched no multi-recording piece group in the train split; "
                f"coverage was {shared_profile}")
    if window_policy == "bar":
        # Geometry is score-only and comes from the locked timing reference; it is
        # attached before any window is cut, so inference can rebuild the same plan.
        train_samples = [attach_bar_geometry(sample, reference[str(sample.sample_id)], horizons=bar_horizons)
                         for sample in train_samples]
        validation_samples = [attach_bar_geometry(sample, reference[str(sample.sample_id)], horizons=bar_horizons)
                              for sample in validation_samples]
    inputs = fit_input_normalizers(train_samples)
    recorded_note_names = note_input_names(shape_names, has_timing_reference=reference is not None)
    if len(recorded_note_names) != len(inputs["note_input_means"]):
        # the name list and the fitted width are built from different code paths, so they can
        # disagree; a checkpoint whose documented inputs do not match its tensor is unusable
        # for anyone trying to reproduce it, which is exactly the mistake this guard ends
        raise TrainingRunError(
            f"recorded note input names ({len(recorded_note_names)}) do not match the fitted "
            f"input width ({len(inputs['note_input_means'])})")
    normalizer = _load_json(dataset_root / "normalizer.json")
    nm, ns = _normalizer_vectors(normalizer, NOTE_TARGET_NAMES, "note_targets")
    if duration_target == "shared_profile":
        # The duration column is no longer the dataset's label distribution, so its standardisation
        # has to be re-fitted on what the model is actually asked to predict.  Leaving the dataset's
        # numbers in place would scale a cross-performer mean as if it were one noisy recording.
        values = np.concatenate([np.asarray(sample.note_targets[
            np.asarray(sample.note_masks[:, SHARED_DURATION_COLUMN], dtype=bool), SHARED_DURATION_COLUMN],
            dtype=np.float64) for sample in train_samples])
        if not values.size:
            raise TrainingRunError("shared duration target has no supervised rows")
        nm[SHARED_DURATION_COLUMN] = float(values.mean())
        ns[SHARED_DURATION_COLUMN] = float(max(values.std(), 1e-6))
    cm, cs = _normalizer_vectors(normalizer, CHORD_TARGET_NAMES, "chord_targets")
    selected_device = _resolve_device(device)
    _seed_everything(seed)
    config: dict[str, Any] = {
        "pipeline": PIPELINE_NAME, "pipeline_version": PIPELINE_VERSION,
        "architecture": architecture, "dataset_root": str(dataset_root),
        "dataset_success_sha256": _sha256(dataset_root / "_SUCCESS"),
        "dataset_validation": validation_report, "device": str(selected_device),
        "torch_version": torch.__version__, "cuda_device": torch.cuda.get_device_name(selected_device)
        if selected_device.type == "cuda" else None,
        "epochs": epochs, "batch_size": batch_size, "batch_unit": "chord_windows",
        "core_size": core_size, "context_size": context_size, "hidden_dim": hidden_dim,
        "dropout": dropout, "learning_rate": learning_rate, "weight_decay": weight_decay,
        "seed": seed, "patience": patience, "target_shuffle": target_shuffle,
        "target_shuffle_scope": "train only; target+native mask rows permuted together within sample",
        "duration_center_per_sample": bool(duration_centering),
        "duration_centering_scope": ("note log_key_duration_ratio minus that recording's own supervised median; "
                                     "applied to both splits so the metric stays in one unit system, and it never "
                                     "enters the inputs, so a fresh performance has no centre to remove"),
        "note_shape_features": bool(shape_features),
        "note_shape_feature_names": list(shape_names),
        "duration_target": duration_target,
        "duration_target_scope": ("shared_profile replaces the train split's duration target with each piece's "
                                  "cross-performer mean of median-centred labels; validation keeps each recording's "
                                  "own centred label, so every metric still answers to a real performance"
                                  if duration_target == "shared_profile" else
                                  "label: each recording is supervised against its own performance"),
        "shared_profile_coverage": shared_profile or None,
        "relative_position": True, "input_fit_split": "selected train only",
        "quality_mask": None, "max_steps_per_epoch": max_steps_per_epoch,
        "cpu_threads": torch.get_num_threads(), "onset_weight": onset_weight,
        "duration_weight": duration_weight, "velocity_weight": velocity_weight,
        "pedal_weight": pedal_weight, "ioi_weight": ioi_weight,
        "micro_weight": micro_weight, "cumulative_weight": cumulative_weight,
        "cumulative_spans": list(cumulative_spans), "cumulative_log_ratio_clip": cumulative_log_ratio_clip,
        "cumulative_objective": {
            "units": "score-relative seconds; performer-wide oracle tempo scale remains removed",
            "loss": "equal active-span mean SmoothL1(relative predicted-minus-target elapsed / target elapsed)",
            "scope": "native-valid contiguous IOIs; both endpoints inside one disjoint core",
            "prediction_clamp": [-cumulative_log_ratio_clip, cumulative_log_ratio_clip],
            "target_clamp": None, "minimum_elapsed_seconds": 1e-6,
        },
        "window_policy": window_policy, "core_size_used_by_policy": window_policy == "fixed",
        "max_window_chords": max_window_chords if window_policy == "bar" else None,
        "bar_horizons": list(bar_horizons), "bar_elapsed_weight": bar_elapsed_weight,
        "bar_tempo_weight": bar_tempo_weight, "bar_local_weight": bar_local_weight,
        "bar_log_ratio_clip": bar_log_ratio_clip,
        "bar_objective": {
            "units": "score-relative seconds; performer-wide oracle tempo scale remains removed",
            "tempo": "b = log(sum(score_IOI*exp(log_ratio))/sum(score_IOI)) per complete in-core span",
            "local": "log_ratio - b; a constant shift across a span moves only tempo",
            "horizon_weighting": "mean of active-horizon means; span counts never weight horizons",
            "layout": "cores cut at recorded bar boundaries, capacity bounds the window with halos",
            "eligibility": "both endpoints in one core and every interval row native-valid; masks never move a cut",
            "excluded": "partial bars, missing endpoints, capacity-forced cuts, spans with any invalid row",
            "prediction_clamp": [-bar_log_ratio_clip, bar_log_ratio_clip], "target_clamp": None,
            "minimum_elapsed_seconds": bar_minimum_elapsed_s,
        },
        "note_target_means": nm.tolist(), "note_target_scales": ns.tolist(),
        "chord_target_means": cm.tolist(), "chord_target_scales": cs.tolist(),
        "note_features": list(recorded_note_names),
        "chord_features": list(chord_input_names(has_timing_reference=reference is not None)),
        "note_targets": list(NOTE_TARGET_NAMES), "chord_targets": list(CHORD_TARGET_NAMES),
        "timing_reference": str(Path(timing_reference).resolve()) if timing_reference is not None else None,
        "timing_reference_sha256": reference_hash(timing_reference),
        "train_sample_id_file": str(train_path), "train_sample_id_file_sha256": _sha256(train_path),
        "train_sample_id_count": len(train_ids), "validation_sample_id_file": str(validation_path),
        "validation_sample_id_file_sha256": _sha256(validation_path), "validation_sample_id_count": len(validation_ids),
        "dataset_validation_cache": str(Path(validation_cache).resolve()) if validation_cache is not None else None,
        "dataset_validation_cache_sha256": _sha256(Path(validation_cache).resolve()) if validation_cache is not None else None,
        **inputs,
    }
    model = make_model(config).to(selected_device)
    config["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    config["implementation_sha256"] = {name: _sha256(Path(__file__).parent / name)
                                       for name in ("sequence_expression.py", "train_sequence_expression.py",
                                                    "rhythm_objective.py", "bar_windows.py", "bar_objective.py",
                                                    "bar_supervision.py", "shape_features.py",
                                                    "shared_profile.py", "piece_groups.py")}
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "_INCOMPLETE").write_text("sequence pilot in progress\n", encoding="ascii")
    (output_root / "train_sample_ids.txt").write_bytes(train_path.read_bytes())
    (output_root / "validation_sample_ids.txt").write_bytes(validation_path.read_bytes())
    _write_json(output_root / "run_config.json", config)
    best, best_epoch, stale = math.inf, 0, 0
    with (output_root / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for epoch in range(1, epochs + 1):
            train_metrics = _run_epoch(model, train_samples, config, device=selected_device, optimizer=optimizer,
                                       seed=seed + epoch, max_steps=max_steps_per_epoch)
            validation_metrics = _run_epoch(model, validation_samples, config, device=selected_device,
                                            optimizer=None, seed=seed)
            record = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            print(json.dumps(record, sort_keys=True), flush=True)
            payload = {"pipeline": PIPELINE_NAME, "pipeline_version": PIPELINE_VERSION,
                       "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                       "run_config": config, "train_metrics": train_metrics, "validation_metrics": validation_metrics}
            torch.save(payload, output_root / "checkpoint_last.pt")
            current = float(validation_metrics["loss"])
            if current < best:
                best, best_epoch, stale = current, epoch, 0
                torch.save(payload, output_root / "checkpoint_best.pt")
            else:
                stale += 1
                if stale >= patience:
                    break
    config.update(best_epoch=best_epoch, best_validation_loss=best, completed_epochs=epoch)
    _write_json(output_root / "run_config.json", config)
    write_integrity(output_root)
    (output_root / "_INCOMPLETE").unlink()
    return {"status": "complete", "output": str(output_root), "architecture": architecture,
            "best_epoch": best_epoch, "best_validation_loss": best, "parameter_count": config["parameter_count"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "output", "train-sample-id-file", "validation-sample-id-file"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--architecture", choices=("mlp", "gru"), required=True)
    parser.add_argument("--timing-reference")
    parser.add_argument("--validation-cache")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    for name, default in (("epochs", 5), ("batch-size", 16), ("core-size", 128), ("context-size", 16),
                          ("hidden-dim", 64), ("seed", 20261003), ("patience", 5)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--max-steps-per-epoch", type=int)
    parser.add_argument("--cpu-threads", type=int)
    for name, default in (("dropout", 0.1), ("learning-rate", 1e-3), ("weight-decay", 1e-4),
                          ("onset-weight", 0.1), ("duration-weight", 1.0), ("velocity-weight", 1.0),
                          ("pedal-weight", 1.0), ("ioi-weight", 1.0)):
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--micro-weight", type=float, default=1.0)
    parser.add_argument("--cumulative-weight", type=float, default=0.0)
    parser.add_argument("--cumulative-spans", type=int, nargs="+", default=list(DEFAULT_CUMULATIVE_SPANS))
    parser.add_argument("--cumulative-log-ratio-clip", type=float, default=4.0)
    parser.add_argument("--window-policy", choices=("fixed", "bar"), default="fixed")
    parser.add_argument("--max-window-chords", type=int, default=DEFAULT_MAX_WINDOW_CHORDS)
    parser.add_argument("--bar-horizons", type=int, nargs="+", default=list(DEFAULT_BAR_HORIZONS))
    for name, default in (("bar-elapsed-weight", 0.0), ("bar-tempo-weight", 0.0), ("bar-local-weight", 0.0),
                          ("bar-log-ratio-clip", 4.0), ("bar-minimum-elapsed-s", 1e-6)):
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--target-shuffle", action="store_true")
    parser.add_argument("--duration-centering", action="store_true",
                        help="train on each recording's duration profile with its own level removed")
    parser.add_argument("--duration-target", choices=("label", "shared_profile"), default="label",
                        help="shared_profile supervises the train split on each piece's cross-performer mean "
                             "shape rather than one noisy recording; requires --duration-centering, and the "
                             "validation split keeps its own centred labels so the metrics stay honest")
    parser.add_argument("--shape-features", action="store_true",
                        help="append derived voice-continuity note columns (same-pitch gaps, voices still "
                             "sounding at each note's end); changes the input width, so inference must use "
                             "the same flag")
    args = vars(parser.parse_args(argv))
    try:
        result = train(**args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"sequence training failed: {exc}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
