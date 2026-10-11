"""Evaluate a completed expression checkpoint on a locked dataset split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from training.midi_features import CHORD_TARGET_NAMES, NOTE_TARGET_NAMES
from training.train_expression import (
    ExpressionMLP,
    TrainingRunError,
    _normalizer_vectors,
    _resolve_device,
    _run_epoch,
)
from training.quality_mask import QualityMaskError, load_quality_mask
from training.validate_pianocore_dataset import validate_dataset


def evaluate(
    *,
    dataset: str | Path,
    checkpoint: str | Path,
    split: str = "test",
    device: str = "auto",
    batch_size: int = 8192,
    quality_mask: str | Path | None = None,
) -> dict[str, object]:
    if split not in {"validation", "test"}:
        raise TrainingRunError("evaluation split must be validation or test")
    if batch_size <= 0:
        raise TrainingRunError("batch_size must be positive")
    dataset_root = Path(dataset).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    quality_mask_data = None
    if quality_mask is not None:
        try:
            quality_mask_data = load_quality_mask(quality_mask)
        except QualityMaskError as exc:
            raise TrainingRunError(str(exc)) from exc
    validation_report = validate_dataset(dataset_root)
    selected_device = _resolve_device(device)
    payload = torch.load(checkpoint_path, map_location=selected_device, weights_only=False)
    run_config = payload.get("run_config")
    if not isinstance(run_config, dict):
        raise TrainingRunError("checkpoint is missing run_config")
    model = ExpressionMLP(
        hidden_dim=int(run_config["hidden_dim"]),
        dropout=float(run_config["dropout"]),
    ).to(selected_device)
    model.load_state_dict(payload["model"])
    model.eval()
    normalizer = json.loads((dataset_root / "normalizer.json").read_text(encoding="utf-8"))
    note_mean_np, note_scale_np = _normalizer_vectors(
        normalizer, NOTE_TARGET_NAMES, "note_targets"
    )
    chord_mean_np, chord_scale_np = _normalizer_vectors(
        normalizer, CHORD_TARGET_NAMES, "chord_targets"
    )
    note_means = torch.from_numpy(note_mean_np).to(selected_device)
    note_scales = torch.from_numpy(note_scale_np).to(selected_device)
    chord_means = torch.from_numpy(chord_mean_np).to(selected_device)
    chord_scales = torch.from_numpy(chord_scale_np).to(selected_device)
    note_weights = torch.tensor(
        [
            1.0,
            float(run_config.get("duration_weight", 1.0)),
            float(run_config.get("velocity_weight", 1.0)),
            float(run_config.get("velocity_weight", 1.0)),
            float(run_config.get("pedal_weight", 1.0)),
            float(run_config.get("pedal_weight", 1.0)),
        ],
        dtype=torch.float32,
        device=selected_device,
    )
    chord_weights = torch.tensor(
        [
            float(run_config.get("onset_weight", 0.1)),
            float(run_config.get("ioi_weight", 1.0)),
        ],
        dtype=torch.float32,
        device=selected_device,
    )
    relative_position = bool(run_config.get("relative_position", False))
    note_input_means = np.asarray(
        run_config.get("note_input_means", np.zeros(len(model.note_net[0].weight[0]))),
        dtype=np.float32,
    )
    note_input_scales = np.asarray(
        run_config.get("note_input_scales", np.ones(len(note_input_means))),
        dtype=np.float32,
    )
    chord_input_means = np.asarray(
        run_config.get("chord_input_means", np.zeros(len(model.chord_net[0].weight[0]))),
        dtype=np.float32,
    )
    chord_input_scales = np.asarray(
        run_config.get("chord_input_scales", np.ones(len(chord_input_means))),
        dtype=np.float32,
    )
    metrics = _run_epoch(
        model,
        dataset_root,
        split,
        optimizer=None,
        device=selected_device,
        batch_size=batch_size,
        note_means=note_means,
        note_scales=note_scales,
        chord_means=chord_means,
        chord_scales=chord_scales,
        note_weights=note_weights,
        chord_weights=chord_weights,
        seed=int(run_config.get("seed", 0)),
        relative_position=relative_position,
        note_input_means=note_input_means,
        note_input_scales=note_input_scales,
        chord_input_means=chord_input_means,
        chord_input_scales=chord_input_scales,
        quality_mask=(quality_mask_data.decisions if quality_mask_data else None),
        quality_mask_splits=frozenset({split}),
    )
    return {
        "status": "valid",
        "dataset_validation": validation_report,
        "checkpoint": str(checkpoint_path),
        "split": split,
        "device": str(selected_device),
        "quality_mask": (
            {
                "path": quality_mask_data.path,
                "sha256": quality_mask_data.sha256,
                "policy_version": quality_mask_data.policy_version,
                "row_count": quality_mask_data.row_count,
                "reviewed_count": quality_mask_data.reviewed_count,
                "applied_split": split,
            }
            if quality_mask_data
            else None
        ),
        "metrics": metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate an expression checkpoint on validation or locked test data."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--quality-mask", default=None)
    args = parser.parse_args(argv)
    try:
        report = evaluate(
            dataset=args.dataset,
            checkpoint=args.checkpoint,
            split=args.split,
            device=args.device,
            batch_size=args.batch_size,
            quality_mask=args.quality_mask,
        )
    except (TrainingRunError, RuntimeError, ValueError, OSError) as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["evaluate", "main"]
