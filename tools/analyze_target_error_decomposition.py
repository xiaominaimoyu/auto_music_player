"""Split a trained model's per-channel error into a constant offset and incoherent scatter.

A score-only model cannot reproduce a label component that does not repeat across
performers, so the question for the duration channel is not "is the error big" but "which
kind of error is it".  For every supervised row of a channel the model's error is

    e = prediction - label = median(e)  +  (e - median(e))

and the two terms are reported separately, per performance and row-weighted:

* ``offset_mae`` -- how wrong the model is about each recording's constant level.  That part
  is only reachable if the level is a score property, which
  ``tools/audit_piece_offset_reproducibility.py`` tests with ICC(1);
* ``scatter_mae`` -- the within-recording shape error, which is what centering the training
  target could actually attack;
* ``shape_mae`` -- prediction and label each centred on their own median, the metric an arm
  trained on centred labels and an arm trained on raw labels can both be judged by, with
  ``label_scatter_mae`` (one constant per recording) as its baseline.

Predictions come from already-trained checkpoints through the same inference path the game
would use, so nothing is retrained and the locked test split is never read.
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
    DURATION_TARGET_COLUMN,
    checkpoint_preparation_flags,
    load_selected_samples,
    predict_sample,
)
from training.train_expression import TrainingRunError, _load_sample_id_file, _sha256, _write_json
from training.tempo_reference import load_timing_reference

# channel name -> (row kind, column), matching tools/audit_target_channels.CHANNELS
CHANNELS = (("onset_delta_ms", "chord", 0), ("log_ioi_ratio", "chord", 1),
            ("log_key_duration_ratio", "note", 1))


def evaluation_label(label: np.ndarray, mask: np.ndarray, *, centred: bool) -> np.ndarray:
    """Put the label in the units the checkpoint was trained to predict.

    A centred-target arm outputs "this recording's duration profile relative to its own
    level", so its reference must be the label with that same per-recording median removed.
    The median comes from the label side, never from the prediction, so the two arms stay
    comparable: each one is judged against its own units, and ``shape_mae`` then removes the
    level from both sides for the cross-arm comparison.
    """
    if not centred:
        return label
    supervised = label[mask]
    if not supervised.size:
        return label
    centred_label = label.copy()
    centred_label[mask] = supervised - float(np.median(supervised))
    return centred_label


def split_prediction_error(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, object]:
    """Row-weighted offset/scatter/shape split of one channel's predictions.

    ``scatter_mae`` gives the model one free constant per recording, chosen from its own
    errors (the median error); ``shape_mae`` instead centres prediction and label each by
    their own median, which is the symmetric metric an arm trained on centred labels can be
    compared against.  ``label_scatter_mae`` is what predicting a constant per recording
    would cost in that same metric, so the three numbers always sit on one ruler.
    """
    if not pairs:
        return {"rows": 0, "performances": 0}
    errors = [prediction - label for prediction, label in pairs]
    medians = [float(np.median(block)) for block in errors]
    sizes = [int(block.size) for block in errors]
    rows = sum(sizes)
    raw = np.concatenate(errors)
    raw_mae = float(np.abs(raw).mean())
    scatter = np.concatenate([block - median for block, median in zip(errors, medians)])
    scatter_mae = float(np.abs(scatter).mean())
    shape = np.concatenate([(prediction - float(np.median(prediction)))
                            - (label - float(np.median(label))) for prediction, label in pairs])
    label_level = np.concatenate([label - float(np.median(label)) for _prediction, label in pairs])
    return {"rows": rows, "performances": len(pairs),
            "raw_mae": raw_mae,
            "offset_mae": float(sum(abs(median) * size for median, size in zip(medians, sizes)) / rows),
            "scatter_mae": scatter_mae,
            "shape_mae": float(np.abs(shape).mean()),
            "label_scatter_mae": float(np.abs(label_level).mean()),
            "scatter_share_of_raw": scatter_mae / raw_mae if raw_mae > 0 else None}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--timing-reference", type=Path, required=True)
    parser.add_argument("--sample-id-file", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="completed run directories")
    parser.add_argument("--checkpoint", default="checkpoint_best.pt")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.output.exists() and any(arguments.output.iterdir()):
        raise SystemExit(f"refusing to overwrite existing output: {arguments.output}")
    torch.set_num_threads(arguments.cpu_threads)
    dataset_root = arguments.dataset.resolve()
    sample_ids = frozenset(str(value) for value in _load_sample_id_file(arguments.sample_id_file))
    reference = load_timing_reference(arguments.timing_reference)
    device = torch.device(arguments.device if torch.cuda.is_available() or arguments.device == "cpu" else "cpu")
    # one preparation per distinct flag combination, so a run never sees samples prepared for
    # another run's input width
    prepared: dict[tuple[bool, bool, bool], list] = {}
    runs = []
    for run_directory in arguments.runs:
        checkpoint = (run_directory / arguments.checkpoint).resolve()
        model, config = load_model(checkpoint, device)
        flags = checkpoint_preparation_flags(config)
        if flags not in prepared:
            prepared[flags] = load_selected_samples(dataset_root, arguments.split, sample_ids,
                                                   timing_reference=reference, relative_position=flags[2],
                                                   duration_centering=flags[0], shape_features=flags[1])
            if not prepared[flags]:
                raise TrainingRunError("no samples selected for the error decomposition")
        samples = prepared[flags]
        centred = flags[0]
        pairs: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {name: [] for name, _kind, _column in CHANNELS}
        chord_rows = 0
        for sample in samples:
            note_output, chord_output = predict_sample(model, sample, config=config, device=device)
            arrays = {"chord": chord_output, "note": note_output}
            labels = {"chord": sample.chord_targets, "note": sample.note_targets}
            masks = {"chord": sample.chord_masks, "note": sample.note_masks}
            for name, kind, column in CHANNELS:
                mask = np.asarray(masks[kind][:, column], dtype=bool)
                label_column = np.asarray(labels[kind][:, column], dtype=np.float64)
                if centred and kind == "note" and column == DURATION_TARGET_COLUMN:
                    label_column = evaluation_label(label_column, mask, centred=True)
                prediction = np.asarray(arrays[kind][:, column], dtype=np.float64)[mask]
                label = label_column[mask]
                if prediction.size:
                    pairs[name].append((prediction, label))
            chord_rows += len(sample.chord_features)
        runs.append({"run": run_directory.name, "checkpoint": str(checkpoint),
                     "checkpoint_sha256": _sha256(checkpoint),
                     "trained_window_policy": config.get("window_policy"),
                     "trained_duration_centering": centred,
                     "trained_shape_features": flags[1],
                     "note_input_width": int(samples[0].note_features.shape[1]),
                     "duration_label_units": "recording-centred" if centred else "raw",
                     "sample_count": len(samples), "chord_rows": chord_rows,
                     "channels": {name: split_prediction_error(pairs[name])
                                  for name, _kind, _column in CHANNELS}})
    report = {"kind": "target_error_offset_versus_scatter", "status": "complete", "trained": False,
              "device": str(device), "dataset": str(dataset_root), "split": arguments.split,
              "sample_id_file": str(arguments.sample_id_file.resolve()),
              "sample_id_file_sha256": _sha256(arguments.sample_id_file),
              "timing_reference_sha256": _sha256(arguments.timing_reference / "summary.json"),
              "definitions": {
                  "offset_mae": "row-weighted mean over recordings of |median error|: the constant level the model "
                                "gets wrong for each recording",
                  "scatter_mae": "mean |error - median error| within each recording: the incoherent part that an "
                                 "oracle per-recording constant would remove",
                  "shape_mae": "prediction and label each centred by their own median, then mean absolute "
                               "difference: the profile-reproduction metric a raw-target arm and a centred-target "
                               "arm can both be compared on",
                  "label_scatter_mae": "the same shape metric for a predictor that outputs one constant per "
                                       "recording, i.e. the baseline shape_mae has to beat",
                  "note": "offset_mae + scatter_mae need not equal raw_mae; the split is a decomposition of the "
                          "error distribution, not an additive identity"},
              "reading": {"decision_rule": "a centred-target arm only pays if its shape_mae beats the raw-target "
                                           "arm's shape_mae by more than seed spread; if both already sit near "
                                           "label_scatter_mae and the cross-performance floor, no target "
                                           "reparameterisation can help",
                          "model_quality_claim": False, "adoption_evidence": False},
              "runs": runs, "model_inputs_modified": False, "adoption_evidence": False}
    arguments.output.mkdir(parents=True, exist_ok=True)
    _write_json(arguments.output / "summary.json", report)
    digest = _sha256(arguments.output / "summary.json")
    (arguments.output / "_SUCCESS").write_text(digest + "\n", encoding="ascii")
    print(json.dumps({"output": str(arguments.output), "summary_sha256": digest,
                      "runs": [{"run": entry["run"], "channels": entry["channels"]} for entry in runs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
