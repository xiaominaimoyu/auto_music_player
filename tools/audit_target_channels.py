"""Read-only audit of what each supervised target channel actually contains.

The channels are scored against the two baselines a model can reach without any
input signal: predict zero (identity) and predict each performance's own median.
If per-performance centering collapses a channel's error, the label carries a
piece-level nuisance offset rather than playable expression, and no model can
recover it from the score.  Nothing here trains, edits masks, or touches playback.
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

from training.midi_features import CHORD_TARGET_NAMES, NOTE_TARGET_NAMES
from training.sequence_expression import load_selected_samples
from training.train_expression import TrainingRunError, _load_sample_id_file, _sha256, _write_json

CHANNELS = (("chord", "onset_delta_ms", 0), ("chord", "log_ioi_ratio", 1),
            ("note", "log_key_duration_ratio", 1), ("note", "micro_offset_ms", 0),
            ("note", "velocity_robust_z", 3))
QUANTS = (50.0, 90.0, 99.0)


def per_channel(rows: list[dict[str, object]]) -> dict[str, object]:
    """Compare predict-zero with predict-each-performance's-own-median, per channel."""
    report: dict[str, object] = {}
    for _kind, name, _column in CHANNELS:
        blocks = [np.asarray(row["channel_values"][name], dtype=np.float64) for row in rows]
        if not any(block.size for block in blocks):
            raise TrainingRunError(f"channel {name} has no supervised rows in this selection")
        values = np.concatenate([block for block in blocks if block.size])
        identity = float(np.abs(values).mean())
        centered = np.concatenate([block - float(np.median(block)) for block in blocks if block.size])
        per_sample_median = np.array([float(np.median(block)) if block.size else 0.0 for block in blocks])
        ordered = np.sort(np.array([float(row["channel_mass"][name]) for row in rows]))[::-1]
        total = float(ordered.sum())
        worst = sorted(rows, key=lambda row: -float(row["channel_mass"][name]))[:5]
        report[name] = {
            "supervised_row_count": int(values.size), "sample_count": len(rows),
            "identity_baseline_mae": identity,
            "per_sample_median_centered_mae": float(np.abs(centered).mean()),
            "median_shift_share": float(1.0 - np.abs(centered).mean() / identity) if identity else None,
            "abs_value_percentiles": {f"p{int(quantile)}": float(np.percentile(np.abs(values), quantile))
                                       for quantile in QUANTS},
            "per_sample_median_abs": {f"p{int(quantile)}": float(np.percentile(np.abs(per_sample_median), quantile))
                                      for quantile in QUANTS},
            "samples_with_median_beyond_half_identity": int((np.abs(per_sample_median) > 0.5 * identity).sum()),
            "error_mass_in_top_1pct_samples": (float(ordered[:max(1, len(ordered) // 100)].sum() / total) if total else None),
            "error_mass_in_top_10pct_samples": (float(ordered[:max(1, len(ordered) // 10)].sum() / total) if total else None),
            "worst_samples": [{"id": str(row["id"]), "source": str(row["source"]),
                               "channel_median": round(float(np.median(row["channel_values"][name])), 4)
                               if np.asarray(row["channel_values"][name]).size else None,
                               "rows": int(np.asarray(row["channel_values"][name]).size)}
                              for row in worst],
        }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sample-id-file", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    for kind, name, column in CHANNELS:
        names = CHORD_TARGET_NAMES if kind == "chord" else NOTE_TARGET_NAMES
        if names[column] != name:
            raise SystemExit(f"channel map drifted: {kind} column {column} is {names[column]!r}, not {name!r}")
    if arguments.output.exists() and any(arguments.output.iterdir()):
        raise SystemExit(f"refusing to overwrite existing output: {arguments.output}")
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
                source = str(row.get("performance_dataset") or "unknown")
                identities[sample_id] = (source, str(row.get("piece_group") or sample_id))
    if set(identities) != set(sample_ids):
        raise TrainingRunError("dataset manifest does not cover every selected sample")
    rows: list[dict[str, object]] = []
    # ``load_selected_samples`` owns the shard layout and sample offsets; this audit
    # only reads the target columns and their native masks.
    for sample in load_selected_samples(dataset_root, arguments.split, sample_ids):
        blocks = {"note": (sample.note_targets, sample.note_masks),
                  "chord": (sample.chord_targets, sample.chord_masks)}
        channel_values: dict[str, np.ndarray] = {}
        for kind, name, column in CHANNELS:
            targets, masks = blocks[kind]
            channel_values[name] = np.asarray(targets[:, column], dtype=np.float64)[
                np.asarray(masks[:, column], dtype=bool)]
        rows.append({"id": sample.sample_id, "source": identities[sample.sample_id][0],
                     "piece_group": identities[sample.sample_id][1],
                     "channel_mass": {key: float(np.abs(values).sum()) for key, values in channel_values.items()},
                     "channel_values": channel_values})
    if {str(row["id"]) for row in rows} != set(sample_ids):
        raise TrainingRunError("selected samples were not all loaded from the dataset")
    report = {"kind": "target_channel_content_audit", "status": "complete", "dataset": str(dataset_root),
              "split": arguments.split, "sample_id_file": str(arguments.sample_id_file.resolve()),
              "sample_id_file_sha256": _sha256(arguments.sample_id_file),
              "sample_count": len(rows), "channels": per_channel(rows),
              "reading": {"identity_baseline_mae": "what a model predicting zero achieves; a model cannot beat its own label noise",
                          "per_sample_median_centered_mae": "the same error after removing each performance's own offset; "
                                                            "a large drop means the channel carries a piece-level nuisance "
                                                            "instead of playable expression",
                          "adoption_gate_changed": False, "masks_modified": False, "model_evaluation": False}}
    arguments.output.mkdir(parents=True, exist_ok=True)
    _write_json(arguments.output / "summary.json", report)
    digest = _sha256(arguments.output / "summary.json")
    (arguments.output / "_SUCCESS").write_text(digest + "\n", encoding="ascii")
    print(json.dumps({"output": str(arguments.output), "summary_sha256": digest,
                      "channels": {name: {key: value for key, value in body.items()
                                          if key in ("supervised_row_count", "identity_baseline_mae",
                                                     "per_sample_median_centered_mae", "median_shift_share",
                                                     "error_mass_in_top_10pct_samples")}
                                   for name, body in report["channels"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
