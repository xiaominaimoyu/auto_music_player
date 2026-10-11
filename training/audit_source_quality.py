"""Audit performance-label quality on deterministic per-source work samples.

This is a read-only companion to the PianoCoRe-A* builder.  It deliberately
re-opens the refined MIDI/alignment triplets instead of trusting only the
derived shards, so timing, duration, pedal-tail and alignment diagnostics can
be reviewed before a model is promoted to playback.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from training.midi_features import build_aligned_features, parse_midi_bytes
from training.pianocore_dataset import RefinedArchive, _load_alignment


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _quantiles(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "mean": None, "median": None, "q1": None, "q3": None, "p95": None}
    q1, median, q3, p95 = np.quantile(array, [0.25, 0.5, 0.75, 0.95])
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "p95": float(p95),
    }


def _choose_rows(rows: list[dict[str, Any]], per_source: int) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("performance_dataset") or "unknown")].append(row)
    chosen: dict[str, list[dict[str, Any]]] = {}
    for source, values in sorted(groups.items()):
        values = sorted(values, key=lambda item: str(item.get("id") or ""))
        count = min(int(per_source), len(values))
        indices = np.unique(np.linspace(0, len(values) - 1, count, dtype=np.int64))
        chosen[source] = [values[int(index)] for index in indices]
    return chosen


def _finite_target_stats(arrays: Any, target_index: int, mask_index: int) -> dict[str, float | int | None]:
    values = arrays.note_targets[:, target_index][arrays.note_masks[:, mask_index]]
    return _quantiles(values.tolist())


def _audit_row(row: Mapping[str, Any], archive: RefinedArchive) -> dict[str, Any]:
    score = parse_midi_bytes(archive.read(str(row["score_midi_path"])))
    performance = parse_midi_bytes(archive.read(str(row["performance_midi_path"])))
    perf_idx, interpolated = _load_alignment(archive.read(str(row["alignment_path"])))
    arrays = build_aligned_features(score, performance, perf_idx, interpolated)
    note_mask = arrays.note_masks[:, 0]
    chord_mask = arrays.chord_masks[:, 1]
    last_note_end = max((note.end_s for note in performance.notes), default=0.0)
    last_pedal_event = max((event.time_s for event in performance.pedal_events), default=0.0)
    pedal_tail = max(0.0, last_pedal_event - last_note_end)
    duration = arrays.note_targets[:, 1][note_mask]
    flags: list[str] = []
    if float(row.get("interpolation_rate") or 0.0) > 0.10:
        flags.append("interpolation_rate_over_10_percent")
    if float(row.get("refined_recall") or 0.0) < 0.90:
        flags.append("refined_recall_below_0.90")
    if not 0.50 <= float(arrays.diagnostics["global_tempo_scale"]) <= 2.0:
        flags.append("tempo_scale_outside_typical_range")
    if len(duration) and abs(float(np.median(duration))) > 1.0:
        flags.append("duration_ratio_median_over_1_log_unit")
    if pedal_tail > 2.0:
        flags.append("pedal_tail_over_2_seconds")
    return {
        "id": row.get("id"),
        "performance_dataset": row.get("performance_dataset"),
        "performance_id": row.get("performance_id"),
        "split": row.get("split"),
        "piece_group": row.get("piece_group"),
        "composer": row.get("composer"),
        "composition": row.get("composition"),
        "movement": row.get("movement"),
        "note_count_score": len(score.notes),
        "note_count_performance": len(performance.notes),
        "chord_count": int(arrays.chord_targets.shape[0]),
        "pedal_event_count": len(performance.pedal_events),
        "last_note_end_s": float(last_note_end),
        "last_pedal_event_s": float(last_pedal_event),
        "pedal_tail_after_last_note_s": float(pedal_tail),
        "refined_recall": float(row.get("refined_recall") or 0.0),
        "interpolation_rate": float(arrays.diagnostics["interpolation_rate"]),
        "global_tempo_scale": float(arrays.diagnostics["global_tempo_scale"]),
        "supervised_note_count": int(note_mask.sum()),
        "supervised_chord_count": int(arrays.chord_masks[:, 0].sum()),
        "onset_delta_abs_p95_ms": float(arrays.diagnostics["onset_delta_abs_p95_ms"]),
        "micro_offset_abs_p95_ms": float(arrays.diagnostics["micro_offset_abs_p95_ms"]),
        "target_stats": {
            "micro_offset_ms": _finite_target_stats(arrays, 0, 0),
            "log_key_duration_ratio": _finite_target_stats(arrays, 1, 1),
            "velocity_0_1": _finite_target_stats(arrays, 2, 2),
            "velocity_robust_z": _finite_target_stats(arrays, 3, 3),
            "pedal_at_onset": _finite_target_stats(arrays, 4, 4),
            "pedal_coverage": _finite_target_stats(arrays, 5, 5),
            "log_ioi_ratio": _quantiles(arrays.chord_targets[:, 1][chord_mask].tolist()),
            "onset_delta_ms": _quantiles(
                arrays.chord_targets[:, 0][arrays.chord_masks[:, 0]].tolist()
            ),
        },
        "flags": flags,
    }


def audit(dataset_root: Path, archive_path: Path, output_path: Path, per_source: int) -> dict[str, Any]:
    rows = _read_jsonl(dataset_root / "manifest.jsonl")
    selected = _choose_rows(rows, per_source)
    sampled: list[dict[str, Any]] = []
    with RefinedArchive(archive_path) as archive:
        for source, source_rows in selected.items():
            for row in source_rows:
                audited = _audit_row(row, archive)
                sampled.append(audited)
    aggregates: dict[str, Any] = {}
    for source in sorted(selected):
        source_rows = [row for row in sampled if row["performance_dataset"] == source]
        aggregates[source] = {
            "sample_count": len(source_rows),
            "flagged_sample_count": sum(bool(row["flags"]) for row in source_rows),
            "tempo_scale": _quantiles(row["global_tempo_scale"] for row in source_rows),
            "interpolation_rate": _quantiles(row["interpolation_rate"] for row in source_rows),
            "refined_recall": _quantiles(row["refined_recall"] for row in source_rows),
            "pedal_tail_after_last_note_s": _quantiles(
                row["pedal_tail_after_last_note_s"] for row in source_rows
            ),
            "duration_log_ratio_median": _quantiles(
                row["target_stats"]["log_key_duration_ratio"]["median"]
                for row in source_rows
                if row["target_stats"]["log_key_duration_ratio"]["median"] is not None
            ),
            "micro_offset_abs_p95_ms": _quantiles(
                row["micro_offset_abs_p95_ms"] for row in source_rows
            ),
            "onset_delta_abs_p95_ms": _quantiles(
                row["onset_delta_abs_p95_ms"] for row in source_rows
            ),
        }
    result = {
        "schema_version": 1,
        "status": "complete",
        "dataset_root": str(dataset_root),
        "archive": str(archive_path),
        "per_source_requested": int(per_source),
        "manifest_rows": len(rows),
        "sampled_rows": len(sampled),
        "source_counts": {source: len(values) for source, values in sorted(selected.items())},
        "aggregates": aggregates,
        "samples": sampled,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=10)
    args = parser.parse_args(argv)
    result = audit(args.dataset_root, args.archive, args.output, max(1, args.per_source))
    print(json.dumps({"status": result["status"], "sampled_rows": result["sampled_rows"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
