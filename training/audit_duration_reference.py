"""Read-only physical key-duration and CC64-release audit for locked samples.

The output is an evaluation-only sidecar, not model input or a replacement
dataset/mask. CC64-derived release is a logical MIDI endpoint, not measured
acoustic decay; missing pedal release is censored rather than imputed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import mido

from training.midi_features import ParsedMidi, build_aligned_features, parse_midi_bytes
from training.pianocore_dataset import RefinedArchive, _load_alignment, write_deterministic_npz
from training.tempo_reference import (
    TimingReferenceError, _canonical, _safe_path, _sha256, _tempo_events,
    load_timing_reference,
)


class DurationReferenceError(TimingReferenceError):
    """Physical references or predictions have invalid source/shape binding."""


def _pedal_release(performance: ParsedMidi) -> dict[str, np.ndarray]:
    grouped = defaultdict(list)
    for event in performance.pedal_events:
        grouped[event.channel].append(event)
    count = len(performance.notes)
    ends = np.full(count, np.nan, dtype=np.float64)
    available, censored, boundary = (np.zeros(count, dtype=bool) for _ in range(3))
    for channel, events in grouped.items():
        times = np.array([event.time_s for event in events])
        values = np.array([event.value for event in events])
        releases = times[values < 64]
        for index, note in enumerate(performance.notes):
            if note.channel != channel:
                continue
            at = int(np.searchsorted(times, note.end_s, side="right")) - 1
            boundary[index] = at >= 0 and math.isclose(float(times[at]), note.end_s, rel_tol=0, abs_tol=1e-12)
            if at < 0 or values[at] < 64:
                ends[index], available[index] = note.end_s, True
                continue
            next_release = int(np.searchsorted(releases, note.end_s, side="right"))
            if next_release < len(releases):
                ends[index], available[index] = releases[next_release], True
            else:
                censored[index] = True
    return {"note_sustain_end_seconds": ends, "note_sustain_end_observed": available,
            "note_sustain_end_censored": censored, "note_pedal_boundary_at_keyoff": boundary}


def _next_pitch_onsets(performance: ParsedMidi) -> np.ndarray:
    result = np.full(len(performance.notes), np.inf)
    following = {}
    for index in range(len(performance.notes) - 1, -1, -1):
        note = performance.notes[index]
        key = (note.channel, note.pitch)
        result[index] = following.get(key, np.inf)
        following[key] = note.start_s
    return result


def make_duration_reference(score_data: bytes, performance_data: bytes,
                            perf_idx: np.ndarray, interpolated: np.ndarray,
                            timing: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Rebuild original labels; retain key-up and logical sustain separately."""
    score, performance = parse_midi_bytes(score_data), parse_midi_bytes(performance_data)
    arrays = build_aligned_features(score, performance, perf_idx, interpolated)
    for first, second in ((arrays.note_to_chord, timing["note_to_chord"]),
                          (arrays.note_masks, timing["note_native_masks"]),
                          (arrays.chord_masks, timing["chord_native_masks"]),
                          (arrays.note_targets[:, 1], timing["note_log_key_duration_ratio"]),
                          (arrays.chord_targets[:, 1], timing["chord_log_ioi_ratio"])):
        if not np.array_equal(first, second):
            raise DurationReferenceError("original source/native labels differ from bound timing reference")
    aligned = [performance.notes[int(index)] for index in perf_idx]
    score_duration = np.array([note.duration_s for note in score.notes])
    if not np.array_equal(score_duration, timing["note_score_duration_seconds"]):
        raise DurationReferenceError("score duration order differs from timing reference")
    scale = float(timing["global_tempo_scale"])
    score_ticks, score_tempos = _tempo_events(score_data)
    perf_ticks, perf_tempos = _tempo_events(performance_data)
    # Endpoint quantization resolution is one source tick, not an estimate of
    # OMR/transcription/alignment accuracy and not a new quality mask.
    score_starts = np.array([note.start_tick for note in score.notes])
    score_ends = np.array([note.end_tick for note in score.notes])
    perf_starts = np.array([note.start_tick for note in aligned])
    perf_ends = np.array([note.end_tick for note in aligned])
    def resolution(ticks, tempo_ticks, tempos, ppq):
        return tempos[np.searchsorted(tempo_ticks, ticks, side="right") - 1] / (1e6 * ppq)
    pedal = _pedal_release(performance)
    selected_pedal = {name: value[perf_idx] for name, value in pedal.items()}
    next_onset = _next_pitch_onsets(performance)[perf_idx]
    key_end = np.array([note.end_s for note in aligned])
    perf_duration = np.array([note.duration_s for note in aligned])
    counts = np.bincount(arrays.note_to_chord, minlength=len(arrays.chord_features))
    result = {
        "note_score_onset_seconds": np.array([note.start_s for note in score.notes]),
        "note_score_duration_seconds": score_duration,
        "note_performance_onset_seconds": np.array([note.start_s for note in aligned]),
        "note_performance_key_end_seconds": key_end,
        "note_performance_key_duration_seconds": perf_duration,
        "note_score_pitch": np.array([note.pitch for note in score.notes], dtype=np.int16),
        "note_performance_channel": np.array([note.channel for note in aligned], dtype=np.int16),
        "note_performance_index": np.asarray(perf_idx, dtype=np.int64),
        "note_interpolated": np.asarray(interpolated, dtype=bool),
        "note_to_chord": arrays.note_to_chord,
        "note_native_masks": arrays.note_masks,
        "note_log_key_duration_ratio": arrays.note_targets[:, 1],
        "note_micro_offset_ms": arrays.note_targets[:, 0],
        "note_singleton_chord": counts[arrays.note_to_chord] == 1,
        "note_score_duration_floored": score_duration * scale < 1e-4,
        "note_performance_duration_floored": perf_duration < 1e-4,
        "note_score_duration_tick_count": score_ends - score_starts,
        "note_performance_duration_tick_count": perf_ends - perf_starts,
        "note_score_tick_resolution_seconds": np.maximum(
            resolution(score_starts, score_ticks, score_tempos, score.ticks_per_beat),
            resolution(score_ends, score_ticks, score_tempos, score.ticks_per_beat)),
        "note_performance_tick_resolution_seconds": np.maximum(
            resolution(perf_starts, perf_ticks, perf_tempos, performance.ticks_per_beat),
            resolution(perf_ends, perf_ticks, perf_tempos, performance.ticks_per_beat)),
        "note_same_pitch_key_overlap": next_onset < key_end,
        "note_same_pitch_sustain_reattack": (next_onset < selected_pedal["note_sustain_end_seconds"]),
        "score_explicit_tempo_event_count": np.asarray(sum(message.type == "set_tempo"
            for track in mido.MidiFile(file=io.BytesIO(score_data)).tracks for message in track), dtype=np.int64),
        "performance_explicit_tempo_event_count": np.asarray(sum(message.type == "set_tempo"
            for track in mido.MidiFile(file=io.BytesIO(performance_data)).tracks for message in track), dtype=np.int64),
        "global_tempo_scale": np.asarray(scale),
        **selected_pedal,
    }
    return result


def reconstruct_key_duration_seconds(reference: Mapping[str, np.ndarray], predicted_log_duration: np.ndarray,
                                     *, calibration: str = "oracle_global",
                                     fixed_scale: float | None = None) -> np.ndarray:
    prediction = np.asarray(predicted_log_duration, dtype=np.float64)
    score = np.asarray(reference["note_score_duration_seconds"], dtype=np.float64)
    if prediction.shape != score.shape or not np.isfinite(prediction).all():
        raise DurationReferenceError("invalid key-duration prediction shape/values")
    if calibration == "oracle_global":
        scale = float(reference["global_tempo_scale"])
    elif calibration == "fixed_score" and fixed_scale is not None:
        scale = float(fixed_scale)
    else:
        raise DurationReferenceError("fixed_score requires explicit fixed_scale; otherwise use oracle_global")
    if not math.isfinite(scale) or scale <= 0:
        raise DurationReferenceError("duration calibration scale must be finite and positive")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.maximum(1e-4, score * scale) * np.exp(prediction)
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise DurationReferenceError("reconstructed key durations must be finite and positive")
    return result


def _errors(values: np.ndarray) -> dict[str, Any]:
    absolute = np.abs(values) * 1000
    return {"count": len(values), "mae_ms": float(absolute.mean()) if len(values) else None,
            "p95_ms": float(np.quantile(absolute, .95)) if len(values) else None,
            "max_ms": float(absolute.max()) if len(values) else None}


def evaluate_duration_prediction(reference: Mapping[str, np.ndarray], predicted_log_duration: np.ndarray | None = None,
                                 *, predicted_duration_seconds: np.ndarray | None = None,
                                 predicted_onset_seconds: np.ndarray | None = None,
                                 prediction_valid: np.ndarray | None = None,
                                 calibration: str = "oracle_global", fixed_scale: float | None = None) -> dict[str, Any]:
    """Compare raw/safe/projected key durations; key-end needs explicit onsets.

    Direct durations must already be on the selected calibration's time axis.
    Invalid/missing projected notes are excluded with explicit coverage; zeros
    are genuine released-immediately predictions, never fabricated coverage.
    No reference-human onset is silently substituted for a prediction.
    """
    if (predicted_log_duration is None) == (predicted_duration_seconds is None):
        raise DurationReferenceError("provide exactly one duration representation")
    score = reference["note_score_duration_seconds"]
    if calibration not in {"oracle_global", "fixed_score"}:
        raise DurationReferenceError("unsupported duration calibration")
    if calibration == "fixed_score" and (fixed_scale is None or not math.isfinite(fixed_scale) or fixed_scale <= 0):
        raise DurationReferenceError("fixed_score requires explicit finite positive fixed_scale")
    duration = (reconstruct_key_duration_seconds(reference, predicted_log_duration, calibration=calibration,
                                                fixed_scale=fixed_scale) if predicted_duration_seconds is None
                else np.asarray(predicted_duration_seconds, dtype=np.float64))
    valid = np.ones(len(score), dtype=bool) if prediction_valid is None else np.asarray(prediction_valid)
    if valid.shape != score.shape or valid.dtype.kind != "b":
        raise DurationReferenceError("prediction_valid must be a note-shaped boolean mask")
    if duration.shape != score.shape or not np.isfinite(duration[valid]).all() or np.any(duration[valid] < 0):
        raise DurationReferenceError("invalid physical key-duration shape/values")
    supervised = reference["note_native_masks"][:, 1].astype(bool)
    measured = supervised & valid
    duration_error = duration - reference["note_performance_key_duration_seconds"]
    key_end_error = np.empty(0)
    if predicted_onset_seconds is not None:
        onset = np.asarray(predicted_onset_seconds, dtype=np.float64)
        if onset.shape != score.shape or not np.isfinite(onset[valid]).all():
            raise DurationReferenceError("invalid explicit predicted onsets")
        key_end_error = (onset + duration - reference["note_performance_key_end_seconds"])[measured]
    return {"calibration": calibration, "uses_performance_calibration": calibration == "oracle_global",
            "deployment_evidence": False, "native_duration_count": int(supervised.sum()),
            "missing_prediction_count": int((supervised & ~valid).sum()),
            "key_duration": _errors(duration_error[measured]), "key_end": _errors(key_end_error),
            "key_end_policy": "explicit prediction onsets only; none are imputed"}


def _sample_review(reference: Mapping[str, np.ndarray]) -> dict[str, Any]:
    valid = reference["note_native_masks"][:, 1]
    micro_valid = reference["note_native_masks"][:, 0]
    singleton = reference["note_singleton_chord"] & micro_valid
    durations = reference["note_performance_key_duration_seconds"][valid]
    ratio = reference["note_log_key_duration_ratio"][valid]
    known = reference["note_sustain_end_observed"] & valid
    extension = reference["note_sustain_end_seconds"] - reference["note_performance_key_end_seconds"]
    flags = {name: int((reference[name] & valid).sum()) for name in (
        "note_score_duration_floored", "note_performance_duration_floored", "note_sustain_end_censored",
        "note_pedal_boundary_at_keyoff", "note_same_pitch_key_overlap", "note_same_pitch_sustain_reattack")}
    return {"note_count": len(valid), "native_duration_count": int(valid.sum()),
            "interpolated_count": int(reference["note_interpolated"].sum()),
            "native_singleton_micro_count": int(singleton.sum()),
            "native_singleton_nonzero_micro_count": int(np.count_nonzero(reference["note_micro_offset_ms"][singleton])),
            "sustain_known_count": int(known.sum()), "sustain_unknown_count": int((valid & ~known).sum()),
            "sustain_extended_count": int((known & (extension > 1e-9)).sum()),
            "key_duration_quantiles_ms": dict(zip(("p05", "p50", "p95"), map(float, np.quantile(durations * 1000, [.05, .5, .95])))) if len(durations) else None,
            "native_log_duration_abs_over_4_count": int((np.abs(ratio) > 4).sum()),
            "native_duration_ratio_outside_listener_bounds_0p4_1p6_count": int(((ratio < math.log(.4)) | (ratio > math.log(1.6))).sum()),
            "native_key_duration_below_40ms_count": int((durations < .04).sum()),
            "score_explicit_tempo_event_count": int(np.asarray(reference["score_explicit_tempo_event_count"]).item()),
            "performance_explicit_tempo_event_count": int(np.asarray(reference["performance_explicit_tempo_event_count"]).item()),
            "score_duration_one_tick_or_less_count": int(((reference["note_score_duration_tick_count"] <= 1) & valid).sum()),
            "performance_duration_one_tick_or_less_count": int(((reference["note_performance_duration_tick_count"] <= 1) & valid).sum()),
            "ambiguity_flags": flags,
            "native_target_roundtrip": evaluate_duration_prediction(reference, reference["note_log_key_duration_ratio"],
                predicted_onset_seconds=reference["note_performance_onset_seconds"]),
            "fixed_score_identity": evaluate_duration_prediction(reference, np.zeros(len(valid)),
                predicted_onset_seconds=reference["note_score_onset_seconds"], calibration="fixed_score", fixed_scale=1.0)}


def audit_duration_reference(*, timing_reference: str | Path, output_dir: str | Path) -> dict[str, Any]:
    root, output = Path(timing_reference).resolve(), Path(output_dir).resolve()
    implementation_hash = _sha256(Path(__file__))
    if root.is_file():
        root = root.parent
    references = load_timing_reference(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    dataset, wanted = Path(summary["dataset_root"]), {row["id"]: row for row in summary["samples"]}
    rows = []
    with (dataset / "manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] in wanted:
                rows.append(row)
    if len(rows) != len(wanted):
        raise DurationReferenceError("selected original source rows are missing")
    if output.exists() and any(output.iterdir()):
        raise DurationReferenceError("output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    (output / "_INCOMPLETE").write_text("duration reference audit in progress\n", encoding="utf-8")
    index, loaded_path, shard = [], None, None
    with RefinedArchive(summary["archive_path"]) as archive:
        for position, row in enumerate(sorted(rows, key=lambda value: (value["shard_path"], value["sample_index"]))):
            source = {}
            for prefix in ("score_midi", "performance_midi", "alignment"):
                source[prefix] = archive.read(row[f"{prefix}_path"])
                if hashlib.sha256(source[prefix]).hexdigest() != row[f"{prefix}_sha256"]:
                    raise DurationReferenceError(f"original member hash mismatch: {row['id']} {prefix}")
            perf_idx, interpolated = _load_alignment(source["alignment"])
            reference = make_duration_reference(source["score_midi"], source["performance_midi"],
                                                perf_idx, interpolated, references[row["id"]])
            if loaded_path != row["shard_path"]:
                loaded_path = row["shard_path"]
                with np.load(_safe_path(dataset, loaded_path), allow_pickle=False) as values:
                    shard = {name: values[name] for name in values.files}
            at = int(row["sample_index"])
            n0, n1 = map(int, shard["sample_note_offsets"][at:at + 2])
            c0, c1 = map(int, shard["sample_chord_offsets"][at:at + 2])
            expected = wanted[row["id"]]
            if str(shard["sample_ids"][at]) != row["id"] or (n0, c0, n1 - n0, c1 - c0) != (
                expected["note_offset"], expected["chord_offset"], expected["note_count"], expected["chord_count"]):
                raise DurationReferenceError("original sample indices/offsets differ")
            for first, second in ((reference["note_native_masks"], shard["note_masks"][n0:n1]),
                                  (reference["note_to_chord"], shard["note_to_chord"][n0:n1] - c0),
                                  (reference["note_log_key_duration_ratio"], shard["note_targets"][n0:n1, 1])):
                if not np.array_equal(first, second):
                    raise DurationReferenceError("original shard note labels/native masks differ")
            relative = f"samples/{hashlib.sha256(row['id'].encode()).hexdigest()}.npz"
            digest = write_deterministic_npz(output / relative, reference)
            index.append({"id": row["id"], "split": row["split"], "performance_dataset": row["performance_dataset"],
                "piece_group": row["piece_group"], "path": relative, "sha256": digest,
                "source_row_sha256": hashlib.sha256(_canonical(row)).hexdigest(),
                "source_members": {prefix: {"path": row[f"{prefix}_path"], "sha256": row[f"{prefix}_sha256"]}
                                   for prefix in ("score_midi", "performance_midi", "alignment")},
                "review": _sample_review(reference)})
            if (position + 1) % 25 == 0 or position + 1 == len(rows):
                print(f"duration_reference={position + 1}/{len(rows)}", flush=True)
    if _sha256(Path(__file__)) != implementation_hash:
        raise DurationReferenceError("implementation changed during duration audit; incomplete output must not be used")
    report = {"schema_version": 1, "kind": "pianocore_duration_reference", "status": "complete",
        "timing_reference_root": str(root), "timing_reference_summary_sha256": _sha256(root / "summary.json"),
        "sample_count": len(index), "split_counts": dict(Counter(row["split"] for row in index)),
        "implementation_sha256": implementation_hash, "samples": sorted(index, key=lambda row: row["id"]),
        "policy": "native original masks unchanged; duration floors retained; no automatic filtering or retraining",
        "sustain_policy": "channel CC64>=64 at keyoff extends to next CC64<64; absent pedal or missing release is unknown/censored; not acoustic decay",
        "calibration_policy": "oracle_global is diagnostic only; fixed_score explicitly 1.0; no performance values are model inputs",
        "key_end_policy": "native target roundtrip uses explicit true onsets solely to isolate duration-label accuracy; fixed identity uses score onsets",
        "ambiguity_policy": "same-pitch overlap, pedal-boundary ties, floors and short notes are indicators, not proven label errors",
        "deployment_evidence": False}
    (output / "summary.json").write_bytes(_canonical(report) + b"\n")
    (output / "_SUCCESS").write_text(_sha256(output / "summary.json") + "\n", encoding="ascii")
    (output / "_INCOMPLETE").unlink()
    return report


def load_duration_reference(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    root = Path(path).resolve()
    if root.is_file():
        root = root.parent
    summary_path = root / "summary.json"
    if (root / "_INCOMPLETE").exists() or not (root / "_SUCCESS").is_file() or (root / "_SUCCESS").read_text().strip() != _sha256(summary_path):
        raise DurationReferenceError("duration reference incomplete or summary hash mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != 1 or summary.get("status") != "complete" or summary.get("kind") != "pianocore_duration_reference":
        raise DurationReferenceError("unsupported duration reference schema/status")
    timing_root = Path(summary["timing_reference_root"])
    if _sha256(timing_root / "summary.json") != summary["timing_reference_summary_sha256"]:
        raise DurationReferenceError("bound timing reference summary changed")
    timing = load_timing_reference(timing_root)
    if (len(summary["samples"]) != summary["sample_count"] or len(summary["samples"]) != len(timing)
            or set(timing) != {row["id"] for row in summary["samples"]}):
        raise DurationReferenceError("duration sample set/count differs from timing source")
    result = {}
    for row in summary["samples"]:
        sample_path = _safe_path(root, row["path"])
        if _sha256(sample_path) != row["sha256"]:
            raise DurationReferenceError("duration sample hash mismatch")
        with np.load(sample_path, allow_pickle=False) as values:
            reference = {name: values[name] for name in values.files}
        reference["global_tempo_scale"] = reference["global_tempo_scale"].reshape(())
        if not np.array_equal(reference["note_native_masks"], timing[row["id"]]["note_native_masks"]):
            raise DurationReferenceError("duration native masks differ from timing source")
        result[row["id"]] = reference
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit_duration_reference(timing_reference=args.timing_reference, output_dir=args.output_dir)
    print(json.dumps({key: result[key] for key in ("status", "sample_count", "split_counts")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
